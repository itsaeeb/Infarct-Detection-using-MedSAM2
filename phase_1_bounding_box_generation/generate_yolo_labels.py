"""
generate_yolo_labels.py
=======================
Stage 1 of the APIS -> YOLO pipeline.

Reads every (ncct, r1_mask) pair, extracts axial slices, windows the CT,
detects infarct regions, generates one YOLO bounding box per connected
component, and writes the organised dataset to disk.

Usage:
    python generate_yolo_labels.py

Outputs (all under OUTPUT_ROOT):
    yolo_dataset/
        images/train/  images/val/  images/test/
        labels/train/  labels/val/  labels/test/
        dataset.yaml

Design decisions (based on inspect_nifti.py results):
  - Mask label 1 = infarct  (only non-zero value found)
  - Multi-box mode: one YOLO box per connected component (up to 6 per slice)
  - CT window: brain window  WL=40 HU, WW=80 HU  ->  [0, 80] HU range
    (configurable via WINDOW_LEVEL / WINDOW_WIDTH)
  - Images saved as 8-bit PNG (uint8, range 0-255)
  - Patient-level train/val/test split (no data leakage)
  - train_019 has an affine mismatch; it is included but flagged in the log
  - Cases with no lesion (train_027 etc.) still produce negative-example slices
"""

import os
import sys
import glob
import json
import random
import logging
import numpy as np

try:
    import nibabel as nib
except ImportError:
    sys.exit("nibabel not installed. Run:  pip install nibabel")

try:
    import cv2
except ImportError:
    sys.exit("opencv-python not installed. Run:  pip install opencv-python")

try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARNING] scipy not found -- single-box mode will be used as fallback.")

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

candidate_dataset = [
    os.path.join(PROJECT_ROOT, "APIS_Dataset"),
    os.path.join(SCRIPT_DIR, "APIS_Dataset"),
]
DATASET_ROOT = next((p for p in candidate_dataset if os.path.isdir(p)), candidate_dataset[0])
OUTPUT_ROOT  = os.path.join(PROJECT_ROOT, "phase_2_yolo26", "dataset")

# CT brain window  (edit these to change the intensity window)
WINDOW_LEVEL = 40    # HU  (centre)
WINDOW_WIDTH = 80    # HU  (full width)
# Resulting HU range: [WL - WW/2, WL + WW/2] = [0, 80]

# Output image size (pixels).  None = keep original per-volume size.
# Set to e.g. (640, 640) to resize all images to a fixed square.
OUTPUT_SIZE = None   # (width, height) or None

# Dataset split ratios (must sum to 1.0)
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# Reproducibility
RANDOM_SEED = 42

# Minimum bounding-box area in pixels to keep (filters out 1-pixel artefacts)
MIN_BOX_AREA_PX = 4

# YOLO class id for infarct
CLASS_ID = 0

# =============================================================================


def setup_logging(output_root):
    """Configure console + file logging."""
    log_path = os.path.join(output_root, "generation.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s  %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        ],
    )
    return log_path


def find_file(directory, *patterns):
    """Return first matching *file* for any pattern, or None."""
    for pat in patterns:
        hits = [m for m in glob.glob(os.path.join(directory, pat))
                if os.path.isfile(m)]
        if hits:
            return hits[0]
    return None


def collect_pairs(dataset_root):
    """Return sorted list of (pid, ncct_path, mask_path)."""
    pairs   = []
    skipped = []
    patient_dirs = sorted([
        d for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d)) and d.startswith("train_")
    ])
    for pid in patient_dirs:
        pdir = os.path.join(dataset_root, pid)
        mdir = os.path.join(pdir, "masks")
        ncct = find_file(pdir, f"{pid}_ncct.nii.gz", f"{pid}_ncct.nii")
        mask = find_file(mdir, f"{pid}_r1_mask.nii.gz", f"{pid}_r1_mask.nii")
        if ncct is None or mask is None:
            reason = []
            if ncct is None: reason.append("ncct missing")
            if mask is None: reason.append("mask missing")
            skipped.append((pid, "; ".join(reason)))
        else:
            pairs.append((pid, ncct, mask))
    return pairs, skipped


def split_patients(pids, train_r, val_r, test_r, seed):
    """Patient-level split -- all slices of a patient stay together."""
    rng = random.Random(seed)
    shuffled = list(pids)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(n * train_r))
    n_val   = int(round(n * val_r))
    train = shuffled[:n_train]
    val   = shuffled[n_train:n_train + n_val]
    test  = shuffled[n_train + n_val:]
    return set(train), set(val), set(test)


def window_ct(ct_data, window_level, window_width):
    """Apply a CT HU window and return a float32 array in [0, 1]."""
    lo = window_level - window_width / 2.0
    hi = window_level + window_width / 2.0
    clipped = np.clip(ct_data, lo, hi)
    return (clipped - lo) / (hi - lo)   # normalised float [0,1]


def slice_to_uint8(slice_2d):
    """Convert a float [0,1] 2D slice to uint8 [0,255]."""
    return (slice_2d * 255).astype(np.uint8)


def get_boxes_from_mask_slice(mask_slice, infarct_label=1):
    """
    Find bounding boxes for each connected infarct component in a 2D mask.

    Returns a list of (row_min, col_min, row_max, col_max) tuples in pixel
    coordinates (0-indexed).  Empty list if no lesion is present.
    """
    binary = (mask_slice == infarct_label)
    if not binary.any():
        return []

    if HAS_SCIPY:
        labeled, n_components = ndimage.label(binary)
    else:
        # Fallback: treat the whole lesion as one component
        labeled, n_components = binary.astype(int), 1

    boxes = []
    for comp_id in range(1, n_components + 1):
        comp = (labeled == comp_id)
        rows = np.where(comp.any(axis=1))[0]
        cols = np.where(comp.any(axis=0))[0]
        r_min, r_max = int(rows.min()), int(rows.max())
        c_min, c_max = int(cols.min()), int(cols.max())
        area = (r_max - r_min + 1) * (c_max - c_min + 1)
        if area >= MIN_BOX_AREA_PX:
            boxes.append((r_min, c_min, r_max, c_max))
    return boxes


def box_to_yolo(r_min, c_min, r_max, c_max, h, w):
    """
    Convert pixel box (row-min, col-min, row-max, col-max) to YOLO format.

    YOLO expects: x_center y_center width height  (all normalised by image W/H)
    Note: 'x' corresponds to columns, 'y' to rows.
    """
    x_center = ((c_min + c_max) / 2.0) / w
    y_center = ((r_min + r_max) / 2.0) / h
    box_w    = (c_max - c_min + 1)    / w
    box_h    = (r_max - r_min + 1)    / h

    # Clamp to [0, 1] just in case of edge touching
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    box_w    = max(0.0, min(1.0, box_w))
    box_h    = max(0.0, min(1.0, box_h))
    return x_center, y_center, box_w, box_h


def save_image(slice_uint8, out_path, output_size):
    """Save a 2D uint8 array as a grayscale PNG, optionally resizing."""
    img = slice_uint8
    if output_size is not None:
        img = cv2.resize(img, output_size, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(out_path, img)


def save_label(yolo_lines, out_path):
    """Write YOLO label lines to a .txt file (empty file for negatives)."""
    with open(out_path, "w") as f:
        if yolo_lines:
            f.write("\n".join(yolo_lines) + "\n")


def make_dataset_yaml(output_root, class_names):
    """Write the YOLO dataset.yaml configuration file."""
    # Use forward slashes for YOLO compatibility
    abs_root = output_root.replace("\\", "/")
    yaml_path = os.path.join(output_root, "dataset.yaml")
    lines = [
        f"path: {abs_root}",
        f"train: images/train",
        f"val:   images/val",
        f"test:  images/test",
        "",
        f"nc: {len(class_names)}",
        f"names: {class_names}",
        "",
        "# Generated by generate_yolo_labels.py",
        "# CT brain window: WL=40 HU, WW=80 HU",
        "# One box per connected infarct component per axial slice.",
    ]
    with open(yaml_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return yaml_path


def process_volume(pid, ncct_path, mask_path, split,
                   output_root, stats):
    """
    Process one (ncct, mask) pair.

    Extracts axial slices, applies brain window, computes bounding boxes,
    and saves images + labels.

    Returns nothing; mutates 'stats' dict in-place.
    """
    img_dir = os.path.join(output_root, "images", split)
    lbl_dir = os.path.join(output_root, "labels", split)

    try:
        ct_img   = nib.load(ncct_path)
        mask_img = nib.load(mask_path)
    except Exception as e:
        logging.warning(f"[SKIP] {pid}: could not load NIfTI -- {e}")
        stats["skipped"].append((pid, str(e)))
        return

    ct_data   = ct_img.get_fdata(dtype=np.float32)
    mask_data = mask_img.get_fdata(dtype=np.float32)

    # Shape check
    if ct_data.shape != mask_data.shape:
        msg = (f"shape mismatch CT={ct_data.shape} vs MASK={mask_data.shape}."
               " Affine mismatch -- including but flagging.")
        logging.warning(f"[FLAG] {pid}: {msg}")
        stats["flagged"].append((pid, msg))
        # Still proceed -- shapes might match even if affines differ slightly

    # Affine check (warn only, do not skip)
    if not np.allclose(ct_img.affine, mask_img.affine, atol=1e-3):
        logging.warning(f"[FLAG] {pid}: affine matrices differ -- verify alignment manually.")

    # Round mask to integer labels
    mask_int = np.round(mask_data).astype(np.uint8)

    # Apply brain CT window
    ct_windowed = window_ct(ct_data, WINDOW_LEVEL, WINDOW_WIDTH)

    n_slices       = ct_data.shape[2]
    pos_slices     = 0
    neg_slices     = 0
    boxes_in_vol   = 0

    for z in range(n_slices):
        ct_slice   = ct_windowed[:, :, z]     # shape (H, W)
        mask_slice = mask_int[:, :, z]        # shape (H, W)

        H, W = ct_slice.shape

        # File name: patientid_slice{z:03d}
        basename = f"{pid}_slice{z:03d}"
        img_path = os.path.join(img_dir, basename + ".png")
        lbl_path = os.path.join(lbl_dir, basename + ".txt")

        # Save image
        img_uint8 = slice_to_uint8(ct_slice)
        save_image(img_uint8, img_path, OUTPUT_SIZE)

        # Compute bounding boxes
        boxes_px = get_boxes_from_mask_slice(mask_slice, infarct_label=1)

        yolo_lines = []
        for (r_min, c_min, r_max, c_max) in boxes_px:
            # If output was resized, YOLO normalisation still works because
            # we normalise by original H,W then the boxes are correct for
            # the resized image too (linear scaling preserves ratios).
            xc, yc, bw, bh = box_to_yolo(r_min, c_min, r_max, c_max, H, W)
            yolo_lines.append(f"{CLASS_ID} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        save_label(yolo_lines, lbl_path)

        if boxes_px:
            pos_slices   += 1
            boxes_in_vol += len(boxes_px)
        else:
            neg_slices += 1

    stats["patients"][pid] = {
        "split"        : split,
        "n_slices"     : n_slices,
        "pos_slices"   : pos_slices,
        "neg_slices"   : neg_slices,
        "n_boxes"      : boxes_in_vol,
    }
    logging.info(f"  {pid} [{split:5s}]: {n_slices} slices | "
                 f"{pos_slices} pos | {neg_slices} neg | {boxes_in_vol} boxes")


def main():
    # ── Output directory setup ────────────────────────────────────────────
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(OUTPUT_ROOT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_ROOT, "labels", split), exist_ok=True)

    log_path = setup_logging(OUTPUT_ROOT)
    logging.info("=" * 60)
    logging.info("  YOLO Label Generator -- APIS Dataset")
    logging.info("=" * 60)
    logging.info(f"  Dataset root : {DATASET_ROOT}")
    logging.info(f"  Output root  : {OUTPUT_ROOT}")
    logging.info(f"  CT window    : WL={WINDOW_LEVEL} HU  WW={WINDOW_WIDTH} HU")
    logging.info(f"  Output size  : {OUTPUT_SIZE if OUTPUT_SIZE else 'original'}")
    logging.info(f"  Split ratios : train={TRAIN_RATIO}  val={VAL_RATIO}  test={TEST_RATIO}")
    logging.info(f"  Random seed  : {RANDOM_SEED}")
    logging.info("")

    # ── Collect pairs ─────────────────────────────────────────────────────
    pairs, skipped_missing = collect_pairs(DATASET_ROOT)
    if skipped_missing:
        for pid, reason in skipped_missing:
            logging.warning(f"[SKIP] {pid}: {reason}")

    pids = [p[0] for p in pairs]
    train_set, val_set, test_set = split_patients(
        pids, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED)

    logging.info(f"  Total patients : {len(pids)}")
    logging.info(f"  Train          : {len(train_set)}")
    logging.info(f"  Val            : {len(val_set)}")
    logging.info(f"  Test           : {len(test_set)}")
    logging.info("")

    # ── Process each volume ───────────────────────────────────────────────
    stats = {
        "patients" : {},
        "skipped"  : list(skipped_missing),
        "flagged"  : [],
    }

    for pid, ncct_path, mask_path in pairs:
        if pid in train_set:
            split = "train"
        elif pid in val_set:
            split = "val"
        else:
            split = "test"
        process_volume(pid, ncct_path, mask_path, split, OUTPUT_ROOT, stats)

    # ── Write dataset.yaml ────────────────────────────────────────────────
    yaml_path = make_dataset_yaml(OUTPUT_ROOT, ["infarct"])
    logging.info(f"\n  Wrote dataset config: {yaml_path}")

    # ── Aggregate summary ─────────────────────────────────────────────────
    total_slices     = sum(v["n_slices"]   for v in stats["patients"].values())
    total_pos        = sum(v["pos_slices"] for v in stats["patients"].values())
    total_neg        = sum(v["neg_slices"] for v in stats["patients"].values())
    total_boxes      = sum(v["n_boxes"]    for v in stats["patients"].values())

    logging.info("")
    logging.info("=" * 60)
    logging.info("  GENERATION SUMMARY")
    logging.info("=" * 60)
    logging.info(f"  Patients processed : {len(stats['patients'])}")
    logging.info(f"  Patients skipped   : {len(stats['skipped'])}")
    logging.info(f"  Patients flagged   : {len(stats['flagged'])}")
    logging.info(f"  Total slices       : {total_slices}")
    logging.info(f"  Positive slices    : {total_pos}  ({100*total_pos/max(total_slices,1):.1f}%)")
    logging.info(f"  Negative slices    : {total_neg}  ({100*total_neg/max(total_slices,1):.1f}%)")
    logging.info(f"  Total boxes        : {total_boxes}")

    for split in ("train", "val", "test"):
        ps = {k: v for k, v in stats["patients"].items() if v["split"] == split}
        sl = sum(v["n_slices"] for v in ps.values())
        po = sum(v["pos_slices"] for v in ps.values())
        bx = sum(v["n_boxes"] for v in ps.values())
        logging.info(f"  [{split:5s}]  {len(ps):2d} patients | {sl:4d} slices | {po:4d} pos | {bx:4d} boxes")

    # Save JSON stats for the validation script
    stats_path = os.path.join(OUTPUT_ROOT, "generation_stats.json")
    with open(stats_path, "w") as f:
        # Convert sets to lists for JSON serialisation
        out = {
            "patients" : stats["patients"],
            "skipped"  : stats["skipped"],
            "flagged"  : stats["flagged"],
            "config"   : {
                "window_level" : WINDOW_LEVEL,
                "window_width" : WINDOW_WIDTH,
                "output_size"  : OUTPUT_SIZE,
                "train_ratio"  : TRAIN_RATIO,
                "val_ratio"    : VAL_RATIO,
                "test_ratio"   : TEST_RATIO,
                "random_seed"  : RANDOM_SEED,
                "class_id"     : CLASS_ID,
                "class_name"   : "infarct",
                "infarct_label": 1,
            }
        }
        json.dump(out, f, indent=2)
    logging.info(f"\n  Stats saved to: {stats_path}")
    logging.info(f"  Log saved to  : {log_path}")
    logging.info("\n  Done. Run validate_boxes.py to visualise and verify labels.\n")


if __name__ == "__main__":
    main()
