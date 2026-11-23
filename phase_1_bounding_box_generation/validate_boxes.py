"""
validate_boxes.py
=================
Step 3 of the APIS -> YOLO pipeline: visual validation.

Reads generated images and labels from yolo_dataset/, overlays bounding
boxes, and saves annotated images for manual inspection.

Usage:
    python validate_boxes.py

Outputs (under VALIDATION_DIR):
    validate/
        positive/   -- slices that have at least one infarct box (annotated)
        negative/   -- a sample of negative (no-lesion) slices
        summary.txt -- overall statistics report

Requirements:
    pip install opencv-python numpy
"""

import os
import sys
import json
import random
import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("opencv-python not installed. Run:  pip install opencv-python")

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT    = os.path.dirname(SCRIPT_DIR)

candidate_yolo = [
    os.path.join(PROJECT_ROOT, "phase_2_yolo26", "dataset"),
    os.path.join(PROJECT_ROOT, "yolo_dataset"),
    os.path.join(SCRIPT_DIR, "yolo_dataset"),
]
YOLO_ROOT       = next((p for p in candidate_yolo if os.path.isdir(p)), candidate_yolo[0])
VALIDATION_DIR  = os.path.join(SCRIPT_DIR, "validation_output")

# How many negative-slice examples to save (set 0 to skip)
N_NEG_SAMPLES   = 20

# Box colour and thickness for visualisation
BOX_COLOR       = (0, 255, 0)   # green  (BGR)
BOX_THICKNESS   = 2

# =============================================================================


def parse_yolo_label(label_path, img_w, img_h):
    """
    Parse a YOLO .txt label file.

    Returns a list of (class_id, x_center, y_center, bw, bh) in pixels.
    Each value is already de-normalised (pixel coordinates).
    """
    boxes = []
    if not os.path.isfile(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            cls, xc, yc, bw, bh = int(parts[0]), *[float(x) for x in parts[1:]]
            # De-normalise
            xc_px = xc * img_w
            yc_px = yc * img_h
            bw_px = bw * img_w
            bh_px = bh * img_h
            x1 = int(xc_px - bw_px / 2)
            y1 = int(yc_px - bh_px / 2)
            x2 = int(xc_px + bw_px / 2)
            y2 = int(yc_px + bh_px / 2)
            # Clamp
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)
            boxes.append((cls, x1, y1, x2, y2))
    return boxes


def draw_boxes(img_bgr, boxes, color, thickness):
    """Draw bounding boxes on a copy of the image."""
    out = img_bgr.copy()
    for (cls, x1, y1, x2, y2) in boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(out, f"infarct", (x1, max(y1 - 4, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return out


def validate_boxes_on_image(img_path, label_path):
    """
    Check that all boxes in the label are within image bounds and non-degenerate.

    Returns (is_ok, list_of_warnings).
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False, ["Could not read image"]
    h, w = img.shape
    warnings = []
    boxes = parse_yolo_label(label_path, w, h)
    for (cls, x1, y1, x2, y2) in boxes:
        if x1 >= x2 or y1 >= y2:
            warnings.append(f"  Degenerate box: ({x1},{y1})-({x2},{y2})")
        if x1 < 0 or y1 < 0 or x2 >= w or y2 >= h:
            warnings.append(f"  Box out of bounds: ({x1},{y1})-({x2},{y2}) vs img {w}x{h}")
    return len(warnings) == 0, warnings


def collect_images_and_labels(yolo_root, splits=("train", "val", "test")):
    """
    Walk all splits and return:
      positive_pairs : list of (img_path, label_path) where label is non-empty
      negative_pairs : list of (img_path, label_path) where label is empty
    """
    positive = []
    negative = []
    for split in splits:
        img_dir = os.path.join(yolo_root, "images", split)
        lbl_dir = os.path.join(yolo_root, "labels", split)
        if not os.path.isdir(img_dir):
            continue
        for fname in sorted(os.listdir(img_dir)):
            if not fname.endswith(".png"):
                continue
            img_path = os.path.join(img_dir, fname)
            lbl_path = os.path.join(lbl_dir, fname.replace(".png", ".txt"))
            # Check if label is non-empty
            has_box = False
            if os.path.isfile(lbl_path) and os.path.getsize(lbl_path) > 0:
                with open(lbl_path) as f:
                    content = f.read().strip()
                has_box = len(content) > 0
            if has_box:
                positive.append((img_path, lbl_path))
            else:
                negative.append((img_path, lbl_path))
    return positive, negative


def main():
    os.makedirs(os.path.join(VALIDATION_DIR, "positive"), exist_ok=True)
    os.makedirs(os.path.join(VALIDATION_DIR, "negative"), exist_ok=True)

    print(f"\n{'='*60}")
    print("  YOLO Bounding Box Validator")
    print(f"{'='*60}")
    print(f"  YOLO dataset : {YOLO_ROOT}")
    print(f"  Output dir   : {VALIDATION_DIR}")

    if not os.path.isdir(YOLO_ROOT):
        sys.exit(f"\n[ERROR] yolo_dataset not found at {YOLO_ROOT}.\n"
                 "  Run generate_yolo_labels.py first.")

    positive, negative = collect_images_and_labels(YOLO_ROOT)
    print(f"\n  Positive slices (have boxes) : {len(positive)}")
    print(f"  Negative slices (no boxes)   : {len(negative)}")
    print()

    # -- Validate and visualise all positive slices --
    n_ok       = 0
    n_warn     = 0
    all_issues = []

    for img_path, lbl_path in positive:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            all_issues.append((img_path, ["Could not read image"]))
            continue
        h, w = img.shape
        boxes = parse_yolo_label(lbl_path, w, h)

        # Geometry validation
        is_ok, warns = validate_boxes_on_image(img_path, lbl_path)
        if not is_ok:
            n_warn += 1
            all_issues.append((img_path, warns))
        else:
            n_ok += 1

        # Draw and save
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        annotated = draw_boxes(img_bgr, boxes, BOX_COLOR, BOX_THICKNESS)
        out_name  = os.path.basename(img_path)
        out_path  = os.path.join(VALIDATION_DIR, "positive", out_name)
        cv2.imwrite(out_path, annotated)

    # -- Save a random sample of negatives for inspection --
    if N_NEG_SAMPLES > 0 and negative:
        rng = random.Random(42)
        sample_neg = rng.sample(negative, min(N_NEG_SAMPLES, len(negative)))
        for img_path, _ in sample_neg:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            # Draw a "NEG" label on the image
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.putText(img_bgr, "NEG", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            out_name = os.path.basename(img_path)
            cv2.imwrite(os.path.join(VALIDATION_DIR, "negative", out_name), img_bgr)

    # -- Load generation stats if available --
    stats_path = os.path.join(YOLO_ROOT, "generation_stats.json")
    gen_stats  = None
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            gen_stats = json.load(f)

    # -- Write summary report --
    summary_path = os.path.join(VALIDATION_DIR, "summary.txt")
    lines = []
    lines.append("=" * 60)
    lines.append("  YOLO Dataset Validation Report")
    lines.append("=" * 60)
    lines.append(f"  Positive slices inspected : {len(positive)}")
    lines.append(f"  Negative slices total     : {len(negative)}")
    lines.append(f"  Boxes OK (in bounds)      : {n_ok}")
    lines.append(f"  Slices with warnings      : {n_warn}")
    lines.append("")

    if gen_stats:
        cfg = gen_stats.get("config", {})
        lines.append("  Generation config:")
        lines.append(f"    CT window : WL={cfg.get('window_level')} HU, WW={cfg.get('window_width')} HU")
        lines.append(f"    Output sz : {cfg.get('output_size', 'original')}")
        lines.append(f"    Seed      : {cfg.get('random_seed')}")
        lines.append("")
        lines.append("  Per-split summary:")
        for split in ("train", "val", "test"):
            ps = {k: v for k, v in gen_stats["patients"].items()
                  if v.get("split") == split}
            sl = sum(v["n_slices"]   for v in ps.values())
            po = sum(v["pos_slices"] for v in ps.values())
            ne = sum(v["neg_slices"] for v in ps.values())
            bx = sum(v["n_boxes"]    for v in ps.values())
            lines.append(f"    [{split:5s}]  patients={len(ps)}  "
                         f"slices={sl}  pos={po}  neg={ne}  boxes={bx}")
        lines.append("")

        flagged = gen_stats.get("flagged", [])
        skipped = gen_stats.get("skipped", [])
        lines.append(f"  Flagged volumes (affine/shape issues) : {len(flagged)}")
        for pid, msg in flagged:
            lines.append(f"    {pid}: {msg}")
        lines.append(f"  Skipped volumes (missing files)       : {len(skipped)}")
        for pid, msg in skipped:
            lines.append(f"    {pid}: {msg}")
        lines.append("")

    if all_issues:
        lines.append(f"  Geometry issues ({len(all_issues)} images):")
        for img_path, warns in all_issues:
            lines.append(f"    {os.path.basename(img_path)}")
            for w in warns:
                lines.append(f"      {w}")
    else:
        lines.append("  No geometry issues found -- all boxes are valid.")

    lines.append("")
    lines.append("=" * 60)
    report = "\n".join(lines)
    print(report)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")

    print(f"\n  Annotated images saved to : {VALIDATION_DIR}")
    print(f"  Summary report saved to   : {summary_path}\n")


if __name__ == "__main__":
    main()
