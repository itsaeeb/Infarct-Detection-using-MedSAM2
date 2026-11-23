"""
inspect_nifti.py
================
Step 1: Inspect the APIS NIfTI dataset before generating YOLO labels.

This script is READ-ONLY -- it never writes to the dataset.

Run:
    python inspect_nifti.py

What it does:
  - Finds every (ncct, mask) pair in the APIS_Dataset directory.
  - Prints shape, dtype, voxel spacing, and unique mask values for a
    representative sample of cases (configurable via NUM_SAMPLE_CASES).
  - Checks whether CT and mask are spatially compatible (same H x W x D).
  - Reports unique mask label values across ALL cases so we know which
    integer values represent infarct.
  - Counts connected components per positive slice to advise on
    single-box vs. multi-box strategy.
  - Reports any missing files or pairing problems.
"""

import os
import sys
import glob
import numpy as np

try:
    import nibabel as nib
except ImportError:
    sys.exit("nibabel is not installed. Run:  pip install nibabel")

try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARNING] scipy not found -- connected-component counting skipped.")

# =============================================================================
# CONFIGURATION  (edit these if your layout differs)
SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT     = os.path.dirname(SCRIPT_DIR)
candidate_roots  = [
    os.path.join(PROJECT_ROOT, "APIS_Dataset"),
    os.path.join(SCRIPT_DIR, "APIS_Dataset"),
]
DATASET_ROOT     = next((p for p in candidate_roots if os.path.isdir(p)), candidate_roots[0])
NUM_SAMPLE_CASES = 5   # how many cases to print detailed info for
# =============================================================================


def find_file(directory, *patterns):
    """Return the first *file* in 'directory' matching any glob pattern."""
    for pat in patterns:
        matches = [m for m in glob.glob(os.path.join(directory, pat))
                   if os.path.isfile(m)]
        if matches:
            return matches[0]
    return None


def collect_pairs(dataset_root):
    """Build a list of (patient_id, ncct_path, mask_path) tuples."""
    pairs   = []
    missing = []

    patient_dirs = sorted([
        d for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d)) and d.startswith("train_")
    ])

    for pid in patient_dirs:
        pdir = os.path.join(dataset_root, pid)
        mdir = os.path.join(pdir, "masks")

        ncct = find_file(pdir, f"{pid}_ncct.nii.gz", f"{pid}_ncct.nii")
        mask = find_file(mdir, f"{pid}_r1_mask.nii.gz", f"{pid}_r1_mask.nii")

        if ncct is None:
            missing.append((pid, "ncct missing"))
        if mask is None:
            missing.append((pid, "mask missing"))
        if ncct and mask:
            pairs.append((pid, ncct, mask))

    return pairs, missing, patient_dirs


def voxel_spacing(img):
    try:
        return img.header.get_zooms()
    except Exception:
        return None


def inspect_case(pid, ncct_path, mask_path, verbose=True):
    """Load one case and return an info dict. Prints details if verbose=True."""
    info = {"pid": pid, "ok": False}

    try:
        ct_img   = nib.load(ncct_path)
        mask_img = nib.load(mask_path)
    except Exception as e:
        info["error"] = str(e)
        if verbose:
            print(f"  [ERROR] Could not load {pid}: {e}")
        return info

    ct_data   = ct_img.get_fdata(dtype=np.float32)
    mask_data = mask_img.get_fdata(dtype=np.float32)

    ct_shape   = ct_data.shape
    mask_shape = mask_data.shape

    unique_vals  = np.unique(np.round(mask_data).astype(int))
    ct_min       = float(ct_data.min())
    ct_max       = float(ct_data.max())
    ct_mean      = float(ct_data.mean())
    spatially_ok = (ct_shape == mask_shape)
    affine_close = np.allclose(ct_img.affine, mask_img.affine, atol=1e-3)
    ct_zooms     = voxel_spacing(ct_img)
    mask_zooms   = voxel_spacing(mask_img)

    info.update({
        "ok"           : True,
        "ct_shape"     : ct_shape,
        "mask_shape"   : mask_shape,
        "spatially_ok" : spatially_ok,
        "affine_close" : affine_close,
        "ct_dtype"     : str(ct_img.header.get_data_dtype()),
        "mask_dtype"   : str(mask_img.header.get_data_dtype()),
        "unique_vals"  : unique_vals.tolist(),
        "ct_min"       : ct_min,
        "ct_max"       : ct_max,
        "ct_mean"      : ct_mean,
        "ct_zooms"     : ct_zooms,
        "mask_zooms"   : mask_zooms,
        "has_lesion"   : bool((unique_vals != 0).any()),
    })

    if verbose:
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"  Patient : {pid}")
        print(f"  NCCT    : {os.path.basename(ncct_path)}")
        print(f"  Mask    : {os.path.basename(mask_path)}")
        print(f"  CT  shape / dtype : {ct_shape}  /  {info['ct_dtype']}")
        print(f"  Mask shape / dtype: {mask_shape}  /  {info['mask_dtype']}")
        print(f"  CT  HU range      : [{ct_min:.1f}, {ct_max:.1f}]  mean={ct_mean:.1f}")
        print(f"  CT  voxel spacing : {ct_zooms}")
        print(f"  Mask voxel spacing: {mask_zooms}")
        print(f"  Spatial match     : {'YES' if spatially_ok else 'NO <-- MISMATCH!'}")
        print(f"  Affine match      : {'YES' if affine_close else 'NO <-- MISMATCH!'}")
        print(f"  Mask unique vals  : {unique_vals.tolist()}")
        print(f"  Has lesion        : {info['has_lesion']}")

    return info


def count_components_per_slice(mask_data, infarct_vals):
    """For each axial slice, count the number of connected lesion components."""
    binary   = np.isin(np.round(mask_data).astype(int), infarct_vals)
    n_slices = mask_data.shape[2]
    comp_counts = []
    for z in range(n_slices):
        sl = binary[:, :, z]
        if sl.any():
            if HAS_SCIPY:
                _, n = ndimage.label(sl)
                comp_counts.append(n)
            else:
                comp_counts.append(1)  # placeholder
    return comp_counts


def main():
    sep = "=" * 60
    print(f"\n{sep}")
    print("  APIS NIfTI Dataset Inspector")
    print(sep)
    print(f"  Dataset root: {DATASET_ROOT}")

    if not os.path.isdir(DATASET_ROOT):
        sys.exit(f"\n[ERROR] Dataset root not found:\n  {DATASET_ROOT}\n"
                 "  Edit DATASET_ROOT at the top of this script.")

    pairs, missing, all_pids = collect_pairs(DATASET_ROOT)
    print(f"\n  Total patient directories : {len(all_pids)}")
    print(f"  Complete (ncct+mask) pairs: {len(pairs)}")
    print(f"  Problems found            : {len(missing)}")
    for pid, reason in missing:
        print(f"    [MISSING] {pid}: {reason}")

    # Detailed sample
    dash = "-" * 60
    print(f"\n{dash}")
    print(f"  Detailed info for first {NUM_SAMPLE_CASES} case(s):")
    for pid, ncct_path, mask_path in pairs[:NUM_SAMPLE_CASES]:
        inspect_case(pid, ncct_path, mask_path, verbose=True)

    # Aggregate sweep
    print(f"\n{dash}")
    print("  Sweeping ALL cases for aggregate statistics ...")
    all_unique_vals = set()
    shape_counts    = {}
    mismatch_cases  = []
    no_lesion_cases = []
    all_comp_counts = []

    for pid, ncct_path, mask_path in pairs:
        info = inspect_case(pid, ncct_path, mask_path, verbose=False)
        if not info["ok"]:
            continue
        for v in info["unique_vals"]:
            all_unique_vals.add(v)
        sk = f"CT{info['ct_shape']} / MASK{info['mask_shape']}"
        shape_counts[sk] = shape_counts.get(sk, 0) + 1
        if not info["spatially_ok"] or not info["affine_close"]:
            mismatch_cases.append(pid)
        if not info["has_lesion"]:
            no_lesion_cases.append(pid)

    infarct_vals = sorted([v for v in all_unique_vals if v != 0])

    # Connected-component sweep
    print("  Counting connected components per positive slice ...")
    for pid, ncct_path, mask_path in pairs:
        try:
            mask_img  = nib.load(mask_path)
            mask_data = mask_img.get_fdata(dtype=np.float32)
            comps = count_components_per_slice(mask_data, infarct_vals)
            all_comp_counts.extend(comps)
        except Exception as e:
            print(f"  [WARN] {pid}: {e}")

    # Final summary
    print(f"\n{sep}")
    print("  AGGREGATE SUMMARY")
    print(sep)
    print(f"  All unique mask label values : {sorted(all_unique_vals)}")
    print(f"  Inferred infarct label(s)    : {infarct_vals}")
    print(f"  Cases with no lesion at all  : {len(no_lesion_cases)}  {no_lesion_cases}")
    print(f"  Spatial/affine mismatches    : {len(mismatch_cases)}  {mismatch_cases}")
    print("\n  Volume shape distribution (CT / MASK):")
    for k, v in sorted(shape_counts.items(), key=lambda x: -x[1]):
        print(f"    {v:3d}x  {k}")

    if all_comp_counts:
        arr = np.array(all_comp_counts)
        print(f"\n  Connected components per positive axial slice:")
        print(f"    Total positive slices inspected : {len(arr)}")
        print(f"    Min  components / slice         : {arr.min()}")
        print(f"    Max  components / slice         : {arr.max()}")
        print(f"    Mean components / slice         : {arr.mean():.2f}")
        print(f"    Slices with > 1 component       : {(arr > 1).sum()}  ({100*(arr>1).mean():.1f}%)")
        print()
        if arr.max() > 1:
            print("  RECOMMENDATION: Some slices have >1 disconnected lesion region.")
            print("    The pipeline will generate one box per connected component")
            print("    on those slices (multi-box mode is more accurate).")
        else:
            print("  All positive slices have a single connected component.")
            print("  A single bounding box per slice is appropriate.")
    else:
        print("  [NOTE] No positive slices found -- masks may all be empty.")

    print(f"\n{sep}\n")


if __name__ == "__main__":
    main()
