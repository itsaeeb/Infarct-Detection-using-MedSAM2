"""
test_single_slice.py
====================
Stage 3 Smoke Test: End-to-end execution on a single patient volume and positive slice.

Performs:
1. Loads original CT NIfTI volume (train_001_ncct.nii.gz) and converts to 4D tensor (23, 3, 512, 512).
2. Initializes MedSAM2 predictor state with the 4D CT tensor.
3. Parses Phase 1 bounding box prompt for positive slice z=16.
4. Generates predicted infarct segmentation mask via predictor wrapper.
5. Loads ground-truth NIfTI mask (train_001_r1_mask.nii.gz) at slice z=16.
6. Computes quantitative Dice and IoU metrics.
7. Saves a 4-panel visual comparison to MEDSAM for infarct/outputs/.
"""

import os
import sys
from typing import Optional
import numpy as np
import torch
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INFARCT_ROOT = SCRIPT_DIR
PROJECT_ROOT = os.path.dirname(INFARCT_ROOT)

sys.path.insert(0, os.path.join(INFARCT_ROOT, "nifti_to_tensor"))
sys.path.insert(0, os.path.join(INFARCT_ROOT, "bbox_format"))
sys.path.insert(0, os.path.join(INFARCT_ROOT, "predictor"))
sys.path.insert(0, os.path.join(INFARCT_ROOT, "evaluation"))

from nifti_to_tensor import load_apis_nifti_volume, ct_volume_to_medsam2_tensor, apply_hu_windowing
from box_parser import parse_yolo_label_file
from predictor_wrapper import MedSAM2InfarctPredictor, DEFAULT_CONFIG_PATH
from metrics import evaluate_slice_metrics
from visualization import save_4panel_visualization

try:
    import nibabel as nib
except ImportError:
    sys.exit("nibabel is required. Run: pip install nibabel")


def run_stage3_smoke_test(patient_id: str = "train_001", slice_idx: int = 16, checkpoint_path: Optional[str] = None):
    print("=" * 80)
    print(f"STAGE 3: SINGLE-PATIENT/SLICE SMOKE TEST ({patient_id}, slice {slice_idx:03d})")
    print("=" * 80)

    # 1. Define Paths
    ct_nii_path = os.path.join(PROJECT_ROOT, "APIS_Dataset", patient_id, f"{patient_id}_ncct.nii.gz")
    mask_nii_path = os.path.join(PROJECT_ROOT, "APIS_Dataset", patient_id, "masks", f"{patient_id}_r1_mask.nii.gz")
    label_path = os.path.join(PROJECT_ROOT, "phase_2_yolo26", "dataset", "labels", "test", f"{patient_id}_slice{slice_idx:03d}.txt")

    if not os.path.isfile(mask_nii_path):
        mask_nii_path = os.path.join(PROJECT_ROOT, "APIS_Dataset", patient_id, "masks", f"{patient_id}_r1_mask.nii")

    print(f"CT Volume Path   : {ct_nii_path}")
    print(f"GT Mask Path     : {mask_nii_path}")
    print(f"Box Prompt Path  : {label_path}")

    # 2. Check Weights (Prioritize fine-tuned checkpoint if available)
    ckpt_dir = os.path.join(INFARCT_ROOT, "checkpoints")
    finetuned_file = os.path.join(ckpt_dir, "finetuned", "best_medsam2_apis.pt")

    if checkpoint_path is not None and os.path.isfile(checkpoint_path):
        ckpt_file = checkpoint_path
        ckpt_type = "SPECIFIED"
    elif os.path.isfile(finetuned_file):
        ckpt_file = finetuned_file
        ckpt_type = "FINE-TUNED (APIS Specialized)"
    else:
        ckpt_file = os.path.join(ckpt_dir, "MedSAM2_CTLesion.pt")
        ckpt_type = "PRETRAINED BASELINE"

    has_weights = os.path.isfile(ckpt_file)
    print(f"Model Checkpoint : {ckpt_type} -> {ckpt_file}")

    # 3. Load Volume & Convert to 4D Tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Step 1/5] Loading 3D NIfTI Volume and Converting to Tensor...")
    raw_vol, meta = load_apis_nifti_volume(ct_nii_path)
    video_tensor, t_meta = ct_volume_to_medsam2_tensor(raw_vol, meta, device=device)
    print(f"  - 4D CT Tensor Shape: {video_tensor.shape} (D, 3, 512, 512)")
    print(f"  - Value Range       : [{t_meta['tensor_min']:.4f}, {t_meta['tensor_max']:.4f}]")

    # 4. Initialize Predictor
    print(f"\n[Step 2/5] Initializing MedSAM2 Predictor State...")
    predictor = MedSAM2InfarctPredictor(
        checkpoint_path=ckpt_file if has_weights else None,
        config_path=DEFAULT_CONFIG_PATH,
        device=device,
        mode="video_npz"
    )
    predictor.init_patient_tensor(
        video_tensor=video_tensor,
        video_height=meta["orig_shape"][0],
        video_width=meta["orig_shape"][1],
        patient_id=patient_id
    )
    print(f"  - State initialized with {video_tensor.shape[0]} slices.")

    # 5. Parse Box Prompt
    print(f"\n[Step 3/5] Parsing Bounding Box Prompt for Slice {slice_idx:03d}...")
    boxes = parse_yolo_label_file(label_path, img_w=512, img_h=512)
    print(f"  - Box count: {len(boxes)}")
    for b_idx, b in enumerate(boxes, 1):
        print(f"    Box {b_idx}: [xmin={b[0]:.1f}, ymin={b[1]:.1f}, xmax={b[2]:.1f}, ymax={b[3]:.1f}]")

    # 6. Predict Mask
    print(f"\n[Step 4/5] Running MedSAM2 Slice Prediction...")
    pred_mask_512 = predictor.predict_slice_from_boxes(frame_idx=slice_idx, boxes=boxes)
    print(f"  - Predicted Mask Shape: {pred_mask_512.shape} (dtype: {pred_mask_512.dtype})")
    print(f"  - Predicted Infarct Pixels: {int(np.sum(pred_mask_512))}")

    # 7. Load Ground-Truth Mask Slice & Evaluate Metrics
    print(f"\n[Step 5/5] Evaluating Against Ground-Truth NIfTI Mask...")
    mask_img = nib.load(mask_nii_path)
    gt_vol = np.round(mask_img.get_fdata(dtype=np.float32)).astype(np.uint8)
    gt_slice_native = gt_vol[:, :, slice_idx]

    # Resize GT slice to 512x512 matching model space
    gt_slice_512 = cv2.resize(gt_slice_native, (512, 512), interpolation=cv2.INTER_NEAREST)

    metrics = evaluate_slice_metrics(pred_mask_512, gt_slice_512)
    print(f"  - Dice Similarity Coefficient (DSC): {metrics['dice']:.4f}")
    print(f"  - Intersection over Union (IoU)    : {metrics['iou']:.4f}")
    print(f"  - GT Infarct Pixels (512x512)      : {metrics['gt_pixels']}")
    print(f"  - Pred Infarct Pixels (512x512)    : {metrics['pred_pixels']}")

    # 8. Extract Slice for Visualization
    ct_slice_raw = raw_vol[:, :, slice_idx]
    ct_slice_windowed = apply_hu_windowing(ct_slice_raw, window_level=40.0, window_width=80.0)
    ct_slice_512 = cv2.resize(ct_slice_windowed.astype(np.uint8), (512, 512), interpolation=cv2.INTER_LINEAR)

    vis_out_path = os.path.join(INFARCT_ROOT, "outputs", f"smoke_test_{patient_id}_slice{slice_idx:03d}.png")
    save_4panel_visualization(
        ct_slice_255=ct_slice_512,
        boxes_px=boxes,
        gt_mask=gt_slice_512,
        pred_mask=pred_mask_512,
        save_path=vis_out_path,
        patient_id=patient_id,
        slice_idx=slice_idx,
        dice=metrics["dice"],
        iou=metrics["iou"]
    )
    print(f"\n  - Saved 4-Panel Visualization to: {vis_out_path}")

    print("\n" + "=" * 80)
    print(f"STAGE 3 SMOKE TEST COMPLETED SUCCESSFULLY for {patient_id} slice {slice_idx:03d}")
    print("=" * 80)
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run MedSAM2 pipeline on a single CT slice")
    parser.add_argument("--patient_id", type=str, default="train_008", help="APIS patient ID (e.g. train_008)")
    parser.add_argument("--slice_idx", type=int, default=7, help="Axial slice index (e.g. 7)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Custom checkpoint path (.pt)")
    args = parser.parse_args()

    success = run_stage3_smoke_test(args.patient_id, args.slice_idx, args.checkpoint)
    sys.exit(0 if success else 1)
