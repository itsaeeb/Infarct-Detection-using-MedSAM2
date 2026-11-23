"""
test_predictor_interface.py
===========================
Validation script for Stage 2.

Validates:
1. Instantiation of MedSAM2InfarctPredictor wrapper.
2. Ingestion of a real APIS 4D CT tensor from `nifti_to_tensor`.
3. Ingestion and parsing of Phase 1 bounding boxes from `bbox_format`.
4. Verification of the `init_patient_tensor()` and `predict_slice_from_boxes()` pipeline.
"""

import os
import sys
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INFARCT_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(INFARCT_ROOT)

sys.path.insert(0, os.path.join(INFARCT_ROOT, "nifti_to_tensor"))
sys.path.insert(0, os.path.join(INFARCT_ROOT, "bbox_format"))
sys.path.insert(0, SCRIPT_DIR)

from nifti_to_tensor import load_apis_nifti_volume, ct_volume_to_medsam2_tensor
from box_parser import parse_yolo_label_file
from predictor_wrapper import MedSAM2InfarctPredictor, DEFAULT_CONFIG_PATH


def run_predictor_interface_validation():
    print("=" * 80)
    print("STAGE 2: MEDSAM2 PREDICTOR INTERFACE VALIDATION")
    print("=" * 80)

    # 1. Check local standalone sam2 directory and config
    local_sam2_dir = os.path.join(INFARCT_ROOT, "sam2")
    config_exists = os.path.isfile(DEFAULT_CONFIG_PATH)
    print(f"1. Local Standalone sam2   : {'EXISTS' if os.path.isdir(local_sam2_dir) else 'NOT FOUND'} ({local_sam2_dir})")
    print(f"2. Config File Check       : {'EXISTS' if config_exists else 'NOT FOUND'} ({DEFAULT_CONFIG_PATH})")

    # 2. Check checkpoints folder
    ckpt_dir = os.path.join(INFARCT_ROOT, "checkpoints")
    print(f"3. Checkpoints Directory   : {'EXISTS' if os.path.isdir(ckpt_dir) else 'NOT FOUND'} ({ckpt_dir})")

    # Check if weights file exists or provide instruction
    ckpt_file = os.path.join(ckpt_dir, "MedSAM2_CTLesion.pt")
    if not os.path.isfile(ckpt_file):
        alt_ckpt = os.path.join(ckpt_dir, "MedSAM2_latest.pt")
        if os.path.isfile(alt_ckpt):
            ckpt_file = alt_ckpt

    has_weights = os.path.isfile(ckpt_file)
    print(f"4. Pretrained Weights Check: {'EXISTS' if has_weights else 'NOT YET DOWNLOADED'} ({ckpt_file})")

    # 3. Instantiate Predictor Wrapper
    print("\n5. Initializing MedSAM2InfarctPredictor Wrapper...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = MedSAM2InfarctPredictor(
        checkpoint_path=ckpt_file if has_weights else None,
        config_path=DEFAULT_CONFIG_PATH,
        device=device,
        mode="video_npz"
    )
    print(f"   - Target Device: {predictor.device}")
    print(f"   - Mode         : {predictor.mode}")

    # 4. End-to-end tensor pipeline test with real APIS scan (train_001)
    test_nii = os.path.join(PROJECT_ROOT, "APIS_Dataset", "train_001", "train_001_ncct.nii.gz")
    test_lbl = os.path.join(PROJECT_ROOT, "phase_2_yolo26", "dataset", "labels", "test", "train_001_slice016.txt")

    print(f"\n6. Testing Tensor Ingestion on {os.path.basename(test_nii)}...")
    raw_vol, meta = load_apis_nifti_volume(test_nii)
    tensor, t_meta = ct_volume_to_medsam2_tensor(raw_vol, meta, device=device)
    print(f"   - Converted 4D CT Tensor Shape: {tensor.shape} (D, 3, 512, 512)")

    # 5. Initialize Predictor State with Tensor
    state = predictor.init_patient_tensor(
        video_tensor=tensor,
        video_height=meta["orig_shape"][0],
        video_width=meta["orig_shape"][1],
        patient_id="train_001"
    )
    print(f"   - Predictor State Initialized : SUCCESS (num_frames={state['num_frames']})")

    # 6. Parse Bounding Box and Run Slice Prediction
    print(f"\n7. Testing Slice Prediction on Slice z=16 with Box Prompt...")
    boxes = parse_yolo_label_file(test_lbl, img_w=512, img_h=512)
    print(f"   - Converted Box Prompt: {boxes.tolist()}")

    pred_mask = predictor.predict_slice_from_boxes(frame_idx=16, boxes=boxes)
    print(f"   - Predicted Mask Shape : {pred_mask.shape} | dtype: {pred_mask.dtype}")
    print(f"   - Infarct Pixels in Mask: {int(np.sum(pred_mask))} pixels")

    assert pred_mask.shape == (512, 512), f"Expected mask shape (512, 512), got {pred_mask.shape}"
    assert np.sum(pred_mask) > 0, "Predicted mask should contain positive infarct pixels"

    print("\n" + "=" * 80)
    print("STAGE 2 VALIDATION PASSED: Predictor wrapper interface verified with 4D tensor and box prompts.")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = run_predictor_interface_validation()
    sys.exit(0 if success else 1)
