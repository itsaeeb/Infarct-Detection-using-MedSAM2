"""
test_nifti_to_tensor.py
========================
Validation script to test loading of multiple original APIS brain CT NIfTI volumes,
converting them to official MedSAM2 video predictor tensors, and verifying compatibility
with MedSAM2's `init_state()` interface without requiring full model weights.
"""

import os
import sys
import glob
from collections import OrderedDict
import numpy as np
import torch

# Ensure local paths and MedSAM2 can be imported
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INFARCT_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(INFARCT_ROOT)
MEDSAM2_ROOT = os.path.join(PROJECT_ROOT, "MedSAM2")

sys.path.insert(0, SCRIPT_DIR)
if os.path.isdir(MEDSAM2_ROOT):
    sys.path.insert(0, MEDSAM2_ROOT)

from nifti_to_tensor import (
    load_apis_nifti_volume,
    ct_volume_to_medsam2_tensor,
    DEFAULT_WINDOW_LEVEL,
    DEFAULT_WINDOW_WIDTH,
    TARGET_SPATIAL_SIZE
)


def mock_test_init_state_compatibility(video_tensor: torch.Tensor, video_height: int, video_width: int):
    """
    Test direct compatibility with official SAM2VideoPredictorNPZ.init_state()
    by verifying all data structure invariants required by lines 64-114 of
    `MedSAM2/sam2/sam2_video_predictor_npz.py`.
    """
    assert isinstance(video_tensor, torch.Tensor), "images must be a torch.Tensor"
    assert len(video_tensor.shape) == 4, f"images must be 4D (D, 3, H, W), got {video_tensor.shape}"
    assert video_tensor.shape[1] == 3, f"images must have 3 channels (RGB), got {video_tensor.shape[1]}"
    assert video_tensor.shape[2] == TARGET_SPATIAL_SIZE and video_tensor.shape[3] == TARGET_SPATIAL_SIZE, \
        f"images spatial size must be {TARGET_SPATIAL_SIZE}x{TARGET_SPATIAL_SIZE}, got {video_tensor.shape[2:]}"
    assert video_tensor.dtype in (torch.float32, torch.bfloat16, torch.float16), \
        f"images dtype must be float, got {video_tensor.dtype}"
    
    # Simulate init_state state dictionary creation
    inference_state = {}
    inference_state["images"] = video_tensor
    inference_state["num_frames"] = len(video_tensor)
    inference_state["video_height"] = video_height
    inference_state["video_width"] = video_width
    inference_state["device"] = video_tensor.device
    inference_state["storage_device"] = video_tensor.device
    inference_state["point_inputs_per_obj"] = {}
    inference_state["mask_inputs_per_obj"] = {}
    inference_state["cached_features"] = {}
    inference_state["output_dict"] = {
        "cond_frame_outputs": {},
        "non_cond_frame_outputs": {},
    }
    
    # Test frame extraction matching MedSAM2 _get_image_feature (line 892)
    sample_frame_0 = inference_state["images"][0].to(inference_state["device"]).float().unsqueeze(0)
    assert sample_frame_0.shape == (1, 3, TARGET_SPATIAL_SIZE, TARGET_SPATIAL_SIZE), \
        f"Frame extraction failed shape check: {sample_frame_0.shape}"
    return True


def run_validation():
    print("=" * 80)
    print("MEDSAM2 NIfTI-TO-TENSOR CONVERSION VALIDATION")
    print("=" * 80)
    
    dataset_dir = os.path.join(PROJECT_ROOT, "APIS_Dataset")
    ncct_files = sorted(glob.glob(os.path.join(dataset_dir, "train_*", "train_*_ncct.nii.gz")))
    
    if not ncct_files:
        # Fallback to .nii
        ncct_files = sorted(glob.glob(os.path.join(dataset_dir, "train_*", "train_*_ncct.nii")))
        
    print(f"Discovered {len(ncct_files)} CT NIfTI volumes in APIS_Dataset.")
    
    # Test on a representative diverse sample of 4 patients
    test_cases = [f for f in ncct_files if os.path.getsize(f) > 0][:4]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target evaluation device: {device}\n")
    
    results = []
    
    for idx, nifti_file in enumerate(test_cases, 1):
        pid = os.path.basename(os.path.dirname(nifti_file))
        print(f"--- Case {idx}/{len(test_cases)}: {pid} ---")
        print(f"File Path: {nifti_file} ({os.path.getsize(nifti_file) / 1024 / 1024:.2f} MB)")
        
        # 1. Load original NIfTI volume
        raw_vol, meta = load_apis_nifti_volume(nifti_file)
        print(f"  [Original NIfTI Volume]")
        print(f"    - Shape (H, W, D):     {meta['orig_shape']}")
        print(f"    - Voxel Spacing (mm):  {meta['spacing']}")
        print(f"    - Orientation Codes:   {meta['orientation']}")
        print(f"    - Raw Intensity (HU):  min={meta['intensity_min_raw']:.1f}, max={meta['intensity_max_raw']:.1f}")
        
        # Check affine matrix properties
        affine = meta['affine']
        print(f"    - Affine Determinant: {np.linalg.det(affine[:3, :3]):.4f}")
        
        # 2. Convert to MedSAM2 Tensor
        tensor, tensor_meta = ct_volume_to_medsam2_tensor(
            volume_hu=raw_vol,
            meta=meta,
            window_level=DEFAULT_WINDOW_LEVEL,
            window_width=DEFAULT_WINDOW_WIDTH,
            target_size=TARGET_SPATIAL_SIZE,
            device=device
        )
        
        print(f"  [Converted MedSAM2 Video Tensor]")
        print(f"    - Tensor Shape:        {tensor_meta['tensor_shape']} -> (D, C, H, W)")
        print(f"    - Number of Slices (D):{tensor_meta['num_slices_D']}")
        print(f"    - Channels (C):        {tensor.shape[1]} (RGB)")
        print(f"    - Spatial Resolution:  {tensor.shape[2]} x {tensor.shape[3]}")
        print(f"    - Data Type:           {tensor_meta['tensor_dtype']}")
        print(f"    - Intensity Min/Max:   min={tensor_meta['tensor_min']:.4f}, max={tensor_meta['tensor_max']:.4f}")
        print(f"    - Intensity Mean:      {tensor_meta['tensor_mean']:.4f}")
        print(f"    - Device:              {tensor_meta['device']}")
        
        # 3. Verify init_state() compatibility
        compat_ok = mock_test_init_state_compatibility(
            video_tensor=tensor,
            video_height=meta['orig_shape'][0],
            video_width=meta['orig_shape'][1]
        )
        print(f"    - init_state() Check:  {'PASSED' if compat_ok else 'FAILED'}")
        print()
        
        results.append({
            "pid": pid,
            "orig_shape": meta['orig_shape'],
            "tensor_shape": tensor_meta['tensor_shape'],
            "compat": compat_ok
        })
        
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    for r in results:
        status = "OK [Compatible]" if r["compat"] else "ERROR"
        print(f"  {r['pid']:12s} | Orig: {str(r['orig_shape']):16s} -> Tensor: {str(r['tensor_shape']):20s} | {status}")
    print("\nAll tested volumes successfully verified for MedSAM2 tensor conversion.\n")


if __name__ == "__main__":
    run_validation()
