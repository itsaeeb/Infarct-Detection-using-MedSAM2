"""
nifti_to_tensor.py
==================
Handles loading original 3D brain CT NIfTI volumes from APIS_Dataset and converting
them into the exact tensor representation expected by official MedSAM2's video
predictor (sam2_video_predictor_npz.py / medsam2_infer_3D_CT.py).

Official MedSAM2 Tensor Specification:
---------------------------------------
1. Dimensionality: 4D PyTorch Tensor of shape (D, 3, 512, 512)
   - D: Number of axial slices in superior-inferior order (frame_idx = 0 .. D-1).
   - 3: RGB color channels (duplicated across grayscale CT).
   - 512 x 512: Spatial height and width expected by Hiera-Tiny backbone.
2. Data Type: torch.float32 (or bfloat16/float16 under autocast)
3. Normalization:
   - Clipped to HU window [WL - WW/2, WL + WW/2] (Brain window default: [0.0, 80.0] HU).
   - Scaled to [0.0, 1.0] (equivalent to uint8 / 255.0).
   - Normalized with ImageNet statistics:
       mean = (0.485, 0.456, 0.406)
       std  = (0.229, 0.224, 0.225)
       tensor = (tensor - mean) / std
4. Metadata Preservation:
   - Preserves original affine, voxel zooms, original spatial dimensions (H_ori, W_ori, D),
     and slice orientation for later 3D reconstruction and evaluation.
"""

import os
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import nibabel as nib
except ImportError:
    raise ImportError("nibabel is required. Run: pip install nibabel")


# ImageNet statistics matching MedSAM2 (medsam2_infer_3D_CT.py lines 248-253)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# Standard CT Brain Window settings
DEFAULT_WINDOW_LEVEL = 40.0   # Center: 40 HU
DEFAULT_WINDOW_WIDTH = 80.0   # Width: 80 HU -> [0.0, 80.0] HU
TARGET_SPATIAL_SIZE  = 512    # MedSAM2 Hiera-Tiny spatial size


def load_apis_nifti_volume(nifti_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load an APIS CT NIfTI volume, preserving true HU intensities and metadata.
    
    Parameters
    ----------
    nifti_path : str
        Path to .nii or .nii.gz file.
        
    Returns
    -------
    data : np.ndarray
        Raw 3D floating-point array with shape (H, W, D).
    meta : dict
        Dictionary containing spatial and orientation metadata:
        - 'affine': 4x4 affine matrix
        - 'spacing': voxel zoom tuple (dx, dy, dz)
        - 'orientation': 3-letter orientation code (e.g. ('L', 'A', 'S'))
        - 'orig_shape': (H, W, D)
        - 'nifti_path': source file path
    """
    if not os.path.isfile(nifti_path):
        raise FileNotFoundError(f"NIfTI file not found: {nifti_path}")
        
    img = nib.load(nifti_path)
    
    # get_fdata() automatically applies scl_slope and scl_inter to yield calibrated HU
    raw_data = img.get_fdata(dtype=np.float32)
    
    affine = img.affine
    header = img.header
    spacing = header.get_zooms()[:3]
    try:
        orientation = nib.aff2axcodes(affine)
    except Exception:
        orientation = None
        
    meta = {
        "nifti_path": os.path.abspath(nifti_path),
        "affine": affine,
        "spacing": tuple(float(s) for s in spacing),
        "orientation": orientation,
        "orig_shape": raw_data.shape,  # (H, W, D)
        "orig_dtype": str(raw_data.dtype),
        "intensity_min_raw": float(np.min(raw_data)),
        "intensity_max_raw": float(np.max(raw_data)),
    }
    
    return raw_data, meta


def apply_hu_windowing(
    volume: np.ndarray,
    window_level: float = DEFAULT_WINDOW_LEVEL,
    window_width: float = DEFAULT_WINDOW_WIDTH
) -> np.ndarray:
    """
    Apply Hounsfield Unit (HU) windowing to CT array and rescale into [0.0, 255.0].
    
    Parameters
    ----------
    volume : np.ndarray
        Raw CT intensity array in Hounsfield Units.
    window_level : float
        Center of window (HU).
    window_width : float
        Full width of window (HU).
        
    Returns
    -------
    scaled_volume : np.ndarray
        Windowed array in range [0.0, 255.0] float32.
    """
    lower = window_level - window_width / 2.0
    upper = window_level + window_width / 2.0
    
    clipped = np.clip(volume, lower, upper)
    
    # Map [lower, upper] to [0.0, 255.0]
    scaled = (clipped - lower) / (upper - lower) * 255.0
    return scaled.astype(np.float32)


def ct_volume_to_medsam2_tensor(
    volume_hu: np.ndarray,
    meta: Dict[str, Any],
    window_level: float = DEFAULT_WINDOW_LEVEL,
    window_width: float = DEFAULT_WINDOW_WIDTH,
    target_size: int = TARGET_SPATIAL_SIZE,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Convert a 3D CT volume array (H, W, D) into the official MedSAM2 video predictor tensor.
    
    Following the exact official implementation in `medsam2_infer_3D_CT.py`:
    1. Clips to HU window and scales to [0, 255] uint8.
    2. Transposes axial slices (axis 2) to leading depth dimension D.
    3. Resizes each 2D slice to (target_size, target_size) and expands to 3 RGB channels.
    4. Scales by 1.0 / 255.0 to range [0.0, 1.0].
    5. Normalizes by ImageNet mean and std.
    
    Parameters
    ----------
    volume_hu : np.ndarray
        Array of shape (H, W, D) in HU.
    meta : dict
        Metadata from `load_apis_nifti_volume`.
    window_level : float
        Center of HU window (default: 40 HU).
    window_width : float
        Width of HU window (default: 80 HU).
    target_size : int
        Spatial size for MedSAM2 backbone (default: 512).
    device : torch.device, optional
        Target device (CPU or CUDA). Defaults to CPU.
        
    Returns
    -------
    video_tensor : torch.Tensor
        4D Tensor of shape (D, 3, target_size, target_size) ready for MedSAM2's `init_state()`.
    tensor_meta : dict
        Updated metadata dict with tensor statistics and spatial dimensions.
    """
    if device is None:
        device = torch.device("cpu")
        
    # Check shape: volume is (H, W, D)
    H_ori, W_ori, D = volume_hu.shape
    
    # 1. Apply HU Windowing and map to [0, 255]
    windowed_255 = apply_hu_windowing(volume_hu, window_level, window_width)
    uint8_vol = np.uint8(np.round(windowed_255))
    
    # 2. Iterate through axial slices (0 .. D-1) and format as (D, 3, target_size, target_size)
    # Matching official `resize_grayscale_to_rgb_and_resize` in medsam2_infer_3D_CT.py
    resized_array = np.zeros((D, 3, target_size, target_size), dtype=np.float32)
    
    for z in range(D):
        slice_2d = uint8_vol[:, :, z]  # (H, W) uint8
        img_pil = Image.fromarray(slice_2d)
        img_rgb = img_pil.convert("RGB")
        if (H_ori, W_ori) != (target_size, target_size):
            img_resized = img_rgb.resize((target_size, target_size), resample=Image.BILINEAR)
        else:
            img_resized = img_rgb
        # PIL to array with shape (3, target_size, target_size)
        img_array = np.array(img_resized, dtype=np.float32).transpose(2, 0, 1)
        resized_array[z] = img_array
        
    # 3. Rescale to [0.0, 1.0]
    resized_array = resized_array / 255.0
    
    # 4. Convert to PyTorch Tensor on target device
    video_tensor = torch.from_numpy(resized_array).to(device)
    
    # 5. Apply ImageNet Mean and Standard Deviation (matching medsam2_infer_3D_CT.py lines 248-253)
    mean_tensor = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=device)[:, None, None]
    std_tensor  = torch.tensor(IMAGENET_STD,  dtype=torch.float32, device=device)[:, None, None]
    
    video_tensor = (video_tensor - mean_tensor) / std_tensor
    
    tensor_meta = dict(meta)
    tensor_meta.update({
        "tensor_shape": tuple(video_tensor.shape),
        "tensor_dtype": str(video_tensor.dtype),
        "num_slices_D": D,
        "video_height": H_ori,
        "video_width": W_ori,
        "model_image_size": target_size,
        "window_level": window_level,
        "window_width": window_width,
        "device": str(device),
        "tensor_min": float(video_tensor.min().item()),
        "tensor_max": float(video_tensor.max().item()),
        "tensor_mean": float(video_tensor.mean().item()),
    })
    
    return video_tensor, tensor_meta
