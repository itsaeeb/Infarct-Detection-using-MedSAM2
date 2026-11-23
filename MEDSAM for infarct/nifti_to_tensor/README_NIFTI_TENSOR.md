# MedSAM2 NIfTI-to-Tensor Conversion Module

This module loads original 3D brain CT NIfTI volumes (`.nii` / `.nii.gz`) from `APIS_Dataset`, preserves spatial metadata, applies calibrated CT Hounsfield Unit (HU) windowing, and converts the volume into the exact 4D tensor representation expected by official MedSAM2 (`sam2_video_predictor_npz.py` / `medsam2_infer_3D_CT.py`).

---

## 1. Official MedSAM2 Tensor Specification

Based on inspection of `MedSAM2/medsam2_infer_3D_CT.py` (lines 240–257) and `MedSAM2/sam2/sam2_video_predictor_npz.py` (lines 45–115, 883–896), the input tensor to `predictor.init_state(images, video_height, video_width)` must meet these specifications:

| Attribute | Specification | Notes |
| :--- | :--- | :--- |
| **Dimensionality** | 4D Tensor `(D, 3, 512, 512)` | $D$: number of axial slices along the depth axis. $3$: RGB channels. $512 \times 512$: spatial resolution. |
| **Data Type** | `torch.float32` | Convertible to `bfloat16`/`float16` under autocast. |
| **Intensity Range** | Normalized by ImageNet statistics | Scaled to $[0.0, 1.0]$, then subtracted by $(0.485, 0.456, 0.406)$ and divided by $(0.229, 0.224, 0.225)$. Min $\approx -2.1179$, Max $\approx 2.6400$. |
| **Slice Ordering** | Axial slices $z = 0 \dots D-1$ | Preserves superior-inferior anatomical ordering matching Phase 1 slice indices. |
| **Spatial Metadata** | Stored in metadata dict | `video_height` ($H_{\text{ori}}$) and `video_width` ($W_{\text{ori}}$) passed into `init_state()` to allow output mask resizing to native resolution. |

---

## 2. Code Structure

- **`MEDSAM for infarct/src/nifti_to_tensor.py`**:
  - `load_apis_nifti_volume(nifti_path)`: Reads `.nii.gz` via `nibabel.get_fdata()` to obtain true calibrated HU values and extracts affine, zooms, and orientation codes.
  - `apply_hu_windowing(volume, window_level, window_width)`: Clips raw HU values to window range (Brain Window: $[0, 80]$ HU) and rescales to $[0.0, 255.0]$.
  - `ct_volume_to_medsam2_tensor(...)`: Re-arranges axial slices, resizes each 2D slice to $512 \times 512 \times 3$, rescales by $1/255.0$, converts to PyTorch tensor, and applies ImageNet mean/std normalization.
- **`MEDSAM for infarct/test_nifti_to_tensor.py`**:
  - Validation test script that tests multiple APIS cases and verifies compatibility with `init_state()`.

---

## 3. How to Run the Validation Script

From the project root directory (`Code files/`), run:

```bash
python "MEDSAM for infarct/test_nifti_to_tensor.py"
```

### Verification Output:
All test cases demonstrate exact tensor shape alignment `(D, 3, 512, 512)` and passed compatibility checks:
- `train_000`: Orig `(568, 455, 26)` $\rightarrow$ Tensor `(26, 3, 512, 512)` | `torch.float32` | `PASSED`
- `train_001`: Orig `(288, 256, 23)` $\rightarrow$ Tensor `(23, 3, 512, 512)` | `torch.float32` | `PASSED`
- `train_002`: Orig `(256, 256, 26)` $\rightarrow$ Tensor `(26, 3, 512, 512)` | `torch.float32` | `PASSED`
- `train_003`: Orig `(256, 256, 26)` $\rightarrow$ Tensor `(26, 3, 512, 512)` | `torch.float32` | `PASSED`
