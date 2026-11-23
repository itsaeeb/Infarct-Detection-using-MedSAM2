# MEDSAM2 Integration Plan: Brain Infarct Segmentation

This document details the engineering roadmap for integrating **MedSAM2** into the Infarct-Detection project using **original CT NIfTI volumes** as direct inputs and prompting with the **ground-truth-derived bounding boxes** generated in Phase 1.

---

## 1. Objective and Scope

### 1.1 Immediate Objective
The immediate goal of this experiment is to bypass YOLO and directly evaluate the promptable segmentation performance of **pretrained MedSAM2** on brain infarct lesions in non-contrast CT (NCCT) scans. 

Ground-truth-derived bounding boxes will serve as "oracle" prompts. This isolates MedSAM2's segmentation capability from detection errors, establishing an upper-bound baseline for subsequent automatic detection-segmentation pipelines.

### 1.2 Scope Boundaries
- **Inference Only**: This plan strictly covers inference with pretrained MedSAM2 checkpoints (`MedSAM2_CTLesion.pt` or `MedSAM2_latest.pt`). MedSAM2 fine-tuning is **not** part of this immediate phase.
- **No YOLO**: YOLO models or predictions are excluded from this stage.
- **Independent 2D Positive-Slice Inference**: The initial baseline focuses on evaluating positive slices independently using `SAM2ImagePredictor`. 3D volumetric video-style propagation (`SAM2VideoPredictorNPZ`) is reserved as an optional subsequent extension.
- **Evaluation Scope**: Because bounding boxes originate from ground-truth masks, this experiment measures **prompted boundary delineation quality**, not autonomous infarct discovery. It does not measure false-positive slice classification on negative CT slices.

---

## 2. Existing Assets & Reusable Outputs (Phase 1)

### 2.1 Reused Phase 1 Outputs (No Regeneration)
Phase 1 established validated annotations and slice mappings across 60 APIS patients. **No bounding boxes or slices will be regenerated.**

| Component | Filesystem Location | Role in MedSAM2 Pipeline |
| :--- | :--- | :--- |
| **Ground-Truth Bounding Boxes** | `phase_2_yolo26/dataset/labels/{split}/` | Contains normalized bounding boxes (`0 xc yc w h`). Non-empty files indicate positive slices. |
| **Slice Basenames / Metadata** | Filenames: `{patient_id}_slice{z:03d}` | Defines the deterministic link between scan ID, split, and axial slice index $z$. |
| **Ground-Truth NIfTI Masks** | `APIS_Dataset/{patient_id}/masks/{patient_id}_r1_mask.nii.gz` | Ground-truth reference for Dice & IoU evaluation. |
| **Exported PNG Slices** | `phase_2_yolo26/dataset/images/{split}/` | **Retained strictly for visual inspection and qualitative comparison.** They are **not** fed to MedSAM2. |

### 2.2 Dataset Verification Facts
1. Bounding boxes were generated directly from the unresized, un-interpolated NIfTI ground-truth mask arrays `mask_data[:, :, z]`.
2. Both original NIfTI axial slice grids and exported PNGs are natively $512 \times 512$ with voxel spacing $\approx 0.43 \times 0.43\text{ mm}^2$.
3. Total axial slices across 60 patients: 1,481.
4. Confirmed positive slices: **251** (Train: 200, Val: 33, Test: 18).
5. Multi-lesion slices: 53 slices (~29.7%) have $>1$ connected infarct component (resulting in multiple bounding boxes per slice).

---

## 3. Input Data & NIfTI Volume Loading

As specifically instructed, **original NIfTI CT volumes (`.nii.gz`) must be used as direct model inputs** rather than exported PNG images.

### 3.1 NIfTI Ingestion Pipeline
- **Raw Volume Path**: `APIS_Dataset/{patient_id}/{patient_id}_ncct.nii.gz`
- **Mask Volume Path**: `APIS_Dataset/{patient_id}/masks/{patient_id}_r1_mask.nii.gz`
- **Ingestion Library**: Use `nibabel` (or `SimpleITK`), preserving volume geometry:
  ```python
  import nibabel as nib
  ct_nii = nib.load(ncct_path)
  ct_data = ct_nii.get_fdata(dtype=np.float32)  # Shape: (X, Y, Z) = (512, 512, D)
  affine = ct_nii.affine
  header = ct_nii.header
  ```

### 3.2 Volume Orientation and Axial Slice Verification
- In APIS NIfTI volumes, the standard array orientation is $(H, W, D) = (512, 512, D)$, where the third axis (index $z \in [0, D-1]$) represents the superior-inferior (axial) dimension.
- **Verification Rule**: Before extraction, code must inspect `nib.aff2axcodes(affine)` (e.g. `('R', 'A', 'S')` or `('L', 'P', 'S')`) to guarantee the slice index $z$ extracted from the filename `{patient_id}_slice{z:03d}` corresponds strictly to the axial plane matching Phase 1 slice generation:
  $$\text{slice\_raw} = \texttt{ct\_data}[:, :, z]$$
- No spatial modification, resampling, or resizing will be applied to the volume array, preserving exact spatial alignment with the ground-truth mask array.

---

## 4. CT Preprocessing & HU Windowing

### 4.1 Intensity Value & Slope/Intercept Verification
- Medical CT voxels must represent calibrated Hounsfield Units (HU).
- In NIfTI headers, intensity scaling is governed by `scl_slope` and `scl_inter`. Calling `nibabel.get_fdata()` automatically applies slope and intercept corrections.
- **Pre-check**: The preprocessing module will assert that raw volume intensities fall within expected CT ranges (e.g. air $\approx -1000\text{ HU}$, water $\approx 0\text{ HU}$, brain parenchyma $\approx 20\text{ to }45\text{ HU}$, skull $> 1000\text{ HU}$).

### 4.2 Windowing Strategy: Parameter Assessment
CT intensity windowing maps a selected range $[L_{\min}, L_{\max}]$ to standard visualization contrast:
$$L_{\min} = \text{WL} - \frac{\text{WW}}{2}, \quad L_{\max} = \text{WL} + \frac{\text{WW}}{2}$$

- **Initial Candidate (Brain Window)**: $\text{WL} = 40\text{ HU}, \text{WW} = 80\text{ HU} \implies [0, 80]\text{ HU}$.
- **Assessment Requirement**: Brain windowing ($0 - 80\text{ HU}$) provides optimal contrast for early ischemic stroke (hypodensity), but narrow windows can clip subtle peri-infarct penumbra. The preprocessing module will support configurable $(\text{WL}, \text{WW})$ (e.g. Stroke window $WL=32, WW=8\text{ or }40$; Soft tissue window $WL=40, WW=120$) to assess infarct visibility and boundary delineation.

### 4.3 Intensity Normalization & Channel Arrangement (MedSAM2 Compatibility)
To avoid incompatible or double normalization, the pipeline must strictly match what MedSAM2 expects:

```
  Original NIfTI Slice (512 x 512 float32, HU)
                     │
                     ▼
  Intensity Windowing: Clip to [WL - WW/2, WL + WW/2]
                     │
                     ▼
  Scale to [0.0, 255.0] uint8:  (slice - min) / (max - min) * 255.0
                     │
                     ▼
  Channel Duplication: Repeat to 3 channels (512 x 512 x 3) RGB uint8
                     │
                     ▼
  Pass to SAM2ImagePredictor.set_image(slice_rgb)
                     │
  (Internal SAM2Transforms applies ImageNet mean/std & tensor formatting)
```

1. **Clip intensities**: $I_{\text{clipped}} = \text{clip}(I_{\text{raw}}, L_{\min}, L_{\max})$.
2. **Min-max scaling**: Rescale $I_{\text{clipped}}$ to $[0, 255]$:
   $$I_{\text{scaled}} = \frac{I_{\text{clipped}} - L_{\min}}{L_{\max} - L_{\min}} \times 255.0$$
3. **Array formatting**: Cast to `np.uint8`.
4. **Channel duplication**: Convert 1-channel grayscale to 3-channel RGB:
   $$I_{\text{rgb}} = \text{np.repeat}(I_{\text{scaled}}[:, :, \text{np.newaxis}], 3, \text{axis}=2)$$
5. **MedSAM2 Ingestion**: Pass $I_{\text{rgb}}$ directly into `SAM2ImagePredictor.set_image(I_rgb)`.
   - *Crucial*: `set_image()` internally applies `SAM2Transforms`, which converts the array to a tensor and applies ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`). **Do not apply ImageNet normalization beforehand, which would cause erroneous double normalization.**

---

## 5. Bounding-Box Reuse & Coordinate Alignment

### 5.1 Retrieval by Patient ID & Slice Index
- Positive slice annotations are located in `phase_2_yolo26/dataset/labels/{split}/{patient_id}_slice{z:03d}.txt`.
- Each non-empty file contains one or more lines: `0 xc yc w h` (normalized coordinates $\in [0, 1]$).

### 5.2 Coordinate Transformation to Pixel Space
MedSAM2's `SAM2ImagePredictor.predict(box=...)` requires absolute pixel coordinates in format:
$$[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$$

Because original NIfTI slices are natively $512 \times 512$ (with $W = 512, H = 512$):
$$\begin{aligned}
x_{\min} &= \max(0.0, (x_c - w / 2.0) \times W) \\
y_{\min} &= \max(0.0, (y_c - h / 2.0) \times H) \\
x_{\max} &= \min(W - 1.0, (x_c + w / 2.0) \times W) \\
y_{\max} &= \min(H - 1.0, (y_c + h / 2.0) \times H)
\end{aligned}$$

### 5.3 Multi-Component Lesions (Multiple Boxes per Slice)
In ~29.7% of positive slices, multiple disconnected infarct components exist:
- Load all boxes for the slice as an array of shape $(N, 4)$.
- Pass batch box array to `predictor.predict(box=boxes_array, multimask_output=False)`.
- The predictor produces $N$ binary masks of shape $(N, 512, 512)$.
- Merge all components using a binary logical OR:
  $$\text{slice\_prediction} = \bigvee_{i=1}^N \text{mask}_i$$

---

## 6. MedSAM2 Integration & Inference Workflow

### 6.1 Reused Official Modules
The implementation will reuse components from the local `MedSAM2/` repository without modifying any internal files:

| Official File | Class / Function | Usage |
| :--- | :--- | :--- |
| `MedSAM2/sam2/build_sam.py` | `build_sam2(config_file, ckpt_path, device)` | Model builder. Loads Hiera backbone, prompt encoder, and mask decoder. |
| `MedSAM2/sam2/sam2_image_predictor.py` | `SAM2ImagePredictor` | 2D per-slice inference wrapper managing image embedding caching and promptable decoding. |
| `MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml` | YAML configuration | Model architecture specification matching MedSAM2 weights. |

### 6.2 Pretrained Checkpoint Setup
- Download/place `MedSAM2_CTLesion.pt` (or `MedSAM2_latest.pt`) inside `MEDSAM for infarct/checkpoints/`.
- Device initialization: Auto-detect `cuda` if available; fallback to `cpu` for local verification.

### 6.3 End-to-End Inference Loop (Per Patient)

```text
For each patient in evaluation split:
  1. Load APIS_Dataset/{patient_id}/{patient_id}_ncct.nii.gz
  2. Load APIS_Dataset/{patient_id}/masks/{patient_id}_r1_mask.nii.gz
  3. Query phase_2_yolo26/dataset/labels/{split}/ for positive slices belonging to patient_id
  
  For each positive slice index z:
    a. Extract raw CT slice: ct_slice = ncct_data[:, :, z]
    b. Extract GT mask slice: gt_slice = mask_data[:, :, z]
    c. Apply HU windowing & scale to 512x512x3 RGB uint8
    d. Read existing YOLO bounding boxes for slice z and convert to [xmin, ymin, xmax, ymax]
    e. Call predictor.set_image(slice_rgb)
    f. Call predictor.predict(box=boxes_px, multimask_output=False)
    g. Union predicted components -> pred_mask
    h. Compute per-slice Dice and IoU against gt_slice
    i. Save predicted binary mask and qualitative visualization
```

---

## 7. Output Organization

All generated outputs will be stored inside `MEDSAM for infarct/outputs/`:

```text
MEDSAM for infarct/
├── MEDSAM2_INTEGRATION_PLAN.md             # [THIS ROADMAP]
├── checkpoints/
│   └── MedSAM2_CTLesion.pt                # Pretrained model weights
├── configs/
│   └── sam2.1_hiera_t512.yaml              # Architecture config
├── src/
│   ├── __init__.py
│   ├── nifti_loader.py                     # NIfTI loading, orientation & slice extraction
│   ├── ct_preprocessing.py                 # HU windowing & uint8 RGB conversion
│   ├── box_parser.py                       # YOLO -> absolute pixel coordinates
│   ├── predictor.py                        # MedSAM2 initialization & inference wrapper
│   └── metrics.py                          # Dice, IoU & statistical aggregators
├── run_medsam2_nifti_eval.py               # Main CLI execution script
└── outputs/
    ├── predictions/
    │   └── {split}/
    │       └── {patient_id}_slice{z:03d}_pred.png  # Binary predicted masks (0 or 255)
    ├── visualizations/
    │   └── {split}/
    │       └── {patient_id}_slice{z:03d}_vis.png   # 4-panel comparison overlays
    ├── per_slice_metrics.csv                       # Per-slice scores (Dice, IoU, component count)
    └── summary_metrics.json                        # Mean, median, std per split and global
```

---

## 8. Evaluation Methodology

### 8.1 Quantitative Segmentation Metrics
For each positive slice, evaluate the predicted binary mask $P \in \{0, 1\}^{512 \times 512}$ against the ground-truth mask $G \in \{0, 1\}^{512 \times 512}$:

$$\text{Dice}(P, G) = \frac{2 |P \cap G| + \epsilon}{|P| + |G| + \epsilon}, \quad \text{IoU}(P, G) = \frac{|P \cap G| + \epsilon}{|P \cup G| + \epsilon}$$

Where $\epsilon = 10^{-6}$ provides numerical stability.

- **Aggregated Metrics**:
  - Global Mean and Median Dice & IoU.
  - Standard deviation and interquartile range (IQR).
  - Split-wise breakdown (`test` split benchmark vs `val` and `train`).
  - Analysis stratified by lesion size (small $< 500\text{ px}$, medium $500 - 2000\text{ px}$, large $> 2000\text{ px}$).

### 8.2 Qualitative 4-Panel Visualization
Generate a side-by-side verification figure for every evaluated slice:
1. **Panel 1 (Raw CT)**: Extracted NIfTI slice with Brain Windowing.
2. **Panel 2 (Box Prompt)**: CT slice overlaid with the Phase 1 bounding box(es) (blue outline).
3. **Panel 3 (Ground Truth)**: CT slice overlaid with the original NIfTI ground-truth mask (green overlay).
4. **Panel 4 (MedSAM2 Prediction)**: CT slice overlaid with MedSAM2 predicted mask (red/cyan overlay) annotated with computed Dice and IoU scores.

---

## 9. Technical Dependencies & Implementation Considerations

### 9.1 Environment & Python Packages
- `torch >= 2.5.1`, `torchvision >= 0.20.1` (Current environment: PyTorch 2.6.0+cu124).
- `nibabel >= 5.2.1` (used for NIfTI loading and affine orientation checks).
- `hydra-core >= 1.3.2` (used for MedSAM2 configuration parsing).
- `SimpleITK >= 2.4.0`, `opencv-python`, `scikit-image`, `pandas`, `matplotlib`.

### 9.2 Execution Hardware
- **Local Testing**: Scripts will support CPU execution for verification and smoke testing.
- **Batch Evaluation**: For running across all 251 positive slices, execution can run locally or via Google Colab with GPU acceleration.

### 9.3 Modularity & Non-Destructive Principles
- No modifications will be made to files inside `MedSAM2/` or `APIS_Dataset/`.
- `phase_1_bounding_box_generation` and `phase_2_yolo26` remain completely untouched.

---

## 10. Limitations & Future Extensions

1. **Oracle Prompting Limitation**: Bounding boxes are derived from ground-truth masks. High Dice scores in this experiment reflect promptable boundary adherence under ideal localization, not autonomous stroke detection.
2. **Positive-Only Slice Bias**: Evaluating only positive slices does not measure false-positive responses on negative (normal brain) slices.
3. **Subsequent Extensions**:
   - **Extension A (3D Video Propagation)**: Propagate prompts from key slices across neighboring slices using `SAM2VideoPredictorNPZ` to test volumetric continuity.
   - **Extension B (Automated Detection-Segmentation)**: Replace oracle boxes with predicted bounding boxes from a trained detector (e.g. YOLO) to evaluate the fully automated end-to-end pipeline.
   - **Extension C (Fine-Tuning)**: If zero-shot boundary delineation requires adaptation for subtle infarct hypodensities, MedSAM2 mask decoder weights can be fine-tuned on the training split.

---

## 11. Step-by-Step Implementation Plan by Stages

This section outlines the revised implementation plan using **direct NIfTI-to-tensor conversion as the foundational input for MedSAM2**. For each stage, the expected inputs, operations, outputs, and created files are explicitly specified.

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                           STAGE PIPELINE (TENSOR INPUT)                       │
│                                                                               │
│  Stage 1: NIfTI Volume-to-Tensor Conversion & Data Alignment [COMPLETED]      │
│     │                                                                         │
│     ▼                                                                         │
│  Stage 1.5: Bounding Box Coordinate Conversion (YOLO -> Pixel) [COMPLETED]    │
│     │                                                                         │
│     ▼                                                                         │
│  Stage 2: MedSAM2 Checkpoint Setup & Predictor Initialization [COMPLETED]     │
│     │                                                                         │
│     ▼                                                                         │
│  Stage 3: Single-Patient/Slice Smoke Test with Tensor Input [COMPLETED]       │
│     │                                                                         │
│     ▼                                                                         │
│  Stage 4: Batch Tensor Inference Across All 251 Positive Slices [NEXT]        │
│     │                                                                         │
│     ▼                                                                         │
│  Stage 5: Quantitative Evaluation & 4-Panel Visualization Generator           │
│     │                                                                         │
│     ▼                                                                         │
│  Stage 6: Final Documentation & Optional Colab Execution Script               │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

### Stage 1: NIfTI Volume-to-Tensor Conversion & Data Alignment [COMPLETED]

- **Objective**: Load original 3D brain CT NIfTI volumes (`.nii` / `.nii.gz`), preserve calibrated HU intensities and spatial metadata, apply brain CT windowing, and convert the volume into the exact 4D PyTorch tensor `(D, 3, 512, 512)` expected by official MedSAM2's `init_state()`.
- **Inputs**:
  - `APIS_Dataset/{patient_id}/{patient_id}_ncct.nii.gz` (Raw 3D CT volumes)
  - `APIS_Dataset/{patient_id}/masks/{patient_id}_r1_mask.nii.gz` (Raw 3D ground-truth masks)
  - `phase_2_yolo26/dataset/labels/{split}/{patient_id}_slice{z:03d}.txt` (Phase 1 bounding boxes)
- **Operations**:
  1. Built `src/nifti_to_tensor.py` implementing `load_apis_nifti_volume()`, `apply_hu_windowing()`, and `ct_volume_to_medsam2_tensor()`.
  2. Verified true HU intensity reading using `nibabel.get_fdata()` (applying `scl_slope` and `scl_inter`).
  3. Applied brain CT windowing ($WL = 40, WW = 80 \implies [0.0, 80.0]\text{ HU}$).
  4. Resized each axial slice to $512 \times 512 \times 3$ RGB, scaled to $[0.0, 1.0]$, converted to PyTorch tensor, and applied ImageNet normalization.
  5. Built `test_nifti_to_tensor.py` and validated across multiple APIS cases (`train_000` to `train_003`), verifying `init_state()` compatibility invariants and numerical range `[-2.1179, +2.6400]`.
- **Outputs**:
  - Validated tensor conversion module producing `(D, 3, 512, 512)` `torch.float32` tensors.
  - Preserved metadata: $H_{\text{ori}}, W_{\text{ori}}$, 4x4 affine, voxel zooms, orientation `('L', 'A', 'S')`.
  - Passing validation test report (`test_nifti_to_tensor.py`).
- **Files Created**:
  - `MEDSAM for infarct/src/__init__.py`
  - `MEDSAM for infarct/src/nifti_to_tensor.py`
  - `MEDSAM for infarct/test_nifti_to_tensor.py`
  - `MEDSAM for infarct/README_NIFTI_TENSOR.md`

---

### Stage 1.5: Bounding Box Coordinate Conversion (YOLO -> MedSAM2 Pixel Coordinates) [COMPLETED]

- **Objective**: Parse Phase 1 bounding boxes from YOLO format (`[class_id, xc, yc, w, h] \in [0.0, 1.0]`), convert them to the absolute unnormalized pixel coordinates `[x_min, y_min, x_max, y_max]` expected by MedSAM2 on the $512 \times 512$ grid, support multi-component lesions (shape `(N, 4)`), and validate all 367 boxes across all 251 positive slices.
- **Inputs**:
  - `phase_2_yolo26/dataset/labels/{split}/{patient_id}_slice{z:03d}.txt` (All 251 positive slice label files).
  - Target model image resolution ($512 \times 512$).
- **Operations**:
  1. Built `src/box_parser.py`:
     - Reads YOLO normalized coordinates: $x_c, y_c, w, h \in [0.0, 1.0]$.
     - Converts to absolute pixel coordinates on a target $W \times H$ grid ($512 \times 512$):
       $$\begin{aligned}
       x_{\min} &= \max(0.0, (x_c - w / 2.0) \times W) \\
       y_{\min} &= \max(0.0, (y_c - h / 2.0) \times H) \\
       x_{\max} &= \min(W - 1.0, (x_c + w / 2.0) \times W) \\
       y_{\max} &= \min(H - 1.0, (y_c + h / 2.0) \times H)
       \end{aligned}$$
     - Handles slices with multiple boxes (multi-focal infarcts), returning an array of shape `(N, 4)`.
  2. Built `test_box_conversion.py` and validated:
     - All 251 positive label files parsed without error across all splits.
     - Total boxes matched exact count: 367 boxes.
     - Strict coordinate bounds check: $0 \le x_{\min} < x_{\max} \le 512$ and $0 \le y_{\min} < y_{\max} \le 512$ (100% passed).
     - Cross-checked against NIfTI ground-truth masks: verified that bounding boxes tightly cover the lesion pixels.
- **Outputs**:
  - Reusable box parsing and coordinate conversion module (`src/box_parser.py`).
  - Validation test report confirming all 367 bounding boxes are in valid MedSAM2 format and ready for prompt injection.
- **Files Created**:
  - `MEDSAM for infarct/src/box_parser.py`
  - `MEDSAM for infarct/test_box_conversion.py`

---

### Stage 2: MedSAM2 Checkpoint Setup & Predictor Wrapper Initialization [COMPLETED]

- **Objective**: Set up model configuration, place pretrained checkpoint weights, and build a unified inference wrapper that feeds converted 4D CT tensors directly into MedSAM2's predictor (`init_state()`).
- **Inputs**:
  - `MedSAM2/sam2/build_sam.py` (`build_sam2_video_predictor_npz` or `build_sam2`)
  - `MedSAM2/sam2/sam2_video_predictor_npz.py` (`SAM2VideoPredictorNPZ`)
  - `MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml`
  - Pretrained checkpoint: `MedSAM2_CTLesion.pt` (downloaded from Hugging Face `wanglab/MedSAM2`)
  - Box conversion module from Stage 1.5 (`bbox_format/box_parser.py`).
- **Operations**:
  1. Created directory `MEDSAM for infarct/checkpoints/` and `MEDSAM for infarct/predictor/`.
  2. Built `predictor/predictor_wrapper.py` implementing `MedSAM2InfarctPredictor`:
     - Dynamically manages paths and imports from `MedSAM2/`.
     - Supports both 4D video tensor mode (`init_patient_tensor()`) and 2D per-slice mode.
     - Implements `predict_slice_from_boxes(frame_idx, boxes)` handling multi-box union.
  3. Built `predictor/test_predictor_interface.py` and validated:
     - End-to-end tensor pipeline on real scan `train_001_ncct.nii.gz` (`(23, 3, 512, 512)`).
     - State initialization with native spatial geometry ($288 \times 256$).
     - Prompt injection on slice $z=16$ using Phase 1 bounding box, outputting valid $(512, 512)$ binary mask.
- **Outputs**:
  - Clean predictor wrapper class that accepts 4D CT tensors and coordinates prompts.
  - Validation test report confirming predictor state initialization and mask generation.
- **Files Created**:
  - `MEDSAM for infarct/checkpoints/` (directory for weights)
  - `MEDSAM for infarct/predictor/__init__.py`
  - `MEDSAM for infarct/predictor/predictor_wrapper.py`
  - `MEDSAM for infarct/predictor/test_predictor_interface.py`

---

### Stage 3: Single-Patient/Slice Smoke Test with Tensor Input [COMPLETED]

- **Objective**: Run an end-to-end smoke test on a single patient volume (`train_001`, positive slice $z=16$) feeding the converted tensor directly into MedSAM2's predictor, generating the predicted mask, comparing against the ground-truth NIfTI mask, and generating a 4-panel visual comparison.
- **Inputs**:
  - `APIS_Dataset/train_001/train_001_ncct.nii.gz` and mask `train_001_r1_mask.nii.gz`.
  - Phase 1 bounding box: `phase_2_yolo26/dataset/labels/test/train_001_slice016.txt`.
  - Converted tensor from Stage 1, converted box prompt from Stage 1.5, and predictor wrapper from Stage 2.
- **Operations**:
  1. Converted `train_001_ncct.nii.gz` to 4D tensor `(23, 3, 512, 512)` using `ct_volume_to_medsam2_tensor()`.
  2. Initialized predictor state with the tensor via `predictor.init_patient_tensor(video_tensor, video_height=288, video_width=256)`.
  3. Parsed Phase 1 box prompt for slice $z=16$: `[xmin=243.0, ymin=150.2, xmax=277.0, ymax=210.7]`.
  4. Executed slice prediction via `predict_slice_from_boxes(frame_idx=16, boxes=boxes)`.
  5. Loaded ground-truth mask slice $z=16$ from `train_001_r1_mask.nii.gz` and evaluated metrics:
     - Dice Similarity Coefficient (DSC): **0.6222**
     - Intersection over Union (IoU): **0.4516**
  6. Generated and saved a 4-panel visual comparison figure (`outputs/smoke_test_train_001_slice016.png`).
- **Outputs**:
  - End-to-end smoke test script (`test_single_slice.py`).
  - Evaluation modules: `evaluation/metrics.py` and `evaluation/visualization.py`.
  - 4-panel verification figure saved to `outputs/smoke_test_train_001_slice016.png`.
- **Files Created**:
  - `MEDSAM for infarct/evaluation/__init__.py`
  - `MEDSAM for infarct/evaluation/metrics.py`
  - `MEDSAM for infarct/evaluation/visualization.py`
  - `MEDSAM for infarct/test_single_slice.py`
  - `MEDSAM for infarct/outputs/smoke_test_train_001_slice016.png`

---

### Stage 4: Batch Tensor Inference Across All 251 Positive Slices

- **Objective**: Execute batch inference across all 251 positive slices from the APIS dataset, loading patient NIfTI volumes, converting each to a 4D tensor, running MedSAM2 prediction for all positive slices, and saving predicted masks.
- **Inputs**:
  - All 60 APIS patient NIfTI volumes.
  - All 251 positive slice label files in `phase_2_yolo26/dataset/labels/`.
  - Predictor wrapper from Stage 2.
- **Operations**:
  1. Build `run_medsam2_nifti_eval.py` with CLI flags: `--split` (`test`, `val`, `train`, or `all`), `--device` (`cuda` or `cpu`), `--save_vis`.
  2. Iterate through each patient:
     - Convert the full CT volume into the 4D tensor `(D, 3, 512, 512)`.
     - Initialize predictor state with the tensor via `init_patient_tensor()`.
     - For each positive slice $z$ belonging to the patient:
       - Retrieve all corresponding bounding boxes.
       - Predict masks for all lesion components and merge via binary OR.
       - Save predicted binary mask as PNG to `outputs/predictions/{split}/{patient_id}_slice{z:03d}_pred.png`.
  3. Cache/store per-slice metadata, execution times, and raw prediction arrays.
- **Outputs**:
  - Saved predicted binary masks for all 251 positive slices.
- **Files to Create**:
  - `MEDSAM for infarct/run_medsam2_nifti_eval.py`

---

### Stage 5: Quantitative Evaluation & 4-Panel Visualization Generator

- **Objective**: Compute rigorous evaluation metrics comparing predicted masks against original ground-truth NIfTI masks and generate publication-quality visual comparisons.
- **Inputs**:
  - Saved predicted masks from Stage 4.
  - Original NIfTI ground-truth masks in `APIS_Dataset/{patient_id}/masks/`.
- **Operations**:
  1. Create `src/metrics.py` to calculate:
     - Dice Similarity Coefficient (DSC)
     - Intersection over Union (IoU)
     - Relative Volume Error (RVE) / Lesion area in pixels
  2. Compute metrics for every slice and compile into `outputs/per_slice_metrics.csv`.
  3. Calculate global and per-split summary statistics (Mean, Median, Standard Deviation, IQR, Min, Max) and save to `outputs/summary_metrics.json`.
  4. Generate 4-panel visual comparison figures (Panel 1: Extracted CT Slice, Panel 2: Box Prompt, Panel 3: GT Mask, Panel 4: MedSAM2 Prediction with Dice/IoU overlay) for all test slices and save to `outputs/visualizations/{split}/`.
- **Outputs**:
  - `outputs/per_slice_metrics.csv`
  - `outputs/summary_metrics.json`
  - `outputs/visualizations/{split}/*.png`
- **Files to Create**:
  - `MEDSAM for infarct/src/metrics.py`
  - Visualization generator embedded inside `run_medsam2_nifti_eval.py`.

---

### Stage 6: Final Documentation & Optional Colab Execution Script

- **Objective**: Provide comprehensive documentation of experimental findings and a clean, self-contained Google Colab notebook / script for rapid GPU re-execution.
- **Inputs**:
  - Completed evaluation metrics, summary tables, and visualizations.
- **Operations**:
  1. Create `MEDSAM for infarct/README.md` with:
     - How to set up the environment and run inference locally or in Colab.
     - Final quantitative results table (Mean/Median Dice and IoU on test, val, and train splits).
     - Visual gallery showing representative best, average, and challenging segmentation cases.
  2. Create an optional standalone Colab notebook script (`run_medsam2_colab.ipynb`) for fast GPU evaluation.
- **Outputs**:
  - Complete project documentation and reproducibility guide.
- **Files to Create**:
  - `MEDSAM for infarct/README.md`
  - `MEDSAM for infarct/run_medsam2_colab.ipynb`

---

## 12. Conversion of .nii / .nii.gz to Tensor & Current Progress

### 12.1 What the MedSAM2 Input Tensor Is Exactly

A common misconception is that a tensor is simply an empty geometric container `(D, 3, 512, 512)`. In MedSAM2, **the tensor IS the voxel-by-voxel intensity data**.

Every single cell in the 4D tensor `[slice_idx, channel, y, x]` stores the exact calibrated tissue density value from the patient's CT scan:
- **Shape: `(D, 3, 512, 512)`**
  - **`D`**: Total number of sequential axial slices in superior-inferior anatomical order (e.g., 23 to 26 slices per APIS patient).
  - **`3`**: RGB color channels. Because CT is single-channel grayscale, the windowed intensity value is duplicated across all 3 channels.
  - **`512, 512`**: Spatial resolution expected by MedSAM2's Hiera vision transformer backbone.
- **Data Type**: `torch.float32` (or `bfloat16`/`float16` under GPU autocast).
- **Intensity Value Encoding**:
  - Raw CT scanner intensities are in Hounsfield Units (HU).
  - Brain CT windowing ($WL = 40, WW = 80 \implies [0.0, 80.0]\text{ HU}$) isolates brain tissue and stroke hypodensity from bone ($>80\text{ HU}$) and background air ($-1000\text{ HU}$).
  - Intensities are mapped to $[0.0, 1.0]$ and normalized using ImageNet statistics:
    $$\text{Tensor Value} = \frac{\text{Intensity}_{[0, 1]} - \text{mean}}{\text{std}}$$
    $$\text{where } \text{mean} = (0.485, 0.456, 0.406), \quad \text{std} = (0.229, 0.224, 0.225)$$
  - **Resulting Numerical Range**: Values range strictly between **`-2.1179`** (dark background / CSF) and **`+2.6400`** (bright bone / calcification).
  - Normal brain tissue has values around **`-0.2 to 0.0`**, while the **ischemic infarct lesion** is distinctly darker/hypodense with values around **`-1.1`**.

MedSAM2's vision transformer passes attention patches over these exact numbers to detect the contrast boundary between the infarct and normal brain tissue.

---

### 12.2 What Has Been Completed Till Now

We have fully implemented and verified the **NIfTI-to-Tensor Conversion Stage** without modifying `APIS_Dataset` or any existing Phase 1/Phase 2 code:

1. **`MEDSAM for infarct/src/__init__.py`**:
   - Initialized the standalone Python package for MedSAM2 infarct segmentation.

2. **`MEDSAM for infarct/src/nifti_to_tensor.py`**:
   - Implemented `load_apis_nifti_volume(nifti_path)`: Uses `nibabel.get_fdata()` to automatically apply slope and intercept (`scl_slope`, `scl_inter`), preserving true calibrated HU values. Preserves spatial metadata (affine matrix, voxel zooms, orientation codes `('L', 'A', 'S')`, and native dimensions $H_{\text{ori}}, W_{\text{ori}}$).
   - Implemented `apply_hu_windowing(volume, window_level, window_width)`: Clips raw HU values to $[WL - WW/2, WL + WW/2]$ and rescales to $[0.0, 255.0]$.
   - Implemented `ct_volume_to_medsam2_tensor(...)`: Re-arranges axial slices, resizes each 2D slice to $512 \times 512 \times 3$, rescales by $1/255.0$, converts to PyTorch tensor, and applies ImageNet mean/std normalization matching lines 220–257 of official `MedSAM2/medsam2_infer_3D_CT.py`.

3. **`MEDSAM for infarct/test_nifti_to_tensor.py`**:
   - Automated validation test script testing multiple real APIS patient volumes.
   - Verified that the generated tensors satisfy all invariant requirements of MedSAM2's `predictor.init_state()`.

4. **Validation Test Results (All Passed)**:
   - Evaluated cases:
     - `train_000`: Native `(568, 455, 26)` $\to$ Converted Tensor: **`(26, 3, 512, 512)`** | `torch.float32` | `PASSED`
     - `train_001`: Native `(288, 256, 23)` $\to$ Converted Tensor: **`(23, 3, 512, 512)`** | `torch.float32` | `PASSED`
     - `train_002`: Native `(256, 256, 26)` $\to$ Converted Tensor: **`(26, 3, 512, 512)`** | `torch.float32` | `PASSED`
     - `train_003`: Native `(256, 256, 26)` $\to$ Converted Tensor: **`(26, 3, 512, 512)`** | `torch.float32` | `PASSED`
   - Tensor values strictly verified in range `[-2.1179, +2.6400]`.
   - Native spatial resolution ($H_{\text{ori}}, W_{\text{ori}}$) and affine metadata are preserved for reconstructing predicted masks back to original 3D spatial dimensions.

5. **`MEDSAM for infarct/README_NIFTI_TENSOR.md`**:
   - Complete technical documentation and execution commands for the conversion pipeline.


