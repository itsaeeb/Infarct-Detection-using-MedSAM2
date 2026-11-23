"""
dataset_apis.py
===============
PyTorch Dataset for APIS Brain CT Infarct segmentation fine-tuning.
Directly reuses:
- `load_apis_nifti_volume` and `apply_hu_windowing` from `nifti_to_tensor`
- `parse_yolo_label_file` from `box_parser`
- Patient split definitions from `phase_2_yolo26/dataset/generation_stats.json`
"""

import os
import sys
import json
import glob
import random
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INFARCT_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(INFARCT_ROOT)

sys.path.insert(0, os.path.join(INFARCT_ROOT, "nifti_to_tensor"))
sys.path.insert(0, os.path.join(INFARCT_ROOT, "bbox_format"))

from nifti_to_tensor import load_apis_nifti_volume, apply_hu_windowing, IMAGENET_MEAN, IMAGENET_STD
from box_parser import parse_yolo_label_file

try:
    import nibabel as nib
except ImportError:
    raise ImportError("nibabel is required. Run: pip install nibabel")


class APISSliceDataset(Dataset):
    """
    Dataset of confirmed positive brain CT slices from the APIS dataset for MedSAM2 fine-tuning.
    """

    def __init__(
        self,
        split: str = "train",
        dataset_root: Optional[str] = None,
        labels_root: Optional[str] = None,
        target_size: int = 512,
        jitter_box: bool = True,
        max_jitter_ratio: float = 0.08,
        cache_volumes: bool = True
    ):
        """
        Parameters
        ----------
        split : str
            One of 'train', 'val', or 'test'.
        dataset_root : str, optional
            Path to APIS_Dataset directory.
        labels_root : str, optional
            Path to phase_2_yolo26/dataset directory.
        target_size : int
            Model image size (512).
        jitter_box : bool
            Whether to apply bounding box jitter during training.
        max_jitter_ratio : float
            Maximum box shift/scale noise ratio (default: 8%).
        cache_volumes : bool
            Cache loaded NIfTI volumes in memory to accelerate epoch iterations.
        """
        self.split = split
        self.target_size = target_size
        self.jitter_box = jitter_box and (split == "train")
        self.max_jitter_ratio = max_jitter_ratio
        self.cache_volumes = cache_volumes

        if dataset_root is None:
            dataset_root = os.path.join(PROJECT_ROOT, "APIS_Dataset")
        if labels_root is None:
            local_labels = os.path.join(INFARCT_ROOT, "labels")
            fallback_labels = os.path.join(PROJECT_ROOT, "phase_2_yolo26", "dataset")
            labels_root = local_labels if os.path.isdir(local_labels) else fallback_labels

        self.dataset_root = dataset_root
        self.labels_root = labels_root

        # Load patient splits from generation_stats.json
        stats_path = os.path.join(labels_root, "generation_stats.json")
        self.patient_splits = {}
        if os.path.isfile(stats_path):
            with open(stats_path, "r") as f:
                stats = json.load(f)
                for pid, pdata in stats.get("patients", {}).items():
                    self.patient_splits[pid] = pdata.get("split", "train")

        # Discover all positive label files for the given split (handles both labels/<split> and <split>)
        if os.path.isdir(os.path.join(labels_root, split)):
            split_labels_dir = os.path.join(labels_root, split)
        else:
            split_labels_dir = os.path.join(labels_root, "labels", split)

        label_files = sorted(glob.glob(os.path.join(split_labels_dir, "*.txt")))

        self.samples = []
        for lf in label_files:
            # Non-empty files contain bounding boxes (positive slices)
            if os.path.getsize(lf) > 0:
                base = os.path.splitext(os.path.basename(lf))[0]
                # parse {patient_id}_slice{slice_idx}
                parts = base.split("_slice")
                if len(parts) == 2:
                    pid = parts[0]
                    slice_idx = int(parts[1])
                    self.samples.append({
                        "patient_id": pid,
                        "slice_idx": slice_idx,
                        "label_path": lf
                    })

        self.mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32)[:, None, None]
        self.std  = torch.tensor(IMAGENET_STD,  dtype=torch.float32)[:, None, None]
        self.volume_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _get_patient_volumes(self, patient_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and cache CT volume and ground-truth mask volume."""
        if self.cache_volumes and patient_id in self.volume_cache:
            return self.volume_cache[patient_id]

        pdir = os.path.join(self.dataset_root, patient_id)
        ct_file = os.path.join(pdir, f"{patient_id}_ncct.nii.gz")
        if not os.path.isfile(ct_file):
            ct_file = os.path.join(pdir, f"{patient_id}_ncct.nii")

        mask_file = os.path.join(pdir, "masks", f"{patient_id}_r1_mask.nii.gz")
        if not os.path.isfile(mask_file):
            mask_file = os.path.join(pdir, "masks", f"{patient_id}_r1_mask.nii")

        ct_raw, _ = load_apis_nifti_volume(ct_file)
        mask_raw = np.round(nib.load(mask_file).get_fdata(dtype=np.float32)).astype(np.uint8)

        if self.cache_volumes:
            self.volume_cache[patient_id] = (ct_raw, mask_raw)

        return ct_raw, mask_raw

    def _apply_box_jitter(self, box: np.ndarray) -> np.ndarray:
        """Add random noise to bounding box coordinates during training."""
        x1, y1, x2, y2 = box
        w = max(4.0, x2 - x1)
        h = max(4.0, y2 - y1)

        dx1 = random.uniform(-self.max_jitter_ratio, self.max_jitter_ratio) * w
        dy1 = random.uniform(-self.max_jitter_ratio, self.max_jitter_ratio) * h
        dx2 = random.uniform(-self.max_jitter_ratio, self.max_jitter_ratio) * w
        dy2 = random.uniform(-self.max_jitter_ratio, self.max_jitter_ratio) * h

        new_x1 = max(0.0, min(float(self.target_size - 1), x1 + dx1))
        new_y1 = max(0.0, min(float(self.target_size - 1), y1 + dy1))
        new_x2 = max(0.0, min(float(self.target_size - 1), x2 + dx2))
        new_y2 = max(0.0, min(float(self.target_size - 1), y2 + dy2))

        if new_x1 >= new_x2:
            new_x1, new_x2 = min(x1, x2), max(x1, x2)
        if new_y1 >= new_y2:
            new_y1, new_y2 = min(y1, y2), max(y1, y2)

        return np.array([new_x1, new_y1, new_x2, new_y2], dtype=np.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_info = self.samples[idx]
        pid = sample_info["patient_id"]
        z = sample_info["slice_idx"]
        lbl_path = sample_info["label_path"]

        ct_vol, mask_vol = self._get_patient_volumes(pid)

        # Extract 2D axial slice
        raw_slice = ct_vol[:, :, z]
        gt_slice = mask_vol[:, :, z]

        # 1. Apply Brain Windowing [0, 80] HU -> [0, 255] uint8
        windowed_255 = apply_hu_windowing(raw_slice, window_level=40.0, window_width=80.0)
        uint8_slice = np.uint8(np.round(windowed_255))

        # 2. Convert to RGB PIL Image and resize to target_size (512x512)
        pil_img = Image.fromarray(uint8_slice).convert("RGB")
        if pil_img.size != (self.target_size, self.target_size):
            pil_img = pil_img.resize((self.target_size, self.target_size), resample=Image.BILINEAR)

        img_np = np.array(pil_img, dtype=np.float32).transpose(2, 0, 1) / 255.0  # (3, 512, 512)
        img_tensor = torch.from_numpy(img_np)
        img_tensor = (img_tensor - self.mean) / self.std

        # 3. Process Ground-Truth Mask
        if gt_slice.shape != (self.target_size, self.target_size):
            gt_resized = cv2.resize(gt_slice, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)
        else:
            gt_resized = gt_slice

        mask_tensor = torch.from_numpy((gt_resized > 0).astype(np.float32)).unsqueeze(0)  # (1, 512, 512)

        # 4. Parse Bounding Box Prompts
        boxes = parse_yolo_label_file(lbl_path, img_w=self.target_size, img_h=self.target_size)

        if len(boxes) == 0:
            # Fallback to mask bounds if file was empty
            y_indices, x_indices = np.where(gt_resized > 0)
            if len(x_indices) > 0:
                prompt_box = np.array([float(x_indices.min()), float(y_indices.min()), float(x_indices.max()), float(y_indices.max())], dtype=np.float32)
            else:
                prompt_box = np.array([0.0, 0.0, float(self.target_size), float(self.target_size)], dtype=np.float32)
        elif len(boxes) == 1:
            prompt_box = boxes[0]
        else:
            # Multi-component lesion: select enclosing outer bounding box
            min_x = float(np.min(boxes[:, 0]))
            min_y = float(np.min(boxes[:, 1]))
            max_x = float(np.max(boxes[:, 2]))
            max_y = float(np.max(boxes[:, 3]))
            prompt_box = np.array([min_x, min_y, max_x, max_y], dtype=np.float32)

        if self.jitter_box:
            prompt_box = self._apply_box_jitter(prompt_box)

        prompt_box_tensor = torch.from_numpy(prompt_box).float()

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "box": prompt_box_tensor,
            "patient_id": pid,
            "slice_idx": z
        }

