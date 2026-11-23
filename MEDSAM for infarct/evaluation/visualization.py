"""
visualization.py
================
Generates 4-panel qualitative visualization figures comparing:
Panel 1: Brain CT Slice (Windowed WL=40, WW=80)
Panel 2: CT Slice with Phase 1 Bounding Box Overlay
Panel 3: CT Slice with Ground-Truth Mask Overlay (Green)
Panel 4: CT Slice with MedSAM2 Predicted Mask Overlay (Red/Cyan) with Dice and IoU
"""

import os
from typing import Optional, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def save_4panel_visualization(
    ct_slice_255: np.ndarray,
    boxes_px: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    save_path: str,
    patient_id: str,
    slice_idx: int,
    dice: float,
    iou: float
) -> str:
    """
    Generate and save a 4-panel visual comparison figure.

    Parameters
    ----------
    ct_slice_255 : np.ndarray
        2D CT slice scaled to [0, 255] uint8 (H, W).
    boxes_px : np.ndarray
        Bounding boxes in pixel coordinates [x_min, y_min, x_max, y_max] of shape (N, 4).
    gt_mask : np.ndarray
        Binary ground-truth mask (H, W).
    pred_mask : np.ndarray
        Binary predicted mask (H, W).
    save_path : str
        Target PNG output filepath.
    patient_id : str
        Identifier (e.g. 'train_001').
    slice_idx : int
        Axial slice index z.
    dice : float
        Dice score for title annotation.
    iou : float
        IoU score for title annotation.

    Returns
    -------
    save_path : str
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="black")

    # Panel 1: Original Brain CT
    axes[0].imshow(ct_slice_255, cmap="gray")
    axes[0].set_title(f"1. Brain CT (WL=40, WW=80)\n{patient_id} - Slice {slice_idx:03d}", color="white", fontsize=11)
    axes[0].axis("off")

    # Panel 2: CT with Bounding Box
    axes[1].imshow(ct_slice_255, cmap="gray")
    if boxes_px is not None and len(boxes_px) > 0:
        for b in boxes_px:
            x1, y1, x2, y2 = b
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor="#00ffff", facecolor="none"
            )
            axes[1].add_patch(rect)
    axes[1].set_title(f"2. Bounding Box Prompt\n(Count: {len(boxes_px)})", color="white", fontsize=11)
    axes[1].axis("off")

    # Panel 3: Ground-Truth Mask Overlay (Green)
    axes[2].imshow(ct_slice_255, cmap="gray")
    gt_overlay = np.zeros((*gt_mask.shape, 4), dtype=np.float32)
    gt_overlay[gt_mask > 0] = [0.0, 1.0, 0.0, 0.5]  # Green with 50% opacity
    axes[2].imshow(gt_overlay)
    axes[2].set_title(f"3. Ground-Truth Mask\n(Pixels: {int(np.sum(gt_mask > 0))})", color="white", fontsize=11)
    axes[2].axis("off")

    # Panel 4: MedSAM2 Predicted Mask Overlay (Cyan)
    axes[3].imshow(ct_slice_255, cmap="gray")
    pred_overlay = np.zeros((*pred_mask.shape, 4), dtype=np.float32)
    pred_overlay[pred_mask > 0] = [1.0, 0.2, 0.2, 0.5]  # Red-cyan with 50% opacity
    axes[3].imshow(pred_overlay)
    axes[3].set_title(f"4. MedSAM2 Prediction\nDice: {dice:.4f} | IoU: {iou:.4f}", color="#39ff14", fontsize=11)
    axes[3].axis("off")

    plt.tight_layout()
    normalized_path = os.path.normpath(save_path)
    if os.path.isfile(normalized_path):
        try:
            os.remove(normalized_path)
        except Exception:
            pass
    plt.savefig(normalized_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)

    return normalized_path
