"""
metrics.py
==========
Quantitative evaluation metrics for Brain Infarct Segmentation.

Computes:
1. Dice Similarity Coefficient (DSC): 2 * |P \cap G| / (|P| + |G|)
2. Intersection over Union (IoU): |P \cap G| / (|P \cup G|)
3. Area statistics: lesion area in pixels and relative difference.
"""

from typing import Tuple, Dict, Any
import numpy as np


def compute_dice_iou(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    eps: float = 1e-6
) -> Tuple[float, float]:
    """
    Compute Dice Similarity Coefficient and IoU between predicted and ground-truth binary masks.

    Parameters
    ----------
    pred_mask : np.ndarray
        Binary predicted mask (values in {0, 1} or boolean).
    gt_mask : np.ndarray
        Binary ground-truth mask (values in {0, 1} or boolean).
    eps : float
        Small epsilon to prevent division by zero (default: 1e-6).

    Returns
    -------
    dice : float
        Dice score in range [0.0, 1.0].
    iou : float
        IoU score in range [0.0, 1.0].
    """
    p = (pred_mask > 0).astype(np.float32)
    g = (gt_mask > 0).astype(np.float32)

    intersection = np.sum(p * g)
    area_pred = np.sum(p)
    area_gt = np.sum(g)
    union = area_pred + area_gt - intersection

    dice = (2.0 * intersection + eps) / (area_pred + area_gt + eps)
    iou = (intersection + eps) / (union + eps)

    return float(dice), float(iou)


def evaluate_slice_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    eps: float = 1e-6
) -> Dict[str, Any]:
    """
    Compute a complete dictionary of evaluation metrics for a single slice:
    Dice, IoU, Precision, Recall, and pixel counts.
    """
    p = (pred_mask > 0).astype(np.float32)
    g = (gt_mask > 0).astype(np.float32)

    intersection = float(np.sum(p * g))
    area_pred = float(np.sum(p))
    area_gt = float(np.sum(g))
    union = area_pred + area_gt - intersection

    dice = (2.0 * intersection + eps) / (area_pred + area_gt + eps)
    iou = (intersection + eps) / (union + eps)
    precision = (intersection + eps) / (area_pred + eps) if area_pred > 0 else 0.0
    recall = (intersection + eps) / (area_gt + eps) if area_gt > 0 else 0.0

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "pred_pixels": int(area_pred),
        "gt_pixels": int(area_gt),
        "pixel_diff": int(area_pred - area_gt)
    }
