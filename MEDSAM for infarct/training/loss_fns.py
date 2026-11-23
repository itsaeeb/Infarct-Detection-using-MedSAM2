"""
loss_fns.py
===========
Loss functions for fine-tuning MedSAM2 on brain CT infarct segmentation:
1. Binary Cross-Entropy (BCE) with logits
2. Soft Dice Loss (handles severe class imbalance in stroke lesions)
3. IoU Prediction Loss (supervises MedSAM2's internal IoU prediction head)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : torch.Tensor
            Predicted raw logits of shape (B, 1, H, W).
        targets : torch.Tensor
            Ground-truth binary mask of shape (B, 1, H, W) in {0.0, 1.0}.
        """
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        cardinality = probs_flat.sum() + targets_flat.sum()

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice


class MedSAM2CompoundLoss(nn.Module):
    """
    Compound Loss matching official MedSAM2 training:
    Loss = w_bce * BCE + w_dice * Dice + w_iou * IoU_loss
    """
    def __init__(
        self,
        weight_bce: float = 20.0,
        weight_dice: float = 1.0,
        weight_iou: float = 1.0,
        smooth: float = 1.0
    ):
        super().__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.weight_iou = weight_iou

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = SoftDiceLoss(smooth=smooth)
        self.iou_loss = nn.SmoothL1Loss()

    def forward(
        self,
        pred_logits: torch.Tensor,
        gt_masks: torch.Tensor,
        pred_ious: torch.Tensor = None
    ) -> dict:
        """
        Parameters
        ----------
        pred_logits : torch.Tensor
            Predicted mask logits of shape (B, 1, H, W).
        gt_masks : torch.Tensor
            Ground truth binary masks of shape (B, 1, H, W).
        pred_ious : torch.Tensor, optional
            Predicted IoU scores of shape (B, 1).
        """
        bce = self.bce_loss(pred_logits, gt_masks.float())
        dice = self.dice_loss(pred_logits, gt_masks.float())

        total_loss = self.weight_bce * bce + self.weight_dice * dice

        loss_dict = {
            "total_loss": total_loss,
            "bce_loss": bce.item(),
            "dice_loss": dice.item()
        }

        if pred_ious is not None:
            # Calculate true IoU target
            probs = torch.sigmoid(pred_logits) > 0.5
            gt_bool = gt_masks > 0.5
            intersection = (probs & gt_bool).float().sum(dim=(1, 2, 3))
            union = (probs | gt_bool).float().sum(dim=(1, 2, 3))
            true_iou = (intersection + 1e-6) / (union + 1e-6)
            true_iou = true_iou.unsqueeze(1)

            iou_err = self.iou_loss(pred_ious, true_iou)
            total_loss = total_loss + self.weight_iou * iou_err
            loss_dict["total_loss"] = total_loss
            loss_dict["iou_loss"] = iou_err.item()

        return loss_dict

