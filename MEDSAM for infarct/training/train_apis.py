"""
train_apis.py
=============
Fine-tuning MedSAM2 specifically on the APIS Brain CT Infarct Dataset.

Workflow:
1. Loads pretrained MedSAM2 checkpoint (`MedSAM2_CTLesion.pt`) to initialize weights.
2. Freezes image encoder trunk by default, focusing fine-tuning on the Mask Decoder and Neck.
3. Loads positive slices from APIS using `dataset_apis.py` (which leverages `nifti_to_tensor`).
4. Trains with multi-task loss (Dice + BCE + IoU) and AdamW optimizer.
5. Evaluates validation Dice every epoch and saves `best_medsam2_apis.pt`.
"""

import os
import sys
import argparse
import time
import json
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INFARCT_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(INFARCT_ROOT)

sys.path.insert(0, INFARCT_ROOT)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(INFARCT_ROOT, "nifti_to_tensor"))
sys.path.insert(0, os.path.join(INFARCT_ROOT, "bbox_format"))

from sam2.build_sam import build_sam2
from dataset_apis import APISSliceDataset
from loss_fns import MedSAM2CompoundLoss


def compute_batch_dice_iou(pred_logits: torch.Tensor, gt_masks: torch.Tensor) -> Tuple[float, float]:
    """Calculate mean Dice and IoU for evaluation."""
    probs = (pred_logits > 0.0).float()
    targets = (gt_masks > 0.5).float()

    intersection = (probs * targets).sum(dim=(2, 3))
    cardinality = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    union = cardinality - intersection

    dice = ((2.0 * intersection + 1e-6) / (cardinality + 1e-6)).mean().item()
    iou = ((intersection + 1e-6) / (union + 1e-6)).mean().item()
    return dice, iou


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    scaler: Optional[torch.amp.GradScaler] = None
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss_accum = 0.0
    bce_loss_accum = 0.0
    dice_loss_accum = 0.0
    iou_loss_accum = 0.0
    steps = 0

    for step, batch in enumerate(dataloader, 1):
        images = batch["image"].to(device)    # (B, 3, 512, 512)
        masks  = batch["mask"].to(device)     # (B, 1, 512, 512)
        boxes  = batch["box"].to(device)      # (B, 4)

        optimizer.zero_grad()

        # Mixed precision forward pass if GPU
        use_amp = scaler is not None
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            # 1. Image Encoder
            out = model.forward_image(images)
            feat = out["backbone_fpn"][-1]
            high_res = out["backbone_fpn"][:2]

            # 2. Prompt Encoder
            sparse, dense = model.sam_prompt_encoder(points=None, boxes=boxes, masks=None)

            # 3. Mask Decoder
            low_res_masks, ious, _, _ = model.sam_mask_decoder(
                image_embeddings=feat,
                image_pe=model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
                repeat_image=False,
                high_res_features=high_res
            )

            # Upsample low-res masks (128x128) to input resolution (512x512)
            pred_logits = F.interpolate(
                low_res_masks.float(),
                size=(512, 512),
                mode="bilinear",
                align_corners=False
            )

            # 4. Compute Loss
            loss_dict = criterion(pred_logits, masks, pred_ious=ious)
            loss = loss_dict["total_loss"]

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=0.1)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=0.1)
            optimizer.step()

        total_loss_accum += loss.item()
        bce_loss_accum += loss_dict.get("bce_loss", 0.0)
        dice_loss_accum += loss_dict.get("dice_loss", 0.0)
        iou_loss_accum += loss_dict.get("iou_loss", 0.0)
        steps += 1

    return {
        "train_loss": total_loss_accum / max(1, steps),
        "train_bce": bce_loss_accum / max(1, steps),
        "train_dice": dice_loss_accum / max(1, steps),
        "train_iou_err": iou_loss_accum / max(1, steps)
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model on validation split."""
    model.eval()
    dice_list = []
    iou_list = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks  = batch["mask"].to(device)
            boxes  = batch["box"].to(device)

            out = model.forward_image(images)
            feat = out["backbone_fpn"][-1]
            high_res = out["backbone_fpn"][:2]

            sparse, dense = model.sam_prompt_encoder(points=None, boxes=boxes, masks=None)

            low_res_masks, _, _, _ = model.sam_mask_decoder(
                image_embeddings=feat,
                image_pe=model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
                repeat_image=False,
                high_res_features=high_res
            )

            pred_logits = F.interpolate(
                low_res_masks.float(),
                size=(512, 512),
                mode="bilinear",
                align_corners=False
            )

            d, j = compute_batch_dice_iou(pred_logits, masks)
            dice_list.append(d)
            iou_list.append(j)

    return {
        "val_mean_dice": float(np.mean(dice_list)) if dice_list else 0.0,
        "val_mean_iou": float(np.mean(iou_list)) if iou_list else 0.0
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MedSAM2 for Brain CT Infarct Segmentation")
    parser.add_argument("--epochs", type=int, default=40, help="Number of fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per training step")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for decoder and neck")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--unfreeze_trunk", action="store_true", help="Unfreeze Hiera backbone trunk")
    parser.add_argument("--device", type=str, default="", help="Compute device ('cuda' or 'cpu')")
    parser.add_argument("--init_ckpt", type=str, default="", help="Pretrained checkpoint for initialization")
    parser.add_argument("--output_dir", type=str, default="", help="Output checkpoint directory")
    parser.add_argument("--sanity_check", action="store_true", help="Run 1 epoch with 2 batches for validation")
    args = parser.parse_args()

    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Compute Device: {device}")

    # Checkpoint paths
    if not args.init_ckpt:
        args.init_ckpt = os.path.join(INFARCT_ROOT, "checkpoints", "MedSAM2_CTLesion.pt")
    if not os.path.isfile(args.init_ckpt):
        sys.exit(f"Pretrained checkpoint not found at: {args.init_ckpt}")

    if not args.output_dir:
        args.output_dir = os.path.join(INFARCT_ROOT, "checkpoints", "finetuned")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("MEDSAM2 BRAIN INFARCT FINE-TUNING")
    print("=" * 80)
    print(f"Pretrained Init Checkpoint: {args.init_ckpt}")
    print(f"Target Output Directory   : {args.output_dir}")
    print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size} | Base LR: {args.lr}")

    # 1. Build Model
    print("\n[Step 1/4] Building MedSAM2 Model and Loading Initial Weights...")
    model = build_sam2(
        config_file="configs/sam2.1_hiera_t512.yaml",
        ckpt_path=args.init_ckpt,
        device=device,
        mode="train"
    )

    # Freeze or Unfreeze Trunk
    if not args.unfreeze_trunk:
        print("  - Freezing Hiera Backbone Trunk (Preserving general CT representations)")
        for p in model.image_encoder.trunk.parameters():
            p.requires_grad = False
    else:
        print("  - Backbone Trunk is Trainable")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  - Trainable Parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.1f}%)")

    # 2. Datasets & Loaders
    print("\n[Step 2/4] Loading APIS CT Slices via dataset_apis...")
    train_dataset = APISSliceDataset(split="train", jitter_box=True)
    val_dataset   = APISSliceDataset(split="val", jitter_box=False)

    if args.sanity_check:
        print("  - [SANITY CHECK MODE] Subsetting to 4 train and 2 val slices")
        train_dataset.samples = train_dataset.samples[:4]
        val_dataset.samples = val_dataset.samples[:2]
        args.epochs = 1

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )
    print(f"  - Train Positive Slices: {len(train_dataset)}")
    print(f"  - Val Positive Slices  : {len(val_dataset)}")

    # 3. Optimizer, Scheduler, & Loss
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.1
    )
    criterion = MedSAM2CompoundLoss(weight_bce=20.0, weight_dice=1.0, weight_iou=1.0)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # 4. Training Loop
    print("\n[Step 3/4] Starting Fine-Tuning Loop...")
    best_val_dice = 0.0
    history = []

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        ep_start = time.time()
        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            scaler=scaler
        )
        val_metrics = evaluate(model=model, dataloader=val_loader, device=device)
        scheduler.step()

        ep_duration = time.time() - ep_start
        val_dice = val_metrics["val_mean_dice"]
        val_iou = val_metrics["val_mean_iou"]

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Loss: {train_metrics['train_loss']:.4f} (Dice: {train_metrics['train_dice']:.4f}) | "
            f"Val DSC: {val_dice:.4f} | Val IoU: {val_iou:.4f} | "
            f"Time: {ep_duration:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["train_loss"],
            "val_dice": val_dice,
            "val_iou": val_iou
        })

        # Save Best Model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_ckpt_path = os.path.join(args.output_dir, "best_medsam2_apis.pt")
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_dice": val_dice}, best_ckpt_path)
            print(f"  >>> Best Checkpoint Saved to {best_ckpt_path} (DSC: {val_dice:.4f})")

        # Save Latest Model
        latest_ckpt_path = os.path.join(args.output_dir, "latest_medsam2_apis.pt")
        torch.save({"model": model.state_dict(), "epoch": epoch, "val_dice": val_dice}, latest_ckpt_path)

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"FINE-TUNING COMPLETED in {total_time / 60.0:.1f} minutes")
    print(f"Best Validation Dice: {best_val_dice:.4f}")
    print("=" * 80)

    # Save training metrics history
    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to: {history_path}")


if __name__ == "__main__":
    main()
