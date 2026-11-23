"""
batch_eval.py
=============
Batch evaluation script for MedSAM2 brain infarct segmentation on the APIS dataset.

Features:
1. Evaluates all slices with non-empty bounding box prompts in a target split (default: test split = 18 slices).
2. Groups slices by patient to load each 3D CT volume and initialize predictor state only once.
3. Computes per-slice Dice, IoU, Precision, Recall, and pixel counts.
4. Generates 4-panel visual overlays for each slice.
5. Optionally compares Fine-Tuned vs. Pretrained model side-by-side.
6. Exports full results to CSV and JSON summaries in MEDSAM for infarct/outputs/.
"""

import os
import sys
import glob
import json
import argparse
from typing import Dict, List, Any, Optional
import numpy as np
import torch
import cv2

# Set Matplotlib cache dir to prevent permission warnings
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", ".mplcache")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INFARCT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROJECT_ROOT = os.path.dirname(INFARCT_ROOT)

sys.path.insert(0, os.path.join(INFARCT_ROOT, "nifti_to_tensor"))
sys.path.insert(0, os.path.join(INFARCT_ROOT, "bbox_format"))
sys.path.insert(0, os.path.join(INFARCT_ROOT, "predictor"))
sys.path.insert(0, os.path.join(INFARCT_ROOT, "evaluation"))

from nifti_to_tensor import load_apis_nifti_volume, ct_volume_to_medsam2_tensor, apply_hu_windowing
from box_parser import parse_yolo_label_file
from predictor_wrapper import MedSAM2InfarctPredictor, DEFAULT_CONFIG_PATH
from metrics import evaluate_slice_metrics
from visualization import save_4panel_visualization

try:
    import nibabel as nib
except ImportError:
    sys.exit("nibabel is required. Run: pip install nibabel")


def find_positive_slices(split: str = "test") -> List[Dict[str, Any]]:
    """
    Find all non-empty label files in the specified split directory.
    Prioritizes self-contained 'MEDSAM for infarct/labels/<split>' directory.
    """
    local_dir = os.path.join(INFARCT_ROOT, "labels", split)
    fallback_dir = os.path.join(PROJECT_ROOT, "phase_2_yolo26", "dataset", "labels", split)
    labels_dir = local_dir if os.path.isdir(local_dir) else fallback_dir

    if not os.path.isdir(labels_dir):
        raise FileNotFoundError(f"Labels directory not found at: {labels_dir}")

    label_files = sorted(glob.glob(os.path.join(labels_dir, "*.txt")))
    positive_slices = []

    for label_path in label_files:
        if os.path.getsize(label_path) == 0:
            continue
        with open(label_path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        if len(lines) == 0:
            continue

        fname = os.path.basename(label_path)
        # Expected format: <patient_id>_slice<slice_idx>.txt
        parts = fname.replace(".txt", "").split("_slice")
        if len(parts) != 2:
            continue
        patient_id = parts[0]
        slice_idx = int(parts[1])

        positive_slices.append({
            "patient_id": patient_id,
            "slice_idx": slice_idx,
            "label_path": label_path,
            "num_boxes": len(lines)
        })

    return positive_slices


def evaluate_model(
    checkpoint_path: str,
    slices: List[Dict[str, Any]],
    output_dir: str,
    device: torch.device,
    save_visuals: bool = True,
    model_tag: str = "finetuned"
) -> List[Dict[str, Any]]:
    """
    Run evaluation for a specific model checkpoint on the slice list.
    """
    print(f"\nEvaluating Model [{model_tag}]: {checkpoint_path}")
    print(f"Total slices to evaluate: {len(slices)}")

    predictor = MedSAM2InfarctPredictor(
        checkpoint_path=checkpoint_path,
        config_path=DEFAULT_CONFIG_PATH,
        device=device,
        mode="video_npz"
    )

    # Group slices by patient
    patient_slices: Dict[str, List[Dict[str, Any]]] = {}
    for item in slices:
        pid = item["patient_id"]
        patient_slices.setdefault(pid, []).append(item)

    results = []
    vis_dir = os.path.join(output_dir, f"visuals_{model_tag}")
    if save_visuals:
        os.makedirs(vis_dir, exist_ok=True)

    for pid_idx, (patient_id, p_items) in enumerate(patient_slices.items(), 1):
        ct_nii_path = os.path.join(PROJECT_ROOT, "APIS_Dataset", patient_id, f"{patient_id}_ncct.nii.gz")
        mask_nii_path = os.path.join(PROJECT_ROOT, "APIS_Dataset", patient_id, "masks", f"{patient_id}_r1_mask.nii.gz")
        if not os.path.isfile(mask_nii_path):
            mask_nii_path = os.path.join(PROJECT_ROOT, "APIS_Dataset", patient_id, "masks", f"{patient_id}_r1_mask.nii")

        if not os.path.isfile(ct_nii_path) or not os.path.isfile(mask_nii_path):
            print(f"  [Skip] Missing NIfTI volume or mask for {patient_id}")
            continue

        print(f"[{pid_idx}/{len(patient_slices)}] Loading patient volume {patient_id} ({len(p_items)} positive slices)...")
        raw_vol, meta = load_apis_nifti_volume(ct_nii_path)
        video_tensor, _ = ct_volume_to_medsam2_tensor(raw_vol, meta, device=device)

        # Initialize predictor state once for this patient
        predictor.init_patient_tensor(
            video_tensor=video_tensor,
            video_height=meta["orig_shape"][0],
            video_width=meta["orig_shape"][1],
            patient_id=patient_id
        )

        # Load 3D GT mask volume
        mask_img = nib.load(mask_nii_path)
        gt_vol = np.round(mask_img.get_fdata(dtype=np.float32)).astype(np.uint8)

        for item in p_items:
            slice_idx = item["slice_idx"]
            label_path = item["label_path"]

            # Parse bounding boxes in 512x512 space
            boxes = parse_yolo_label_file(label_path, img_w=512, img_h=512)

            # Run inference
            pred_mask = predictor.predict_slice_from_boxes(frame_idx=slice_idx, boxes=boxes)

            # Ground truth slice resized to 512x512
            gt_slice_native = gt_vol[:, :, slice_idx]
            gt_slice_512 = cv2.resize(gt_slice_native, (512, 512), interpolation=cv2.INTER_NEAREST)

            # Compute metrics
            m = evaluate_slice_metrics(pred_mask, gt_slice_512)

            row = {
                "patient_id": patient_id,
                "slice_idx": slice_idx,
                "num_boxes": len(boxes),
                "dice": m["dice"],
                "iou": m["iou"],
                "precision": m["precision"],
                "recall": m["recall"],
                "gt_pixels": m["gt_pixels"],
                "pred_pixels": m["pred_pixels"],
                "pixel_diff": m["pixel_diff"],
                "model": model_tag
            }
            results.append(row)

            print(f"    - Slice {slice_idx:03d} | Boxes: {len(boxes)} | DSC: {m['dice']:.4f} | IoU: {m['iou']:.4f} | Prec: {m['precision']:.4f} | Rec: {m['recall']:.4f}")

            if save_visuals:
                ct_slice_raw = raw_vol[:, :, slice_idx]
                ct_slice_win = apply_hu_windowing(ct_slice_raw, window_level=40.0, window_width=80.0)
                ct_slice_512 = cv2.resize(ct_slice_win.astype(np.uint8), (512, 512), interpolation=cv2.INTER_LINEAR)

                vis_path = os.path.join(vis_dir, f"{patient_id}_slice{slice_idx:03d}_{model_tag}.png")
                save_4panel_visualization(
                    ct_slice_255=ct_slice_512,
                    boxes_px=boxes,
                    gt_mask=gt_slice_512,
                    pred_mask=pred_mask,
                    save_path=vis_path,
                    patient_id=patient_id,
                    slice_idx=slice_idx,
                    dice=m["dice"],
                    iou=m["iou"]
                )

    return results


def export_results(
    results: List[Dict[str, Any]],
    output_dir: str,
    prefix: str
) -> Dict[str, float]:
    """
    Save CSV and summary JSON metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{prefix}_per_slice.csv")
    json_path = os.path.join(output_dir, f"{prefix}_summary.json")

    # Write CSV
    if results:
        headers = list(results[0].keys())
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(headers) + "\n")
            for r in results:
                row_str = ",".join(str(r[h]) for h in headers)
                f.write(row_str + "\n")

    # Compute summary
    dices = [r["dice"] for r in results]
    ious = [r["iou"] for r in results]
    precs = [r["precision"] for r in results]
    recs = [r["recall"] for r in results]

    summary = {
        "num_slices": len(results),
        "mean_dice": float(np.mean(dices)) if dices else 0.0,
        "std_dice": float(np.std(dices)) if dices else 0.0,
        "median_dice": float(np.median(dices)) if dices else 0.0,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "std_iou": float(np.std(ious)) if ious else 0.0,
        "median_iou": float(np.median(ious)) if ious else 0.0,
        "mean_precision": float(np.mean(precs)) if precs else 0.0,
        "mean_recall": float(np.mean(recs)) if recs else 0.0
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="MedSAM2 Batch Evaluation")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "train"], help="Split to evaluate")
    parser.add_argument("--finetuned_ckpt", type=str, default=None, help="Path to fine-tuned checkpoint")
    parser.add_argument("--pretrained_ckpt", type=str, default=None, help="Path to pretrained baseline checkpoint")
    parser.add_argument("--compare_pretrained", action="store_true", help="Compare fine-tuned vs pretrained baseline")
    parser.add_argument("--no_visuals", action="store_true", help="Disable saving 4-panel visual overlays")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("MEDSAM2 QUANTITATIVE BATCH EVALUATION")
    print(f"Device: {device} | Target Split: {args.split}")
    print("=" * 80)

    # Resolve checkpoints
    finetuned_path = args.finetuned_ckpt or os.path.join(INFARCT_ROOT, "checkpoints", "finetuned", "best_medsam2_apis.pt")
    pretrained_path = args.pretrained_ckpt or os.path.join(INFARCT_ROOT, "checkpoints", "MedSAM2_CTLesion.pt")

    if not os.path.isfile(finetuned_path):
        sys.exit(f"Fine-tuned checkpoint not found at: {finetuned_path}")

    # Discover positive slices
    slices = find_positive_slices(args.split)
    print(f"Discovered {len(slices)} positive slices with bounding boxes in split '{args.split}'.")

    output_dir = os.path.join(INFARCT_ROOT, "outputs", f"eval_{args.split}")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Run Fine-Tuned Evaluation
    ft_results = evaluate_model(
        checkpoint_path=finetuned_path,
        slices=slices,
        output_dir=output_dir,
        device=device,
        save_visuals=not args.no_visuals,
        model_tag="finetuned"
    )
    ft_summary = export_results(ft_results, output_dir, prefix="metrics_finetuned")

    # 2. Run Pretrained Evaluation (if requested)
    pt_summary = None
    if args.compare_pretrained and os.path.isfile(pretrained_path):
        pt_results = evaluate_model(
            checkpoint_path=pretrained_path,
            slices=slices,
            output_dir=output_dir,
            device=device,
            save_visuals=not args.no_visuals,
            model_tag="pretrained"
        )
        pt_summary = export_results(pt_results, output_dir, prefix="metrics_pretrained")

        # Export side-by-side comparison CSV
        comparison_path = os.path.join(output_dir, "comparison_per_slice.csv")
        with open(comparison_path, "w", encoding="utf-8") as f:
            f.write("patient_id,slice_idx,num_boxes,pretrained_dice,finetuned_dice,dice_gain,pretrained_iou,finetuned_iou,iou_gain\n")
            for r_pt, r_ft in zip(pt_results, ft_results):
                dice_gain = r_ft["dice"] - r_pt["dice"]
                iou_gain = r_ft["iou"] - r_pt["iou"]
                f.write(
                    f"{r_ft['patient_id']},{r_ft['slice_idx']},{r_ft['num_boxes']},"
                    f"{r_pt['dice']:.4f},{r_ft['dice']:.4f},{dice_gain:+.4f},"
                    f"{r_pt['iou']:.4f},{r_ft['iou']:.4f},{iou_gain:+.4f}\n"
                )

    # Print Final Summary Table
    print("\n" + "=" * 80)
    print(f"QUANTITATIVE EVALUATION SUMMARY (Split: {args.split}, {len(slices)} Slices)")
    print("=" * 80)
    print(f"{'Metric':<20} | {'Fine-Tuned (best_medsam2_apis.pt)':<32} | {'Pretrained (Baseline)':<25} | {'Gain':<10}")
    print("-" * 95)

    if pt_summary is not None:
        d_gain = ft_summary["mean_dice"] - pt_summary["mean_dice"]
        iou_gain = ft_summary["mean_iou"] - pt_summary["mean_iou"]
        p_gain = ft_summary["mean_precision"] - pt_summary["mean_precision"]
        r_gain = ft_summary["mean_recall"] - pt_summary["mean_recall"]

        print(f"{'Mean Dice (DSC)':<20} | {ft_summary['mean_dice']:.4f} +/- {ft_summary['std_dice']:.4f}{'':<14} | {pt_summary['mean_dice']:.4f} +/- {pt_summary['std_dice']:.4f}{'':<7} | {d_gain:+.4f} ({d_gain*100:+.2f}%)")
        print(f"{'Median Dice':<20} | {ft_summary['median_dice']:.4f}{'':<26} | {pt_summary['median_dice']:.4f}{'':<19} | {ft_summary['median_dice'] - pt_summary['median_dice']:+.4f}")
        print(f"{'Mean IoU':<20} | {ft_summary['mean_iou']:.4f} +/- {ft_summary['std_iou']:.4f}{'':<15} | {pt_summary['mean_iou']:.4f} +/- {pt_summary['std_iou']:.4f}{'':<8} | {iou_gain:+.4f} ({iou_gain*100:+.2f}%)")
        print(f"{'Median IoU':<20} | {ft_summary['median_iou']:.4f}{'':<26} | {pt_summary['median_iou']:.4f}{'':<19} | {ft_summary['median_iou'] - pt_summary['median_iou']:+.4f}")
        print(f"{'Mean Precision':<20} | {ft_summary['mean_precision']:.4f}{'':<26} | {pt_summary['mean_precision']:.4f}{'':<19} | {p_gain:+.4f}")
        print(f"{'Mean Recall':<20} | {ft_summary['mean_recall']:.4f}{'':<26} | {pt_summary['mean_recall']:.4f}{'':<19} | {r_gain:+.4f}")
    else:
        print(f"{'Mean Dice (DSC)':<20} | {ft_summary['mean_dice']:.4f} +/- {ft_summary['std_dice']:.4f}")
        print(f"{'Median Dice':<20} | {ft_summary['median_dice']:.4f}")
        print(f"{'Mean IoU':<20} | {ft_summary['mean_iou']:.4f} +/- {ft_summary['std_iou']:.4f}")
        print(f"{'Median IoU':<20} | {ft_summary['median_iou']:.4f}")
        print(f"{'Mean Precision':<20} | {ft_summary['mean_precision']:.4f}")
        print(f"{'Mean Recall':<20} | {ft_summary['mean_recall']:.4f}")

    print("=" * 80)
    print(f"Results exported to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

