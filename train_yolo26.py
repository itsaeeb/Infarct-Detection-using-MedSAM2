"""
train_yolo26.py
===============
Stage 2 of the APIS -> YOLO26 pipeline.

Fine-tunes a pretrained YOLO26 model (yolo26n.pt) on the brain CT infarct
dataset and evaluates performance metrics (Precision, Sensitivity/Recall,
mAP@0.5, mAP@0.5:0.95, and IoU).

Usage:
    python train_yolo26.py --epochs 30 --batch 16 --imgsz 512

Arguments:
    --model     Pretrained YOLO26 weights (default: 'yolo26n.pt')
    --data      Path to dataset.yaml (default: 'dataset/dataset.yaml')
    --epochs    Number of training epochs (default: 30)
    --batch     Batch size (default: 16)
    --imgsz     Image size in pixels (default: 512)
    --device    Device to train on ('cpu', 'cuda', or auto-detect)
    --project   Directory to save runs (default: 'runs/detect')
    --name      Run experiment name (default: 'yolo26_infarct')
"""

import os
import sys
import json
import argparse
import logging
import torch

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("Ultralytics not installed. Run:  pip install ultralytics")

# =============================================================================
# DEFAULTS
# =============================================================================
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA    = os.path.join(SCRIPT_DIR, "dataset", "dataset.yaml")
DEFAULT_MODEL   = os.path.join(SCRIPT_DIR, "yolo26n.pt")
DEFAULT_PROJECT = os.path.join(SCRIPT_DIR, "runs", "detect")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO26 on Brain Infarct Dataset")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Path to pretrained YOLO26 weights (e.g. yolo26n.pt)")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA,
                        help="Path to dataset.yaml")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16,
                        help="Training batch size")
    parser.add_argument("--imgsz", type=int, default=512,
                        help="Image resolution (square)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 'cuda', 'cpu', or None for auto-detect")
    parser.add_argument("--project", type=str, default=DEFAULT_PROJECT,
                        help="Save project directory (absolute path)")
    parser.add_argument("--name", type=str, default="yolo26_infarct",
                        help="Experiment name")
    return parser.parse_args()


def determine_device(requested_device):
    """Determine the optimal computation device."""
    if requested_device is not None:
        return requested_device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"[INFO] CUDA GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        num_threads = os.cpu_count() or 4
        torch.set_num_threads(num_threads)
        print(f"[INFO] CUDA GPU not detected. Training on CPU with {num_threads} threads.")
    return device


def main():
    args = parse_args()
    device = determine_device(args.device)

    # Verify dataset.yaml existence
    dataset_yaml = os.path.abspath(args.data)
    if not os.path.isfile(dataset_yaml):
        sys.exit(f"[ERROR] Dataset configuration not found: {dataset_yaml}\n"
                 f"  Make sure generate_yolo_labels.py has been run first.")

    # Ensure project output directory exists
    os.makedirs(args.project, exist_ok=True)

    print("\n" + "=" * 65)
    print("  YOLO26 Fine-Tuning -- Brain Infarct Detection")
    print("=" * 65)
    print(f"  Pretrained model : {args.model}")
    print(f"  Dataset config   : {dataset_yaml}")
    print(f"  Epochs           : {args.epochs}")
    print(f"  Batch size       : {args.batch}")
    print(f"  Image size       : {args.imgsz}")
    print(f"  Device           : {device}")
    print(f"  Output directory : {os.path.join(args.project, args.name)}")
    print("=" * 65 + "\n")

    # 1. Load Pretrained YOLO26
    print(f"[1/3] Loading pretrained YOLO26 model from '{args.model}'...")
    model = YOLO(args.model)

    # 2. Fine-tune on the brain CT infarct dataset
    print(f"\n[2/3] Starting fine-tuning for {args.epochs} epochs...\n")
    # For medical CT scans:
    # - fliplr=0.5: brain has bilateral symmetry
    # - flipud=0.0: avoid anatomically inverted scans
    # - mosaic=0.0: avoid 4-image random tiling composites
    # - degrees=5.0: minor patient head rotation tolerance
    # - cache=True: keep processed images in RAM for fast epoch transitions
    # - workers=0: avoid Windows multiprocessing IPC bottlenecks
    train_results = model.train(
        data=dataset_yaml,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        cache=True,
        mosaic=0.0,
        fliplr=0.5,
        flipud=0.0,
        degrees=5.0,
        workers=0,
        save=True,
        verbose=True,
    )

    # Resolve training output directory
    save_dir = getattr(train_results, "save_dir", None)
    if not save_dir or not os.path.isdir(str(save_dir)):
        save_dir = os.path.join(args.project, args.name)
    else:
        save_dir = str(save_dir)

    print(f"\n[INFO] Training complete. Run saved to: {save_dir}")

    # 3. Final Evaluation on Held-Out Test Set
    print("\n" + "=" * 65)
    print("  [3/3] Evaluating Fine-Tuned Model on Test Set")
    print("=" * 65)
    best_weights = os.path.join(save_dir, "weights", "best.pt")
    if not os.path.isfile(best_weights):
        # Fallback to model if best.pt is not at standard path
        best_weights = os.path.join(args.project, args.name, "weights", "best.pt")

    print(f"  Loading best weights: {best_weights}")
    eval_model = YOLO(best_weights) if os.path.isfile(best_weights) else model

    test_metrics = eval_model.val(
        data=dataset_yaml,
        split="test",
        imgsz=args.imgsz,
        device=device,
        project=os.path.join(save_dir, "eval"),
        name="test_split",
        exist_ok=True,
    )

    # Extract performance metrics
    precision = float(test_metrics.box.p[0]) if len(test_metrics.box.p) > 0 else 0.0
    sensitivity = float(test_metrics.box.r[0]) if len(test_metrics.box.r) > 0 else 0.0
    map50 = float(test_metrics.box.map50)
    map75 = float(test_metrics.box.map75)
    map50_95 = float(test_metrics.box.map)

    f1 = (2 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) > 0 else 0.0

    print("\n" + "=" * 65)
    print("  FINAL TEST SET PERFORMANCE METRICS")
    print("=" * 65)
    print(f"  Precision                 : {precision:.4f}  ({precision*100:.2f}%)")
    print(f"  Sensitivity (Recall)      : {sensitivity:.4f}  ({sensitivity*100:.2f}%)")
    print(f"  F1-Score                  : {f1:.4f}  ({f1*100:.2f}%)")
    print(f"  mAP @ IoU 0.50            : {map50:.4f}  ({map50*100:.2f}%)")
    print(f"  mAP @ IoU 0.75            : {map75:.4f}  ({map75*100:.2f}%)")
    print(f"  mAP @ IoU 0.50:0.95       : {map50_95:.4f}  ({map50_95*100:.2f}%)")
    print("=" * 65)

    # Save summary reports
    summary = {
        "model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": device,
        "save_dir": save_dir,
        "best_weights": best_weights if os.path.isfile(best_weights) else "in_memory",
        "test_metrics": {
            "precision": precision,
            "sensitivity_recall": sensitivity,
            "f1_score": f1,
            "map_50": map50,
            "map_75": map75,
            "map_50_95": map50_95,
        }
    }
    
    # Save inside run directory and root workspace directory
    summary_path_run = os.path.join(save_dir, "test_evaluation_summary.json")
    with open(summary_path_run, "w") as f:
        json.dump(summary, f, indent=2)

    summary_path_root = os.path.join(SCRIPT_DIR, "test_evaluation_summary.json")
    with open(summary_path_root, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[INFO] Summary saved to: {summary_path_run}")
    print(f"[INFO] Summary copy saved to: {summary_path_root}")
    print(f"[INFO] Best model weights: {best_weights}\n")


if __name__ == "__main__":
    main()

