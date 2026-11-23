"""
evaluation package
"""
try:
    from .metrics import compute_dice_iou, evaluate_slice_metrics
    from .visualization import save_4panel_visualization
except (ImportError, ValueError):
    from metrics import compute_dice_iou, evaluate_slice_metrics
    from visualization import save_4panel_visualization

__all__ = [
    "compute_dice_iou",
    "evaluate_slice_metrics",
    "save_4panel_visualization",
]
