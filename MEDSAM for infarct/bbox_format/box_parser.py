"""
box_parser.py
=============
Module for parsing Phase 1 YOLO bounding boxes and converting them to absolute
pixel coordinates expected by MedSAM2 ([x_min, y_min, x_max, y_max]).

Format Details:
---------------
- YOLO Format (Input):
    <class_id> <x_center> <y_center> <width> <height>
    where all coordinate values are normalized between [0.0, 1.0].
- MedSAM2 Format (Output):
    [x_min, y_min, x_max, y_max] in unnormalized pixel values.
    For standard MedSAM2 Hiera backbone: W = 512, H = 512.
- Multi-component Support:
    If a slice has N lesions (multiple lines in label file), returns np.ndarray of shape (N, 4).
"""

import os
import re
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


TARGET_GRID_SIZE = 512  # Standard MedSAM2 model image size


def yolo_to_pixel_box(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    img_w: int = TARGET_GRID_SIZE,
    img_h: int = TARGET_GRID_SIZE
) -> np.ndarray:
    """
    Convert a single normalized YOLO bounding box to absolute pixel coordinates.

    Parameters
    ----------
    x_center : float
        Horizontal box center in range [0.0, 1.0].
    y_center : float
        Vertical box center in range [0.0, 1.0].
    width : float
        Box width in range [0.0, 1.0].
    height : float
        Box height in range [0.0, 1.0].
    img_w : int
        Image width in pixels (default: 512).
    img_h : int
        Image height in pixels (default: 512).

    Returns
    -------
    box : np.ndarray
        Array of [x_min, y_min, x_max, y_max] as float32.
    """
    x_min = (x_center - width / 2.0) * img_w
    y_min = (y_center - height / 2.0) * img_h
    x_max = (x_center + width / 2.0) * img_w
    y_max = (y_center + height / 2.0) * img_h

    # Clamp coordinates to valid image bounds [0, size - 1]
    x_min = max(0.0, min(float(img_w - 1), x_min))
    y_min = max(0.0, min(float(img_h - 1), y_min))
    x_max = max(0.0, min(float(img_w - 1), x_max))
    y_max = max(0.0, min(float(img_h - 1), y_max))

    # Ensure min <= max
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    if y_min > y_max:
        y_min, y_max = y_max, y_min

    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)


def parse_yolo_label_file(
    label_path: str,
    img_w: int = TARGET_GRID_SIZE,
    img_h: int = TARGET_GRID_SIZE
) -> np.ndarray:
    """
    Parse a YOLO .txt label file and convert all bounding boxes to MedSAM2 pixel format.

    Parameters
    ----------
    label_path : str
        Path to the YOLO label .txt file.
    img_w : int
        Target width (default: 512).
    img_h : int
        Target height (default: 512).

    Returns
    -------
    boxes : np.ndarray
        Array of shape (N, 4) containing [x_min, y_min, x_max, y_max] for each box.
        If file is empty (negative slice), returns empty array of shape (0, 4).
    """
    if not os.path.isfile(label_path):
        return np.empty((0, 4), dtype=np.float32)

    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                # Format: class_id, xc, yc, w, h
                xc = float(parts[1])
                yc = float(parts[2])
                w  = float(parts[3])
                h  = float(parts[4])
                box = yolo_to_pixel_box(xc, yc, w, h, img_w=img_w, img_h=img_h)
                boxes.append(box)

    if not boxes:
        return np.empty((0, 4), dtype=np.float32)

    return np.stack(boxes, axis=0)


def parse_slice_identifier(filename: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Extract patient_id and slice index z from a slice filename or path.
    Example: 'train_001_slice015.txt' -> ('train_001', 15)

    Parameters
    ----------
    filename : str
        Basename or path of slice file.

    Returns
    -------
    patient_id : str or None
    slice_idx : int or None
    """
    base = os.path.basename(filename)
    match = re.search(r"^(train_\d+)_slice(\d+)", base)
    if match:
        pid = match.group(1)
        z = int(match.group(2))
        return pid, z
    return None, None
