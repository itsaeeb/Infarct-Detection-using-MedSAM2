"""
bbox_format package
"""
try:
    from .box_parser import (
        yolo_to_pixel_box,
        parse_yolo_label_file,
        parse_slice_identifier,
        TARGET_GRID_SIZE,
    )
except (ImportError, ValueError):
    from box_parser import (
        yolo_to_pixel_box,
        parse_yolo_label_file,
        parse_slice_identifier,
        TARGET_GRID_SIZE,
    )

__all__ = [
    "yolo_to_pixel_box",
    "parse_yolo_label_file",
    "parse_slice_identifier",
    "TARGET_GRID_SIZE",
]
