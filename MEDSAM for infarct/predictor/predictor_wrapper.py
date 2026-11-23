"""
predictor_wrapper.py
====================
Unified inference wrapper around official MedSAM2 model for Brain Infarct Detection.

Supports both:
1. Volumetric 4D CT Tensor Inference (via SAM2VideoPredictorNPZ):
   - Ingests preprocessed 4D CT tensor `(D, 3, 512, 512)` via `init_state()`.
   - Prompts positive slices with bounding boxes via `add_new_points_or_box()`.
   - Merges multi-component lesion masks via logical OR.
2. 2D Per-Slice Predictor (via SAM2ImagePredictor):
   - Ingests individual 2D slice RGB arrays `(512, 512, 3)`.
   - Generates predicted masks directly.
"""

import os
import sys
from typing import Dict, Any, Optional, Tuple, Union, List
import numpy as np
import torch

# Dynamically locate INFARCT_ROOT and prioritize local standalone sam2 package
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INFARCT_ROOT = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(INFARCT_ROOT)

# Prioritize local standalone sam2 inside MEDSAM for infarct
LOCAL_SAM2_DIR = INFARCT_ROOT
if LOCAL_SAM2_DIR not in sys.path:
    sys.path.insert(0, LOCAL_SAM2_DIR)

DEFAULT_CONFIG_PATH = os.path.join(INFARCT_ROOT, "sam2", "configs", "sam2.1_hiera_t512.yaml")
DEFAULT_CHECKPOINT_PATH = os.path.join(INFARCT_ROOT, "checkpoints", "MedSAM2_CTLesion.pt")


class MedSAM2InfarctPredictor:
    """
    Unified predictor wrapper for MedSAM2 brain infarct segmentation.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config_path: str = DEFAULT_CONFIG_PATH,
        device: Optional[Union[str, torch.device]] = None,
        mode: str = "video_npz",
        auto_load: bool = True
    ):
        """
        Initialize the MedSAM2 Infarct Predictor.

        Parameters
        ----------
        checkpoint_path : str, optional
            Path to pretrained weights (.pt). Defaults to MedSAM2_CTLesion.pt if present.
        config_path : str
            Path to YAML model configuration.
        device : str or torch.device, optional
            Compute device ('cuda' or 'cpu'). Defaults to best available.
        mode : str
            'video_npz' for 3D volumetric/tensor predictor, or 'image' for 2D slice predictor.
        auto_load : bool
            Whether to automatically call load_model() on initialization.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if checkpoint_path is None and os.path.isfile(DEFAULT_CHECKPOINT_PATH):
            checkpoint_path = DEFAULT_CHECKPOINT_PATH

        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.mode = mode
        self.model = None
        self.inference_state = None
        self.current_patient_id = None
        self.orig_hw = None

        if checkpoint_path is not None and not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        if auto_load and checkpoint_path is not None:
            self.load_model()

    def load_model(self):
        """
        Build and instantiate the MedSAM2 model using official factory functions.
        """
        try:
            from sam2.build_sam import build_sam2_video_predictor_npz, build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as e:
            raise ImportError(
                f"Could not import MedSAM2 modules. "
                f"Ensure sam2 package is available. Error: {e}"
            )

        config_name = os.path.basename(self.config_path)
        hydra_config = f"configs/{config_name}"

        if self.mode == "video_npz":
            # Uses build_sam2_video_predictor_npz for direct 4D tensor consumption
            self.model = build_sam2_video_predictor_npz(
                config_file=hydra_config,
                ckpt_path=self.checkpoint_path,
                device=self.device,
                mode="eval",
                apply_postprocessing=True
            )
        elif self.mode == "image":
            # Uses build_sam2 and wraps in SAM2ImagePredictor
            base_model = build_sam2(
                config_file=hydra_config,
                ckpt_path=self.checkpoint_path,
                device=self.device,
                mode="eval"
            )
            self.model = SAM2ImagePredictor(base_model)
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Must be 'video_npz' or 'image'.")

        return self.model

    def init_patient_tensor(
        self,
        video_tensor: torch.Tensor,
        video_height: int,
        video_width: int,
        patient_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Initialize the predictor state for a 3D CT volume tensor (D, 3, 512, 512).

        Parameters
        ----------
        video_tensor : torch.Tensor
            4D tensor matching official MedSAM2 format of shape (D, 3, 512, 512).
        video_height : int
            Original CT slice height (e.g. 568 or 256).
        video_width : int
            Original CT slice width (e.g. 455 or 256).
        patient_id : str, optional
            Identifier for logging.

        Returns
        -------
        inference_state : dict
        """
        if self.mode != "video_npz":
            raise RuntimeError("init_patient_tensor requires mode='video_npz'.")

        self.current_patient_id = patient_id
        self.orig_hw = (video_height, video_width)

        # Move tensor to target device if not already there
        video_tensor = video_tensor.to(self.device)

        if self.model is None:
            raise RuntimeError(
                "MedSAM2 model is not loaded. Cannot initialize predictor state without model weights. "
                "Ensure checkpoints/MedSAM2_CTLesion.pt exists and call load_model()."
            )

        # Video predictor expects coordinates in tensor spatial size (512x512)
        tensor_h, tensor_w = video_tensor.shape[2], video_tensor.shape[3]
        self.inference_state = self.model.init_state(
            images=video_tensor,
            video_height=tensor_h,
            video_width=tensor_w
        )

        return self.inference_state

    def predict_slice_from_boxes(
        self,
        frame_idx: int,
        boxes: np.ndarray,
        multimask_output: bool = False
    ) -> np.ndarray:
        """
        Prompt MedSAM2 on slice index `frame_idx` using absolute pixel bounding boxes
        and return the merged binary mask of shape (512, 512).

        Parameters
        ----------
        frame_idx : int
            Axial slice index z (0 <= frame_idx < D).
        boxes : np.ndarray
            Array of shape (N, 4) or (4,) containing [x_min, y_min, x_max, y_max].
        multimask_output : bool
            Whether to output multiple mask candidates (default: False).

        Returns
        -------
        combined_mask : np.ndarray
            Binary mask of shape (512, 512) where 1 = Infarct, 0 = Background.
        """
        if boxes.ndim == 1:
            boxes = boxes[None, :]  # shape: (1, 4)

        N = len(boxes)
        if N == 0:
            return np.zeros((512, 512), dtype=np.uint8)

        if self.model is None:
            raise RuntimeError(
                "MedSAM2 model is not loaded. Mock mode is disabled. "
                "Please load valid checkpoint weights."
            )

        component_masks = []

        if self.mode == "video_npz":
            # Reset any prior tracking state on this session for clean independent prompting
            if hasattr(self.model, "reset_state"):
                self.model.reset_state(self.inference_state)

            last_masks = None
            for obj_id, bbox in enumerate(boxes, start=1):
                box_np = np.asarray(bbox, dtype=np.float32)
                _, out_obj_ids, out_mask_logits = self.model.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    box=box_np,
                    clear_old_points=True
                )
                last_masks = out_mask_logits

            if last_masks is not None:
                # last_masks shape: (N, 1, H, W)
                binary_masks = (last_masks > 0.0).cpu().numpy().astype(np.uint8)
                if binary_masks.ndim == 4:
                    return np.logical_or.reduce(binary_masks[:, 0, :, :], axis=0).astype(np.uint8)
                elif binary_masks.ndim == 3:
                    return np.logical_or.reduce(binary_masks, axis=0).astype(np.uint8)
                else:
                    return binary_masks.squeeze().astype(np.uint8)
            return np.zeros((512, 512), dtype=np.uint8)
        elif self.mode == "image":
            # SAM2ImagePredictor accepts (N, 4) in batch
            masks, _, _ = self.model.predict(
                box=boxes,
                multimask_output=multimask_output
            )
            for m in masks:
                component_masks.append(m.astype(np.uint8))

        # Merge multiple components on the slice using logical OR
        if len(component_masks) == 1:
            return component_masks[0]
        else:
            return np.logical_or.reduce(component_masks).astype(np.uint8)
