"""Dependency-free ONNX inference core for the omni-detector.

Ported verbatim (behavior-wise) from ``src/onnx_vision_service/onnx_vision_service.py``.
The full pipeline:

    PIL RGB image
      -> resize to model input (H, W) with PIL BILINEAR
      -> uint8 HWC numpy
      -> (optional) k-means background strip
      -> transpose to CHW, add batch dim -> uint8 NCHW [1, C, H, W]
      -> session.run(None, {"image": x})
      -> unpack outputs in DATA order: boxes, categories, scores
      -> filter by min_confidence
      -> map 1-indexed class indices to labels
      -> normalize box coords to [0, 1] by model input size

NO normalization (mean/std) is applied to pixels — the model expects raw uint8.

Expected ONNX model contract (from convert_to_onnx.py):
    Input:  'image'    — uint8 tensor [1, C, H, W] in range [0, 255]
    Output: position 0 — float32 [N, 4] bounding boxes (x_min, y_min, x_max, y_max)
            position 1 — float32 [N]   class indices (1-indexed; output NAME is misleading)
            position 2 — float32 [N]   confidence scores (output NAME is misleading)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

# Input tensor name matching convert_to_onnx.py.
ONNX_INPUT_NAME = "image"

# NOTE: The ONNX output names from convert_to_onnx.py are misleading.
# The model returns (boxes, labels_float, scores) but the export names them:
#   output[0] 'location' = boxes          (correct)
#   output[1] 'score'    = labels (float)  (misleading name — it's class indices)
#   output[2] 'category' = scores          (misleading name — it's confidence)
# We therefore always unpack by POSITION, never by name.


@dataclass
class RawDetection:
    """A single detection with normalized [0, 1] coordinates.

    Coordinates are normalized relative to the model's input dimensions,
    which (for fixed-input models) equals normalizing relative to the
    original image — the resize is uniform per-axis.
    """

    class_name: str
    confidence: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class OmniOnnxInference:
    """Runs ONNX object detection inference with no viam/torch dependency."""

    def __init__(
        self,
        model_path: str,
        labels: List[str],
        min_confidence: float = 0.0,
        background_strip_dist: float = 0.0,
    ):
        """Create an inference session.

        Args:
            model_path: Path to the ``model.onnx`` file.
            labels: Ordered class labels (0-indexed; no background entry).
            min_confidence: Minimum confidence to keep a detection.
            background_strip_dist: If > 0, strip the k-means background within
                this Euclidean distance (8-bit RGB space) before inference.
        """
        self.labels: List[str] = list(labels)
        self.min_confidence: float = min_confidence
        self.background_strip_dist: float = background_strip_dist

        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )

        # Extract input shape [batch, channels, height, width] from metadata.
        input_shape = self.session.get_inputs()[0].shape  # e.g. [1, 3, 480, 640]
        if len(input_shape) == 4:
            _, _, h, w = input_shape
            # Handle dynamic dimensions (symbolic strings) -> 0 = unknown.
            self.input_height: int = int(h) if isinstance(h, int) else 0
            self.input_width: int = int(w) if isinstance(w, int) else 0
        else:
            self.input_height = 0
            self.input_width = 0

    # ------------------------------------------------------------------ #
    #  Constructors
    # ------------------------------------------------------------------ #

    @classmethod
    def from_model_dir(
        cls,
        model_dir: Union[str, Path],
        min_confidence: float = 0.0,
        background_strip_dist: float = 0.0,
    ) -> "OmniOnnxInference":
        """Build from a directory containing ``model.onnx`` + ``labels.txt``."""
        model_dir = Path(model_dir)
        model_path = model_dir / "model.onnx"
        labels_path = model_dir / "labels.txt"
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model file not found: {model_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
        labels = cls._load_labels(labels_path)
        return cls(
            model_path=str(model_path),
            labels=labels,
            min_confidence=min_confidence,
            background_strip_dist=background_strip_dist,
        )

    # ------------------------------------------------------------------ #
    #  Public inference
    # ------------------------------------------------------------------ #

    def infer(self, image: Image.Image) -> List[RawDetection]:
        """Run the full pipeline on a PIL RGB image.

        Args:
            image: PIL RGB image.

        Returns:
            List of RawDetection with normalized [0, 1] coordinates,
            filtered by ``min_confidence``.
        """
        input_tensor = self._preprocess(image)
        outputs = self.session.run(None, {ONNX_INPUT_NAME: input_tensor})
        # Unpack in DATA order (not name order — see NOTE above):
        #   output[0] = boxes, output[1] = labels (float), output[2] = scores
        boxes, categories, scores = outputs
        return self._decode(boxes, scores, categories)

    # ------------------------------------------------------------------ #
    #  Internal: preprocess / decode / background strip
    # ------------------------------------------------------------------ #

    def _preprocess(self, img_pil: Image.Image) -> np.ndarray:
        """Resize image and convert to uint8 numpy tensor [1, C, H, W]."""
        if self.input_height > 0 and self.input_width > 0:
            img_resized = img_pil.resize(
                (self.input_width, self.input_height), Image.BILINEAR
            )
        else:
            img_resized = img_pil

        # [H, W, C] uint8 -> [C, H, W] uint8 -> [1, C, H, W] uint8
        img_np = np.array(img_resized, dtype=np.uint8)
        if self.background_strip_dist and self.background_strip_dist > 0:
            img_np = self._background_strip_np(
                img_np, dist=self.background_strip_dist
            )
        img_chw = img_np.transpose(2, 0, 1)
        return np.expand_dims(img_chw, axis=0)

    def _decode(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        categories: np.ndarray,
    ) -> List[RawDetection]:
        """Convert ONNX outputs to RawDetection with normalized coords.

        The model outputs boxes in the coordinate space of its input tensor
        (input_height x input_width). We normalize them to [0, 1] by dividing
        by the input dimensions.

        Args:
            boxes:      [N, 4] float32 — (x_min, y_min, x_max, y_max) in model coords
            scores:     [N] float32    — confidence scores
            categories: [N] float32    — class indices (float, cast to int)
        """
        if len(scores) == 0:
            return []

        if self.input_width > 0 and self.input_height > 0:
            norm_w = float(self.input_width)
            norm_h = float(self.input_height)
        else:
            norm_w = 1.0
            norm_h = 1.0

        detections: List[RawDetection] = []
        for i in range(len(scores)):
            score = float(scores[i])
            if score < self.min_confidence:
                continue

            # Map class index to label.
            # Faster R-CNN uses 0 = background, 1..N = actual classes.
            # The labels list is 0-indexed (no background entry), so subtract 1.
            cat_idx = int(round(categories[i])) - 1
            if 0 <= cat_idx < len(self.labels):
                class_name = self.labels[cat_idx]
            else:
                class_name = str(cat_idx + 1)  # fallback: show original index

            detections.append(
                RawDetection(
                    class_name=class_name,
                    confidence=score,
                    x_min=float(boxes[i][0]) / norm_w,
                    y_min=float(boxes[i][1]) / norm_h,
                    x_max=float(boxes[i][2]) / norm_w,
                    y_max=float(boxes[i][3]) / norm_h,
                )
            )

        return detections

    @staticmethod
    def _background_strip_np(
        img_hwc_u8: np.ndarray, dist: float = 150
    ) -> np.ndarray:
        """Strip pixels within Euclidean distance ``dist`` of the k-means background.

        This is the canonical (numpy/cv2) background strip — a numpy->numpy port
        of ``src/utils/transforms.py::background_strip`` with the same behavior:
        - Distance computed in **8-bit RGB space** (values 0-255)
        - Background color estimated via k-means on a resized 100x100 image

        Args:
            img_hwc_u8: uint8 image of shape [H, W, 3] in RGB order.
            dist: Euclidean distance threshold in 8-bit RGB space.

        Returns:
            uint8 image [H, W, 3] (zeros where stripped).
        """
        if not isinstance(img_hwc_u8, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(img_hwc_u8)}")
        if img_hwc_u8.ndim != 3 or img_hwc_u8.shape[2] != 3:
            raise ValueError(f"Expected [H, W, 3], got shape {img_hwc_u8.shape}")
        if img_hwc_u8.dtype != np.uint8:
            raise ValueError(f"Expected dtype uint8, got {img_hwc_u8.dtype}")

        # Mirror `get_background_from_img_tensor()`:
        # - reshape image as 100x100 float32 in [0, 1]
        # - run cv2.kmeans with k=5 and count the most common label
        resized = cv2.resize(
            img_hwc_u8, (100, 100), interpolation=cv2.INTER_LINEAR
        )
        data = (
            (resized.astype(np.float32) / 255.0)
            .reshape((-1, 3))
            .astype(np.float32)
        )

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100,
            0.85,
        )
        _compactness, labels, centers = cv2.kmeans(
            data, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        labels = labels.reshape(-1)
        max_label = int(
            np.bincount(labels, minlength=centers.shape[0]).argmax()
        )
        background_color_01 = centers[max_label]  # float32 in [0,1]
        bg_rgb_255 = background_color_01 * 255.0  # float32 in [0,255]

        diff = img_hwc_u8.astype(np.float32) - bg_rgb_255.reshape((1, 1, 3))
        dist_sq_map = np.sum(diff * diff, axis=2)  # (H, W)
        dist_sq = float(dist) * float(dist)
        mask = dist_sq_map <= dist_sq

        out = img_hwc_u8.copy()
        out[mask] = 0
        return out

    @staticmethod
    def _load_labels(labels_path: Union[str, Path]) -> List[str]:
        """Load class labels from a text file (one label per line)."""
        labels: List[str] = []
        with open(labels_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    labels.append(line)
        return labels
