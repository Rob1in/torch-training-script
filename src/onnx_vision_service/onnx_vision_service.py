"""Viam Vision Service module for ONNX object detection models.

Loads an ONNX detection model (exported by this training pipeline) and serves
detections through the standard Viam Vision API. Only depends on onnxruntime
for ML inference — no PyTorch required.

Expected ONNX model contract (from convert_to_onnx.py):
    Input:  'image'    — uint8 tensor [1, C, H, W] in range [0, 255]
    Output: 'location' — float32 [N, 4] bounding boxes (x_min, y_min, x_max, y_max)
            'score'    — float32 [N]   confidence scores
            'category' — float32 [N]   class indices
"""

from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import onnxruntime as ort
from PIL import Image
from typing_extensions import Self

from viam.components.camera import Camera
from viam.logging import getLogger
from viam.media.video import ViamImage
from viam.module.types import Reconfigurable
from viam.proto.app.robot import ServiceConfig
from viam.proto.common import PointCloudObject, ResourceName
from viam.proto.service.vision import Classification, Detection
from viam.resource.base import ResourceBase
from viam.resource.types import Model, ModelFamily
from viam.services.vision import Vision, CaptureAllResult
from viam.utils import ValueTypes

from src.onnx_vision_service.utils import decode_image

LOGGER = getLogger(__name__)

# Hardcoded output tensor names matching convert_to_onnx.py
ONNX_INPUT_NAME = "image"
# NOTE: The ONNX output names from convert_to_onnx.py are misleading.
# The model returns (boxes, labels_float, scores) but the export names them:
#   output[0] 'location' = boxes          (correct)
#   output[1] 'score'    = labels (float)  (misleading name — it's class indices)
#   output[2] 'category' = scores          (misleading name — it's confidence)


@dataclass
class Properties:
    """Vision service properties."""

    classifications_supported: bool = False
    detections_supported: bool = False
    object_point_clouds_supported: bool = False


class OnnxVisionService(Vision, Reconfigurable):
    """Vision Service that performs object detection using an ONNX model."""

    MODEL: ClassVar[Model] = Model(
        ModelFamily("viam", "vision"), "onnx-detector"
    )

    def __init__(self, name: str):
        super().__init__(name=name)
        self.camera_name: str = ""
        self.camera: Optional[Camera] = None
        self.session: Optional[ort.InferenceSession] = None
        self.labels: List[str] = []
        self.min_confidence: float = 0.0
        self.input_height: int = 0
        self.input_width: int = 0
        self.properties = Properties(
            classifications_supported=False,
            detections_supported=True,
            object_point_clouds_supported=False,
        )

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #

    @classmethod
    def new_service(
        cls,
        config: ServiceConfig,
        dependencies: Mapping[ResourceName, ResourceBase],
    ) -> Self:
        """Create and configure a new instance."""
        service = cls(config.name)
        service.reconfigure(config, dependencies)
        return service

    @classmethod
    def validate_config(
        cls, config: ServiceConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        """Validate JSON configuration.

        Returns (dependencies, optional_dependencies).
        """
        model_path = config.attributes.fields["model_path"].string_value
        camera_name = config.attributes.fields["camera_name"].string_value
        labels_path = config.attributes.fields["labels_path"].string_value

        if not model_path:
            raise Exception(
                "A 'model_path' to an ONNX model is required."
            )
        if not Path(model_path).exists():
            raise Exception(
                f"ONNX model file not found: {model_path}"
            )
        if not camera_name:
            raise Exception(
                "A 'camera_name' is required for this vision service module."
            )
        if not labels_path:
            raise Exception(
                "A 'labels_path' pointing to a labels.txt file is required."
            )
        if not Path(labels_path).exists():
            raise Exception(
                f"Labels file not found: {labels_path}"
            )
        return [camera_name], []

    def reconfigure(
        self,
        config: ServiceConfig,
        dependencies: Mapping[ResourceName, ResourceBase],
    ):
        """Handle attribute reconfiguration."""
        self.dependencies = dependencies

        # -- Camera dependency ----------------------------------------- #
        self.camera_name = config.attributes.fields[
            "camera_name"
        ].string_value
        self.camera = self.dependencies[
            Camera.get_resource_name(self.camera_name)
        ]

        # -- Labels ---------------------------------------------------- #
        labels_path = config.attributes.fields["labels_path"].string_value
        self.labels = self._load_labels(labels_path)
        LOGGER.info(f"Loaded {len(self.labels)} labels from {labels_path}: {self.labels}")

        # -- Min confidence -------------------------------------------- #
        self.min_confidence = 0.0
        if "min_confidence" in config.attributes.fields:
            self.min_confidence = config.attributes.fields[
                "min_confidence"
            ].number_value

        # -- ONNX model ------------------------------------------------ #
        model_path = config.attributes.fields["model_path"].string_value
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )

        # Extract input shape from ONNX metadata
        input_info = self.session.get_inputs()[0]
        input_shape = input_info.shape  # e.g. [1, 3, 1080, 1920]
        LOGGER.info(
            f"Loaded ONNX model: {model_path} | "
            f"input: {input_info.name} {input_shape} ({input_info.type})"
        )

        # The model expects [batch, channels, height, width]
        if len(input_shape) == 4:
            _, _, h, w = input_shape
            # Handle dynamic dimensions (symbolic strings)
            self.input_height = int(h) if isinstance(h, int) else 0
            self.input_width = int(w) if isinstance(w, int) else 0
        else:
            self.input_height = 0
            self.input_width = 0

        if self.input_height == 0 or self.input_width == 0:
            LOGGER.warning(
                "Could not determine fixed input size from ONNX model metadata. "
                "Images will be passed without resizing."
            )
        else:
            LOGGER.info(
                f"Model input size: {self.input_height}x{self.input_width}"
            )

        # Log output info
        for out in self.session.get_outputs():
            LOGGER.info(f"  output: {out.name} {out.shape} ({out.type})")

    # ------------------------------------------------------------------ #
    #  Vision API — Detections
    # ------------------------------------------------------------------ #

    async def get_detections(
        self,
        image: Union[Image.Image, ViamImage],
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        """Get detections from an image."""
        img_pil = decode_image(image)
        orig_w, orig_h = img_pil.size  # PIL uses (width, height)

        # Preprocess: resize → numpy uint8 [1, C, H, W]
        input_tensor = self._preprocess(img_pil)

        # Run ONNX inference
        outputs = self.session.run(
            None, {ONNX_INPUT_NAME: input_tensor}
        )
        # Unpack in DATA order (not name order — see NOTE above):
        #   output[0] = boxes, output[1] = labels (float), output[2] = scores
        boxes, categories, scores = outputs

        # Post-process → List[Detection]
        return self._postprocess(
            boxes, scores, categories, orig_w, orig_h
        )

    async def get_detections_from_camera(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        """Get detections from the configured camera."""
        if camera_name not in (self.camera_name, ""):
            raise ValueError(
                f"Camera name '{camera_name}' does not match "
                f"configured camera '{self.camera_name}'."
            )
        image = await self._get_image_from_camera()
        return await self.get_detections(image, extra=extra, timeout=timeout)

    # ------------------------------------------------------------------ #
    #  Vision API — CaptureAll
    # ------------------------------------------------------------------ #

    async def capture_all_from_camera(
        self,
        camera_name: str,
        return_image: bool = False,
        return_classifications: bool = False,
        return_detections: bool = False,
        return_object_point_clouds: bool = False,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> CaptureAllResult:
        """Capture image and detections from camera."""
        result = CaptureAllResult()

        if camera_name not in (self.camera_name, ""):
            raise ValueError(
                f"Camera name '{camera_name}' does not match "
                f"configured camera '{self.camera_name}'."
            )

        images, _ = await self.camera.get_images()
        if images is None or len(images) == 0:
            raise ValueError("No images returned by get_images")

        if return_image:
            result.image = images[0]

        if return_detections:
            try:
                detections = await self.get_detections(
                    images[0], extra=extra, timeout=timeout
                )
                result.detections = detections
            except Exception as e:
                LOGGER.info(f"get_detections failed: {e}")

        return result

    # ------------------------------------------------------------------ #
    #  Vision API — Not supported
    # ------------------------------------------------------------------ #

    async def get_classifications(
        self,
        image: Union[Image.Image, ViamImage],
        count: int,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        raise NotImplementedError("Classifications not supported by this module.")

    async def get_classifications_from_camera(
        self,
        camera_name: str,
        count: int,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        raise NotImplementedError("Classifications not supported by this module.")

    async def get_object_point_clouds(
        self,
        camera_name: str,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> List[PointCloudObject]:
        raise NotImplementedError("Object point clouds not supported by this module.")

    # ------------------------------------------------------------------ #
    #  Vision API — Properties & DoCommand
    # ------------------------------------------------------------------ #

    async def get_properties(
        self,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Properties:
        """Return vision service properties."""
        return self.properties

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_labels(labels_path: str) -> List[str]:
        """Load class labels from a text file (one label per line)."""
        path = Path(labels_path)
        labels = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    labels.append(line)
        return labels

    def _preprocess(self, img_pil: Image.Image) -> np.ndarray:
        """Resize image and convert to uint8 numpy tensor [1, C, H, W].

        Args:
            img_pil: PIL RGB image.

        Returns:
            numpy array of shape [1, 3, H, W], dtype uint8.
        """
        if self.input_height > 0 and self.input_width > 0:
            img_resized = img_pil.resize(
                (self.input_width, self.input_height), Image.BILINEAR
            )
        else:
            img_resized = img_pil

        # [H, W, C] uint8 → [C, H, W] uint8 → [1, C, H, W] uint8
        img_np = np.array(img_resized, dtype=np.uint8)
        img_chw = img_np.transpose(2, 0, 1)
        return np.expand_dims(img_chw, axis=0)

    def _postprocess(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        categories: np.ndarray,
        orig_width: int,
        orig_height: int,
    ) -> List[Detection]:
        """Convert ONNX outputs to Viam Detection objects.

        The model outputs bounding boxes in the coordinate space of its
        input tensor (input_height × input_width). We scale them back
        to the original image dimensions.

        Args:
            boxes:      [N, 4] float32 — (x_min, y_min, x_max, y_max) in model coords
            scores:     [N] float32    — confidence scores
            categories: [N] float32    — class indices (float, cast to int)
            orig_width:  Original image width  (before resize)
            orig_height: Original image height (before resize)

        Returns:
            Filtered list of Detection objects.
        """
        if len(scores) == 0:
            return []

        # Scale factors to map from model input coords → original image coords
        if self.input_width > 0 and self.input_height > 0:
            sx = orig_width / self.input_width
            sy = orig_height / self.input_height
        else:
            sx = 1.0
            sy = 1.0

        detections: List[Detection] = []
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

            # Scale box to original image coordinates
            x_min = float(boxes[i][0]) * sx
            y_min = float(boxes[i][1]) * sy
            x_max = float(boxes[i][2]) * sx
            y_max = float(boxes[i][3]) * sy

            detection = Detection(
                x_min=int(x_min),
                y_min=int(y_min),
                x_max=int(x_max),
                y_max=int(y_max),
                confidence=score,
                class_name=class_name,
            )

            # Add normalized coordinates if original dimensions are valid
            if orig_width > 0 and orig_height > 0:
                detection = Detection(
                    x_min=int(x_min),
                    y_min=int(y_min),
                    x_max=int(x_max),
                    y_max=int(y_max),
                    x_min_normalized=x_min / orig_width,
                    y_min_normalized=y_min / orig_height,
                    x_max_normalized=x_max / orig_width,
                    y_max_normalized=y_max / orig_height,
                    confidence=score,
                    class_name=class_name,
                )

            detections.append(detection)

        return detections

    async def _get_image_from_camera(self) -> ViamImage:
        """Grab the first image from the camera dependency."""
        images, _ = await self.camera.get_images()
        if images is None or len(images) == 0:
            raise ValueError("No images returned by get_images")
        return images[0]
