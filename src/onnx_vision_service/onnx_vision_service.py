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
from viam.services.vision import CaptureAllResult, Vision
from viam.utils import ValueTypes

from omni_inference import OmniOnnxInference, RawDetection
from src.onnx_vision_service.utils import decode_image

LOGGER = getLogger(__name__)


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
        # Dependency-free inference core (holds the onnxruntime session,
        # labels, input dims, min_confidence, and background-strip config).
        self.inference: Optional[OmniOnnxInference] = None
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
        labels = self._load_labels(labels_path)
        LOGGER.info(f"Loaded {len(labels)} labels from {labels_path}: {labels}")

        # -- Min confidence -------------------------------------------- #
        min_confidence = 0.0
        if "min_confidence" in config.attributes.fields:
            min_confidence = config.attributes.fields[
                "min_confidence"
            ].number_value

        # -- Optional preprocessing ------------------------------------ #
        background_strip_dist = 0.0
        if "background_strip_dist" in config.attributes.fields:
            background_strip_dist = config.attributes.fields[
                "background_strip_dist"
            ].number_value

        # -- ONNX model (via dependency-free inference core) ------------ #
        model_path = config.attributes.fields["model_path"].string_value
        self.inference = OmniOnnxInference(
            model_path=model_path,
            labels=labels,
            min_confidence=min_confidence,
            background_strip_dist=background_strip_dist,
        )

        input_info = self.inference.session.get_inputs()[0]
        LOGGER.info(
            f"Loaded ONNX model: {model_path} | "
            f"input: {input_info.name} {input_info.shape} ({input_info.type})"
        )
        if self.inference.input_height == 0 or self.inference.input_width == 0:
            LOGGER.warning(
                "Could not determine fixed input size from ONNX model metadata. "
                "Images will be passed without resizing."
            )
        else:
            LOGGER.info(
                f"Model input size: "
                f"{self.inference.input_height}x{self.inference.input_width}"
            )

        # Log output info
        for out in self.inference.session.get_outputs():
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
        if self.inference is None:
            raise RuntimeError("Service not configured: no ONNX model loaded.")

        img_pil = decode_image(image)
        orig_w, orig_h = img_pil.size  # PIL uses (width, height)

        # Run the dependency-free inference core (normalized [0,1] coords).
        raw_detections = self.inference.infer(img_pil)

        return self._to_viam_detections(raw_detections, orig_w, orig_h)

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

    @staticmethod
    def _to_viam_detections(
        raw_detections: List[RawDetection],
        orig_width: int,
        orig_height: int,
    ) -> List[Detection]:
        """Convert core RawDetection (normalized coords) to Viam Detection.

        The output contract carries BOTH absolute pixel coords (in ORIGINAL
        image space) AND normalized coords. Pixel coords are computed as
        ``normalized * original_dimension``.

        Args:
            raw_detections: Detections from the inference core (already
                filtered by min_confidence; coords normalized to [0, 1]).
            orig_width:  Original image width  (before resize).
            orig_height: Original image height (before resize).
        """
        detections: List[Detection] = []
        for raw in raw_detections:
            # Map normalized coords back to original-image pixel space.
            x_min = raw.x_min * orig_width
            y_min = raw.y_min * orig_height
            x_max = raw.x_max * orig_width
            y_max = raw.y_max * orig_height

            if orig_width > 0 and orig_height > 0:
                detection = Detection(
                    x_min=int(x_min),
                    y_min=int(y_min),
                    x_max=int(x_max),
                    y_max=int(y_max),
                    x_min_normalized=raw.x_min,
                    y_min_normalized=raw.y_min,
                    x_max_normalized=raw.x_max,
                    y_max_normalized=raw.y_max,
                    confidence=raw.confidence,
                    class_name=raw.class_name,
                )
            else:
                detection = Detection(
                    x_min=int(x_min),
                    y_min=int(y_min),
                    x_max=int(x_max),
                    y_max=int(y_max),
                    confidence=raw.confidence,
                    class_name=raw.class_name,
                )

            detections.append(detection)

        return detections

    async def _get_image_from_camera(self) -> ViamImage:
        """Grab the first image from the camera dependency."""
        images, _ = await self.camera.get_images()
        if images is None or len(images) == 0:
            raise ValueError("No images returned by get_images")
        return images[0]
