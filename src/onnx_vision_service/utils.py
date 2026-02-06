"""Utility file including methods for image decoding (PIL-based, no PyTorch)."""

from io import BytesIO
from typing import Union

from PIL import Image
from viam.logging import getLogger
from viam.media.video import CameraMimeType, ViamImage

LOGGER = getLogger(__name__)

SUPPORTED_IMAGE_TYPE = [
    CameraMimeType.JPEG,
    CameraMimeType.PNG,
    CameraMimeType.VIAM_RGBA,
]
LIBRARY_SUPPORTED_FORMATS = ["JPEG", "PNG", "VIAM_RGBA"]


def decode_image(image: Union[Image.Image, ViamImage]) -> Image.Image:
    """Decode a ViamImage or PIL Image into a PIL RGB Image.

    Args:
        image: Input image (ViamImage or PIL Image)

    Returns:
        PIL.Image.Image in RGB mode
    """
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, ViamImage):
        if image.mime_type not in SUPPORTED_IMAGE_TYPE:
            LOGGER.error(
                f"Unsupported image type: {image.mime_type}. "
                f"Supported types are {SUPPORTED_IMAGE_TYPE}."
            )
            raise ValueError(f"Unsupported image type: {image.mime_type}.")
        im = Image.open(
            BytesIO(image.data), formats=LIBRARY_SUPPORTED_FORMATS
        ).convert("RGB")
        return im

    raise TypeError(
        f"Unsupported image type: {type(image)}. "
        "Expected PIL.Image.Image or ViamImage."
    )
