"""
Transforms for object detection datasets.
Handles image and bounding box transformations for training and evaluation.
"""

import logging
import math
import random
from typing import Dict, List, Optional, Tuple

import torch
import cv2
import numpy as np
import torchvision.transforms.functional as F
from omegaconf import DictConfig

log = logging.getLogger(__name__)


class DetectionTransform:
    """CPU-side transform that applies augmentations to both images and bounding boxes."""

    def __init__(self, transforms: List[Dict]):
        self.transforms = transforms

    def __call__(self, image: torch.Tensor, target: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Apply transforms to image and target.

        Args:
            image: Tensor of shape [C, H, W] (already converted to tensor)
            target: Dictionary with 'boxes', 'labels', 'image_id'

        Returns:
            Transformed image and target
        """
        boxes = target["boxes"].clone()
        labels = target["labels"].clone()

        # Get image dimensions
        _, h, w = image.shape

        for transform_config in self.transforms:
            transform_name = transform_config["name"]
            params = transform_config.get("params", {})

            if transform_name == "Resize":
                size = params.get("size", [h, w])
                # Convert ListConfig to regular list if needed (Hydra config)
                if hasattr(size, "__iter__") and not isinstance(size, (str, bytes)):
                    size = list(size)
                if isinstance(size, list) and len(size) == 2:
                    new_h, new_w = int(size[0]), int(size[1])
                else:
                    new_h = new_w = int(size)

                # Resize image
                image = F.resize(image, [new_h, new_w])

                # Scale bounding boxes
                scale_x = new_w / w
                scale_y = new_h / h
                boxes[:, [0, 2]] *= scale_x  # x coordinates
                boxes[:, [1, 3]] *= scale_y  # y coordinates

                h, w = new_h, new_w

            elif transform_name == "BackgroundStrip":
                # strip the background of the image, boxes stay the same
                dist = params.get("dist", 150)
                image = background_strip(image, dist)

            elif transform_name == "RandomRotation":
                degrees = params.get("degrees", 10)
                expand = params.get("expand", False)

                angle = random.uniform(-degrees, degrees)
                image, boxes = self._rotate_image_and_boxes(image, boxes, angle, expand)
                _, h, w = image.shape

            elif transform_name == "RandomHorizontalFlip":
                if random.random() < params.get("p", 0.5):
                    image = F.hflip(image)
                    # Flip x coordinates: x -> w - x
                    boxes[:, [0, 2]] = w - boxes[:, [2, 0]]

            elif transform_name == "RandomVerticalFlip":
                if random.random() < params.get("p", 0.5):
                    image = F.vflip(image)
                    # Flip y coordinates: y -> h - y
                    boxes[:, [1, 3]] = h - boxes[:, [3, 1]]

            elif transform_name == "Normalize":
                mean = params.get("mean", [0.485, 0.456, 0.406])
                std = params.get("std", [0.229, 0.224, 0.225])
                image = F.normalize(image, mean=mean, std=std)

            elif transform_name == "ColorJitter":
                if random.random() < params.get("p", 1.0):
                    brightness = params.get("brightness", 0.2)
                    contrast = params.get("contrast", 0.2)
                    saturation = params.get("saturation", 0.2)
                    hue = params.get("hue", 0.1)

                    image = F.adjust_brightness(
                        image, random.uniform(1 - brightness, 1 + brightness)
                    )
                    image = F.adjust_contrast(image, random.uniform(1 - contrast, 1 + contrast))
                    image = F.adjust_saturation(
                        image, random.uniform(1 - saturation, 1 + saturation)
                    )
                    image = F.adjust_hue(image, random.uniform(-hue, hue))

            elif transform_name == "RandomGaussianNoise":
                if random.random() < params.get("p", 1.0):
                    mean = params.get("mean", 0.0)
                    std = params.get("std", 0.1)
                    noise = torch.randn_like(image) * std + mean
                    image = image + noise
                    image = torch.clamp(image, 0, 1)

            elif transform_name == "RandomGamma":
                if random.random() < params.get("p", 1.0):
                    gamma = params.get("gamma", 1.0)
                    gamma_value = random.uniform(max(0.1, 1 - gamma), 1 + gamma)
                    image = F.adjust_gamma(image, gamma_value)

        # Ensure boxes are valid
        boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], 0, w)
        boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], 0, h)

        # Remove invalid boxes (width or height <= 0)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        if not valid.all():
            boxes = boxes[valid]
            labels = labels[valid]
            log.warning(f"Removed {len(valid) - valid.sum()} invalid boxes after transforms")

        target["boxes"] = boxes
        target["labels"] = labels

        return image, target

    def _rotate_image_and_boxes(
        self, image: torch.Tensor, boxes: torch.Tensor, angle: float, expand: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rotate image and adjust bounding boxes."""
        _, h, w = image.shape
        center = (w / 2, h / 2)

        # Rotate image
        image = F.rotate(image, angle, expand=expand)

        if expand:
            _, new_h, new_w = image.shape
            # Adjust center for expanded image
            center = (new_w / 2, new_h / 2)
        else:
            new_h, new_w = h, w

        # Convert angle to radians
        angle_rad = math.radians(angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # Rotate bounding boxes
        # Convert boxes to center format temporarily
        boxes_centered = torch.zeros_like(boxes)
        boxes_centered[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2 - center[0]  # cx
        boxes_centered[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2 - center[1]  # cy
        boxes_centered[:, 2] = boxes[:, 2] - boxes[:, 0]  # width
        boxes_centered[:, 3] = boxes[:, 3] - boxes[:, 1]  # height

        # Rotate centers
        new_cx = boxes_centered[:, 0] * cos_a - boxes_centered[:, 1] * sin_a + center[0]
        new_cy = boxes_centered[:, 0] * sin_a + boxes_centered[:, 1] * cos_a + center[1]

        # Convert back to x1, y1, x2, y2 format
        new_boxes = torch.zeros_like(boxes)
        new_boxes[:, 0] = new_cx - boxes_centered[:, 2] / 2
        new_boxes[:, 1] = new_cy - boxes_centered[:, 3] / 2
        new_boxes[:, 2] = new_cx + boxes_centered[:, 2] / 2
        new_boxes[:, 3] = new_cy + boxes_centered[:, 3] / 2

        return image, new_boxes


class GPUCollate:
    """Collate function that moves CPU-transformed samples to the target device."""

    def __init__(self, device: torch.device):
        self.device = device

    def __call__(self, batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Collate batch of samples.

        Args:
            batch: List of (image, target) tuples with CPU tensors

        Returns:
            Batched images tensor [B, C, H, W] and list of targets
        """
        images = []
        targets = []

        for image, target in batch:
            image_on_device = image.to(self.device)

            target_on_device = {}
            for key, value in target.items():
                if isinstance(value, torch.Tensor):
                    target_on_device[key] = value.to(self.device)
                else:
                    target_on_device[key] = value

            images.append(image_on_device)
            targets.append(target_on_device)

        images = torch.stack(images, dim=0)

        return images, targets


def attach_dataset_transform(dataset, transform: Optional[DetectionTransform]):
    """Attach a transform to a ViamDataset or Subset wrapping a ViamDataset.

    Subsets get a shallow-cloned underlying dataset so train/val can use
    different transforms without sharing mutable state.
    """
    from torch.utils.data import Subset

    from datasets.viam_dataset import ViamDataset

    if isinstance(dataset, Subset):
        underlying = dataset.dataset
        if not isinstance(underlying, ViamDataset):
            raise TypeError(
                f"Expected Subset of ViamDataset, got Subset of {type(underlying).__name__}"
            )
        return Subset(underlying.with_transform(transform), dataset.indices)

    if isinstance(dataset, ViamDataset):
        dataset.transform = transform
        return dataset

    raise TypeError(f"Unexpected dataset type: {type(dataset).__name__}")


def compute_dataset_stats(
    dataset,
    max_samples: Optional[int] = None,
) -> Tuple[List[float], List[float]]:
    """Compute per-channel mean and std over a dataset of [C, H, W] float [0,1] images.

    Uses a two-pass approach (first mean, then std) which is simple and
    numerically stable enough for typical dataset sizes.

    Args:
        dataset: A ``ViamDataset`` or ``torch.utils.data.Subset`` whose
            ``__getitem__`` returns ``(image_tensor, target)``.
        max_samples: If set, cap the number of images sampled (randomly).

    Returns:
        ``(mean, std)`` — each a list of 3 floats [R, G, B] in [0, 1] range.
    """
    from torch.utils.data import Subset

    indices: List[int]
    if isinstance(dataset, Subset):
        indices = list(dataset.indices)
        base_dataset = dataset.dataset
    else:
        indices = list(range(len(dataset)))
        base_dataset = dataset

    if max_samples is not None and max_samples < len(indices):
        import random as _rng

        indices = _rng.sample(indices, max_samples)

    n = len(indices)
    log.info(f"Computing dataset normalization stats over {n} images ...")

    # Pass 1: mean
    channel_sum = torch.zeros(3, dtype=torch.float64)
    pixel_count = 0
    for idx in indices:
        img, _ = base_dataset[idx]  # [C, H, W] float [0,1]
        channel_sum += img.to(torch.float64).sum(dim=[1, 2])
        pixel_count += img.shape[1] * img.shape[2]

    mean = (channel_sum / pixel_count).tolist()

    # Pass 2: std
    channel_sq_sum = torch.zeros(3, dtype=torch.float64)
    mean_t = torch.tensor(mean, dtype=torch.float64).view(3, 1, 1)
    for idx in indices:
        img, _ = base_dataset[idx]
        channel_sq_sum += ((img.to(torch.float64) - mean_t) ** 2).sum(dim=[1, 2])

    std = (channel_sq_sum / pixel_count).sqrt().tolist()

    # Round for cleaner config output
    mean = [round(v, 6) for v in mean]
    std = [round(v, 6) for v in std]

    log.info(f"Dataset stats — mean: {mean}, std: {std}")
    return mean, std


def build_transforms(
    cfg: DictConfig, is_train: bool = True, test: bool = False
) -> Optional[DetectionTransform]:
    """
    Build transforms from config.

    Args:
        cfg: Hydra config
        is_train: Whether this is for training
        test: Whether this is for testing

    Returns:
        DetectionTransform or None
    """
    if test:
        transform_config = cfg.dataset.transform.test
    elif is_train:
        transform_config = cfg.dataset.transform.train
    else:
        transform_config = cfg.dataset.transform.val

    if not transform_config:
        return None

    return DetectionTransform(transform_config)


def _background_strip_np(img_hwc_u8: np.ndarray, dist: float = 150) -> np.ndarray:
    """Strip pixels within Euclidean distance ``dist`` of the k-means background.

    Distance is computed in **8-bit RGB space** (values 0–255). Background color
    is estimated via k-means on a resized 100×100 image.

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

    resized = cv2.resize(img_hwc_u8, (100, 100), interpolation=cv2.INTER_LINEAR)
    data = (resized.astype(np.float32) / 255.0).reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    _compactness, labels, centers = cv2.kmeans(
        data, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    labels = labels.reshape(-1)
    max_label = int(np.bincount(labels, minlength=centers.shape[0]).argmax())
    background_color_01 = centers[max_label]
    bg_rgb_255 = background_color_01 * 255.0

    diff = img_hwc_u8.astype(np.float32) - bg_rgb_255.reshape((1, 1, 3))
    dist_sq_map = np.sum(diff * diff, axis=2)
    dist_sq = float(dist) * float(dist)
    mask = dist_sq_map <= dist_sq

    out = img_hwc_u8.copy()
    out[mask] = 0
    return out


def background_strip(image: torch.Tensor, dist: float = 150) -> torch.Tensor:
    """
    Strip pixels within Euclidean distance ``dist`` of the k-means background.

    **Input:** ``float32`` ``[3, H, W]`` on CPU, values in ``[0, 1]``.

    **Output:** ``float32`` ``[3, H, W]`` on CPU; ``[0, 1]`` (zeros where stripped).
    """
    if image.dim() != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected [3, H, W], got shape {tuple(image.shape)}")

    t01 = image.detach().float().clamp(0.0, 1.0)
    rgb_u8_chw = (t01 * 255.0).round().clamp(0, 255).to(torch.uint8)
    rgb_u8_hwc = rgb_u8_chw.permute(1, 2, 0).contiguous().numpy()
    result_hwc = _background_strip_np(rgb_u8_hwc, dist=dist)
    result_chw = torch.from_numpy(result_hwc.transpose(2, 0, 1).copy()).float().div_(255.0)
    return result_chw
