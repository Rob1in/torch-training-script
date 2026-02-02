import logging

import torch.nn as nn
from omegaconf import DictConfig
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.models.detection.transform import GeneralizedRCNNTransform

log = logging.getLogger(__name__)

class SSDLiteDetector(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(SSDLiteDetector, self).__init__()
  
        input_h, input_w = cfg.model.transform.input_size
        
        # Load model with proper weights enum
        if cfg.model.pretrained:
            weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
            self.model = ssdlite320_mobilenet_v3_large(weights=weights)
            
            log.info("Loaded SSDLite with COCO_V1 pretrained weights")
            log.info("Model's built-in transform: min_size=320, max_size=320")
            log.info(f"Using normalization from weights - mean: {self.model.transform.image_mean}, std: {self.model.transform.image_std}")
            log.warning("⚠️  When using pretrained weights, the model's built-in transform will handle resizing.")
            log.warning("⚠️  Remove 'Resize' from your config transforms to avoid double-resizing!")
        else:
            self.model = ssdlite320_mobilenet_v3_large(weights=None)
            
            # Only override transform if NOT using pretrained weights
            # When training from scratch, we can use custom normalization
            self.model.transform = GeneralizedRCNNTransform(
                min_size=input_h,
                max_size=input_w,
                image_mean=cfg.model.transform.get('image_mean', [0.485, 0.456, 0.406]),
                image_std=cfg.model.transform.get('image_std', [0.229, 0.224, 0.225])
            )
            log.info(f"Training from scratch with custom transform - mean: {self.model.transform.image_mean}, std: {self.model.transform.image_std}")
            log.info("✓ Config transforms (including Resize) will be applied correctly.")

        # Update classification head for custom number of classes if needed
        # Only modify if num_classes is different from default (91 classes in COCO)
        self.num_classes = cfg.model.num_classes
        if cfg.model.num_classes != 90:  # 90 classes + 1 background = 91 total
            num_anchors = self.model.anchor_generator.num_anchors_per_location()
            size = (320, 320)
            in_channels = det_utils.retrieve_out_channels(self.model.backbone, size)
            
            if cfg.model.pretrained:
                log.info("Transfer learning: Replacing classification head")
                log.info("  - COCO classes: 91 (90 + background)")
                log.info(f"  - Your classes: {cfg.model.num_classes + 1} ({cfg.model.num_classes} + background)")
                log.info("  - Keeping pretrained: Backbone (MobileNetV3), box regression head")
                log.info("  - Replacing with random init: Classification head only")
            
            self.model.head.classification_head = SSDLiteClassificationHead(
                in_channels=in_channels,
                num_anchors=num_anchors,
                num_classes=cfg.model.num_classes + 1,  # +1 for background
                norm_layer=nn.BatchNorm2d
            )
        else:
            if cfg.model.pretrained:
                log.info("Using COCO classes (90), keeping all pretrained weights")
    
    def to(self, device):
        """Override to method to ensure all components are moved to device."""
        self.model = self.model.to(device)
        return super().to(device)

    def forward(self, data, targets=None):
        return self.model(data, targets)

     

