import logging

import torch.nn as nn
from omegaconf import DictConfig
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.faster_rcnn import (
    FasterRCNN,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
)
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.mobilenetv3 import mobilenet_v3_large

log = logging.getLogger(__name__)
#resizes to 800x1333 (will resize any image to this size)
class FasterRCNNDetector(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(FasterRCNNDetector, self).__init__()
        
        input_h, input_w = cfg.model.transform.input_size
        
        # Check if pretrained option exists in config
        pretrained = cfg.model.get('pretrained', False)
        
        if pretrained:
            # Load with pretrained weights
            weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
            from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
            self.model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
            
            log.info("Loaded Faster R-CNN with COCO_V1 pretrained weights")
            log.info("Model's built-in transform: min_size=800, max_size=1333")
            log.info(f"Using normalization from weights - mean: {self.model.transform.image_mean}, std: {self.model.transform.image_std}")
            log.warning("⚠️  When using pretrained weights, the model's built-in transform will handle resizing.")
            log.warning("⚠️  Remove 'Resize' from your config transforms to avoid double-resizing!")
            
            # Update num_classes if different from COCO (91 classes)
            if cfg.model.num_classes != 90:  # 90 classes + 1 background
                from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
                in_features = self.model.roi_heads.box_predictor.cls_score.in_features
                
                log.info("Transfer learning: Replacing box predictor head")
                log.info("  - COCO classes: 91 (90 + background)")
                log.info(f"  - Your classes: {cfg.model.num_classes + 1} ({cfg.model.num_classes} + background)")
                log.info("  - Keeping pretrained: Backbone, FPN, RPN (feature extraction)")
                log.info("  - Replacing with random init: Box predictor (classification + regression)")
                
                self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, cfg.model.num_classes + 1)
            else:
                log.info("Using COCO classes (90), keeping all pretrained weights")
        else:
            # Build from scratch with custom backbone
            backbone = mobilenet_v3_large(weights=None)

            backbone_fpn = BackboneWithFPN(backbone.features,
                                                return_layers={'4': '0', '6': '1', '12': '2', '16': '3'},
                                            in_channels_list=[40, 40, 112, 960],
                                            out_channels=256)
                                            
            self.model = FasterRCNN(
                    backbone=backbone_fpn,
                    num_classes=cfg.model.num_classes + 1,
                    box_nms_thresh=0.5,
            )
            
            # Use custom transform from config
            self.model.transform = GeneralizedRCNNTransform(
                min_size=input_h,
                max_size=input_w,
                image_mean=cfg.model.transform.get('image_mean', [0.485, 0.456, 0.406]),
                image_std=cfg.model.transform.get('image_std', [0.229, 0.224, 0.225])
            )
            log.info(f"Training from scratch with custom transform - mean: {self.model.transform.image_mean}, std: {self.model.transform.image_std}")
            log.info("✓ Config transforms (including Resize) will be applied correctly.")
    
    def forward(self, data, targets=None):
        outputs = self.model(data, targets)
        return outputs

