import logging

import torch
import torch.nn as nn
from omegaconf import DictConfig

log = logging.getLogger(__name__)

class SimpleDetector(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(SimpleDetector, self).__init__()
         
        # config vals
        backbone_channels = cfg.model.backbone.channels
        kernel_size = cfg.model.backbone.kernel_size
        padding = cfg.model.backbone.padding
        hidden_size = cfg.model.detector.hidden_size
        dropout = cfg.model.detector.dropout
        num_classes = cfg.model.num_classes
        num_anchors = cfg.model.num_anchors

        self.expected_size = cfg.model.transform.input_size
        
        # CNN backbone (RGB: 3 channels)
        self.backbone = nn.Sequential(
            # input -> 32
            nn.Conv2d(3, backbone_channels[0], kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(backbone_channels[0]),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 32 -> 64
            nn.Conv2d(backbone_channels[0], backbone_channels[1], kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(backbone_channels[1]),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 64 -> 128
            nn.Conv2d(backbone_channels[1], backbone_channels[2], kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(backbone_channels[2]),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # size is reduced by factor of 8 after 3 maxpool layers 
        h, w = self.expected_size
        feature_h, feature_w = h // 8, w // 8
        feature_size = feature_h * feature_w * backbone_channels[-1]
        
        # detection heads (separate for bbox and classification)
        self.bbox_head = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_anchors * 4)  # allowig for many bboxes
        )

        self.cls_head = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_anchors * num_classes)  # classification for each bbox 
        )
        
    def forward(self, x, targets=None):
        # Expects batched tensor [B, C, H, W]
        features = self.backbone(x)
        
        # Flatten features for the heads
        batch_size = features.shape[0]
        features = features.view(batch_size, -1)
        # predictions from both heads
        boxes = self.bbox_head(features)
        boxes = boxes.view(batch_size, -1, 4)
        scores = self.cls_head(features)
        scores = scores.view(batch_size, -1, 1)  # batch_size, num_anchors, num_classes which is 1 for binary classification (hardcoded)
        
        # return in torchvision format (Dict[Tensor])
        return {'boxes': boxes, 'scores': scores, 'labels': torch.ones(batch_size, dtype=torch.int64) } 

