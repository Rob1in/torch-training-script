#designed as a classificer - might not be as helfpul as a detector since required to modify detection heads 
import math

import hydra
import torch
import torch.nn as nn
import torchvision.models as models
from omegaconf import DictConfig
from torchvision.models.efficientnet import EfficientNet_B0_Weights, efficientnet_b0
from torchvision.ops import box_iou


class EfficientNetDetector(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(EfficientNetDetector, self).__init__()
        
        weights = EfficientNet_B0_Weights.DEFAULT
        self.backbone = efficientnet_b0(weights=weights)

        # Remove classifier to get features
        self.backbone.classifier = nn.Identity()
        
        # Get number of output features from the backbone's last layer
        self.num_features = self.backbone.features[-1].out_channels
        
        self.expected_size = cfg.model.transform.input_size
    
        self.bbox_head = nn.Sequential(
            nn.Linear(self.num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4)  # bounding box 
        )
        
        self.cls_head = nn.Sequential(
            nn.Linear(self.num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, cfg.model.num_classes)
        )
    
    def forward(self, x, targets=None):
        # Expects batched tensor [B, C, H, W]
        features = self.backbone(x)
        
        # Flatten features for the heads
        batch_size = features.shape[0]
        features = features.view(batch_size, -1)
        
        # predictions from both heads
        bbox_pred = self.bbox_head(features)
        cls_pred = self.cls_head(features)
        
        # Return in torchvision format
        return {'boxes': bbox_pred, 'labels': torch.ones(bbox_pred.shape[0], dtype=torch.int64), 'scores': cls_pred} #dict of tensors format to match torchvision 

