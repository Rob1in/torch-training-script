import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.ssdlite import SSDLiteHead, SSDLiteClassificationHead, SSDLiteRegressionHead, SSDLiteFeatureExtractorMobileNet
from torchvision.models.detection import _utils as det_utils
import hydra
from omegaconf import DictConfig
import logging
log = logging.getLogger(__name__)
from torchvision.models.mobilenetv3 import mobilenet_v3_large
from torchvision.models.detection.ssd import SSD, DefaultBoxGenerator
#requires image of size 320x320

class SSDLiteDetector(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(SSDLiteDetector, self).__init__()
  
        input_h, input_w = cfg.model.transform.input_size
        backbone = mobilenet_v3_large(weights=None)

        feature_extractor = SSDLiteFeatureExtractorMobileNet(backbone.features, c4_pos=15, norm_layer=nn.BatchNorm2d, width_mult=0.75, min_depth=16)

        # Default SSD anchor configuration (as in torchvision)
        anchor_generator = DefaultBoxGenerator(
            [[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
            min_ratio=15,
            max_ratio=90,
            aspect_ratios=[[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]
        )
        self.model = SSD(
            feature_extractor=feature_extractor,
            anchor_generator=anchor_generator,
            size=(input_h, input_w),
            num_classes=cfg.model.num_classes +1, #for background     
        )
        from torchvision.models.detection.transform import GeneralizedRCNNTransform
        self.model.transform = GeneralizedRCNNTransform(
            min_size=input_h,
            max_size=input_w,
            image_mean=cfg.model.transform.image_mean if hasattr(cfg.model.transform, 'image_mean') else [0.485, 0.456, 0.406],
            image_std=cfg.model.transform.image_std if hasattr(cfg.model.transform, 'image_std') else [0.229, 0.224, 0.225]
        )

        # specifying classification and regeresion heads
        num_anchors = self.model.anchor_generator.num_anchors_per_location()
        size = (320, 320)
        in_channels = det_utils.retrieve_out_channels(self.model.backbone, size) #retrievng out channels from backbone --> last layer of backbone is first layer of classification head (320 is the expected size for ssdlite)
        # Create new classification head
        self.model.head.classification_head = SSDLiteClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=cfg.model.num_classes + 1
        )

        self.model.head.regression_head = SSDLiteRegressionHead(
            in_channels=in_channels,
            num_anchors=num_anchors)
        

    def forward(self, data, targets=None):
        return self.model(data, targets)

     

