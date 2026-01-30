import torch.nn as nn
from torchvision.models.detection.faster_rcnn import FasterRCNN
from omegaconf import DictConfig
import logging
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.mobilenetv3 import mobilenet_v3_large

log = logging.getLogger(__name__)
#resizes to 800x1333 (will resize any image to this size)
class FasterRCNNDetector(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(FasterRCNNDetector, self).__init__()
        
        backbone = mobilenet_v3_large(weights=None)

        backbone_fpn = BackboneWithFPN(backbone.features,
                                        return_layers={'4': '0', '6': '1', '12': '2', '16': '3'},  # typical MobileNetV3 FPN layers
                                        in_channels_list=[40, 40, 112, 960],
                                        out_channels=256)
                                        
        
        # Load pretrained model withOUT default weights
        self.model = FasterRCNN(
            backbone= backbone_fpn,
            num_classes=cfg.model.num_classes +1,
            box_nms_thresh=0.5,     # NMS IoU threshold
        )
        
        input_h, input_w = cfg.model.transform.input_size
        self.model.transform = GeneralizedRCNNTransform(
            min_size=input_h,
            max_size=input_w,
            image_mean=cfg.model.transform.image_mean,
            image_std=cfg.model.transform.image_std
        )
    
    def forward(self, data, targets=None):
        outputs = self.model(data, targets)
        return outputs

