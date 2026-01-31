"""
Utility functions for freezing/unfreezing model layers.
Used for transfer learning strategies.
"""
import logging

import torch.nn as nn

log = logging.getLogger(__name__)


def freeze_layers(model: nn.Module, layer_names: list = None):
    """
    Freeze parameters in specified layers (disable gradient computation).
    
    Args:
        model: PyTorch model
        layer_names: List of layer name prefixes to freeze (e.g., ['backbone', 'fpn'])
                     If None, freezes all parameters
    """
    if layer_names is None:
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        log.info("Froze ALL model parameters")
        return
    
    frozen_count = 0
    total_count = 0
    
    for name, param in model.named_parameters():
        total_count += 1
        # Check if this parameter belongs to any of the specified layers
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = False
            frozen_count += 1
    
    log.info(f"Froze {frozen_count}/{total_count} parameters in layers: {layer_names}")


def unfreeze_layers(model: nn.Module, layer_names: list = None):
    """
    Unfreeze parameters in specified layers (enable gradient computation).
    
    Args:
        model: PyTorch model
        layer_names: List of layer name prefixes to unfreeze (e.g., ['roi_heads'])
                     If None, unfreezes all parameters
    """
    if layer_names is None:
        # Unfreeze all parameters
        for param in model.parameters():
            param.requires_grad = True
        log.info("Unfroze ALL model parameters")
        return
    
    unfrozen_count = 0
    total_count = 0
    
    for name, param in model.named_parameters():
        total_count += 1
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = True
            unfrozen_count += 1
    
    log.info(f"Unfroze {unfrozen_count}/{total_count} parameters in layers: {layer_names}")


def configure_model_for_transfer_learning(model, model_name: str, freeze_config: dict):
    """
    Configure which layers to freeze/unfreeze based on transfer learning strategy.
    
    Architecture-specific components:
    - Faster R-CNN: backbone, fpn, rpn, roi_heads
    - SSDLite: backbone, head
    - EfficientNet: backbone, bbox_head, cls_head
    - SimpleDetector: backbone, detection heads
    
    Args:
        model: Detection model (Faster R-CNN, SSDLite, etc.)
        model_name: Name of the model ('faster_rcnn', 'ssdlite', etc.)
        freeze_config: Dict with freeze settings:
            - freeze_backbone: bool (all models)
            - freeze_fpn: bool (Faster R-CNN ONLY)
            - freeze_rpn: bool (Faster R-CNN ONLY)
            - freeze_all: bool (overrides others, all models)
    
    Returns:
        Number of trainable parameters
    """
    # Get the actual model (unwrap if it's a wrapper)
    actual_model = model.model if hasattr(model, 'model') else model
    
    # Start with everything trainable
    for param in actual_model.parameters():
        param.requires_grad = True
    
    # Apply freeze settings
    if freeze_config.get('freeze_all', False):
        freeze_layers(actual_model, layer_names=None)
        log.warning("⚠️  freeze_all=True: ALL layers frozen! Only unfreezing detection head...")
        # Unfreeze only the detection head
        if model_name == 'faster_rcnn':
            unfreeze_layers(actual_model, layer_names=['roi_heads.box_predictor'])
        elif model_name == 'ssdlite':
            unfreeze_layers(actual_model, layer_names=['head.classification_head'])
    else:
        layers_to_freeze = []
        
        # Backbone (all models have this)
        if freeze_config.get('freeze_backbone', False):
            layers_to_freeze.append('backbone')
        
        # FPN (Faster R-CNN only)
        if freeze_config.get('freeze_fpn', False):
            if model_name == 'faster_rcnn':
                layers_to_freeze.append('fpn')
            else:
                log.warning(f"⚠️  freeze_fpn=True ignored for {model_name} (only Faster R-CNN has FPN)")
        
        # RPN (Faster R-CNN only)
        if freeze_config.get('freeze_rpn', False):
            if model_name == 'faster_rcnn':
                layers_to_freeze.append('rpn')
            else:
                log.warning(f"⚠️  freeze_rpn=True ignored for {model_name} (only Faster R-CNN has RPN)")
        
        if layers_to_freeze:
            freeze_layers(actual_model, layer_names=layers_to_freeze)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in actual_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in actual_model.parameters())
    
    log.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.1f}%)")
    
    return trainable_params
