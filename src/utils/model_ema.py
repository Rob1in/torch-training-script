"""
Exponential Moving Average (EMA) for model weights.
Based on PyTorch's reference detection implementation.
"""
import copy
from collections import OrderedDict
import torch
import torch.nn as nn


class ModelEMA:
    """
    Exponential Moving Average of model parameters.
    
    Maintains shadow copy of model parameters that are updated as:
        ema_param = decay * ema_param + (1 - decay) * model_param
    
    Args:
        model: Model to track
        decay: Decay rate (typically 0.9998 or 0.9999)
        device: Device to store EMA parameters
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9998, device: torch.device = None):
        self.decay = decay
        self.device = device if device is not None else torch.device('cpu')
        
        # Create shadow copy of model
        self.module = copy.deepcopy(model)
        self.module.eval()
        
        # Move to device if specified
        if device is not None:
            self.module = self.module.to(device)
        
        # Disable gradients for EMA model
        for param in self.module.parameters():
            param.requires_grad_(False)
    
    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        Update EMA parameters.
        
        Args:
            model: Current model with updated parameters
        """
        # Update all parameters
        model_params = dict(model.named_parameters())
        ema_params = dict(self.module.named_parameters())
        
        for name, param in model_params.items():
            if name in ema_params:
                ema_params[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)
        
        # Update all buffers (batch norm running stats, etc.)
        model_buffers = dict(model.named_buffers())
        ema_buffers = dict(self.module.named_buffers())
        
        for name, buffer in model_buffers.items():
            if name in ema_buffers:
                ema_buffers[name].copy_(buffer.data)
    
    def state_dict(self):
        """Return state dict of EMA model."""
        return self.module.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load state dict into EMA model."""
        self.module.load_state_dict(state_dict)
    
    def to(self, device):
        """Move EMA model to device."""
        self.device = device
        self.module = self.module.to(device)
        return self
    
    def eval(self):
        """Set EMA model to eval mode."""
        self.module.eval()
        return self.module
    
    def __call__(self, *args, **kwargs):
        """Forward pass through EMA model."""
        return self.module(*args, **kwargs)
