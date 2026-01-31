"""
Learning rate schedulers for object detection training.
Based on PyTorch's reference detection implementation.
"""
import math
from typing import List

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WarmupMultiStepLR(_LRScheduler):
    """
    Multi-step learning rate scheduler with linear warmup.
    
    Combines warmup period with step decay at specified milestones.
    Based on PyTorch's detection reference implementation.
    
    Args:
        optimizer: Wrapped optimizer
        milestones: List of epoch indices to decay LR
        gamma: Multiplicative factor of learning rate decay
        warmup_iters: Number of iterations for warmup
        warmup_factor: Starting LR = base_lr * warmup_factor
        warmup_method: 'linear' or 'constant'
        last_epoch: The index of last epoch
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        milestones: List[int],
        gamma: float = 0.1,
        warmup_iters: int = 1000,
        warmup_factor: float = 0.001,
        warmup_method: str = "linear",
        last_epoch: int = -1,
    ):
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_iters = warmup_iters
        self.warmup_factor = warmup_factor
        self.warmup_method = warmup_method
        self._step = 0
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        warmup_factor = self._get_warmup_factor_at_iter(self._step)
        
        # Compute step decay multiplier
        if self.last_epoch not in self.milestones:
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        
        # Apply step decay
        return [
            base_lr * warmup_factor * self.gamma ** self.milestones[:self.last_epoch + 1].count(self.last_epoch)
            for base_lr in self.base_lrs
        ]
    
    def _get_warmup_factor_at_iter(self, step: int) -> float:
        """Compute warmup factor based on current iteration."""
        if step >= self.warmup_iters:
            return 1.0
        
        if self.warmup_method == "constant":
            return self.warmup_factor
        elif self.warmup_method == "linear":
            alpha = step / self.warmup_iters
            return self.warmup_factor * (1 - alpha) + alpha
        else:
            raise ValueError(f"Unknown warmup method: {self.warmup_method}")
    
    def step(self, epoch=None):
        """Step the scheduler."""
        if epoch is None:
            epoch = self.last_epoch + 1
            self._step += 1
        else:
            self._step = epoch
        self.last_epoch = epoch
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class WarmupCosineAnnealingLR(_LRScheduler):
    """
    Cosine annealing learning rate scheduler with linear warmup.
    
    Args:
        optimizer: Wrapped optimizer
        T_max: Maximum number of iterations
        eta_min: Minimum learning rate
        warmup_iters: Number of iterations for warmup
        warmup_factor: Starting LR = base_lr * warmup_factor
        last_epoch: The index of last epoch
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        eta_min: float = 0,
        warmup_iters: int = 1000,
        warmup_factor: float = 0.001,
        last_epoch: int = -1,
    ):
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_iters = warmup_iters
        self.warmup_factor = warmup_factor
        self._step = 0
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        warmup_factor = self._get_warmup_factor_at_iter(self._step)
        
        if self._step < self.warmup_iters:
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        
        # Cosine annealing after warmup
        progress = (self._step - self.warmup_iters) / (self.T_max - self.warmup_iters)
        return [
            self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress)) / 2
            for base_lr in self.base_lrs
        ]
    
    def _get_warmup_factor_at_iter(self, step: int) -> float:
        """Compute warmup factor based on current iteration."""
        if step >= self.warmup_iters:
            return 1.0
        
        alpha = step / self.warmup_iters
        return self.warmup_factor * (1 - alpha) + alpha
    
    def step(self, epoch=None):
        """Step the scheduler."""
        if epoch is None:
            epoch = self.last_epoch + 1
            self._step += 1
        else:
            self._step = epoch
        self.last_epoch = epoch
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
