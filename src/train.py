#training script for object detection models
import gc
import logging
import math
import sys
from pathlib import Path

import hydra
import torch
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets.viam_dataset import ViamDataset
from models.faster_rcnn_detector import FasterRCNNDetector
from models.ssdlite_detector import SSDLiteDetector
from utils.freeze import configure_model_for_transfer_learning
from utils.model_ema import ModelEMA
from utils.seed import set_seed
from utils.transforms import DetectionTransform, GPUCollate

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("eval", eval)


def train_one_epoch(model, optimizer, data_loader, device, epoch, cfg, model_ema=None):
    """
    Train for one epoch. Matches PyTorch Vision reference implementation.
    
    Args:
        model: The model to train
        optimizer: Optimizer
        data_loader: Training data loader
        device: Device to train on
        epoch: Current epoch number
        cfg: Hydra config
        model_ema: Optional EMA model
        
    Returns:
        Dictionary of training metrics
    """
    model.train()
    
    # PyTorch reference: Warmup only in epoch 0
    warmup_scheduler = None
    if epoch == 0:
        warmup_factor = cfg.training.get('warmup_factor', 0.001)  # 1/1000
        warmup_iters = min(cfg.training.get('warmup_iters', 1000), len(data_loader) - 1)
        
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=warmup_factor,
            total_iters=warmup_iters
        )
        log.info(f"Epoch 0: Applying warmup for {warmup_iters} iterations (start_factor={warmup_factor})")
    
    train_loss = 0.0
    train_losses = {
        'loss_classifier': 0.0,
        'loss_box_reg': 0.0,
        'loss_objectness': 0.0,
        'loss_rpn_box_reg': 0.0
    }
    
    pbar = tqdm(data_loader, desc=f'Epoch {epoch+1} [Train]')
    
    for batch_idx, (images, targets) in enumerate(pbar):
        # Forward pass - model returns loss dict in training mode
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Check for nan/inf losses (PyTorch reference pattern)
        loss_value = losses.item()
        if not math.isfinite(loss_value):
            log.error(f"Loss is {loss_value}, stopping training")
            log.error(f"Loss dict: {loss_dict}")
            sys.exit(1)
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        
        # Gradient clipping (optional, PyTorch reference doesn't use it by default)
        if cfg.training.get('gradient_clip', 0.0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.training.gradient_clip)
        
        optimizer.step()
        
        # Update EMA after optimizer step
        if model_ema is not None:
            model_ema.update(model)
        
        # PyTorch reference: Warmup scheduler only in epoch 0, stepped per-iteration
        if warmup_scheduler is not None:
            warmup_scheduler.step()
        
        # Accumulate losses for logging
        train_loss += loss_value
        for k, v in loss_dict.items():
            train_losses[k] += v.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_value:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
    
    # Average losses
    avg_train_loss = train_loss / len(data_loader)
    avg_losses = {k: v / len(data_loader) for k, v in train_losses.items()}
    
    return {
        'loss': avg_train_loss,
        'loss_dict': avg_losses
    }


def evaluate(model, data_loader, device, epoch, cfg):
    """
    Evaluate on validation set. Returns average validation loss.
    
    Args:
        model: The model to evaluate
        data_loader: Validation data loader
        device: Device to evaluate on
        epoch: Current epoch number
        cfg: Hydra config
        
    Returns:
        Average validation loss
    """
    was_training = model.training
    
    # For torchvision models: set to train mode to get loss dict
    # But freeze BatchNorm and Dropout to prevent training
    model.train()
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.Dropout)):
            m.eval()
    
    val_loss = 0.0
    
    with torch.no_grad():
        for images, targets in data_loader:
            loss_dict = model(images, targets)
            batch_loss = sum(loss_dict.values()).item()
            val_loss += batch_loss
    
    # Restore original training state
    model.train(was_training)
    
    return val_loss / len(data_loader)


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    """Main training function."""
    
    log.info(f"check device: {torch.cuda.is_available()}")
    log.info(f"config:\n{OmegaConf.to_yaml(cfg)}")
    
    # Set random seed for reproducibility
    set_seed(cfg.experiment.seed)
    log.info(f"Set random seed to {cfg.experiment.seed} for reproducibility")
    
    # CUDA multiprocessing: Set spawn method to avoid fork issues
    if torch.cuda.is_available():
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
    
    # Device selection: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        log.warning("CUDA requested but not available. Falling back to MPS (macOS GPU).")
        device = torch.device('mps')
    else:
        log.warning("Neither CUDA nor MPS available. Falling back to CPU.")
        device = torch.device('cpu')
    
    log.info(f"Using device: {device}")
    
    # Create model
    if cfg.model.name == "faster_rcnn":
        model = FasterRCNNDetector(cfg).to(device)
        log.info("faster rcnn model created and moved to device")
    elif cfg.model.name == "ssdlite":
        model = SSDLiteDetector(cfg).to(device)
        log.info("ssdlite model created and moved to device")
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}. Supported models: faster_rcnn, ssdlite")
    
    # Apply transfer learning configuration (freezing layers if specified)
    freeze_config = {
        'freeze_backbone': cfg.training.get('freeze_backbone', False),
        'freeze_fpn': cfg.training.get('freeze_fpn', False),
        'freeze_rpn': cfg.training.get('freeze_rpn', False),
        'freeze_all': cfg.training.get('freeze_all', False),
    }
    configure_model_for_transfer_learning(model, cfg.model.name, freeze_config)
    
    # Create Model EMA if enabled
    model_ema = None
    if cfg.training.get('use_ema', False):
        ema_decay = cfg.training.get('ema_decay', 0.9998)
        model_ema = ModelEMA(model, decay=ema_decay, device=device)
        log.info(f"Created Model EMA with decay={ema_decay}")
    
    # Determine classes
    classes = cfg.get('classes', None)
    if classes is None:
        log.info("No classes specified in config. Auto-discovering from training data...")
        temp_dataset = ViamDataset(
            jsonl_path=cfg.dataset.data.train_jsonl,
            data_dir=cfg.dataset.data.train_data_dir,
            classes=None,
            transform=None
        )
        classes = temp_dataset.get_classes()
        log.info(f"Auto-discovered {len(classes)} classes: {classes}")
    
    cfg.model.num_classes = len(classes)
    log.info(f"Training with {cfg.model.num_classes} classes: {classes}")
    
    # Build transforms
    train_transform = DetectionTransform(cfg.dataset.transform.train) if cfg.dataset.transform.train else None
    val_transform = DetectionTransform(cfg.dataset.transform.val) if cfg.dataset.transform.val else None
    
    # Create datasets
    train_dataset = ViamDataset(
        jsonl_path=cfg.dataset.data.train_jsonl,
        data_dir=cfg.dataset.data.train_data_dir,
        classes=classes,
        transform=None  # Transform applied in collate_fn
    )
    
    val_dataset = ViamDataset(
        jsonl_path=cfg.dataset.data.val_jsonl,
        data_dir=cfg.dataset.data.val_data_dir,
        classes=classes,
        transform=None  # Transform applied in collate_fn
    )
    
    # Create dataloaders
    # PyTorch reference: batch_size per GPU, we're using 1 GPU
    num_workers = 0 if device.type == 'mps' else cfg.training.num_workers
    pin_memory = cfg.training.pin_memory and device.type != 'mps'
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=GPUCollate(device, train_transform)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=GPUCollate(device, val_transform)
    )
    
    # Create optimizer with normalization layer separation (PyTorch reference approach)
    norm_weight_decay = cfg.training.get('norm_weight_decay', None)
    
    if norm_weight_decay is None:
        # Simple case: all parameters get same weight decay
        parameters = [p for p in model.parameters() if p.requires_grad]
        log.info("Optimizer: Single parameter group (all trainable params)")
        log.info(f"  - {sum(p.numel() for p in parameters):,} total trainable params")
    else:
        # Split normalization layers from other parameters (PyTorch reference approach)
        from torchvision.ops._utils import split_normalization_params
        param_groups = split_normalization_params(model)
        wd_groups = [norm_weight_decay, cfg.training.weight_decay]
        parameters = [{"params": p, "weight_decay": w} for p, w in zip(param_groups, wd_groups) if p]
        
        log.info("Optimizer: Separate normalization layer weight decay")
        log.info(f"  - Norm layers: {sum(p.numel() for p in param_groups[0]):,} params, weight_decay={norm_weight_decay}")
        log.info(f"  - Other layers: {sum(p.numel() for p in param_groups[1]):,} params, weight_decay={cfg.training.weight_decay}")
    
    # Use SGD with momentum (PyTorch reference standard)
    if cfg.training.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            parameters,
            lr=cfg.training.learning_rate,
            momentum=cfg.training.momentum,
            weight_decay=cfg.training.weight_decay,
            nesterov=cfg.training.get('nesterov', False)
        )
    elif cfg.training.optimizer == "adam":
        # Fallback to Adam if specified
        optimizer = torch.optim.Adam(
            parameters,
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.training.optimizer}")
    
    # Learning rate scheduler (PyTorch reference: MultiStepLR, stepped per-epoch)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg.training.lr_steps,
        gamma=cfg.training.lr_gamma
    )
    
    log.info(f"Scheduler: MultiStepLR (milestones={cfg.training.lr_steps}, gamma={cfg.training.lr_gamma})")
    log.info(f"Warmup: Will be applied in epoch 0 only ({cfg.training.get('warmup_iters', 1000)} iterations)")
    
    # Training loop
    writer = SummaryWriter(Path(cfg.logging.save_dir) / 'tensorboard')
    best_val_loss = float('inf')
    patience_counter = 0
    
    log.info("Starting training...")
    
    for epoch in range(cfg.training.num_epochs):
        # Train one epoch
        train_metrics = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            cfg=cfg,
            model_ema=model_ema
        )
        
        # Evaluate on validation set
        # Use EMA model if available (typically performs better)
        eval_model = model_ema.module if model_ema is not None else model
        val_loss = evaluate(eval_model, val_loader, device, epoch, cfg)
        
        # Log metrics
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        for loss_name, loss_value in train_metrics['loss_dict'].items():
            writer.add_scalar(f'EpochLoss/{loss_name}', loss_value, epoch)
        
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        log.info(f'Epoch {epoch+1}/{cfg.training.num_epochs}: '
                f'Train Loss: {train_metrics["loss"]:.4f}, '
                f'Val Loss: {val_loss:.4f}, '
                f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # PyTorch reference: Step scheduler per-epoch (after validation)
        scheduler.step()
        
        # Save checkpoint if best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = Path(cfg.logging.save_dir) / 'best_model.pth'
            
            checkpoint_dict = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
            }
            
            if model_ema is not None:
                checkpoint_dict['model_ema_state_dict'] = model_ema.state_dict()
            
            torch.save(checkpoint_dict, checkpoint_path)
            log.info(f'Saved best model to {checkpoint_path}')
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= cfg.training.early_stopping_patience:
            log.info(f'Early stopping triggered after {epoch + 1} epochs')
            break
    
    writer.close()
    log.info(f"Training complete. Best validation loss: {best_val_loss:.4f}")
    
    # Cleanup for Optuna
    gc.collect()
    torch.cuda.empty_cache()
    
    return best_val_loss


if __name__ == "__main__":
    main()
