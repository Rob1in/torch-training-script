#trainign script for all models except yolo
import gc
import logging
from pathlib import Path

import hydra
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from tqdm import tqdm

from datasets.viam_dataset import ViamDataset
from models.custom_detector import SimpleDetector
from models.effnet_detector import EfficientNetDetector
from models.faster_rcnn_detector import FasterRCNNDetector
from models.ssdlite_detector import SSDLiteDetector
from utils.freeze import configure_model_for_transfer_learning
from utils.model_ema import ModelEMA
from utils.seed import set_seed
from utils.transforms import GPUCollate, build_transforms

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("eval", eval)
def train_model(model, train_loader, val_loader, optimizer, scheduler, device, cfg: DictConfig, model_ema=None):
    # Create tensorboard writer using Hydra's output directory
    writer = SummaryWriter(Path(cfg.logging.save_dir) / 'tensorboard')  # Use Hydra's output directory
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(cfg.training.num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_losses = {'loss_classifier': 0.0,
                        'loss_box_reg': 0.0,
                        'loss_objectness': 0.0,
                        'loss_rpn_box_reg': 0.0}
        
        # PyTorch reference: Warmup only in epoch 0
        warmup_scheduler = None
        if epoch == 0:
            warmup_factor = cfg.training.get('warmup_factor', 0.001)  # 1/1000
            warmup_iters = min(cfg.training.get('warmup_iters', 1000), len(train_loader) - 1)
            
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=warmup_factor,
                total_iters=warmup_iters
            )
            log.info(f"Epoch 0: Applying warmup for {warmup_iters} iterations (start_factor={warmup_factor})")
                        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{cfg.training.num_epochs} [Train]')
        for batch_idx, (data, targets) in enumerate(train_pbar):
            optimizer.zero_grad()

            if cfg.model.name in ["faster_rcnn", "ssdlite"]:
                # Torchvision models return loss dict - use default weights (sum all losses)
                loss_dict = model(data, targets)
                loss = sum(loss for loss in loss_dict.values()) 
            
            elif cfg.model.name in ["effnet", "custom_detector"]: # For custom models that don't compute loss internally
                outputs = model(data, targets)
                # Compute loss manually
                boxes_pred = outputs['boxes']  # Shape: [batch_size, 4]
                scores_pred = outputs['scores']  # Shape: [batch_size, num_classes]
                
                # For effnet/custom_detector: they predict 1 box per image
                # Use the first target box from each image as ground truth
                target_boxes = torch.stack([t['boxes'][0] for t in targets], dim=0)  # [batch_size, 4]
                target_labels = torch.stack([t['labels'][0] for t in targets], dim=0)  # [batch_size]
                
                # Simple MSE loss for boxes and CE loss for classification
                box_loss = nn.functional.mse_loss(boxes_pred, target_boxes)
                cls_loss = nn.functional.cross_entropy(scores_pred, target_labels - 1)  # -1 because labels are 1-indexed
                
                loss = box_loss * cfg.training.loss.box_loss_weight + cls_loss * cfg.training.loss.cls_loss_weight
                
                # Create loss_dict for logging
                loss_dict = {
                    'loss_classifier': cls_loss,
                    'loss_box_reg': box_loss,
                    'loss_objectness': torch.tensor(0.0, device=device),
                    'loss_rpn_box_reg': torch.tensor(0.0, device=device)
                }
            else:
                raise ValueError(f"Unknown model name: {cfg.model.name}")

            loss.backward()
            
            # Gradient clipping
            if cfg.training.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.training.gradient_clip)
            
            optimizer.step()
            
            # Update EMA after optimizer step
            if model_ema is not None:
                model_ema.update(model)
            
            # PyTorch reference: Warmup scheduler only in epoch 0, stepped per-iteration
            if warmup_scheduler is not None:
                warmup_scheduler.step()
            
            train_loss += loss.item()

            # Log individual losses per batch to tensorboard
            for loss_name, loss_value in loss_dict.items():
                train_losses[loss_name] += loss_value.item()
                writer.add_scalar(f'BatchLoss/{loss_name}', loss_value.item(), 
                                epoch * len(train_loader) + batch_idx)

                # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })

        avg_losses = {k: v/len(train_loader) for k, v in train_losses.items()}
        avg_train_loss = train_loss / len(train_loader) 

        # Run validation for all model types
        # Use EMA model if available (typically performs better)
        eval_model = model_ema.module if model_ema is not None else model
        avg_val_loss = evaluate_validation(eval_model, val_loader, device, epoch, cfg)

        # Log epoch metrics to tensorboard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        
        # Log individual loss components
        for loss_name, loss_value in avg_losses.items():
            writer.add_scalar(f'EpochLoss/{loss_name}', loss_value, epoch)
        
        # Log learning rate
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Log epoch metrics
        log.info(f'Epoch {epoch+1}/{cfg.training.num_epochs}: '
                f'Avg Train Loss: {avg_train_loss:.4f}, '
                f'Avg Val Loss: {avg_val_loss:.4f}')
        
        # PyTorch reference: Step scheduler per-epoch (after validation)
        scheduler.step()

        # Save checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = Path(cfg.logging.save_dir) / 'best_model.pth'
            
            checkpoint_dict = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
            }
            
            # Save EMA state if available
            if model_ema is not None:
                checkpoint_dict['model_ema_state_dict'] = model_ema.state_dict()
            
            torch.save(checkpoint_dict, checkpoint_path)
            log.info(f'Saved best model checkpoint to {checkpoint_path}')
            patience_counter = 0
        else:
             patience_counter += 1
        
        #early stopping
        if patience_counter >= cfg.training.early_stopping_patience: #10 epochs without improvement 
            log.info(f'Early stopping triggered after {epoch + 1} epochs')
            break

    writer.close()
    return best_val_loss  # for optuna to minimize the validation loss 

def evaluate_validation(model, val_loader, device, epoch, cfg: DictConfig):
    
    was_training = model.training #boolean flag to return model to mode it was in before evaluation

    # force loss-returning behavior for torchvision models
    if cfg.model.name in ["faster_rcnn", "ssdlite"]:
        model.train()
        # workaround to freeze batchnorm and dropout layers to avoid training them
        for m in model.modules():
            if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.Dropout)):
                m.eval()
    else:
        model.eval()

    val_loss = 0.0
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            if cfg.model.name in ["faster_rcnn", "ssdlite"]:
                loss_dict = model(images, targets)
                batch_loss = sum(loss_dict.values()).item()
            elif cfg.model.name in ["effnet", "custom_detector"]:
                outputs = model(images, targets)
                boxes_pred = outputs['boxes']  # [batch_size, 4]
                scores_pred = outputs['scores']  # [batch_size, num_classes]
                
                # Use the first target box from each image
                target_boxes = torch.stack([t['boxes'][0] for t in targets], dim=0)  # [batch_size, 4]
                target_labels = torch.stack([t['labels'][0] for t in targets], dim=0)  # [batch_size]
                
                # Simple MSE loss for boxes and CE loss for classification
                box_loss = nn.functional.mse_loss(boxes_pred, target_boxes)
                cls_loss = nn.functional.cross_entropy(scores_pred, target_labels - 1)
                
                batch_loss = (box_loss * cfg.training.loss.box_loss_weight + 
                            cls_loss * cfg.training.loss.cls_loss_weight).item()
            else:
                raise ValueError(f"Unknown model name: {cfg.model.name}")
            
            val_loss += batch_loss
    
    # back to original mode 
    model.train(was_training)
    
    return val_loss / len(val_loader)

@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    # On macOS, 'spawn' is default, but we only set it if not already set
    try:
        mp.set_start_method('spawn', force=False)
    except RuntimeError:
        # Already set, ignore
        pass
    
    # Log config
    log.info(f"check device: {torch.cuda.is_available()}")
    log.info(f"config: \n{OmegaConf.to_yaml(cfg)}")
    
    # Set random seed for reproducibility
    set_seed(cfg.experiment.seed)
    log.info(f"Set random seed to {cfg.experiment.seed} for reproducibility")
    # Device selection with fallback: CUDA -> MPS -> CPU
    requested_device = cfg.model.device
    if requested_device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            log.warning("CUDA requested but not available. Falling back to MPS (macOS GPU).")
            device = torch.device("mps")
            cfg.model.device = "mps"
        else:
            log.warning("CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
            cfg.model.device = "cpu"
    elif requested_device == "mps":
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            log.warning("MPS requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
            cfg.model.device = "cpu"
    else:
        device = torch.device(requested_device)
    log.info(f"Using device: {device}")
    
    # Get classes from top-level config
    classes = cfg.get('classes', None)
    
    # If classes not specified, discover from training dataset
    if classes is None:
        log.info("Classes not specified in config, auto-discovering from training dataset...")
        temp_dataset = ViamDataset(
            jsonl_path=cfg.dataset.data.train_jsonl,
            data_dir=cfg.dataset.data.train_data_dir,
            classes=None,  # Will trigger auto-discovery
        )
        # Get discovered classes from dataset
        classes = sorted(temp_dataset.label_to_id.keys())
        log.info(f"Auto-discovered {len(classes)} classes: {classes}")
    
    # Set model.num_classes based on classes
    num_classes = len(classes)
    if cfg.model.num_classes != num_classes:
        log.info(f"Setting model.num_classes to {num_classes} (from {len(classes)} classes)")
        cfg.model.num_classes = num_classes
    
    # Create model with correct num_classes
    if cfg.model.name == "custom_detector":
        model = SimpleDetector(cfg).to(device)
        log.info("Simple detector model created and moved to device")
    elif cfg.model.name == "faster_rcnn":
        model = FasterRCNNDetector(cfg).to(device)
        log.info("faster rcnn model created and moved to device")
    elif cfg.model.name == "effnet":
        model = EfficientNetDetector(cfg).to(device)
        log.info("efficientnet model created and moved to device")
    elif cfg.model.name == "ssdlite":
        model = SSDLiteDetector(cfg).to(device)
        log.info("ssdlite model created and moved to device")
        summary(model, (32, 1, 320, 320))
    else:
        raise ValueError(f"Unknown model type: {cfg.model.name}")
   
    # Configure transfer learning strategy (freeze/unfreeze layers)
    # Only applies when using pretrained weights
    if cfg.model.get('pretrained', False):
        freeze_config = {
            'freeze_backbone': cfg.training.get('freeze_backbone', False),
            'freeze_fpn': cfg.training.get('freeze_fpn', False),
            'freeze_rpn': cfg.training.get('freeze_rpn', False),
            'freeze_all': cfg.training.get('freeze_all', False),
        }
        
        if any(freeze_config.values()):
            log.info("=" * 60)
            log.info("TRANSFER LEARNING: Configuring layer freezing")
            log.info("=" * 60)
            configure_model_for_transfer_learning(
                model, 
                cfg.model.name, 
                freeze_config
            )
            log.info("=" * 60)
        else:
            log.info("Transfer learning: Full fine-tuning (all layers trainable)")
    else:
        log.info("Training from scratch (no pretrained weights)")
    
    # Create Exponential Moving Average (EMA) of model weights
    # EMA provides more stable models and typically improves accuracy by 0.5-1.0 mAP
    model_ema = None
    if cfg.training.get('use_ema', True):  # Enable by default
        ema_decay = cfg.training.get('ema_decay', 0.9998)
        model_ema = ModelEMA(model, decay=ema_decay, device=device)
        log.info(f"Created Model EMA with decay={ema_decay}")

    train_transform = build_transforms(cfg, is_train=True, test=False) #investigate whether each image can be transformed differently
    val_transform = build_transforms(cfg, is_train=False, test=False)    

    # Create datasets with classes from config
    train_dataset = ViamDataset(
        jsonl_path=cfg.dataset.data.train_jsonl,
        data_dir=cfg.dataset.data.train_data_dir,
        classes=classes,
    )

    val_dataset = ViamDataset(
        jsonl_path=cfg.dataset.data.val_jsonl,
        data_dir=cfg.dataset.data.val_data_dir,
        classes=classes,
    )
    
    # Create dataloaders
    # MPS doesn't support multiprocessing, so set num_workers=0
    # Also disable pin_memory for MPS (only useful for CUDA)
    num_workers = 0 if device.type == 'mps' else cfg.training.num_workers
    pin_memory = cfg.training.pin_memory and device.type == 'cuda'
    
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
    # PyTorch separates normalization layers (BatchNorm, LayerNorm) from other parameters
    # for differential weight decay, NOT backbone vs other layers
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
    
    # train model and get best validation loss
    best_val_loss = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        cfg=cfg,
        model_ema=model_ema
    )
    gc.collect() #to avoid CUDA out of memory error on optuna
    torch.cuda.empty_cache()
    # best validation loss for Optuna 
    return best_val_loss

if __name__ == "__main__":
    main() 

