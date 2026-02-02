#training script for object detection models
import gc
import io
import logging
import math
import sys
from contextlib import redirect_stdout
from pathlib import Path

import hydra
import torch
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets.viam_dataset import ViamDataset
from models.faster_rcnn_detector import FasterRCNNDetector
from models.ssdlite_detector import SSDLiteDetector
from utils.coco_converter import jsonl_to_coco
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
    train_losses = {}  # Dynamic loss tracking - will auto-populate based on model's loss keys
    
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
        
        # Accumulate losses for logging (dynamic keys for different model architectures)
        train_loss += loss_value
        for k, v in loss_dict.items():
            if k not in train_losses:
                train_losses[k] = 0.0
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


def evaluate_loss(model, data_loader, device, epoch, cfg):
    """
    Evaluate validation loss. Keeps model in train mode but freezes BatchNorm/Dropout.
    This allows computing loss during validation (detection models don't compute loss in eval mode).
    
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
    
    # Keep model in train mode to get loss dict, but freeze BatchNorm/Dropout
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


def convert_to_xywh(boxes):
    """Convert boxes from [x1, y1, x2, y2] to [x, y, w, h] format."""
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def evaluate_coco(model, data_loader, device, coco_gt):
    """
    Evaluate using COCO metrics (mAP). Follows torchvision reference implementation.
    
    Args:
        model: The model to evaluate
        data_loader: Validation data loader
        device: Device to evaluate on
        coco_gt: COCO ground truth object
        
    Returns:
        Dictionary with COCO metrics (AP, AP50, AP75, etc.)
    """
    model.eval()
    
    # Collect predictions
    coco_results = []
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc='COCO Eval'):
            # Run inference (no targets passed)
            outputs = model(images)
            
            # Process each image's predictions
            for img_idx, (target, output) in enumerate(zip(targets, outputs)):
                image_id = target['image_id'].item()
                
                if len(output['boxes']) == 0:
                    continue
                
                # Get boxes (currently in transformed image coordinates)
                boxes = output['boxes'].cpu()
                scores = output['scores'].cpu()
                labels = output['labels'].cpu()
                
                # Scale boxes back to original image dimensions
                # Model returns boxes in the coordinate system of images passed to it
                # But COCO ground truth is in original image coordinates
                if 'orig_size' in target:
                    orig_size = target['orig_size']
                    orig_h = orig_size[0].item() if torch.is_tensor(orig_size[0]) else orig_size[0]
                    orig_w = orig_size[1].item() if torch.is_tensor(orig_size[1]) else orig_size[1]
                    
                    # Current image dimensions (after dataset transform)
                    curr_h, curr_w = images[img_idx].shape[-2:]
                    
                    # Scale boxes to original coordinates
                    scale_h = orig_h / curr_h
                    scale_w = orig_w / curr_w
                    boxes = boxes.clone()  # Don't modify original
                    boxes[:, [0, 2]] *= scale_w  # x coordinates
                    boxes[:, [1, 3]] *= scale_h  # y coordinates
                
                # Convert boxes from [x1,y1,x2,y2] to COCO format [x,y,w,h]
                boxes = convert_to_xywh(boxes)
                
                # Add all detections for this image
                for box, score, label in zip(boxes, scores, labels):
                    coco_results.append({
                        'image_id': image_id,
                        'category_id': label.item(),
                        'bbox': box.tolist(),
                        'score': score.item(),
                    })
    
    # Debug information about predictions
    log.info(f"Total predictions collected: {len(coco_results)}")
    log.info("Note: Predictions scaled back to original image coordinates for COCO eval")
    
    if len(coco_results) == 0:
        log.warning("No predictions made during COCO evaluation!")
        return {
            'AP': 0.0,
            'AP50': 0.0,
            'AP75': 0.0,
            'APs': 0.0,
            'APm': 0.0,
            'APl': 0.0,
            'AR1': 0.0,
            'AR10': 0.0,
            'AR100': 0.0,
            'ARs': 0.0,
            'ARm': 0.0,
            'ARl': 0.0,
        }
    
    # Debug: Check prediction details
    pred_image_ids = set(r['image_id'] for r in coco_results)
    pred_cat_ids = set(r['category_id'] for r in coco_results)
    gt_image_ids = set(coco_gt.imgs.keys())
    gt_cat_ids = set(coco_gt.cats.keys())
    
    log.info(f"Prediction image IDs: {len(pred_image_ids)} unique ({list(pred_image_ids)[:5]}...)")
    log.info(f"Ground truth image IDs: {len(gt_image_ids)} unique ({list(gt_image_ids)[:5]}...)")
    log.info(f"Prediction category IDs: {pred_cat_ids}")
    log.info(f"Ground truth category IDs: {gt_cat_ids}")
    
    matching_img_ids = pred_image_ids.intersection(gt_image_ids)
    matching_cat_ids = pred_cat_ids.intersection(gt_cat_ids)
    log.info(f"Matching image IDs: {len(matching_img_ids)}")
    log.info(f"Matching category IDs: {matching_cat_ids}")
    
    # Sample some predictions
    scores = [r['score'] for r in coco_results]
    log.info(f"Score range: min={min(scores):.4f}, max={max(scores):.4f}, mean={sum(scores)/len(scores):.4f}")
    
    # Run COCO evaluation
    with redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(coco_results)
    
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    
    # Print summary (suppress output)
    log.info("COCO Evaluation Results:")
    with redirect_stdout(io.StringIO()):
        coco_eval.summarize()
    
    # Return metrics as dict
    return {
        'AP': coco_eval.stats[0],      # AP @ IoU=0.50:0.95
        'AP50': coco_eval.stats[1],    # AP @ IoU=0.50
        'AP75': coco_eval.stats[2],    # AP @ IoU=0.75
        'APs': coco_eval.stats[3],     # AP for small objects
        'APm': coco_eval.stats[4],     # AP for medium objects
        'APl': coco_eval.stats[5],     # AP for large objects
        'AR1': coco_eval.stats[6],     # AR with 1 detection per image
        'AR10': coco_eval.stats[7],    # AR with 10 detections per image
        'AR100': coco_eval.stats[8],   # AR with 100 detections per image
        'ARs': coco_eval.stats[9],     # AR for small objects
        'ARm': coco_eval.stats[10],    # AR for medium objects
        'ARl': coco_eval.stats[11],    # AR for large objects
    }


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
    
    # Build transforms - always use config transforms
    train_transform = DetectionTransform(cfg.dataset.transform.train) if cfg.dataset.transform.train else None
    val_transform = DetectionTransform(cfg.dataset.transform.val) if cfg.dataset.transform.val else None
    
    # Info about resize behavior
    if cfg.model.get('pretrained', False):
        log.info("=" * 80)
        log.info(f"PRETRAINED MODEL: Using {cfg.model.name} with COCO weights")
        log.info(f"Config resize size: {cfg.model.transform.input_size}")
        log.info("Both dataset and model will resize to this size (redundant but safe)")
        log.info("=" * 80)
    
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
    
    # Prepare COCO ground truth for validation evaluation
    log.info("Preparing COCO ground truth for validation...")
    val_coco_path = Path(cfg.logging.save_dir) / 'val_ground_truth_coco.json'
    jsonl_to_coco(
        jsonl_path=cfg.dataset.data.val_jsonl,
        data_dir=cfg.dataset.data.val_data_dir,
        output_path=val_coco_path,
        classes=classes,
    )
    coco_gt = COCO(str(val_coco_path))
    log.info(f"COCO ground truth created: {len(coco_gt.imgs)} images, {len(coco_gt.anns)} annotations")
    
    # Training loop
    writer = SummaryWriter(Path(cfg.logging.save_dir) / 'tensorboard')
    best_map = 0.0  # Track best mAP (AP@0.50:0.95)
    best_map50 = 0.0  # Also track AP50 for reference
    patience_counter = 0
    
    log.info("Starting training...")
    log.info("=" * 80)
    
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
        
        # 1. Compute validation loss (for monitoring)
        val_loss = evaluate_loss(eval_model, val_loader, device, epoch, cfg)
        
        # 2. Compute COCO metrics (for model selection)
        log.info("Running COCO evaluation on validation set...")
        coco_metrics = evaluate_coco(eval_model, val_loader, device, coco_gt)
        
        # Log all metrics to TensorBoard
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        for loss_name, loss_value in train_metrics['loss_dict'].items():
            writer.add_scalar(f'EpochLoss/{loss_name}', loss_value, epoch)
        
        # Log COCO metrics
        writer.add_scalar('COCO/AP', coco_metrics['AP'], epoch)
        writer.add_scalar('COCO/AP50', coco_metrics['AP50'], epoch)
        writer.add_scalar('COCO/AP75', coco_metrics['AP75'], epoch)
        writer.add_scalar('COCO/AR100', coco_metrics['AR100'], epoch)
        
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        log.info(f'Epoch {epoch+1}/{cfg.training.num_epochs}:')
        log.info(f'  Train Loss: {train_metrics["loss"]:.4f} | Val Loss: {val_loss:.4f}')
        log.info(f'  AP: {coco_metrics["AP"]:.4f} | AP50: {coco_metrics["AP50"]:.4f} | AP75: {coco_metrics["AP75"]:.4f}')
        log.info(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # PyTorch reference: Step scheduler per-epoch (after validation)
        scheduler.step()
        
        # Save checkpoint if best mAP (following torchvision's approach of using AP for model selection)
        current_map = coco_metrics['AP']  # AP @ IoU=0.50:0.95
        
        if current_map > best_map:
            best_map = current_map
            best_map50 = coco_metrics['AP50']
            checkpoint_path = Path(cfg.logging.save_dir) / 'best_model.pth'
            
            checkpoint_dict = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'coco_metrics': coco_metrics,
                'best_map': best_map,
                'best_map50': best_map50,
            }
            
            if model_ema is not None:
                checkpoint_dict['model_ema_state_dict'] = model_ema.state_dict()
            
            torch.save(checkpoint_dict, checkpoint_path)
            log.info(f'✓ New best model! AP: {best_map:.4f} (AP50: {best_map50:.4f}) - Saved to {checkpoint_path}')
            patience_counter = 0
        else:
            patience_counter += 1
            log.info(f'  No improvement (best AP: {best_map:.4f}), patience: {patience_counter}/{cfg.training.early_stopping_patience}')
        
        log.info("=" * 80)
        
        # Early stopping
        if patience_counter >= cfg.training.early_stopping_patience:
            log.info(f'Early stopping triggered after {epoch + 1} epochs')
            break
    
    writer.close()
    log.info("Training complete!")
    log.info(f"Best mAP: {best_map:.4f} (AP50: {best_map50:.4f})")
    
    # Cleanup for Optuna
    gc.collect()
    torch.cuda.empty_cache()
    
    return best_map  # Return mAP instead of loss for hyperparameter optimization


if __name__ == "__main__":
    main()
