import json
import logging
import multiprocessing as mp
import random
from pathlib import Path

import hydra
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.viam_dataset import ViamDataset
from models.faster_rcnn_detector import FasterRCNNDetector
from models.ssdlite_detector import SSDLiteDetector
from utils.coco_converter import jsonl_to_coco
from utils.coco_eval import collect_predictions, evaluate_coco_predictions
from utils.transforms import GPUCollate, build_transforms

log = logging.getLogger(__name__)

def visualize_predictions(image,predictions,targets,cfg: DictConfig, title="", output_dir=None):
    # Convert from [C, H, W] to [H, W, C] for RGB display
    # Clamp values to [0, 1] range for proper display
    img_np = image.cpu().numpy().transpose(1, 2, 0)  # RGB image [H, W, C]
    img_np = np.clip(img_np, 0, 1)  # Ensure values are in [0, 1] range
    fig, ax = plt.subplots(1)
    ax.imshow(img_np)
    
    # Plot predicted boxes in red
    if predictions is not None and len(predictions['boxes']) > 0:
        for box, score in zip(predictions['boxes'], predictions['scores']):
            x1, y1, x2, y2 = box.cpu().numpy()
            if score > cfg.evaluation.confidence_threshold:  # Only plot boxes with confidence > threshold
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, 
                                    edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1-5, f'{score:.2f}', color='red')
    
    # Plot ground truth boxes in green
    if targets is not None and targets['boxes'].numel() > 0:
        boxes = targets['boxes'].view(-1, 4)
        for box in boxes:
            x1, y1, x2, y2 = box.cpu().numpy()
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, 
                                  edgecolor='g', facecolor='none')
            ax.add_patch(rect)
    
    plt.title(title)
    plt.axis('off')
    
    # Save figure 
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"{title.replace(' ', '_')}.png"
        plt.savefig(save_path)
    plt.close()  

def evaluate_model(model, data_loader, cfg: DictConfig, device: torch.device):
    """
    Evaluate model on test set, visualize samples, and collect predictions for COCO metrics.
    
    NOTE: This function collects ALL predictions (no confidence filtering) for COCO evaluation.
    The confidence threshold in cfg is only used for visualization.
    """
    model.eval()
    
    # Visualize random samples
    num_images_to_visualize = 7 
    total_images = len(data_loader.dataset)
    images_to_plot = set(random.sample(range(total_images), min(num_images_to_visualize, total_images)))
    vis_dir = Path(cfg.logging.save_dir) / "visualizations"
    
    # Track confidence score statistics for reporting
    all_scores = []
    total_boxes = 0
    
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(tqdm(data_loader, desc="Visualizing")):
            model_output = model(data)
            
            # Visualize random sample of images
            for i in range(len(data)):
                global_image_idx = batch_idx * cfg.training.batch_size + i
                if global_image_idx in images_to_plot:
                    visualize_predictions(
                        data[i], 
                        model_output[i],
                        targets[i],
                        cfg=cfg,
                        output_dir=vis_dir, 
                        title=f"Image {targets[i]['image_id']}",
                    )
                    images_to_plot.remove(global_image_idx)
            
            # Collect confidence statistics
            for pred in model_output:
                if len(pred['scores']) > 0:
                    all_scores.extend(pred['scores'].cpu().numpy())
                    total_boxes += len(pred['scores'])
    
    # Log confidence score statistics
    if total_boxes > 0:
        all_scores = np.array(all_scores)
        boxes_above_threshold = np.sum(all_scores > cfg.evaluation.confidence_threshold)
        boxes_below_threshold = np.sum(all_scores <= cfg.evaluation.confidence_threshold)
        
        log.info(f"Total boxes detected: {total_boxes}")
        log.info(f"Boxes with confidence > {cfg.evaluation.confidence_threshold}: {boxes_above_threshold} ({boxes_above_threshold/total_boxes*100:.1f}%)")
        log.info(f"Boxes with confidence <= {cfg.evaluation.confidence_threshold}: {boxes_below_threshold} ({boxes_below_threshold/total_boxes*100:.1f}%)")
        log.info(f"Score range: min={all_scores.min():.4f}, max={all_scores.max():.4f}, mean={all_scores.mean():.4f}")
    else:
        log.warning("No predictions made!")
    
    # Collect ALL predictions for COCO evaluation (no confidence filtering!)
    # This matches the behavior during training
    log.info("Collecting predictions for COCO evaluation (no confidence threshold applied)...")
    predictions = collect_predictions(
        model=model,
        data_loader=data_loader,
        device=device,
        scale_to_original=True  # Scale back to original image coordinates
    )
    
    return predictions

@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    base_dir = Path(get_original_cwd())
    mp.set_start_method('spawn', force=True)
    
    # ============================================================
    # NEW: Load training config from run_dir if provided
    # ============================================================
    # Usage: python src/eval.py run_dir=outputs/20-15-26
    # This will automatically load the training config and checkpoint
    if 'run_dir' in cfg:
        run_dir = Path(cfg.run_dir)
        if not run_dir.is_absolute():
            run_dir = base_dir / run_dir
        
        # Load the training config from .hydra/config.yaml
        training_config_path = run_dir / ".hydra" / "config.yaml"
        if not training_config_path.exists():
            raise FileNotFoundError(
                f"Training config not found: {training_config_path}\n"
                f"Make sure the run_dir contains a .hydra folder from training."
            )
        
        log.info(f"Loading training config from: {training_config_path}")
        training_cfg = OmegaConf.load(training_config_path)
        
        # Set struct mode to False to allow adding new keys
        OmegaConf.set_struct(training_cfg, False)
        
        # Store CLI overrides before replacing config
        cli_dataset_overrides = OmegaConf.to_container(cfg.get('dataset', {}))
        cli_evaluation_overrides = OmegaConf.to_container(cfg.get('evaluation', {}))
        cli_model_device = cfg.get('model', {}).get('device', None)
        
        # Replace current config with training config
        cfg = training_cfg
        
        # Apply dataset overrides from CLI (if any were provided)
        if cli_dataset_overrides and 'data' in cli_dataset_overrides:
            for key, value in cli_dataset_overrides['data'].items():
                # Only override if it looks like a user-provided path (not placeholder)
                if value and not value.startswith('path/to/'):
                    cfg.dataset.data[key] = value
        
        # Apply evaluation overrides from CLI (if any were provided)
        if cli_evaluation_overrides:
            for key, value in cli_evaluation_overrides.items():
                if key in cfg.evaluation:
                    cfg.evaluation[key] = value
                    log.info(f"Applied CLI override: evaluation.{key}={value}")
        
        # Apply model.device override from CLI (if provided)
        if cli_model_device is not None:
            cfg.model.device = cli_model_device
            log.info(f"Applied CLI override: model.device={cli_model_device}")
        
        # Auto-set checkpoint_path if not explicitly provided
        if 'checkpoint_path' not in cfg or cfg.checkpoint_path is None:
            checkpoint_path = run_dir / "best_model.pth"
            if checkpoint_path.exists():
                OmegaConf.update(cfg, "checkpoint_path", str(checkpoint_path))
                log.info(f"Auto-detected checkpoint: {checkpoint_path}")
            else:
                log.warning(f"No checkpoint found at {checkpoint_path}, you'll need to specify +checkpoint_path=...")
        
        log.info(f"Loaded training config from run: {run_dir.name}")
        log.info(f"  Model: {cfg.model.name}")
        log.info(f"  Classes: {cfg.get('classes', 'auto-discover')}")
        log.info(f"  Test dataset: {cfg.dataset.data.test_jsonl}")
    # ============================================================
    
    # Device selection with fallback: CUDA -> CPU
    requested_device = cfg.model.device
    if requested_device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            log.warning("CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
            cfg.model.device = "cpu"
    else:
        device = torch.device(requested_device)
    log.info(f"Using device: {device}")
    
    # Get classes from top-level config
    classes = cfg.get('classes', None)
    
    # If classes not specified, discover from test dataset
    if classes is None:
        log.info("Classes not specified in config, auto-discovering from test dataset...")
        temp_dataset = ViamDataset(
            jsonl_path=cfg.dataset.data.test_jsonl,
            data_dir=cfg.dataset.data.test_data_dir,
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
    
    # Create model
    log.info(f"Creating model: {cfg.model.name}")
    if cfg.model.name == "faster_rcnn":
        model = FasterRCNNDetector(cfg).to(device)
        log.info("Faster R-CNN model created and moved to device")
    elif cfg.model.name == "ssdlite":
        model = SSDLiteDetector(cfg).to(device)
        log.info("SSDLite model created and moved to device")
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}. Supported models: faster_rcnn, ssdlite")
 
    # Create test dataset with classes from config
    test_dataset = ViamDataset(
        jsonl_path=cfg.dataset.data.test_jsonl,
        data_dir=cfg.dataset.data.test_data_dir,
        classes=classes,
    )
 
    # Checkpoint path
    checkpoint_path = cfg.get('checkpoint_path', None)
    if checkpoint_path is None:
        raise ValueError(
            "No checkpoint specified. Use one of:\n"
            "  1. run_dir=outputs/20-15-26  (loads config and checkpoint automatically)\n"
            "  2. +checkpoint_path=path/to/model.pth  (manual checkpoint path)"
        )
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = base_dir / checkpoint_path
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Prefer EMA weights if available (better for evaluation)
    if 'model_ema_state_dict' in checkpoint:
        log.info("Loading Model EMA weights for evaluation")
        model.load_state_dict(checkpoint['model_ema_state_dict'])
    else:
        log.info("Loading standard model weights for evaluation")
        model.load_state_dict(checkpoint['model_state_dict'])

    test_transform = build_transforms(cfg, is_train=False, test=True)

    # Set dataloader parameters
    num_workers = cfg.training.num_workers
    pin_memory = cfg.training.pin_memory and device.type == 'cuda'
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=GPUCollate(device, test_transform)
    )

    # Evaluate model (visualize samples + collect predictions)
    predictions = evaluate_model(model, test_loader, cfg, device)

    # Create output directory
    output_dir = Path(cfg.logging.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions to JSON
    predictions_file = output_dir / f"{cfg.model.name}_predictions.json"
    with open(predictions_file, "w") as f:
        json.dump(predictions, f)
    log.info(f"Saved {len(predictions)} predictions to {predictions_file}")

    # Convert JSONL to COCO format if needed
    gt_path_str = cfg.dataset.data.get('test_annotations_coco')
    gt_path = Path(gt_path_str) if gt_path_str else None
    
    # If no COCO file specified or it doesn't exist, convert from JSONL
    if not gt_path or not gt_path.exists():
        log.info("No COCO format ground truth found, converting from JSONL...")
        coco_gt_path = output_dir / "ground_truth_coco.json"
        jsonl_to_coco(
            jsonl_path=cfg.dataset.data.test_jsonl,
            data_dir=cfg.dataset.data.test_data_dir,
            output_path=coco_gt_path,
            classes=cfg.get('classes', None),
        )
        gt_path = coco_gt_path
    
    # Load COCO ground truth
    coco_gt = COCO(str(gt_path))
    log.info(f"Loaded ground truth: {len(coco_gt.imgs)} images, {len(coco_gt.anns)} annotations")
    
    # Evaluate using shared COCO evaluation function
    # This matches the evaluation done during training
    log.info("="*80)
    log.info("Running COCO evaluation (same as during training)...")
    log.info("="*80)
    metrics = evaluate_coco_predictions(
        predictions=predictions,
        coco_gt=coco_gt,
        verbose=True
    )
    
    # Add metadata to metrics
    metrics['checkpoint'] = str(checkpoint_path)
    metrics['num_predictions'] = len(predictions)
    
    # Save metrics
    metrics_file = output_dir / f"{cfg.model.name}_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
    log.info(f"Saved metrics to {metrics_file}")
    
    log.info("="*80)
    log.info("Final Results:")
    log.info(f"  AP (IoU=0.50:0.95): {metrics['AP']:.4f}")
    log.info(f"  AP50 (IoU=0.50):    {metrics['AP50']:.4f}")
    log.info(f"  AP75 (IoU=0.75):    {metrics['AP75']:.4f}")
    log.info("="*80)

if __name__ == "__main__":
    main()

