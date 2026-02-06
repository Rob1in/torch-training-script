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

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

log = logging.getLogger(__name__)


class ONNXModelWrapper:
    """Wrapper to make ONNX model behave like PyTorch model for evaluation.
    
    Expects float32 input in [0, 1] range from the dataloader (no normalization).
    Converts to uint8 [0, 255] for the ONNX model which handles normalization internally.
    
    Note: ONNX models are typically exported with batch_size=1, so we process
    images one at a time to handle variable batch sizes from the DataLoader.
    """
    
    def __init__(self, onnx_path: str, device: str = 'cpu'):
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime not installed. Install with: pip install onnxruntime")
        
        # Set execution providers based on device
        if device == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.device = device
        
        # Get input shape info
        input_info = self.session.get_inputs()[0]
        self.input_name = input_info.name
        input_shape = input_info.shape
        
        log.info(f"Loaded ONNX model from: {onnx_path}")
        log.info(f"ONNX Runtime providers: {self.session.get_providers()}")
        log.info(f"ONNX input name: {self.input_name}, shape: {input_shape}")
        log.info("ONNX model expects uint8 input [0, 255] (normalization handled by model's GeneralizedRCNNTransform)")
        
        # Check if model expects fixed batch size
        if len(input_shape) > 0 and isinstance(input_shape[0], int) and input_shape[0] > 0:
            self.expected_batch_size = input_shape[0]
            log.info(f"ONNX model expects fixed batch size: {self.expected_batch_size}")
        else:
            self.expected_batch_size = None
            log.info("ONNX model supports variable batch size")
    
    def eval(self):
        """No-op for compatibility with PyTorch model interface."""
        return self
    
    def __call__(self, images):
        """
        Run inference on images.
        
        Args:
            images: List of torch tensors [C, H, W] OR batched tensor [B, C, H, W] in range [0, 1]
        
        Returns:
            List of dicts with keys: 'boxes', 'scores', 'labels'
        """
        # Handle both list and batch tensor inputs
        if isinstance(images, list):
            image_list = images
        else:
            # Convert batched tensor to list
            image_list = [images[i] for i in range(images.shape[0])]
        
        # Process each image individually (ONNX model expects batch_size=1)
        # This handles the case where DataLoader provides batches > 1
        results = []
        for img in image_list:
            # Ensure image is [1, C, H, W] for ONNX
            if img.dim() == 3:
                img_batch = img.unsqueeze(0)  # [1, C, H, W]
            else:
                img_batch = img
            
            # Dataloader provides float32 [0, 1] (no normalization).
            # Convert to uint8 [0, 255] for ONNX model.
            # The model's built-in GeneralizedRCNNTransform handles normalization.
            img_np = (img_batch.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            
            # Run ONNX inference (expects batch_size=1)
            outputs = self.session.run(None, {self.input_name: img_np})
            boxes, labels, scores = outputs
            
            # Convert outputs to tensors
            # boxes: [1, N, 4] -> [N, 4]
            # labels: [1, N] -> [N]
            # scores: [1, N] -> [N]
            if boxes.ndim == 3 and boxes.shape[0] == 1:
                boxes = boxes[0]  # [N, 4]
            if labels.ndim == 2 and labels.shape[0] == 1:
                labels = labels[0]  # [N]
            if scores.ndim == 2 and scores.shape[0] == 1:
                scores = scores[0]  # [N]
            
            # Filter out invalid detections (scores > 0)
            valid_mask = scores > 0
            
            boxes_tensor = torch.from_numpy(boxes[valid_mask]).float()
            scores_tensor = torch.from_numpy(scores[valid_mask]).float()
            labels_tensor = torch.from_numpy(labels[valid_mask]).float()
            
            results.append({
                'boxes': boxes_tensor,
                'scores': scores_tensor,
                'labels': labels_tensor
            })
        
        return results


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

@hydra.main(config_path="../configs", config_name="eval", version_base=None)
def main(cfg: DictConfig):
    base_dir = Path(get_original_cwd())
    mp.set_start_method('spawn', force=True)
    
    # ============================================================
    # Handle dataset_dir (required, passed via Hydra override)
    # ============================================================
    dataset_dir_str = cfg.get('dataset_dir', None)
    if dataset_dir_str is None:
        raise ValueError(
            "dataset_dir is required. "
            "Usage: python src/eval.py dataset_dir=<dir> [other args]\n"
            "Example: python src/eval.py dataset_dir=triangles_dataset_small run_dir=outputs/18-08-48"
        )
    
    dataset_dir = Path(dataset_dir_str)
    if not dataset_dir.is_absolute():
        dataset_dir = base_dir / dataset_dir
    
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    dataset_jsonl = dataset_dir / "dataset.jsonl"
    dataset_data_dir = dataset_dir / "data"
    
    if not dataset_jsonl.exists():
        raise FileNotFoundError(f"dataset.jsonl not found in {dataset_dir}")
    if not dataset_data_dir.exists():
        raise FileNotFoundError(f"data/ directory not found in {dataset_dir}")
    
    dataset_jsonl = dataset_jsonl.resolve()
    dataset_data_dir = dataset_data_dir.resolve()
    
    # ============================================================
    # Load training config from run_dir (required)
    # Merge order: training config (base) + eval config (overrides) + CLI overrides
    # ============================================================
    if 'run_dir' not in cfg or not cfg.run_dir:
        raise ValueError(
            "run_dir is required. "
            "Usage: python src/eval.py dataset_dir=<dir> run_dir=outputs/18-08-48\n"
            "The run_dir must contain a .hydra/config.yaml file from training."
        )
    
    # run_dir is required, so we can proceed
    run_dir = Path(cfg.run_dir)
    if not run_dir.is_absolute():
        run_dir = base_dir / run_dir
    
    training_config_path = run_dir / ".hydra" / "config.yaml"
    if not training_config_path.exists():
        raise FileNotFoundError(
            f"Training config not found: {training_config_path}\n"
            f"Make sure the run_dir contains a .hydra folder from training."
        )
    
    log.info(f"Loading training config from: {training_config_path}")
    training_cfg = OmegaConf.load(training_config_path)
    OmegaConf.set_struct(training_cfg, False)
    
    # Store eval-specific and CLI overrides before replacing config
    # We want to preserve: evaluation, dataset_dir, run_dir, checkpoint_path, and any CLI overrides
    eval_specific = {
        'evaluation': cfg.get('evaluation', {}),
        'dataset_dir': cfg.get('dataset_dir'),
        'run_dir': cfg.get('run_dir'),
        'checkpoint_path': cfg.get('checkpoint_path'),
    }
    
    # Start with training config as base (preserves model architecture, classes, etc.)
    cfg = training_cfg
    OmegaConf.set_struct(cfg, False)
    
    # Apply eval-specific settings and CLI overrides on top
    if eval_specific.get('evaluation'):
        cfg.evaluation = eval_specific['evaluation']
    if eval_specific.get('dataset_dir'):
        cfg.dataset_dir = eval_specific['dataset_dir']
    if eval_specific.get('run_dir'):
        cfg.run_dir = eval_specific['run_dir']
    if eval_specific.get('checkpoint_path'):
        cfg.checkpoint_path = eval_specific['checkpoint_path']
    
    # Auto-set checkpoint_path if not provided
    if 'checkpoint_path' not in cfg or cfg.checkpoint_path is None:
        checkpoint_path = run_dir / "best_model.pth"
        if checkpoint_path.exists():
            cfg.checkpoint_path = str(checkpoint_path)
            log.info(f"Auto-detected checkpoint: {checkpoint_path}")
    
    log.info(f"Loaded training config from run: {run_dir.name}")
    
    # Set dataset paths (always override with dataset_dir)
    cfg.dataset.data.test_jsonl = str(dataset_jsonl)
    cfg.dataset.data.test_data_dir = str(dataset_data_dir)
    
    log.info(f"Using dataset from dataset_dir: {dataset_dir}")
    log.info(f"  Model: {cfg.model.name}")
    log.info(f"  Classes: {cfg.get('classes', 'auto-discover')}")
    log.info(f"  Test dataset: {cfg.dataset.data.test_jsonl}")
    
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
    log.info(f"  Model config: pretrained={cfg.model.get('pretrained', False)}, num_classes={cfg.model.num_classes}")
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
 
    # Checkpoint path - auto-detect from run_dir if not provided
    checkpoint_path = cfg.get('checkpoint_path', None)
    if checkpoint_path is None:
        # Auto-detect from run_dir (which is required)
        run_dir_path = Path(cfg.run_dir)
        if not run_dir_path.is_absolute():
            run_dir_path = base_dir / run_dir_path
        
        checkpoint_path = run_dir_path / "best_model.pth"
        if checkpoint_path.exists():
            cfg.checkpoint_path = str(checkpoint_path)
            log.info(f"Auto-detected checkpoint: {checkpoint_path}")
        else:
            raise FileNotFoundError(
                f"No checkpoint found. Expected: {checkpoint_path}\n"
                f"Provide checkpoint_path explicitly: checkpoint_path=path/to/model.pth"
            )
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = base_dir / checkpoint_path
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Check if ONNX model
    is_onnx = checkpoint_path.suffix == '.onnx'
    
    # Set output directory to run_dir/eval_<dataset_name>_<checkpoint_name>_<format>/
    # run_dir is required, so we always set this
    run_dir_path = Path(cfg.run_dir)
    if not run_dir_path.is_absolute():
        run_dir_path = base_dir / run_dir_path
    
    # Extract dataset name from dataset_dir
    dataset_name = dataset_dir.name
    
    # Get checkpoint filename (without extension)
    checkpoint_name = checkpoint_path.stem
    
    # Determine format from checkpoint extension (pth or onnx)
    checkpoint_format = checkpoint_path.suffix[1:] if checkpoint_path.suffix else "pth"  # Remove leading dot
    
    # Set output directory to run_dir/eval_<dataset_name>_<checkpoint_name>_<format>/
    eval_output_dir = run_dir_path / f"eval_{dataset_name}_{checkpoint_name}_{checkpoint_format}"
    OmegaConf.set_struct(cfg, False)
    cfg.logging.save_dir = str(eval_output_dir)
    log.info(f"Evaluation outputs will be saved to: {eval_output_dir}")
    
    if is_onnx:
        log.info("="*80)
        log.info("ONNX MODEL EVALUATION")
        log.info("="*80)
        # Use ONNX wrapper instead of loading PyTorch weights
        model = ONNXModelWrapper(str(checkpoint_path), device=cfg.model.device)
        log.info(f"Loaded ONNX model: {checkpoint_path}")
    else:
        # Load PyTorch checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Prefer EMA weights if available (better for evaluation)
        if 'model_ema_state_dict' in checkpoint:
            log.info("Loading Model EMA weights for evaluation")
            model.load_state_dict(checkpoint['model_ema_state_dict'])
        else:
            log.info("Loading standard model weights for evaluation")
            model.load_state_dict(checkpoint['model_state_dict'])

    # Check if model has a built-in transform (normalization + resize).
    # All torchvision detection models (Faster R-CNN, SSDLite) have GeneralizedRCNNTransform.
    has_builtin_transform = (
        hasattr(model, 'model') and hasattr(model.model, 'transform')
    ) or isinstance(model, ONNXModelWrapper)
    
    if has_builtin_transform:
        # Warn if Normalize is in the transforms — it would cause double normalization
        test_transforms_cfg = cfg.dataset.transform.test
        if test_transforms_cfg and any(t.get('name') == 'Normalize' for t in test_transforms_cfg):
            log.warning("=" * 80)
            log.warning("⚠️  Normalize found in dataset transforms, but model has a built-in transform!")
            log.warning("    This will cause DOUBLE NORMALIZATION and broken results.")
            log.warning("    Remove Normalize from configs/dataset/jsonl.yaml")
            log.warning("=" * 80)
    else:
        log.warning("Model does NOT have a built-in transform.")
        log.warning("Make sure your dataset config includes Normalize in the transforms!")

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
    metrics['is_onnx'] = is_onnx
    metrics['dataset'] = {
        'jsonl': str(cfg.dataset.data.test_jsonl),
        'data_dir': str(cfg.dataset.data.test_data_dir)
    }
    
    # Save metrics with appropriate filename
    model_type = "onnx" if is_onnx else cfg.model.name
    metrics_file = output_dir / f"{model_type}_metrics.json"
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
    # Hydra handles CLI overrides automatically - no manual code needed!
    # Just call main() and Hydra will process all CLI arguments
    main()

