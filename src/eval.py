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
from omegaconf import DictConfig
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.viam_dataset import ViamDataset
from models.custom_detector import SimpleDetector
from models.effnet_detector import EfficientNetDetector
from models.faster_rcnn_detector import FasterRCNNDetector
from models.ssdlite_detector import SSDLiteDetector
from utils.coco_converter import jsonl_to_coco
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

def evaluate_model(model, data_loader, cfg: DictConfig):
    """
    Evaluate model on test set and compute COCO metrics
    """
    model.eval()
    results = []
    total_predictions = 0
    total_boxes = 0
    
    # Track confidence score statistics
    all_scores = []
    boxes_above_threshold = 0
    boxes_below_threshold = 0

    num_images_to_visualize = 7 
    total_images = len(data_loader.dataset)
    images_to_plot = set(random.sample(range(total_images), min(num_images_to_visualize, total_images)))
    # save in visualizations directory
    vis_dir = Path(cfg.logging.save_dir) / "visualizations"
    
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(tqdm(data_loader)):
            model_output = model(data)
            
            # Convert model output to list of predictions per image
            if cfg.model.name in ["faster_rcnn", "ssdlite"]:
                predictions = model_output  # Already a list
            elif cfg.model.name in ["effnet", "custom_detector"]:
                # Model returns dict with batch predictions, need to split by image
                predictions = []
                batch_size = model_output['boxes'].shape[0]
                for i in range(batch_size):
                    predictions.append({
                        'boxes': model_output['boxes'][i:i+1],  # Keep as 2D tensor
                        'scores': model_output['scores'][i],
                        'labels': model_output['labels'][i:i+1] if 'labels' in model_output else torch.ones(1, dtype=torch.int64)
                    })
            else:
                raise ValueError(f"Unknown model name: {cfg.model.name}")
        
            # Visualize random sample of images
            for i in range(len(data)):
                global_image_idx = batch_idx * cfg.training.batch_size + i
                if global_image_idx in images_to_plot:
                    visualize_predictions(
                        data[i], 
                        predictions[i],
                        targets[i],
                        cfg=cfg,
                        output_dir=vis_dir, 
                        title=f"Image {targets[i]['image_id']}",
                    )
                    images_to_plot.remove(global_image_idx)  # avoid duplicates
                        
            for pred, target in zip(predictions, targets):
                image_id = target['image_id'].item()
                boxes = pred['boxes']
                scores = pred['scores']
                total_predictions += 1
                total_boxes += len(boxes)
                
                # statistics about confidence scores for prediction boxes
                if len(scores) > 0:
                    all_scores.extend(scores.cpu().numpy())
                    boxes_above_threshold += (scores > cfg.evaluation.confidence_threshold).sum().item()
                    boxes_below_threshold += (scores <= cfg.evaluation.confidence_threshold).sum().item()
    
                if len(boxes) > 0:
                    # Only include boxes with confidence > threshold
                    mask = scores > cfg.evaluation.confidence_threshold
                    boxes = boxes[mask]
                    scores = scores[mask]
                    
                    if len(boxes) > 0:  # Check if any boxes remain after filtering
                        # convert from [x1,y1,x2,y2] to COCO format [x,y,w,h]
                        boxes_coco = torch.zeros_like(boxes)
                        boxes_coco[:, 0] = boxes[:, 0]  # x
                        boxes_coco[:, 1] = boxes[:, 1]  # y
                        boxes_coco[:, 2] = boxes[:, 2] - boxes[:, 0]  # w
                        boxes_coco[:, 3] = boxes[:, 3] - boxes[:, 1]  # h
                        
                        # Add all detections for this image
                        # Get category_id from predictions (labels tensor)
                        pred_labels = pred.get('labels', torch.ones(len(boxes_coco), dtype=torch.int64))
                        if len(pred_labels) != len(boxes_coco):
                            # If labels don't match boxes, use default category_id=1
                            pred_labels = torch.ones(len(boxes_coco), dtype=torch.int64)
                        
                        results.extend([
                            {
                                'image_id': image_id,
                                'category_id': label.item() if isinstance(label, torch.Tensor) else label,
                                'bbox': box.tolist(),
                                'score': score.item()
                            }
                            for box, score, label in zip(boxes_coco, scores, pred_labels)
                        ])
    
    # Log confidence score statistics
    all_scores = np.array(all_scores)
    log.info(f"Total boxes detected: {total_boxes}")
    log.info(f"Boxes with confidence > {cfg.evaluation.confidence_threshold}: {boxes_above_threshold} ({boxes_above_threshold/total_boxes*100:.1f}%)")
    log.info(f"Boxes with confidence <= {cfg.evaluation.confidence_threshold}: {boxes_below_threshold} ({boxes_below_threshold/total_boxes*100:.1f}%)")
    
    return results

@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    base_dir = Path(get_original_cwd())
    mp.set_start_method('spawn', force=True)
    
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
    
    #Create model
    if cfg.model.name == "custom_detector":
        model = SimpleDetector(cfg).to(device)
    elif cfg.model.name == "faster_rcnn":
        model = FasterRCNNDetector(cfg).to(device)
    elif cfg.model.name == "effnet":
        model = EfficientNetDetector(cfg).to(device)
    elif cfg.model.name == "ssdlite":
        model = SSDLiteDetector(cfg).to(device)
    else:
        raise ValueError(f"Unknown model type: {cfg.model.name}")
 
    # Create test dataset with classes from config
    test_dataset = ViamDataset(
        jsonl_path=cfg.dataset.data.test_jsonl,
        data_dir=cfg.dataset.data.test_data_dir,
        classes=classes,
    )
 
    # Checkpoint path - update this to your trained model
    # Example: outputs/2026-01-30/14-25-30/best_model.pth
    checkpoint_path = cfg.get('checkpoint_path', 'checkpoints/best_model.pth')
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Please specify checkpoint path with: checkpoint_path=<path/to/model.pth>"
        )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_transform = build_transforms(cfg, is_train=False, test=True)

    # MPS doesn't support multiprocessing, so set num_workers=0
    # Also disable pin_memory for MPS (only useful for CUDA)
    num_workers = 0 if device.type == 'mps' else cfg.training.num_workers
    pin_memory = cfg.training.pin_memory and device.type == 'cuda'
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=GPUCollate(device, test_transform)
    )

    results = evaluate_model(model, test_loader, cfg)

    # saving predictions
    output_dir = Path(cfg.logging.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_file = output_dir / f"{cfg.model.name}_predictions.json"
    with open(predictions_file, "w") as f:
        json.dump(results, f)

    # COCO metrics
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
    
    with open(predictions_file, 'r') as f:
        pred_data = json.load(f)
    log.info(f"no of predictions: {len(pred_data)}")
    
    # Load COCO format ground truth
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    log.info(f"ground truth images: {len(gt_data.get('images', []))}")
    log.info(f"ground truth annotations: {len(gt_data.get('annotations', []))}")

    # Check if there are any matching image IDs
    pred_img_ids = set(p['image_id'] for p in pred_data)
    gt_img_ids = set(ann['image_id'] for ann in gt_data['annotations'])
    matching_ids = pred_img_ids.intersection(gt_img_ids)
    log.info(f"matching image IDs: {len(matching_ids)}")

    coco_gt = COCO(str(gt_path))
    
    # Check if there are any predictions before loading
    if len(results) == 0:
        log.warning("No predictions with confidence above threshold. Skipping COCO evaluation.")
        log.warning(f"Consider lowering the confidence threshold (current: {cfg.evaluation.confidence_threshold})")
        log.warning("Or check if the model is properly trained and outputting reasonable predictions.")
        return
    
    coco_dt = coco_gt.loadRes(str(predictions_file))
    
    # Get all category IDs from ground truth (excluding background 0)
    cat_ids = coco_gt.getCatIds()
    log.info(f"Evaluating {len(cat_ids)} categories: {cat_ids}")
    
    coco_eval = COCOeval(cocoGt=coco_gt, cocoDt=coco_dt)
    coco_eval.params.catIds = cat_ids  # Evaluate all categories
    coco_eval.params.iouType = 'bbox'
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Save metrics
    metrics = { #AP = average precision
        'AP': coco_eval.stats[0],  # AP at IoU=0.50:0.95
        'AP50': coco_eval.stats[1],  # AP at IoU=0.50
        'AP75': coco_eval.stats[2],  # AP at IoU=0.75
        'APs': coco_eval.stats[3],   # AP for small objects
        'APm': coco_eval.stats[4],   # AP for medium objects
        'APl': coco_eval.stats[5],   # AP for large objects
    }
    
    metrics_file = output_dir / f"{cfg.model.name}_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
        json.dump(checkpoint_path, f, indent=4)
    
    log.info(f"AP50: {metrics['AP50']:.3f}")

if __name__ == "__main__":
    main()

