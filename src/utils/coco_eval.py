"""
COCO evaluation utilities for object detection.
Shared between training and evaluation scripts.
"""
import io
import logging
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, List, Optional

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

log = logging.getLogger(__name__)


def convert_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from [x1, y1, x2, y2] to [x, y, w, h] format.
    
    Args:
        boxes: Tensor of shape [N, 4] in [x1, y1, x2, y2] format
        
    Returns:
        Tensor of shape [N, 4] in [x, y, w, h] format
    """
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def collect_predictions(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    scale_to_original: bool = True,
) -> List[Dict]:
    """
    Collect predictions from model on a dataset.
    
    Args:
        model: The detection model
        data_loader: DataLoader for the dataset
        device: Device to run inference on
        scale_to_original: Whether to scale boxes back to original image coordinates
        
    Returns:
        List of prediction dictionaries in COCO format
    """
    model.eval()
    coco_results = []
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc='Collecting Predictions'):
            # Run inference (no targets passed to model)
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
                
                # Scale boxes back to original image dimensions if requested
                if scale_to_original and 'orig_size' in target:
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
                
                # Add all detections for this image (no confidence filtering!)
                # COCO evaluation handles confidence thresholding internally
                for box, score, label in zip(boxes, scores, labels):
                    coco_results.append({
                        'image_id': image_id,
                        'category_id': label.item(),
                        'bbox': box.tolist(),
                        'score': score.item(),
                    })
    
    return coco_results


def evaluate_coco_predictions(
    predictions: List[Dict],
    coco_gt: COCO,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Evaluate predictions using COCO metrics.
    
    Args:
        predictions: List of prediction dictionaries
        coco_gt: COCO ground truth object
        verbose: Whether to print detailed statistics
        
    Returns:
        Dictionary of COCO metrics
    """
    # Debug information about predictions
    if verbose:
        log.info(f"Total predictions collected: {len(predictions)}")
    
    if len(predictions) == 0:
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
    if verbose:
        pred_image_ids = set(r['image_id'] for r in predictions)
        pred_cat_ids = set(r['category_id'] for r in predictions)
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
        scores = [r['score'] for r in predictions]
        log.info(f"Score range: min={min(scores):.4f}, max={max(scores):.4f}, mean={sum(scores)/len(scores):.4f}")
    
    # Run COCO evaluation
    with redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(predictions)
    
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    
    # Print summary (suppress output)
    if verbose:
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


def evaluate_coco(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    coco_gt: COCO,
    scale_to_original: bool = True,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Complete COCO evaluation pipeline.
    
    Args:
        model: The detection model
        data_loader: DataLoader for the dataset
        device: Device to run inference on
        coco_gt: COCO ground truth object
        scale_to_original: Whether to scale boxes back to original coordinates
        verbose: Whether to print detailed statistics
        
    Returns:
        Dictionary of COCO metrics
    """
    # Collect predictions
    predictions = collect_predictions(model, data_loader, device, scale_to_original)
    
    # Evaluate predictions
    metrics = evaluate_coco_predictions(predictions, coco_gt, verbose)
    
    return metrics
