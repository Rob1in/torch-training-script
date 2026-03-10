"""
COCO evaluation utilities for object detection.
Shared between training and evaluation scripts.
"""
import io
import logging
from collections import defaultdict
from contextlib import redirect_stdout
from typing import Dict, List, Optional

import numpy as np
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


def _iou_xywh(box_a: List[float], box_b: List[float]) -> float:
    """Compute IoU between two boxes in COCO [x, y, w, h] format."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax + aw, bx + bw)
    inter_y2 = min(ay + ah, by + bh)

    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    union_area = aw * ah + bw * bh - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def compute_precision_recall(
    predictions: List[Dict],
    coco_gt: COCO,
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.5,
) -> Dict:
    """
    Compute precision, recall, and F1 at a specific IoU and confidence threshold.

    Uses greedy matching: predictions are sorted by score (descending) and each
    is matched to the highest-IoU unmatched ground-truth box (if IoU >= threshold).

    Args:
        predictions: List of COCO-format prediction dicts
                     (keys: image_id, category_id, bbox [x,y,w,h], score).
        coco_gt: COCO ground-truth object.
        iou_threshold: Minimum IoU to count a detection as a true positive.
        confidence_threshold: Minimum score to consider a prediction.

    Returns:
        Dict with 'overall' (precision, recall, f1, tp, fp, fn) and
        'per_class' keyed by category name with the same fields.
    """
    cat_id_to_name = {c['id']: c['name'] for c in coco_gt.dataset['categories']}
    all_cat_ids = set(cat_id_to_name.keys())

    # Filter predictions by confidence
    preds = [p for p in predictions if p['score'] >= confidence_threshold]

    # Group predictions by (image_id, category_id), sorted by score desc
    pred_groups: Dict[tuple, List[Dict]] = defaultdict(list)
    for p in preds:
        pred_groups[(p['image_id'], p['category_id'])].append(p)
    for key in pred_groups:
        pred_groups[key].sort(key=lambda x: x['score'], reverse=True)

    # Group ground-truth annotations by (image_id, category_id)
    gt_groups: Dict[tuple, List[Dict]] = defaultdict(list)
    for ann in coco_gt.dataset['annotations']:
        gt_groups[(ann['image_id'], ann['category_id'])].append(ann)

    # Collect all (image_id, category_id) keys from both predictions and GT
    all_keys = set(pred_groups.keys()) | set(gt_groups.keys())

    per_class_counts: Dict[int, Dict[str, int]] = {
        cid: {'tp': 0, 'fp': 0, 'fn': 0} for cid in all_cat_ids
    }

    for key in all_keys:
        img_id, cat_id = key
        if cat_id not in all_cat_ids:
            continue

        gt_boxes = [ann['bbox'] for ann in gt_groups.get(key, [])]
        matched = [False] * len(gt_boxes)

        for pred in pred_groups.get(key, []):
            best_iou = 0.0
            best_idx = -1
            for gi, gt_box in enumerate(gt_boxes):
                if matched[gi]:
                    continue
                iou = _iou_xywh(pred['bbox'], gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = gi

            if best_iou >= iou_threshold and best_idx >= 0:
                per_class_counts[cat_id]['tp'] += 1
                matched[best_idx] = True
            else:
                per_class_counts[cat_id]['fp'] += 1

        per_class_counts[cat_id]['fn'] += sum(1 for m in matched if not m)

    def _metrics(tp: int, fp: int, fn: int) -> Dict:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}

    per_class = {}
    total_tp, total_fp, total_fn = 0, 0, 0
    for cid, counts in per_class_counts.items():
        per_class[cat_id_to_name[cid]] = _metrics(counts['tp'], counts['fp'], counts['fn'])
        total_tp += counts['tp']
        total_fp += counts['fp']
        total_fn += counts['fn']

    return {
        'overall': _metrics(total_tp, total_fp, total_fn),
        'per_class': per_class,
        'iou_threshold': iou_threshold,
        'confidence_threshold': confidence_threshold,
    }


def compute_det_curves(
    predictions: List[Dict],
    coco_gt: COCO,
    iou_threshold: float = 0.5,
) -> Dict:
    """
    Compute precision-recall curve data for object detection.

    For each (image, category) pair, predictions are sorted by confidence
    (descending) and greedily matched to ground-truth boxes.  Each prediction
    is labelled TP or FP.  The cumulative counts as a function of the
    confidence threshold give the precision-recall curve.

    Returns a dict with 'overall' and 'per_class' entries, each containing
    precision / recall / scores arrays plus the interpolated AP.
    """
    cat_id_to_name = {c['id']: c['name'] for c in coco_gt.dataset['categories']}
    all_cat_ids = set(cat_id_to_name.keys())

    gt_count_per_cat: Dict[int, int] = defaultdict(int)
    gt_groups: Dict[tuple, List[Dict]] = defaultdict(list)
    for ann in coco_gt.dataset['annotations']:
        gt_count_per_cat[ann['category_id']] += 1
        gt_groups[(ann['image_id'], ann['category_id'])].append(ann)

    pred_groups: Dict[tuple, List[Dict]] = defaultdict(list)
    for p in predictions:
        pred_groups[(p['image_id'], p['category_id'])].append(p)
    for key in pred_groups:
        pred_groups[key].sort(key=lambda x: x['score'], reverse=True)

    # Greedy matching: label each prediction as TP or FP
    per_cat_dets: Dict[int, List[tuple]] = defaultdict(list)

    all_keys = set(pred_groups.keys()) | set(gt_groups.keys())
    for key in all_keys:
        _, cat_id = key
        if cat_id not in all_cat_ids:
            continue

        gt_boxes = [ann['bbox'] for ann in gt_groups.get(key, [])]
        matched = [False] * len(gt_boxes)

        for pred in pred_groups.get(key, []):
            best_iou, best_idx = 0.0, -1
            for gi, gt_box in enumerate(gt_boxes):
                if matched[gi]:
                    continue
                iou = _iou_xywh(pred['bbox'], gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = gi

            is_tp = best_iou >= iou_threshold and best_idx >= 0
            if is_tp:
                matched[best_idx] = True
            per_cat_dets[cat_id].append((pred['score'], is_tp))

    def _build_curve(scores_arr, tp_arr, n_gt):
        """Return precision, recall, scores, and interpolated AP."""
        if len(scores_arr) == 0:
            return {'precision': [], 'recall': [], 'scores': [], 'n_gt': n_gt, 'ap': 0.0}

        order = np.argsort(-scores_arr)
        scores_sorted = scores_arr[order]
        tp_sorted = tp_arr[order]

        cum_tp = np.cumsum(tp_sorted)
        cum_fp = np.cumsum(~tp_sorted)

        precision = cum_tp / (cum_tp + cum_fp)
        recall = cum_tp / n_gt if n_gt > 0 else np.zeros_like(cum_tp, dtype=float)

        # All-points interpolated AP
        rec = np.concatenate([[0.0], recall, [recall[-1]]])
        prec = np.concatenate([[1.0], precision, [0.0]])
        for i in range(len(prec) - 2, -1, -1):
            prec[i] = max(prec[i], prec[i + 1])
        ap = float(np.sum((rec[1:] - rec[:-1]) * prec[1:]))

        return {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'scores': scores_sorted.tolist(),
            'n_gt': n_gt,
            'ap': ap,
        }

    result: Dict = {'per_class': {}, 'iou_threshold': iou_threshold}

    all_scores_list: list = []
    all_tp_list: list = []

    for cat_id in sorted(all_cat_ids):
        dets = per_cat_dets.get(cat_id, [])
        n_gt = gt_count_per_cat.get(cat_id, 0)
        cat_name = cat_id_to_name[cat_id]
        if not dets and n_gt == 0:
            continue

        scores = np.array([d[0] for d in dets])
        tp_flags = np.array([d[1] for d in dets], dtype=bool)
        result['per_class'][cat_name] = _build_curve(scores, tp_flags, n_gt)

        all_scores_list.extend(scores.tolist())
        all_tp_list.extend(tp_flags.tolist())

    total_gt = sum(gt_count_per_cat.values())
    result['overall'] = _build_curve(
        np.array(all_scores_list),
        np.array(all_tp_list, dtype=bool),
        total_gt,
    )

    return result


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
