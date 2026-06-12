"""
Evaluation Metrics for Object Detection
Compute IoU, Average Precision, mAP
"""

import numpy as np
from typing import List, Dict, Tuple


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute Intersection over Union for 2D bounding boxes
    
    Args:
        box1, box2: [x1, y1, x2, y2] (top-left, bottom-right corners)
        
    Returns:
        IoU value between 0 and 1
    """
    # Intersection area
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])
    
    # No overlap
    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    
    # Union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def compute_average_precision(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute Average Precision for a class
    
    Args:
        predictions: List of {'bbox': [...], 'confidence': ...}
        ground_truths: List of {'bbox': [...]}
        iou_threshold: IoU threshold for match (default 0.5)
        
    Returns:
        ap: Average Precision value (0-1)
        precision: Precision values
        recall: Recall values
    """
    if len(predictions) == 0 or len(ground_truths) == 0:
        return 0.0, np.array([]), np.array([])
    
    # Sort predictions by confidence
    sorted_preds = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
    
    tp = np.zeros(len(sorted_preds))
    fp = np.zeros(len(sorted_preds))
    gt_matched = set()
    
    # For each prediction, find best matching ground truth
    for pred_idx, pred in enumerate(sorted_preds):
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in gt_matched:
                continue
            
            iou = compute_iou(pred['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        # Check if matched
        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp[pred_idx] = 1
            gt_matched.add(best_gt_idx)
        else:
            fp[pred_idx] = 1
    
    # Compute precision-recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    recall = tp_cumsum / len(ground_truths)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    # All-point interpolated AP (Pascal VOC 2010+ convention):
    # take the precision envelope (monotonically non-increasing), then
    # integrate over recall. Raw trapezoidal integration of the un-enveloped
    # PR curve under-/over-estimates AP depending on confidence ordering.
    mrec = np.concatenate(([0.0], recall, [recall[-1]] if len(recall) else [0.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1]))
    
    return ap, precision, recall


def compute_map(
    all_predictions: Dict[str, List[Dict]],
    all_ground_truths: Dict[str, List[Dict]],
    iou_threshold: float = 0.5
) -> Tuple[float, Dict[str, float]]:
    """
    Compute mean Average Precision across all classes
    
    Args:
        all_predictions: Dict mapping class -> predictions
        all_ground_truths: Dict mapping class -> ground truths
        iou_threshold: IoU threshold
        
    Returns:
        mAP: Mean Average Precision
        per_class_ap: Dict of AP per class
    """
    aps = {}
    
    for class_name in all_ground_truths.keys():
        predictions = all_predictions.get(class_name, [])
        ground_truths = all_ground_truths[class_name]
        
        ap, _, _ = compute_average_precision(
            predictions, ground_truths, iou_threshold
        )
        aps[class_name] = ap
    
    mAP = np.mean(list(aps.values())) if len(aps) > 0 else 0.0
    
    return float(mAP), aps


# Test it
if __name__ == "__main__":
    # Test IoU
    box1 = [0, 0, 10, 10]
    box2 = [5, 5, 15, 15]
    iou = compute_iou(box1, box2)
    print(f"IoU test: {iou:.3f} (expected ~0.143)")
    
    # Test AP
    predictions = [
        {'bbox': [0, 0, 10, 10], 'confidence': 0.9},
        {'bbox': [5, 5, 15, 15], 'confidence': 0.7},
    ]
    ground_truths = [
        {'bbox': [1, 1, 11, 11]},
        {'bbox': [4, 4, 14, 14]},
    ]
    
    ap, precision, recall = compute_average_precision(predictions, ground_truths)
    print(f"AP test: {ap:.3f}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
