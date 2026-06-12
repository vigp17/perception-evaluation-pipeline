"""
Class taxonomy mapping for cross-dataset evaluation.

Problem: COCO-pretrained detectors output COCO class names ('car', 'person'),
while nuScenes ground truth uses its own taxonomy ('vehicle.car',
'human.pedestrian.adult'). Comparing them directly yields zero matches.

Solution: map BOTH predictions and ground truths to a shared canonical
taxonomy before scoring. This is the standard approach for evaluating a
detector trained on one label space against another (cf. nuScenes detection
challenge class merging).

Notes:
- COCO has no 'barrier' or 'traffic cone' classes, so a COCO-pretrained
  baseline cannot detect them. Those GT classes are kept in the canonical
  taxonomy (the baseline scores 0 on them, correctly) — a fine-tuned model
  trained on nuScenes labels can score on them.
- nuScenes subcategories are merged (adult/child/worker/officer -> 'person';
  rigid/bendy bus -> 'bus') because a COCO detector cannot distinguish them.
"""

from typing import Dict, List, Optional

# Canonical evaluation taxonomy
CANONICAL_CLASSES = [
    'car',
    'truck',
    'bus',
    'person',
    'bicycle',
    'motorcycle',
    'barrier',
    'traffic_cone',
]

# COCO detector output -> canonical
COCO_TO_CANONICAL: Dict[str, str] = {
    'car': 'car',
    'truck': 'truck',
    'bus': 'bus',
    'person': 'person',
    'bicycle': 'bicycle',
    'motorcycle': 'motorcycle',
    # COCO classes with no nuScenes counterpart (traffic light, stop sign,
    # etc.) are intentionally absent -> predictions dropped before scoring.
}

# nuScenes category_name -> canonical
NUSCENES_TO_CANONICAL: Dict[str, str] = {
    'vehicle.car': 'car',
    'vehicle.truck': 'truck',
    'vehicle.bus.rigid': 'bus',
    'vehicle.bus.bendy': 'bus',
    'vehicle.construction': 'truck',
    'vehicle.emergency.ambulance': 'truck',
    'vehicle.emergency.police': 'car',
    'vehicle.trailer': 'truck',
    'human.pedestrian.adult': 'person',
    'human.pedestrian.child': 'person',
    'human.pedestrian.construction_worker': 'person',
    'human.pedestrian.police_officer': 'person',
    'human.pedestrian.personal_mobility': 'person',
    'human.pedestrian.stroller': 'person',
    'human.pedestrian.wheelchair': 'person',
    'vehicle.bicycle': 'bicycle',
    'vehicle.motorcycle': 'motorcycle',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    # Unmapped nuScenes classes (debris, pushable_pullable, bicycle_rack,
    # animal, ego) are excluded from evaluation entirely.
}


def map_prediction_class(coco_name: str) -> Optional[str]:
    """Map a COCO class name to canonical. Returns None if not evaluable."""
    return COCO_TO_CANONICAL.get(coco_name)


def map_ground_truth_class(nuscenes_name: str) -> Optional[str]:
    """Map a nuScenes category_name to canonical. Returns None if excluded."""
    return NUSCENES_TO_CANONICAL.get(nuscenes_name)


def filter_and_map_predictions(predictions: List[Dict]) -> List[Dict]:
    """Map prediction class names to canonical taxonomy, dropping unmappable."""
    mapped = []
    for pred in predictions:
        canonical = map_prediction_class(pred['class'])
        if canonical is not None:
            mapped.append({**pred, 'class': canonical})
    return mapped


def filter_and_map_ground_truths(ground_truths: List[Dict]) -> List[Dict]:
    """Map GT class names to canonical taxonomy, dropping excluded classes."""
    mapped = []
    for gt in ground_truths:
        canonical = map_ground_truth_class(gt['class'])
        if canonical is not None:
            mapped.append({**gt, 'class': canonical})
    return mapped
