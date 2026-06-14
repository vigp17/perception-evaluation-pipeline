#!/usr/bin/env python
"""
Multi-object tracking evaluation on nuScenes CAM_FRONT sequences.

Runs ByteTrack (Ultralytics) on each val scene in temporal order, matches
predicted tracks against ground-truth tracks (GT identity = nuScenes
instance_token), computes MOT metrics with per-scene accumulation, and slices
ID-switch rate by day/night.

Two correctness requirements learned the hard way:
  - Class-aware, class-filtered matching: only evaluate GT classes the detector
    actually predicts (COCO-mapped vehicle/person set). Otherwise GT barriers,
    cones, debris the model never targets all count as misses and tank MOTA.
  - Per-scene accumulators: track ids and frame timelines must not cross scene
    boundaries, or reused GT ids create phantom ID switches.
"""
import os
import json
from collections import defaultdict

import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points, BoxVisibility
from nuscenes.utils.splits import val as VAL_SCENES
from ultralytics import YOLO

from tracking_metrics import TrackingEvaluator, iou_distance_matrix

WEIGHTS = 'full_results/yolov8s_nuscenes_best.pt'
DATA_ROOT = 'data/nuscenes'
VERSION = 'v1.0-trainval'
OUT = 'analysis/tracking_eval.json'
IMG_W, IMG_H = 1600, 900
IOU_THR = 0.5
CONF = 0.25

# nuScenes class id (from data.yaml order) -> coarse category the COCO-trained
# detector can actually produce. Only these are evaluated for tracking.
# The fine-tuned model predicts the full 23, but we restrict tracking eval to
# the well-supported dynamic classes (vehicles + pedestrian) where tracking is
# meaningful; static furniture (barrier/cone) isn't a tracking target.
NUSC_NAMES = ['animal', 'human.pedestrian.adult', 'human.pedestrian.child',
    'human.pedestrian.construction_worker', 'human.pedestrian.personal_mobility',
    'human.pedestrian.police_officer', 'human.pedestrian.stroller',
    'human.pedestrian.wheelchair', 'movable_object.barrier',
    'movable_object.debris', 'movable_object.pushable_pullable',
    'movable_object.trafficcone', 'static_object.bicycle_rack',
    'vehicle.bicycle', 'vehicle.bus.bendy', 'vehicle.bus.rigid',
    'vehicle.car', 'vehicle.construction', 'vehicle.emergency.ambulance',
    'vehicle.emergency.police', 'vehicle.motorcycle', 'vehicle.trailer',
    'vehicle.truck']
NAME_TO_ID = {n: i for i, n in enumerate(NUSC_NAMES)}

# dynamic classes worth tracking (model predicts these by their own ids)
TRACK_CLASS_IDS = {
    NAME_TO_ID['human.pedestrian.adult'],
    NAME_TO_ID['vehicle.bicycle'],
    NAME_TO_ID['vehicle.bus.rigid'],
    NAME_TO_ID['vehicle.bus.bendy'],
    NAME_TO_ID['vehicle.car'],
    NAME_TO_ID['vehicle.motorcycle'],
    NAME_TO_ID['vehicle.trailer'],
    NAME_TO_ID['vehicle.truck'],
}
# map any nuScenes GT category_name to one of the tracked class ids, or None
GT_NAME_TO_TRACKCLASS = {
    'vehicle.car': NAME_TO_ID['vehicle.car'],
    'vehicle.truck': NAME_TO_ID['vehicle.truck'],
    'vehicle.bus.rigid': NAME_TO_ID['vehicle.bus.rigid'],
    'vehicle.bus.bendy': NAME_TO_ID['vehicle.bus.bendy'],
    'vehicle.trailer': NAME_TO_ID['vehicle.trailer'],
    'vehicle.bicycle': NAME_TO_ID['vehicle.bicycle'],
    'vehicle.motorcycle': NAME_TO_ID['vehicle.motorcycle'],
    'human.pedestrian.adult': NAME_TO_ID['human.pedestrian.adult'],
    'human.pedestrian.construction_worker': NAME_TO_ID['human.pedestrian.adult'],
    'human.pedestrian.police_officer': NAME_TO_ID['human.pedestrian.adult'],
}

_TOKEN2INT = {}
def _tid(token):
    if token not in _TOKEN2INT:
        _TOKEN2INT[token] = len(_TOKEN2INT) + 1
    return _TOKEN2INT[token]


def project_gt(nusc, cam_token):
    """Return (track_int_id, [x1,y1,x2,y2], class_id) for trackable GT only."""
    _, boxes, K = nusc.get_sample_data(cam_token, box_vis_level=BoxVisibility.ANY)
    out = []
    for b in boxes:
        cls = GT_NAME_TO_TRACKCLASS.get(b.name)
        if cls is None:
            continue  # skip non-trackable classes (barrier, cone, debris, ...)
        corners = b.corners()
        infront = corners[2, :] > 0.1
        if infront.sum() < 1:
            continue
        c2d = view_points(corners[:, infront], K, normalize=True)[:2, :]
        x1, x2 = max(0, float(c2d[0].min())), min(IMG_W, float(c2d[0].max()))
        y1, y2 = max(0, float(c2d[1].min())), min(IMG_H, float(c2d[1].max()))
        if (x2-x1) < 2 or (y2-y1) < 2:
            continue
        ann = nusc.get('sample_annotation', b.token)
        out.append((_tid(ann['instance_token']), [x1, y1, x2, y2], cls))
    return out


def main():
    print("Loading nuScenes trainval...")
    nusc = NuScenes(version=VERSION, dataroot=DATA_ROOT, verbose=False)
    model = YOLO(WEIGHTS)

    overall = TrackingEvaluator(IOU_THR)
    cond_eval = {'day': TrackingEvaluator(IOU_THR),
                 'night': TrackingEvaluator(IOU_THR)}

    val_scenes = [s for s in nusc.scene if s['name'] in VAL_SCENES]
    print(f"Evaluating tracking on {len(val_scenes)} val scenes...")

    for si, scene in enumerate(val_scenes):
        if si % 25 == 0:
            print(f"  scene {si}/{len(val_scenes)}")
        cond = 'night' if 'night' in scene['description'].lower() else 'day'

        overall.new_scene(scene['name'])
        cond_eval[cond].new_scene(scene['name'])

        frames = []
        tok = scene['first_sample_token']
        while tok:
            sample = nusc.get('sample', tok)
            frames.append(sample['data']['CAM_FRONT'])
            tok = sample['next']

        for fi, cam_token in enumerate(frames):
            img_path = nusc.get_sample_data_path(cam_token)
            res = model.track(img_path, persist=(fi > 0), conf=CONF,
                              tracker='bytetrack.yaml', verbose=False,
                              device=0)[0]
            pred_ids, pred_boxes = [], []
            if res.boxes is not None and res.boxes.id is not None:
                for b, tid in zip(res.boxes, res.boxes.id):
                    cls = int(b.cls)
                    if cls not in TRACK_CLASS_IDS:
                        continue  # only match classes we track
                    pred_boxes.append([float(x) for x in b.xyxy[0]])
                    pred_ids.append(int(tid))

            gt = project_gt(nusc, cam_token)
            gt_ids = [g[0] for g in gt]
            gt_boxes = [g[1] for g in gt]

            overall.add_frame(gt_ids, gt_boxes, pred_ids, pred_boxes)
            cond_eval[cond].add_frame(gt_ids, gt_boxes, pred_ids, pred_boxes)

        if hasattr(model, 'predictor') and model.predictor is not None:
            try:
                model.predictor.trackers[0].reset()
            except Exception:
                pass

    print("\n=== Overall tracking (CAM_FRONT, dynamic classes, val scenes) ===")
    r = overall.summary()
    print(f"MOTA: {r['mota']:.3f}   MOTP: {r['motp']:.3f}   IDF1: {r['idf1']:.3f}")
    print(f"ID switches: {int(r['num_switches'])}   "
          f"Fragmentations: {int(r['num_fragmentations'])}")
    print(f"Mostly tracked: {int(r['mostly_tracked'])}   "
          f"Mostly lost: {int(r['mostly_lost'])}")
    print(f"FP: {int(r['num_false_positives'])}   "
          f"Misses: {int(r['num_misses'])}   Objects: {int(r['num_objects'])}")

    print("\n=== By condition ===")
    cond_report = {}
    for c in ['day', 'night']:
        rc = cond_eval[c].summary()
        if rc is None:
            continue
        sw, objs = int(rc['num_switches']), int(rc['num_objects'])
        rate = (sw/objs*1000) if objs else 0
        print(f"{c:<8} MOTA={rc['mota']:.3f}  IDF1={rc['idf1']:.3f}  "
              f"switches={sw} ({rate:.2f}/1k obj)  objects={objs}")
        cond_report[c] = {'mota': round(float(rc['mota']), 4),
                          'idf1': round(float(rc['idf1']), 4),
                          'switches': sw, 'objects': objs,
                          'switches_per_1k': round(rate, 2)}

    os.makedirs('analysis', exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump({'overall': {k: (float(r[k]) if not np.isnan(float(r[k])) else None)
                               for k in ['mota','motp','idf1','num_switches',
                                         'num_fragmentations','mostly_tracked',
                                         'mostly_lost','num_false_positives',
                                         'num_misses','num_objects']},
                   'by_condition': cond_report}, f, indent=2)
    print(f"\nSaved {OUT}")


if __name__ == '__main__':
    main()
