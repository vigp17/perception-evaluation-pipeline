#!/usr/bin/env python
"""
Diagnostic 3: run ONE scene through the REAL TrackingEvaluator (same code path
as evaluate_tracking.py) and dump actual SWITCH events from its accumulator -
ground truth of what's being counted, no separate reimplementation.
"""
from nuscenes import NuScenes
from nuscenes.utils.splits import val as VAL_SCENES
from ultralytics import YOLO
from evaluate_tracking import project_gt, TRACK_CLASS_IDS, WEIGHTS, DATA_ROOT, VERSION, CONF
from tracking_metrics import TrackingEvaluator

nusc = NuScenes(version=VERSION, dataroot=DATA_ROOT, verbose=False)
model = YOLO(WEIGHTS)
scene = [s for s in nusc.scene if s['name'] in VAL_SCENES][0]
print(f"Scene: {scene['name']}")

ev = TrackingEvaluator(0.5)
ev.new_scene(scene['name'])

frames = []
tok = scene['first_sample_token']
while tok:
    s = nusc.get('sample', tok); frames.append(s['data']['CAM_FRONT']); tok = s['next']

for fi, cam_token in enumerate(frames):
    img_path = nusc.get_sample_data_path(cam_token)
    res = model.track(img_path, persist=(fi>0), conf=CONF,
                      tracker='bytetrack.yaml', verbose=False, device=0)[0]
    pred_ids, pred_boxes = [], []
    if res.boxes is not None and res.boxes.id is not None:
        for b, tid in zip(res.boxes, res.boxes.id):
            if int(b.cls) in TRACK_CLASS_IDS:
                pred_boxes.append([float(x) for x in b.xyxy[0]])
                pred_ids.append(int(tid))
    gt = project_gt(nusc, cam_token)
    gt_ids = [g[0] for g in gt]
    gt_boxes = [g[1] for g in gt]
    ev.add_frame(gt_ids, gt_boxes, pred_ids, pred_boxes)

acc = ev.accs[0]
events = acc.events
switches = events[events['Type'] == 'SWITCH']
matches = events[events['Type'] == 'MATCH']
print(f"\nTotal MATCH events: {len(matches)}")
print(f"Total SWITCH events: {len(switches)}")
print("\nFirst 15 switches:")
print(switches.head(15))

# also: how many DISTINCT predicted ids did ByteTrack ever emit for this scene?
all_pred_ids = set()
for fi, cam_token in enumerate(frames):
    img_path = nusc.get_sample_data_path(cam_token)
    res = model.track(img_path, persist=True, conf=CONF,
                      tracker='bytetrack.yaml', verbose=False, device=0)[0]
    if res.boxes is not None and res.boxes.id is not None:
        for tid in res.boxes.id:
            all_pred_ids.add(int(tid))
print(f"\nDistinct ByteTrack ids used across scene (2nd pass): {sorted(all_pred_ids)}")
