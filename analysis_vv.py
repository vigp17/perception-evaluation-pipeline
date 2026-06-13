#!/usr/bin/env python
"""
Perception V&V analysis: condition-sliced, distance-binned, and failure-mode
evaluation of the fine-tuned model on the nuScenes val split.

Runs inference ONCE over the val images, then slices results three ways:
  1. By operating condition  (day/night, rain/clear)  -> mAP per condition
  2. By object distance       (0-20m / 20-40m / 40m+)  -> recall per range bin
  3. Failure cases            (missed / false-positive examples) -> annotated images

This is evaluation the way AV perception is actually assessed: not a single mAP,
but how detection quality varies with the conditions and ranges that matter for
safety. Inference-only -- no retraining.

Expects:
  - trained weights at WEIGHTS
  - nuScenes trainval metadata at DATA_ROOT (v1.0-trainval tables + samples)
  - the YOLO val image set at VAL_IMG_DIR (filenames are sample_tokens)
"""
import os
import glob
import json
from collections import defaultdict

import numpy as np
import cv2
from nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points, BoxVisibility
from ultralytics import YOLO

# ---------------- config ----------------
WEIGHTS = 'runs/detect/runs/full_finetuned/weights/best.pt'
DATA_ROOT = 'data/nuscenes'
VERSION = 'v1.0-trainval'
VAL_IMG_DIR = 'data/yolo_dataset_full/images/val'
OUT_DIR = 'analysis'
IMG_W, IMG_H = 1600, 900
IOU_THR = 0.5
CONF_THR = 0.25            # operating-point conf for slicing/failure analysis
DIST_BINS = [(0, 20), (20, 40), (40, 1e9)]
N_FAILURE_IMAGES = 8

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, 'failures'), exist_ok=True)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def classify_condition(desc):
    """Map a scene description string to (time_of_day, weather)."""
    d = desc.lower()
    tod = 'night' if 'night' in d else 'day'
    weather = 'rain' if 'rain' in d else 'clear'
    return tod, weather


def main():
    print("Loading nuScenes trainval metadata...")
    nusc = NuScenes(version=VERSION, dataroot=DATA_ROOT, verbose=False)

    # sample_token -> (scene description, condition)
    sample_meta = {}
    for scene in nusc.scene:
        tod, weather = classify_condition(scene['description'])
        tok = scene['first_sample_token']
        while tok:
            sample_meta[tok] = {'tod': tod, 'weather': weather}
            tok = nusc.get('sample', tok)['next']

    model = YOLO(WEIGHTS)
    images = sorted(glob.glob(os.path.join(VAL_IMG_DIR, '*.jpg')))
    print(f"Running inference on {len(images)} val images...")

    # accumulators
    # condition slices: per (slice_name) -> matched/total for recall, and tp/fp for precision
    cond_stats = defaultdict(lambda: {'gt': 0, 'tp': 0, 'fp': 0})
    dist_stats = {b: {'gt': 0, 'tp': 0} for b in DIST_BINS}
    failure_candidates = []   # (n_missed, n_fp, img_path, gt_boxes, pred_boxes, missed_idx)

    for n, img_path in enumerate(images):
        if n % 500 == 0:
            print(f"  {n}/{len(images)}")
        token = os.path.splitext(os.path.basename(img_path))[0]
        if token not in sample_meta:
            continue
        cam_token = nusc.get('sample', token)['data']['CAM_FRONT']
        _, boxes3d, K = nusc.get_sample_data(
            cam_token, box_vis_level=BoxVisibility.ANY)

        # ground-truth 2D boxes + distance (camera-frame z = forward range)
        gts = []
        for b in boxes3d:
            corners = b.corners()
            infront = corners[2, :] > 0.1
            if infront.sum() < 1:
                continue
            z_mean = float(corners[2, infront].mean())   # forward distance (m)
            c2d = view_points(corners[:, infront], K, normalize=True)[:2, :]
            x1, x2 = float(c2d[0].min()), float(c2d[0].max())
            y1, y2 = float(c2d[1].min()), float(c2d[1].max())
            x1, x2 = max(0, x1), min(IMG_W, x2)
            y1, y2 = max(0, y1), min(IMG_H, y2)
            if (x2-x1) < 2 or (y2-y1) < 2:
                continue
            gts.append({'box': [x1, y1, x2, y2], 'dist': z_mean})

        # predictions at operating conf
        r = model.predict(img_path, conf=CONF_THR, verbose=False)[0]
        preds = []
        if r.boxes is not None:
            for b in r.boxes:
                preds.append([float(x) for x in b.xyxy[0]])

        # greedy match preds<->gts (class-agnostic localization quality)
        used = [False]*len(gts)
        matched_pred = [False]*len(preds)
        for pi, pb in enumerate(preds):
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gts):
                if used[j]:
                    continue
                i = iou(pb, g['box'])
                if i > best_iou:
                    best_iou, best_j = i, j
            if best_iou >= IOU_THR and best_j >= 0:
                used[best_j] = True
                matched_pred[pi] = True

        cond = sample_meta[token]
        slices = [cond['tod'], cond['weather'],
                  f"{cond['tod']}_{cond['weather']}"]

        for j, g in enumerate(gts):
            # condition recall
            for s in slices:
                cond_stats[s]['gt'] += 1
                if used[j]:
                    cond_stats[s]['tp'] += 1
            # distance recall
            for b in DIST_BINS:
                if b[0] <= g['dist'] < b[1]:
                    dist_stats[b]['gt'] += 1
                    if used[j]:
                        dist_stats[b]['tp'] += 1
                    break

        for pi in range(len(preds)):
            if not matched_pred[pi]:
                for s in slices:
                    cond_stats[s]['fp'] += 1

        # failure candidates: images with both misses and false positives
        n_missed = sum(1 for j in range(len(gts)) if not used[j])
        n_fp = sum(1 for pi in range(len(preds)) if not matched_pred[pi])
        if n_missed >= 2 and n_fp >= 1:
            failure_candidates.append({
                'score': n_missed + n_fp, 'img': img_path,
                'gts': gts, 'preds': preds,
                'used': used, 'matched_pred': matched_pred,
            })

    # ---------------- report: conditions ----------------
    def recall(d):
        return d['tp'] / d['gt'] if d['gt'] else 0.0

    def precision(d):
        denom = d['tp'] + d['fp']
        return d['tp'] / denom if denom else 0.0

    print("\n=== Condition-sliced detection (IoU 0.5, conf 0.25) ===")
    print(f"{'Condition':<16}{'GT':>8}{'Recall':>9}{'Precision':>11}")
    cond_report = {}
    for s in ['day', 'night', 'clear', 'rain',
              'day_clear', 'day_rain', 'night_clear', 'night_rain']:
        d = cond_stats.get(s)
        if not d or d['gt'] == 0:
            continue
        print(f"{s:<16}{d['gt']:>8}{recall(d):>9.3f}{precision(d):>11.3f}")
        cond_report[s] = {'gt': d['gt'], 'recall': round(recall(d), 4),
                          'precision': round(precision(d), 4)}

    # ---------------- report: distance ----------------
    print("\n=== Distance-binned recall (IoU 0.5, conf 0.25) ===")
    print(f"{'Range (m)':<16}{'GT':>8}{'Recall':>9}")
    dist_report = {}
    for b in DIST_BINS:
        d = dist_stats[b]
        label = f"{b[0]}-{b[1] if b[1] < 1e8 else 'inf'}"
        rc = d['tp'] / d['gt'] if d['gt'] else 0.0
        print(f"{label:<16}{d['gt']:>8}{rc:>9.3f}")
        dist_report[label] = {'gt': d['gt'], 'recall': round(rc, 4)}

    # ---------------- failure visualizations ----------------
    print("\n=== Writing failure-case visualizations ===")
    failure_candidates.sort(key=lambda x: -x['score'])
    for k, fc in enumerate(failure_candidates[:N_FAILURE_IMAGES]):
        img = cv2.imread(fc['img'])
        for j, g in enumerate(fc['gts']):
            x1, y1, x2, y2 = map(int, g['box'])
            # missed GT = red, detected GT = green
            color = (0, 200, 0) if fc['used'][j] else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        for pi, pb in enumerate(fc['preds']):
            if not fc['matched_pred'][pi]:
                x1, y1, x2, y2 = map(int, pb)
                # false positive = yellow
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 215, 255), 2)
        out = os.path.join(OUT_DIR, 'failures', f"failure_{k:02d}.jpg")
        cv2.imwrite(out, img)
    print(f"  wrote {min(len(failure_candidates), N_FAILURE_IMAGES)} images to "
          f"{OUT_DIR}/failures/")
    print("  legend: green=detected GT, red=missed GT, yellow=false positive")

    # ---------------- save json ----------------
    report = {
        'operating_point': {'iou': IOU_THR, 'conf': CONF_THR},
        'condition_sliced': cond_report,
        'distance_binned': dist_report,
        'num_val_images': len(images),
    }
    with open(os.path.join(OUT_DIR, 'vv_analysis.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n✓ Saved {OUT_DIR}/vv_analysis.json")


if __name__ == '__main__':
    main()
