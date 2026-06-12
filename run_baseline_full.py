#!/usr/bin/env python
"""
Baseline scorer for the FULL nuScenes val split.

Runs pretrained YOLOv8s (COCO) over the full val images and scores per-class
AP@0.5 against the YOLO ground-truth labels, using the SAME val split the
fine-tuned model was evaluated on. COCO predictions are mapped to the 23-class
nuScenes taxonomy in data.yaml; nuScenes-only classes score 0 by construction.
"""
import glob
import os
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

VAL_IMG_DIR = '/data/yolo_dataset_full/images/val'
VAL_LBL_DIR = '/data/yolo_dataset_full/labels/val'
IMG_W, IMG_H = 1600, 900
IOU_THR = 0.5
CONF_THR = 0.001

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

# COCO id -> nuScenes 23-class id
COCO_TO_NUSC = {
    0: 1,    # person -> pedestrian.adult
    1: 13,   # bicycle
    2: 16,   # car
    3: 20,   # motorcycle
    5: 15,   # bus -> bus.rigid
    7: 22,   # truck
}


def yolo_to_xyxy(cx, cy, w, h):
    return [(cx-w/2)*IMG_W, (cy-h/2)*IMG_H, (cx+w/2)*IMG_W, (cy+h/2)*IMG_H]


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0


def ap_from_pr(recall, precision):
    mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 0.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx+1]-mrec[idx])*mpre[idx+1]))


def main():
    model = YOLO('yolov8s.pt')
    images = sorted(glob.glob(os.path.join(VAL_IMG_DIR, '*.jpg')))
    print(f"Scoring pretrained yolov8s on {len(images)} val images...")

    preds_by_cls = defaultdict(list)
    gt_count = defaultdict(int)

    for n, img_path in enumerate(images):
        if n % 500 == 0:
            print(f"  {n}/{len(images)}")
        stem = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(VAL_LBL_DIR, stem + '.txt')

        gts = defaultdict(list)
        if os.path.exists(lbl_path):
            for line in open(lbl_path):
                p = line.split()
                if len(p) != 5:
                    continue
                cid = int(p[0])
                gts[cid].append(yolo_to_xyxy(*map(float, p[1:])))
                gt_count[cid] += 1

        r = model.predict(img_path, conf=CONF_THR, verbose=False)[0]
        dets = []
        if r.boxes is not None:
            for b in r.boxes:
                coco_id = int(b.cls)
                if coco_id not in COCO_TO_NUSC:
                    continue
                dets.append((COCO_TO_NUSC[coco_id], float(b.conf),
                             [float(x) for x in b.xyxy[0]]))

        for cid in set(d[0] for d in dets):
            cdets = sorted([d for d in dets if d[0] == cid], key=lambda x: -x[1])
            used = [False]*len(gts.get(cid, []))
            for _, conf, box in cdets:
                best_iou, best_j = 0.0, -1
                for j, gtb in enumerate(gts.get(cid, [])):
                    if used[j]:
                        continue
                    i = iou(box, gtb)
                    if i > best_iou:
                        best_iou, best_j = i, j
                if best_iou >= IOU_THR and best_j >= 0:
                    used[best_j] = True
                    preds_by_cls[cid].append((conf, 1))
                else:
                    preds_by_cls[cid].append((conf, 0))

    print(f"\n{'Class':<40}{'AP50':>8}")
    print('-'*48)
    aps = []
    for cid in range(len(NUSC_NAMES)):
        n_gt = gt_count.get(cid, 0)
        if n_gt == 0:
            continue
        dets = sorted(preds_by_cls.get(cid, []), key=lambda x: -x[0])
        if not dets:
            ap = 0.0
        else:
            tp = np.array([d[1] for d in dets])
            fp = 1-tp
            tpc, fpc = np.cumsum(tp), np.cumsum(fp)
            ap = ap_from_pr(tpc/n_gt, tpc/(tpc+fpc))
        aps.append(ap)
        print(f"{NUSC_NAMES[cid]:<40}{ap:>8.3f}")
    print('-'*48)
    print(f"{'mAP50 (classes present in val)':<40}{np.mean(aps):>8.3f}")


if __name__ == '__main__':
    main()
