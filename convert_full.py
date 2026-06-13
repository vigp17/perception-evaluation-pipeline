#!/usr/bin/env python
"""
Convert the FULL nuScenes trainval set to YOLO format.

Unlike a naive index split, this assigns each sample to train/val using the
OFFICIAL nuScenes scene split (nuscenes.utils.splits). This prevents scene
leakage: adjacent near-identical frames from the same drive never straddle
train and val, so val mAP reflects real generalization rather than memorized
neighbors.

Only CAM_FRONT keyframes are converted (the project's scope). Boxes are
projected via the fixed camera-frame transform in dataset_converter.py.
"""
import os
import sys
import shutil
from pathlib import Path
import numpy as np
from tqdm import tqdm
from nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points, BoxVisibility
from nuscenes.utils.splits import train as TRAIN_SCENES, val as VAL_SCENES

sys.path.insert(0, str(Path(__file__).parent))
from src.dataset_converter import NuScenesYOLOConverter

DATA_ROOT = 'data/nuscenes'
VERSION = 'v1.0-trainval'
OUTPUT = 'data/yolo_dataset_full'
IMG_W, IMG_H = 1600, 900


def main():
    print("Loading nuScenes trainval (this takes ~30-60s)...")
    conv = NuScenesYOLOConverter(DATA_ROOT, output_dir=OUTPUT, version=VERSION)
    nusc = conv.nusc

    train_tokens, val_tokens = set(), set()
    for scene in nusc.scene:
        if scene['name'] in TRAIN_SCENES:
            bucket = train_tokens
        elif scene['name'] in VAL_SCENES:
            bucket = val_tokens
        else:
            continue
        # walk all samples in the scene
        tok = scene['first_sample_token']
        while tok:
            bucket.add(tok)
            tok = nusc.get('sample', tok)['next']

    print(f"Train samples: {len(train_tokens)}  Val samples: {len(val_tokens)}")

    def convert_one(sample_token, split):
        sample = nusc.get('sample', sample_token)
        cam_token = sample['data']['CAM_FRONT']
        image_path, boxes, K = nusc.get_sample_data(
            cam_token, box_vis_level=BoxVisibility.ANY)

        out_img = Path(OUTPUT) / 'images' / split / f"{sample_token}.jpg"
        try:
            shutil.copy(image_path, out_img)
        except (OSError, shutil.Error):
            return

        lines = []
        for box in boxes:
            corners = box.corners()
            infront = corners[2, :] > 0.1
            if infront.sum() < 1:
                continue
            c2d = view_points(corners[:, infront], K, normalize=True)[:2, :]
            x_min = float(np.clip(c2d[0].min(), 0, IMG_W))
            x_max = float(np.clip(c2d[0].max(), 0, IMG_W))
            y_min = float(np.clip(c2d[1].min(), 0, IMG_H))
            y_max = float(np.clip(c2d[1].max(), 0, IMG_H))
            if (x_max - x_min) < 2 or (y_max - y_min) < 2:
                continue
            cx = (x_min + x_max) / 2 / IMG_W
            cy = (y_min + y_max) / 2 / IMG_H
            w = (x_max - x_min) / IMG_W
            h = (y_max - y_min) / IMG_H
            cid = conv.class_mapping[box.name]
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if lines:
            lbl = Path(OUTPUT) / 'labels' / split / f"{sample_token}.txt"
            with open(lbl, 'w') as f:
                f.write('\n'.join(lines))

    for tok in tqdm(train_tokens, desc='train'):
        convert_one(tok, 'train')
    for tok in tqdm(val_tokens, desc='val'):
        convert_one(tok, 'val')

    conv._save_classes()
    conv._create_yaml()
    print(f"\n✓ Done. YOLO dataset at {OUTPUT}")
    print(f"  train images: {len(list((Path(OUTPUT)/'images'/'train').glob('*.jpg')))}")
    print(f"  val images:   {len(list((Path(OUTPUT)/'images'/'val').glob('*.jpg')))}")


if __name__ == '__main__':
    main()
