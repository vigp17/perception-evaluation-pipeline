"""
Dataset Loader for nuScenes
Load and prepare autonomous driving data
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points, BoxVisibility
from tqdm import tqdm


class NuScenesLoader:
    """Load nuScenes autonomous driving dataset"""
    
    def __init__(self, data_root: str, version: str = 'v1.0-mini'):
        """
        Initialize loader
        
        Args:
            data_root: Path to nuScenes directory
            version: Dataset version (v1.0-mini or v1.0-trainval)
        """
        self.data_root = data_root
        self.version = version
        
        try:
            self.nusc = NuScenes(version=version, dataroot=data_root, verbose=True)
            print(f"✓ Loaded nuScenes {version}")
            print(f"  Scenes: {len(self.nusc.scene)}")
            print(f"  Samples: {len(self.nusc.sample)}")
        except Exception as e:
            raise ValueError(f"Failed to load nuScenes: {e}")
    
    def get_sample_image_and_objects(self, sample_token: str, 
                                     camera: str = 'CAM_FRONT') -> Dict:
        """
        Get camera image and ground truth 2D boxes for a sample.

        Uses nusc.get_sample_data(), which transforms each annotation box
        through the full chain (global -> ego -> camera frame) and filters
        to boxes visible in this camera. Projecting raw ann['translation']
        (global/map coordinates) directly through the intrinsic matrix —
        as a naive implementation does — produces meaningless pixel
        coordinates, including huge negative values for objects behind
        the camera.

        Args:
            sample_token: Token identifying the sample
            camera: Camera name (CAM_FRONT, CAM_BACK, etc)

        Returns:
            Dict with 'image' and 'objects' keys
        """
        sample = self.nusc.get('sample', sample_token)
        camera_token = sample['data'][camera]

        # Boxes come back already in the camera frame, filtered to those
        # with at least one corner visible in the image.
        image_path, boxes, camera_intrinsic = self.nusc.get_sample_data(
            camera_token, box_vis_level=BoxVisibility.ANY
        )

        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]

        objects = []
        for box in boxes:
            corners_3d = box.corners()  # (3, 8) in camera frame

            # Keep only corners in front of the camera plane; projecting
            # negative-depth points flips signs and produces garbage.
            in_front = corners_3d[2, :] > 0.1
            if in_front.sum() < 1:
                continue
            corners_3d = corners_3d[:, in_front]

            corners_2d = view_points(
                corners_3d, camera_intrinsic, normalize=True
            )[:2, :]

            # Axis-aligned 2D box, clipped to image bounds
            x_min = float(np.clip(corners_2d[0].min(), 0, img_w))
            x_max = float(np.clip(corners_2d[0].max(), 0, img_w))
            y_min = float(np.clip(corners_2d[1].min(), 0, img_h))
            y_max = float(np.clip(corners_2d[1].max(), 0, img_h))

            # Skip degenerate boxes (fully clipped or sliver-thin)
            if (x_max - x_min) < 2 or (y_max - y_min) < 2:
                continue

            objects.append({
                'class': box.name,
                'bbox': [x_min, y_min, x_max, y_max],
                'confidence': 1.0,  # Ground truth
            })

        return {
            'image': image,
            'objects': objects,
            'sample_token': sample_token,
            'image_path': image_path
        }
    
    def iterate_samples(self, num_samples: int = None):
        """
        Iterator through dataset samples
        
        Args:
            num_samples: Limit to first N samples (None for all)
        """
        samples = self.nusc.sample[:num_samples] if num_samples else self.nusc.sample
        
        for sample in tqdm(samples, desc="Loading samples"):
            try:
                data = self.get_sample_image_and_objects(sample['token'])
                yield data
            except Exception as e:
                print(f"⚠ Warning: Failed to load sample {sample['token']}: {e}")
                continue


# Test it
if __name__ == "__main__":
    loader = NuScenesLoader(data_root='data/nuscenes', version='v1.0-mini')
    
    # Load first 5 samples
    count = 0
    for data in loader.iterate_samples(num_samples=5):
        print(f"\nSample: {data['sample_token']}")
        print(f"Image shape: {data['image'].shape}")
        print(f"Objects: {len(data['objects'])}")
        print(f"Classes: {[obj['class'] for obj in data['objects']]}")
        count += 1