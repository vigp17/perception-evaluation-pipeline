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
from nuscenes.utils.geometry_utils import view_points
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
        Get camera image and ground truth objects for a sample
        
        Args:
            sample_token: Token identifying the sample
            camera: Camera name (CAM_FRONT, CAM_BACK, etc)
            
        Returns:
            Dict with 'image' and 'objects' keys
        """
        sample = self.nusc.get('sample', sample_token)
        camera_token = sample['data'][camera]
        camera_data = self.nusc.get('sample_data', camera_token)
        
        # Load image
        image_path = os.path.join(self.data_root, camera_data['filename'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get annotations
        objects = []
        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)
            
            # Get 3D box
            from nuscenes.utils.data_classes import Quaternion
            box = Box(
                ann['translation'],
                ann['size'],
                Quaternion(ann['rotation']),
                name=ann['category_name']
            )
            
            # Project to camera
            camera_intrinsic = np.array(
                self.nusc.get(
                    'calibrated_sensor',
                    camera_data['calibrated_sensor_token']
                )['camera_intrinsic']
            )
            
            corners_3d = box.corners()
            corners_2d = view_points(
                corners_3d,
                camera_intrinsic,
                normalize=True
            )[:2, :]
            
            # Get 2D bounding box
            x_min, x_max = corners_2d[0].min(), corners_2d[0].max()
            y_min, y_max = corners_2d[1].min(), corners_2d[1].max()
            
            # Only include objects visible in camera
            if x_min < 1600 and x_max > 0 and y_min < 900 and y_max > 0:
                objects.append({
                    'class': ann['category_name'],
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