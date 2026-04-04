"""
Convert nuScenes dataset to YOLO format for fine-tuning
"""

import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points


class NuScenesYOLOConverter:
    """Convert nuScenes to YOLO format"""
    
    def __init__(self, data_root: str, output_dir: str = 'yolo_dataset'):
        """
        Initialize converter
        
        Args:
            data_root: Path to nuScenes data
            output_dir: Output directory for YOLO format
        """
        self.data_root = data_root
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
        
        # Load nuScenes
        self.nusc = NuScenes(version='v1.0-mini', dataroot=data_root, verbose=False)
        
        # Map nuScenes classes to YOLO format
        self.class_mapping = self._create_class_mapping()
        print(f"✓ Converter initialized with {len(self.class_mapping)} classes")
    
    def _create_class_mapping(self):
        """Create mapping from nuScenes classes to numeric IDs"""
        classes = set()
        for ann in self.nusc.sample_annotation:
            classes.add(ann['category_name'])
        
        class_mapping = {name: idx for idx, name in enumerate(sorted(classes))}
        return class_mapping
    
    def convert_dataset(self, num_samples: int = 100, train_ratio: float = 0.8):
        """
        Convert nuScenes to YOLO format
        
        Args:
            num_samples: Number of samples to convert
            train_ratio: Fraction for training (rest for validation)
        """
        samples = self.nusc.sample[:num_samples]
        num_train = int(len(samples) * train_ratio)
        
        print(f"\nConverting {len(samples)} samples to YOLO format")
        print(f"Train: {num_train}, Val: {len(samples) - num_train}")
        
        for idx, sample in enumerate(tqdm(samples)):
            split = 'train' if idx < num_train else 'val'
            self._convert_sample(sample, split)
        
        # Save class names
        self._save_classes()
        
        # Create data.yaml
        self._create_yaml()
        
        print(f"✓ Conversion complete! Data saved to {self.output_dir}")
    
    def _convert_sample(self, sample, split: str):
        """Convert a single sample"""
        
        # Get camera data
        camera_token = sample['data']['CAM_FRONT']
        camera_data = self.nusc.get('sample_data', camera_token)
        image_path = os.path.join(self.data_root, camera_data['filename'])
        
        # Copy image
        import shutil
        sample_name = f"{sample['token']}.jpg"
        output_image = self.output_dir / 'images' / split / sample_name
        
        try:
            shutil.copy(image_path, output_image)
        except:
            return  # Skip if image copy fails
        
        # Get annotations
        annotations = []
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
                # Convert to YOLO format (center_x, center_y, width, height, normalized)
                img_width, img_height = 1600, 900
                
                center_x = (x_min + x_max) / 2 / img_width
                center_y = (y_min + y_max) / 2 / img_height
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height
                
                # Clamp to [0, 1]
                center_x = max(0, min(1, center_x))
                center_y = max(0, min(1, center_y))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                class_id = self.class_mapping[ann['category_name']]
                annotations.append(f"{class_id} {center_x} {center_y} {width} {height}")
        
        # Save annotations
        if annotations:
            label_path = self.output_dir / 'labels' / split / f"{sample['token']}.txt"
            with open(label_path, 'w') as f:
                f.write('\n'.join(annotations))
    
    def _save_classes(self):
        """Save class names to file"""
        classes_file = self.output_dir / 'classes.txt'
        with open(classes_file, 'w') as f:
            for name, idx in sorted(self.class_mapping.items(), key=lambda x: x[1]):
                f.write(f"{name}\n")
    
    def _create_yaml(self):
        """Create data.yaml for YOLOv5"""
        yaml_content = f"""path: {self.output_dir.absolute()}
train: images/train
val: images/val

nc: {len(self.class_mapping)}
names: {list(self.class_mapping.keys())}
"""
        
        yaml_path = self.output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"✓ Created {yaml_path}")


if __name__ == "__main__":
    converter = NuScenesYOLOConverter('data/nuscenes')
    converter.convert_dataset(num_samples=100)