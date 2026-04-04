"""
Object Detection Model - YOLOv5
Run inference on images
"""

import torch
import numpy as np
from typing import List, Dict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class ObjectDetector:
    """YOLOv5 object detection wrapper"""
    
    def __init__(self, model_name: str = 'yolov5s', device: str = None):
        """
        Initialize detector
        
        Args:
            model_name: YOLOv5 model (yolov5s, yolov5m, yolov5l, yolov5x)
            device: 'cuda' or 'cpu' (auto-detects if None)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        print(f"Using device: {device}")
        
        # Load model
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        self.model.to(device)
        self.model.conf = 0.25  # Confidence threshold
        self.model.iou = 0.45   # NMS IoU threshold
        
        print(f"✓ Loaded YOLOv5 {model_name}")
        print(f"  Classes: {len(self.model.names)}")
    
    def infer(self, image: np.ndarray) -> List[Dict]:
        """
        Run inference on image
        
        Args:
            image: numpy array (H, W, 3) in RGB format
            
        Returns:
            List of detections with bbox, confidence, class
        """
        # Run inference
        results = self.model(image, size=640)
        
        # Parse results
        detections = []
        predictions = results.pandas().xyxy[0]
        
        for _, row in predictions.iterrows():
            detection = {
                'bbox': [row['xmin'], row['ymin'], row['xmax'], row['ymax']],
                'confidence': float(row['confidence']),
                'class': row['name'],
            }
            detections.append(detection)
        
        return detections
    
    def batch_infer(self, images: List[np.ndarray]) -> List[List[Dict]]:
        """Run inference on multiple images"""
        all_detections = []
        
        for image in tqdm(images, desc="Running inference"):
            detections = self.infer(image)
            all_detections.append(detections)
        
        return all_detections


# Test it
if __name__ == "__main__":
    from src.dataset_loader import NuScenesLoader
    
    # Load dataset
    loader = NuScenesLoader(data_root='data/nuscenes', version='v1.0-mini')
    
    # Initialize detector
    detector = ObjectDetector(model_name='yolov5s')
    
    # Run on first 3 samples
    count = 0
    for data in loader.iterate_samples(num_samples=3):
        image = data['image']
        detections = detector.infer(image)
        
        print(f"\nSample: {data['sample_token']}")
        print(f"Ground truth objects: {len(data['objects'])}")
        print(f"Detections: {len(detections)}")
        
        # Show first 3 detections
        for det in detections[:3]:
            print(f"  {det['class']}: {det['confidence']:.2f}")
        
        count += 1
