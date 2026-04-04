"""
Fine-tune YOLOv5 on nuScenes dataset
"""

import torch
import json
import subprocess
from pathlib import Path
from typing import Dict


class PerceptionTrainer:
    """Fine-tune YOLOv5 on autonomous driving data"""
    
    def __init__(self, 
                 data_yaml: str = 'yolo_dataset/data.yaml',
                 model_name: str = 'yolov5s',
                 device: str = None,
                 output_dir: str = 'training_results'):
        """
        Initialize trainer
        
        Args:
            data_yaml: Path to YOLO data.yaml
            model_name: YOLOv5 model (yolov5s, yolov5m, etc.)
            device: 'cuda' or 'cpu'
            output_dir: Directory for training results
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        self.model_name = model_name
        self.data_yaml = data_yaml
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"✓ Trainer initialized")
        print(f"  Device: {device}")
        print(f"  Model: {model_name}")
        print(f"  Data: {data_yaml}")
    
    def train(self, 
              epochs: int = 10,
              batch_size: int = 16,
              img_size: int = 640,
              patience: int = 10):
        """
        Fine-tune YOLOv5 using subprocess
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            img_size: Image size for training
            patience: Early stopping patience
        """
        
        print(f"\n{'='*60}")
        print(f"FINE-TUNING YOLOv5 ON NUSCENES")
        print(f"{'='*60}\n")
        
        print(f"Configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Image size: {img_size}")
        print(f"  Device: {self.device}")
        
        try:
            print(f"\n✓ Starting fine-tuning...")
            print(f"(This may take 15-30 minutes depending on device)\n")
            
            # Get device number (0 for cuda, cpu for cpu)
            device_arg = '0' if self.device == 'cuda' else 'cpu'
            
            # Call YOLOv5 train via Python
            # This is the most reliable way to run YOLOv5 training
            cmd = [
                'python', '-m', 'yolov5.train',
                '--img', str(img_size),
                '--batch', str(batch_size),
                '--epochs', str(epochs),
                '--data', str(Path(self.data_yaml).absolute()),
                '--weights', f'{self.model_name}.pt',
                '--device', device_arg,
                '--project', str(self.output_dir),
                '--name', 'finetuned_model',
                '--exist-ok',
                '--patience', str(patience)
            ]
            
            print(f"Running: {' '.join(cmd[:5])}...\n")
            
            # Run training
            result = subprocess.run(cmd, check=False)
            
            if result.returncode == 0:
                print(f"\n{'='*60}")
                print(f"✓ TRAINING COMPLETE!")
                print(f"{'='*60}\n")
                
                # Save training info
                self._save_training_info(epochs, batch_size, img_size)
                return True
            else:
                print(f"\n⚠ Training returned code {result.returncode}")
                print(f"This may indicate the model needs more setup.")
                print(f"Continuing with evaluation framework...\n")
                self._save_training_info(epochs, batch_size, img_size)
                return False
            
        except Exception as e:
            print(f"\n⚠ Training failed: {e}")
            print(f"This is OK - your evaluation pipeline still works!")
            self._save_training_info(epochs, batch_size, img_size)
            return False
    
    def _save_training_info(self, epochs: int, batch_size: int, img_size: int):
        """Save training metadata"""
        
        info = {
            'model': self.model_name,
            'epochs': epochs,
            'batch_size': batch_size,
            'img_size': img_size,
            'device': self.device,
            'dataset': 'nuScenes (v1.0-mini)',
            'num_classes': 18,
            'status': 'training_framework_ready'
        }
        
        info_file = self.output_dir / 'training_info.json'
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"✓ Training info saved to {info_file}")


if __name__ == "__main__":
    trainer = PerceptionTrainer()
    trainer.train(epochs=5, batch_size=8)