"""
Main Perception Evaluation Pipeline
Orchestrate loading, inference, and metrics
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

# Assumes these files are in the same directory or in path
from src.dataset_loader import NuScenesLoader
from src.model_inference import ObjectDetector
from src.metrics import compute_iou, compute_map


class PerceptionEvaluator:
    """Complete evaluation pipeline"""
    
    def __init__(self, 
                 data_root: str = 'data/nuscenes',
                 model_name: str = 'yolov5s',
                 output_dir: str = 'results',
                 iou_threshold: float = 0.5):
        """
        Initialize evaluator
        
        Args:
            data_root: Path to nuScenes data
            model_name: YOLOv5 model name
            output_dir: Directory for results
            iou_threshold: IoU threshold for matching
        """
        self.data_root = data_root
        self.model_name = model_name
        self.iou_threshold = iou_threshold
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        print("\n" + "="*60)
        print("PERCEPTION EVALUATION PIPELINE")
        print("="*60)
        
        print("\n[1/2] Loading dataset...")
        self.loader = NuScenesLoader(data_root, version='v1.0-mini')
        
        print("\n[2/2] Loading model...")
        self.detector = ObjectDetector(model_name=model_name)
        
        # Metrics storage
        self.class_predictions = defaultdict(list)
        self.class_ground_truths = defaultdict(list)
        self.per_class_ap = {}
        self.mAP = 0.0
        self.num_samples = 0
    
    def evaluate(self, num_samples: int = 50):
        """
        Run evaluation on dataset
        
        Args:
            num_samples: Number of samples to evaluate
        """
        print(f"\n{'='*60}")
        print(f"Evaluating on {num_samples} samples")
        print(f"{'='*60}\n")
        
        # Load and evaluate
        for sample_data in self.loader.iterate_samples(num_samples=num_samples):
            # Get predictions
            predictions = self.detector.infer(sample_data['image'])
            
            # Get ground truths
            ground_truths = sample_data['objects']
            
            # Map both label spaces to the shared canonical taxonomy.
            # Without this, COCO names ('car') never match nuScenes names
            # ('vehicle.car') and every class scores AP = 0.
            from src.class_mapping import (
                filter_and_map_predictions,
                filter_and_map_ground_truths,
            )
            predictions = filter_and_map_predictions(predictions)
            ground_truths = filter_and_map_ground_truths(ground_truths)

            # Organize by class
            for pred in predictions:
                self.class_predictions[pred['class']].append(pred)
            
            for gt in ground_truths:
                self.class_ground_truths[gt['class']].append(gt)
            
            self.num_samples += 1
        
        # Compute metrics
        self._compute_metrics()
        self._print_results()
        self._save_results()
        self._generate_plots()
    
    def _compute_metrics(self):
        """Compute mAP and per-class AP"""
        self.mAP, self.per_class_ap = compute_map(
            self.class_predictions,
            self.class_ground_truths,
            iou_threshold=self.iou_threshold
        )
    
    def _print_results(self):
        """Print evaluation results"""
        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}\n")
        
        print(f"Samples evaluated: {self.num_samples}")
        print(f"IoU threshold: {self.iou_threshold}")
        print(f"mAP@{self.iou_threshold}: {self.mAP:.3f}\n")
        
        print(f"{'Class':<20} {'AP':<10}")
        print("-" * 30)
        
        for class_name in sorted(self.per_class_ap.keys()):
            ap = self.per_class_ap[class_name]
            print(f"{class_name:<20} {ap:.3f}")
        
        print(f"\n{'='*60}\n")
    
    def _save_results(self):
        """Save results to JSON"""
        results = {
            'mAP': float(self.mAP),
            'iou_threshold': self.iou_threshold,
            'num_samples': self.num_samples,
            'per_class_ap': {k: float(v) for k, v in self.per_class_ap.items()}
        }
        
        filepath = self.output_dir / 'evaluation_results.json'
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Results saved: {filepath}")
    
    def _generate_plots(self):
        """Generate visualization plots"""
        if not self.per_class_ap:
            print("⚠ No results to plot")
            return
        
        classes = sorted(self.per_class_ap.keys())
        aps = [self.per_class_ap[c] for c in classes]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.bar(classes, aps, color='steelblue', edgecolor='navy', alpha=0.7)
        ax.axhline(y=self.mAP, color='red', linestyle='--', linewidth=2,
                   label=f'mAP={self.mAP:.3f}')
        
        ax.set_xlabel('Object Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Precision', fontsize=12, fontweight='bold')
        ax.set_title(f'Per-Class Performance (IoU={self.iou_threshold})', 
                    fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save
        viz_dir = self.output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        filepath = viz_dir / 'per_class_ap.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved: {filepath}")
        
        plt.close()


# Main execution
if __name__ == "__main__":
    evaluator = PerceptionEvaluator(
        data_root='data/nuscenes',
        model_name='yolov5s',
        output_dir='results',
        iou_threshold=0.5
    )
    
    # Run evaluation
    # Start with 10 for quick test, then 50, then 200+
    evaluator.evaluate(num_samples=10)
