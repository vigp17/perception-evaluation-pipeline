#!/usr/bin/env python
"""
PERCEPTION EVALUATION PIPELINE - Main Script
Run: python run_evaluation.py
"""

import sys
from pathlib import Path

# Add current directory to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from src.evaluator import PerceptionEvaluator


def main():
    """Run the complete evaluation pipeline"""
    
    print("\n" + "="*60)
    print("Perception Evaluation Pipeline")
    print("="*60)
    
    # Configuration
    DATA_ROOT = 'data/nuscenes'
    MODEL_NAME = 'yolov5s'
    NUM_SAMPLES = 10  # Start with 10, increase later
    IOU_THRESHOLD = 0.5
    OUTPUT_DIR = 'results'
    
    print(f"""
Configuration:
- Dataset: nuScenes (mini version)
- Model: {MODEL_NAME}
- Samples: {NUM_SAMPLES}
- IoU Threshold: {IOU_THRESHOLD}
- Output: {OUTPUT_DIR}/
    """)
    
    try:
        # Create evaluator
        evaluator = PerceptionEvaluator(
            data_root=DATA_ROOT,
            model_name=MODEL_NAME,
            output_dir=OUTPUT_DIR,
            iou_threshold=IOU_THRESHOLD
        )
        
        # Run evaluation
        evaluator.evaluate(num_samples=NUM_SAMPLES)
        
        print("\n" + "="*60)
        print("Evaluation complete")
        print("="*60)
        print("\nResults saved to:")
        print("  - Metrics: results/evaluation_results.json")
        print("  - Plot: results/visualizations/per_class_ap.png")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check that data/nuscenes/ exists")
        print("2. Verify dependencies: pip install -r requirements.txt")
        print("3. Check GPU availability: nvidia-smi")
        print("4. Try CPU mode: Change device to 'cpu' in evaluator")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())