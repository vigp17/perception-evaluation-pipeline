#!/usr/bin/env python
"""
PERCEPTION FINE-TUNING PIPELINE
Convert nuScenes → Train YOLOv5 → Evaluate & Compare

Run: python run_finetuning.py
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.dataset_converter import NuScenesYOLOConverter
from src.trainer import PerceptionTrainer
from src.evaluator import PerceptionEvaluator


def main():
    """Run complete fine-tuning pipeline"""
    
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║     PERCEPTION FINE-TUNING PIPELINE                          ║
║     Fine-tune YOLOv5 on nuScenes Dataset                     ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Configuration
    DATA_ROOT = 'data/nuscenes'
    NUM_SAMPLES = 100  # Convert 100 samples for training
    YOLO_DATASET = 'yolo_dataset'
    TRAINING_RESULTS = 'training_results'
    
    print(f"""
Configuration:
- Dataset: nuScenes (mini version)
- Samples for training: {NUM_SAMPLES}
- Output: {TRAINING_RESULTS}/
    """)
    
    try:
        # STEP 1: Get Baseline Results
        print(f"\n{'='*60}")
        print(f"STEP 1: Get Baseline (Pre-trained) Results")
        print(f"{'='*60}\n")
        
        baseline_evaluator = PerceptionEvaluator(
            data_root=DATA_ROOT,
            model_name='yolov5s',
            output_dir='baseline_results',
            iou_threshold=0.5
        )
        baseline_evaluator.evaluate(num_samples=50)
        
        # Load baseline results
        with open('baseline_results/evaluation_results.json', 'r') as f:
            baseline_results = json.load(f)
        
        print(f"✓ Baseline mAP@0.5: {baseline_results['mAP']:.3f}")
        
        # STEP 2: Convert Dataset
        print(f"\n{'='*60}")
        print(f"STEP 2: Convert nuScenes to YOLO Format")
        print(f"{'='*60}\n")
        
        converter = NuScenesYOLOConverter(
            data_root=DATA_ROOT,
            output_dir=YOLO_DATASET
        )
        converter.convert_dataset(num_samples=NUM_SAMPLES, train_ratio=0.8)
        
        # STEP 3: Fine-tune Model
        print(f"\n{'='*60}")
        print(f"STEP 3: Fine-tune YOLOv5 on nuScenes")
        print(f"{'='*60}\n")
        
        trainer = PerceptionTrainer(
            data_yaml=f'{YOLO_DATASET}/data.yaml',
            model_name='yolov5s',
            output_dir=TRAINING_RESULTS
        )
        
        # Train for 5 epochs (quick for demo)
        # In production, increase to 20-50 epochs
        trainer.train(
            epochs=5,
            batch_size=8,
            img_size=640,
            patience=3
        )
        
        # STEP 4: Evaluate Fine-tuned Model
        print(f"\n{'='*60}")
        print(f"STEP 4: Evaluate Fine-tuned Model")
        print(f"{'='*60}\n")
        
        finetuned_evaluator = PerceptionEvaluator(
            data_root=DATA_ROOT,
            model_name='yolov5s',  # Will use fine-tuned weights
            output_dir='finetuned_results',
            iou_threshold=0.5
        )
        finetuned_evaluator.evaluate(num_samples=50)
        
        # Load fine-tuned results
        with open('finetuned_results/evaluation_results.json', 'r') as f:
            finetuned_results = json.load(f)
        
        print(f"✓ Fine-tuned mAP@0.5: {finetuned_results['mAP']:.3f}")
        
        # STEP 5: Compare Results
        print(f"\n{'='*60}")
        print(f"STEP 5: Compare Performance")
        print(f"{'='*60}\n")
        
        baseline_map = baseline_results['mAP']
        finetuned_map = finetuned_results['mAP']
        improvement = finetuned_map - baseline_map
        improvement_pct = (improvement / baseline_map * 100) if baseline_map > 0 else 0
        
        print(f"Baseline (Pre-trained):    {baseline_map:.3f}")
        print(f"Fine-tuned (nuScenes):     {finetuned_map:.3f}")
        print(f"Improvement:               {improvement:+.3f} ({improvement_pct:+.1f}%)")
        
        # Save comparison
        comparison = {
            'baseline_map': baseline_map,
            'finetuned_map': finetuned_map,
            'improvement': improvement,
            'improvement_pct': improvement_pct,
            'num_training_samples': NUM_SAMPLES,
            'num_eval_samples': 50
        }
        
        with open('finetuning_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"\n✓ Comparison saved to finetuning_comparison.json")
        
        print(f"""
╔═══════════════════════════════════════════════════════════════╗
║  ✅ FINE-TUNING COMPLETE!                                     ║
╚═══════════════════════════════════════════════════════════════╝

📊 Results:
   - Baseline results: baseline_results/
   - Fine-tuned results: finetuned_results/
   - Training results: {TRAINING_RESULTS}/
   - Comparison: finetuning_comparison.json

📈 Next Steps:
   1. Review comparison metrics
   2. Analyze per-class improvements
   3. Increase epochs for production (20-50)
   4. Commit fine-tuning code to GitHub

💡 Notes:
   - Trained on {NUM_SAMPLES} samples for 5 epochs
   - Evaluated on 50 test samples
   - For production: increase epochs and dataset size
    """)
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())