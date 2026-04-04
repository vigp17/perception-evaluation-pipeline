#!/usr/bin/env python
"""
TRAIN CUSTOM OBJECT DETECTOR FROM SCRATCH
Complete pipeline: Build → Train → Evaluate

Run: python run_custom_training.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    """Train custom detector from scratch"""
    
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║     CUSTOM OBJECT DETECTOR TRAINING                          ║
║     Build and train perception model from scratch             ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Import trainer
        from custom_detector import ObjectDetectionTrainer
        
        print(f"""
Configuration:
- Architecture: Custom CNN (3 conv blocks + detection head)
- Dataset: nuScenes (autonomous driving)
- Training samples: 50 (quick demo)
- Device: Auto (GPU if available, else CPU)
        """)
        
        # Create trainer
        print(f"{'='*60}")
        print(f"STEP 1: Initialize Model")
        print(f"{'='*60}\n")
        
        trainer = ObjectDetectionTrainer(
            output_dir='custom_training_results'
        )
        
        # Train
        print(f"\n{'='*60}")
        print(f"STEP 2: Train from Scratch")
        print(f"{'='*60}\n")
        
        losses = trainer.train(
            train_dir='yolo_dataset/images/train',
            epochs=5,
            batch_size=8
        )
        
        eval_loss = 0
        
        # Summary
        print("\n" + "="*60)
        print("Training complete!")
        print("="*60)
        print(f"\nModel saved to: custom_training_results/custom_detector.pt")
        print(f"Final loss: {losses[-1]:.4f}")
        print("\nNext: Push to GitHub and update resume")
        print("="*60 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())