# Troubleshooting Guide - Perception Pipeline Issues & Solutions

## **BEFORE YOU START**

### Check You Have:
- [ ] Python 3.8+ installed: `python --version`
- [ ] pip works: `pip --version`
- [ ] 20+ GB free disk space: `df -h`
- [ ] Git installed: `git --version`
- [ ] nuScenes downloaded to `~/Downloads/v1.0-mini/` or similar

---

## **COMMON ERRORS & SOLUTIONS**

### **ERROR 1: "ModuleNotFoundError: No module named 'nuscenes'"**

**Cause:** nuscenes-devkit not installed

**Solution:**
```bash
# Make sure venv is activated first!
source venv/bin/activate

# Then install
pip install nuscenes-devkit --upgrade

# Verify
python -c "from nuscenes import NuScenes; print('✓ Works!')"
```

---

### **ERROR 2: "No such file or directory: data/nuscenes"**

**Cause:** Data symlink not created correctly

**Solution:**

Check where your data is:
```bash
ls ~/Downloads/
# You should see: v1.0-mini (or similar)

ls ~/Downloads/v1.0-mini/
# Should contain: maps, samples, sweeps, v1.0-mini
```

**Fix the symlink:**

Mac/Linux:
```bash
# Remove broken symlink
rm -f data/nuscenes

# Create correct one (adjust path to YOUR data location)
ln -s ~/Downloads/v1.0-mini data/nuscenes

# Verify
ls data/nuscenes/maps
# Should show map files
```

Windows (Command Prompt as Admin):
```bash
# Remove broken symlink
rmdir data\nuscenes

# Create correct one (adjust path)
mklink /D data\nuscenes C:\Users\YourName\Downloads\v1.0-mini

# Verify
dir data\nuscenes\maps
```

---

### **ERROR 3: "CUDA out of memory"**

**Cause:** GPU doesn't have enough memory for model

**Solution:**

**Option A: Use CPU instead**
```bash
# Edit evaluator.py, find:
self.detector = ObjectDetector(model_name='yolov5s')

# Change to:
self.detector = ObjectDetector(model_name='yolov5s', device='cpu')
```

**Option B: Use smaller model**
```bash
# Change from yolov5s to yolov5n (nano = tiny)
self.detector = ObjectDetector(model_name='yolov5n')
```

**Option C: Reduce batch size**
In `run_evaluation.py`, evaluate fewer samples first:
```python
evaluator.evaluate(num_samples=5)  # Start tiny
```

**Check GPU:**
```bash
# See if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True (if GPU available)

# See GPU memory
nvidia-smi
```

---

### **ERROR 4: "torch not found" or "ImportError: No module named torch"**

**Cause:** PyTorch not installed

**Solution:**
```bash
# Reinstall PyTorch
pip install torch torchvision --upgrade

# Verify
python -c "import torch; print(torch.__version__)"
```

---

### **ERROR 5: "FileNotFoundError: [Errno 2] No such file or directory: 'samples/...'"**

**Cause:** nuScenes data corrupted or incomplete

**Solution:**

**Check data integrity:**
```bash
ls data/nuscenes/v1.0-mini/
# Should show: metadata, sample_data, sample_annotation, etc.

# If files are there but error persists:
# Download fresh copy
```

**Re-download:**
- Go to https://www.nuscenes.org/download
- Download **v1.0-mini** fresh
- Extract to new folder
- Update symlink to point to new location

---

### **ERROR 6: "Expected 'CAM_FRONT' in sample_data"**

**Cause:** nuScenes version mismatch or corrupted data

**Solution:**
```bash
# Verify you have v1.0-mini (not v1.0-trainval or other)
ls data/nuscenes/
# Check folder name

# If it says "v1.0-trainval", download v1.0-mini instead
```

---

### **ERROR 7: "No predictions found" or "Empty results"**

**Cause:** Dataset loaded but no objects detected or very low confidence

**Solution:**

This is actually OK! It means:
- Dataset loaded correctly ✓
- Model ran successfully ✓
- Just no detections in those samples

Try:
```python
# In run_evaluation.py, increase samples:
evaluator.evaluate(num_samples=50)

# More samples = more likely to find objects
```

---

### **ERROR 8: "requirements.txt not found"**

**Cause:** File wasn't created properly

**Solution:**

Create it manually:
```bash
cat > requirements.txt << 'EOF'
numpy==1.23.5
pandas==1.5.3
matplotlib==3.7.1
opencv-python==4.7.0.72
torch==2.0.0
torchvision==0.15.0
yolov5==7.0.10
nuscenes-devkit==1.1.9
scipy==1.10.1
tqdm==4.65.0
Pillow==9.5.0
scikit-learn==1.2.2
EOF

# Then install
pip install -r requirements.txt
```

---

### **ERROR 9: "permission denied: './run_evaluation.py'"**

**Cause:** File not executable

**Solution:**
```bash
# Make it executable (Mac/Linux)
chmod +x run_evaluation.py

# Or just run with python
python run_evaluation.py
```

---

### **ERROR 10: "No space left on device"**

**Cause:** Disk full

**Solution:**
```bash
# Check disk space
df -h

# Free up space:
# - Delete old files/projects
# - Empty trash
# - Delete large downloads

# Need ~20GB minimum
```

---

## **PERFORMANCE ISSUES**

### **It's Running Very Slowly**

**Check 1: Which device?**
```python
# In evaluator.py, add print statement:
print(f"Using: {self.detector.device}")

# If showing 'cpu', it's much slower
# Use 'cuda' if you have GPU
```

**Check 2: Sample count**
```python
# In run_evaluation.py
evaluator.evaluate(num_samples=10)  # Start small

# Time estimate:
# 10 samples on GPU: 1-2 minutes
# 10 samples on CPU: 5-10 minutes
```

**Check 3: System load**
```bash
# Check CPU/GPU usage
# Mac: Activity Monitor
# Linux: top or htop
# Windows: Task Manager
```

---

## **TESTING INDIVIDUAL COMPONENTS**

### **Test Data Loading**
```bash
python -c "
from src.dataset_loader import NuScenesLoader
loader = NuScenesLoader('data/nuscenes')
print(f'✓ Loaded {len(loader.nusc.scene)} scenes')
"
```

### **Test Model**
```bash
python -c "
from src.model_inference import ObjectDetector
detector = ObjectDetector(model_name='yolov5s')
print('✓ Model loaded')
"
```

### **Test Metrics**
```bash
python -c "
from src.metrics import compute_iou
iou = compute_iou([0,0,10,10], [5,5,15,15])
print(f'✓ IoU test: {iou:.3f}')
"
```

### **Test Evaluator**
```bash
python -c "
from src.evaluator import PerceptionEvaluator
evaluator = PerceptionEvaluator()
print('✓ Evaluator initialized')
"
```

---

## **DEBUGGING TIPS**

### **Add Print Statements**

In `evaluator.py`, add debug info:
```python
def evaluate(self, num_samples: int = 50):
    """Run evaluation"""
    print(f"\n{'='*60}")
    print(f"Starting evaluation...")
    print(f"{'='*60}\n")
    
    for i, sample_data in enumerate(self.loader.iterate_samples(num_samples=num_samples)):
        print(f"[{i+1}/{num_samples}] Processing sample...")
        
        predictions = self.detector.infer(sample_data['image'])
        print(f"  Predictions: {len(predictions)}")
        
        ground_truths = sample_data['objects']
        print(f"  Ground truth: {len(ground_truths)}")
        
        # ... rest of code
```

### **Check Intermediate Files**

```bash
# After first run, check results:
ls -la results/
cat results/evaluation_results.json
ls -la results/visualizations/
```

### **Use Python Interactive Mode**

```bash
python

# Then in Python:
from src.dataset_loader import NuScenesLoader
loader = NuScenesLoader('data/nuscenes')

# Load one sample manually
data = next(loader.iterate_samples(num_samples=1))
print(data.keys())
print(f"Image shape: {data['image'].shape}")
print(f"Objects: {len(data['objects'])}")

# Exit
exit()
```

---

## **WHEN EVERYTHING FAILS**

### **Nuclear Option: Start Fresh**

```bash
# Remove everything
rm -rf venv results data/nuscenes

# Start over
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Re-create symlink
ln -s ~/Downloads/v1.0-mini data/nuscenes

# Try again
python run_evaluation.py
```

### **Check Basics**

```bash
# 1. Python version
python --version  # Should be 3.8+

# 2. Virtual environment
which python  # Should show path inside venv

# 3. Packages
pip list | grep -E "torch|nuscenes|yolov5"

# 4. Data
ls -la data/nuscenes/v1.0-mini/

# 5. Code
ls -la src/
```

---

## **GET HELP**

If you're still stuck:

1. **Read the error message carefully** - it tells you exactly what's wrong
2. **Google the error** - someone has probably had it
3. **Check file paths** - most errors are "file not found"
4. **Verify installations** - `pip list` to see what's actually installed
5. **Ask Claude** - I can help debug!

---

## **SUCCESS INDICATORS**

You know it's working when:

✅ All imports work (no ModuleNotFoundError)
✅ Data loads (prints scene/sample counts)
✅ Model loads (prints "Loaded YOLOv5")
✅ Inference runs (shows progress bars)
✅ Results saved (JSON file created)
✅ Plot generated (PNG file created)

---

**Remember: Most issues are simple - usually just missing data or wrong paths!**

Start simple, test each part, then combine them. 💪

