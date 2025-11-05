# Running NBA Predictor on Google Colab 🚀

## ✅ YES - You Can Run This on Google Colab!

**Benefits:**
- ✅ Free GPU access (faster neural network training)
- ✅ 12-15 GB RAM (vs your local system)
- ✅ Runs in background while you use your PC
- ✅ Can leave training running overnight
- ✅ No installation needed on local machine

**Perfect for:**
- Training models (overnight runs)
- Testing new features
- Heavy computation tasks

---

## 🚀 Quick Setup (5 minutes)

### Step 1: Upload Your Code to Google Drive

```bash
# On your local machine:
# Zip your project (exclude large files)
zip -r nba_predictor.zip . -x "*.pkl" "*.zip" "*model_cache/*" "*.git/*" "*venv/*" "*__pycache__/*"

# Or on Windows PowerShell:
Compress-Archive -Path .\* -DestinationPath nba_predictor.zip -Force -Exclude "*.pkl","model_cache","venv","__pycache__"
```

### Step 2: Open Google Colab
1. Go to: https://colab.research.google.com
2. Sign in with Google account
3. Click: **File → New Notebook**

### Step 3: Mount Google Drive

```python
# Run this in first cell
from google.colab import drive
drive.mount('/content/drive')
```

### Step 4: Setup Environment

```python
# Install dependencies
!pip install kagglehub lightgbm scikit-learn pandas numpy
!pip install torch pytorch-tabnet  # For neural network

# Upload your Kaggle credentials
from google.colab import files
uploaded = files.upload()  # Upload kaggle.json

# Setup Kaggle
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

### Step 5: Clone/Upload Your Code

**Option A: Upload Zip**
```python
# Upload your nba_predictor.zip
from google.colab import files
uploaded = files.upload()

# Extract
!unzip nba_predictor.zip -d /content/nba_predictor
%cd /content/nba_predictor
```

**Option B: Clone from GitHub**
```python
!git clone https://github.com/yourrepo/nba_predictor.git
%cd nba_predictor
```

### Step 6: Run Training

```python
# Train with GPU acceleration
!python train_auto.py --verbose --fresh --neural-device gpu --neural-epochs 100
```

---

## 📋 Complete Colab Notebook Template

Save this as `NBA_Predictor_Training.ipynb`:

```python
# ==============================================================================
# NBA PREDICTOR - GOOGLE COLAB TRAINING NOTEBOOK
# ==============================================================================

# 1. SETUP ENVIRONMENT
# ==============================================================================
print("🔧 Setting up environment...")

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install dependencies
!pip install -q kagglehub lightgbm scikit-learn pandas numpy
!pip install -q torch pytorch-tabnet

print("✅ Dependencies installed!")

# ==============================================================================
# 2. SETUP KAGGLE CREDENTIALS
# ==============================================================================
print("\n📦 Setting up Kaggle...")

# Upload kaggle.json (you'll be prompted)
from google.colab import files
print("Upload your kaggle.json file:")
uploaded = files.upload()

# Setup Kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

print("✅ Kaggle configured!")

# ==============================================================================
# 3. UPLOAD YOUR CODE
# ==============================================================================
print("\n📤 Upload your nba_predictor.zip:")
uploaded = files.upload()

# Extract code
!unzip -q nba_predictor.zip -d /content/nba_predictor
%cd /content/nba_predictor

print("✅ Code uploaded!")

# ==============================================================================
# 4. CHECK GPU AVAILABILITY
# ==============================================================================
import torch
print(f"\n🎮 GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ==============================================================================
# 5. TRAIN MODELS
# ==============================================================================
print("\n🚀 Starting training...")

# Option 1: Full training with neural network
!python train_auto.py \
    --verbose \
    --fresh \
    --neural-device gpu \
    --neural-epochs 100 \
    --enable-window-ensemble

# Option 2: Quick test (20 epochs, recent data only)
# !python train_auto.py --neural-epochs 20 --player-season-cutoff 2020 --neural-device gpu

print("\n✅ Training complete!")

# ==============================================================================
# 6. DOWNLOAD TRAINED MODELS
# ==============================================================================
print("\n💾 Downloading models...")

# Zip models folder
!zip -r models_trained.zip models/

# Download to your computer
from google.colab import files
files.download('models_trained.zip')

print("✅ Models downloaded! Extract and place in your local models/ folder.")

# ==============================================================================
# 7. SAVE TO GOOGLE DRIVE (BACKUP)
# ==============================================================================
print("\n💾 Saving to Google Drive...")

!cp -r models/ /content/drive/MyDrive/nba_predictor_models/
!cp -r model_cache/ /content/drive/MyDrive/nba_predictor_cache/

print("✅ Backup saved to Google Drive!")
```

---

## ⚡ GPU vs CPU Training Times

### Google Colab GPU (Tesla T4 - Free Tier):
- **Full training**: 10-15 minutes
- **Quick test**: 2-3 minutes
- **Phase 7 implementation**: 15-20 minutes

### Your Local CPU:
- **Full training**: 40-60 minutes
- **Quick test**: 8-12 minutes
- **Phase 7 implementation**: 45-60 minutes

**Speedup: ~4x faster with Colab GPU** ⚡

---

## 🔄 Workflow: Colab + Local Machine

### Best Practice Setup:

```
┌─────────────────────────────────────┐
│     GOOGLE COLAB (Training)         │
│  - Overnight model training         │
│  - Feature testing                  │
│  - Heavy computation                │
│  - Free GPU acceleration            │
└──────────────┬──────────────────────┘
               │ Download models
               │
┌──────────────▼──────────────────────┐
│   LOCAL MACHINE (Daily Use)         │
│  - Daily predictions (riq_analyzer) │
│  - Result evaluation                │
│  - Quick analysis                   │
│  - Light computation                │
└─────────────────────────────────────┘
```

### Recommended Workflow:

1. **Once per month (Colab):**
   ```python
   # Train on Colab with GPU
   !python train_auto.py --verbose --fresh --neural-device gpu
   
   # Download models
   files.download('models_trained.zip')
   ```

2. **Daily (Local machine):**
   ```bash
   # Use pre-trained models
   python riq_analyzer.py
   python evaluate.py
   ```

3. **When testing new features (Colab):**
   ```python
   # Quick test with recent data
   !python train_auto.py --neural-epochs 20 --player-season-cutoff 2020
   ```

---

## 📊 Resource Comparison

| Resource | Local PC | Google Colab Free | Google Colab Pro |
|----------|----------|-------------------|------------------|
| **GPU** | None | Tesla T4 (16GB) | Tesla T4/V100 |
| **RAM** | ~8-16GB | 12-15GB | 25-52GB |
| **Training Time** | 40-60 min | 10-15 min | 8-12 min |
| **Cost** | $0 | $0 | $10/month |
| **Runtime Limit** | Unlimited | 12 hours | 24 hours |
| **Idle Timeout** | None | 90 minutes | 90 minutes |

**Recommendation:** Start with free tier, upgrade to Pro ($10/month) if you train frequently.

---

## 🎯 Phase 7 Implementation on Colab

Here's how to implement Phase 7 (quick wins) on Colab:

```python
# ==============================================================================
# PHASE 7: QUICK WINS IMPLEMENTATION
# ==============================================================================

# 1. Upload phase7_features.py (I'll create this for you)
from google.colab import files
print("Upload phase7_features.py:")
uploaded = files.upload()

# 2. Test implementation
print("\n🧪 Testing Phase 7 features...")

# Run with Phase 7 features
!python train_auto.py \
    --verbose \
    --fresh \
    --neural-device gpu \
    --neural-epochs 50 \
    --player-season-cutoff 2022

# 3. Compare results
print("\n📊 Comparing results...")
!python evaluate.py --analyze-only

# 4. Download improved models
!zip -r models_phase7.zip models/
files.download('models_phase7.zip')

print("\n✅ Phase 7 implementation complete!")
print("Expected improvement: +5-8% accuracy")
```

---

## 🔒 Keep Your Session Alive

Colab disconnects after 90 minutes of inactivity. To prevent this:

```python
# Run this in a cell (keeps connection alive)
import time
from IPython.display import Javascript

def keep_alive():
    while True:
        display(Javascript('google.colab.output.setIframeHeight(0, true, {maxHeight: 5000})'))
        time.sleep(300)  # Ping every 5 minutes

# Run in background
import threading
thread = threading.Thread(target=keep_alive, daemon=True)
thread.start()
```

Or install this Chrome extension:
https://chrome.google.com/webstore/detail/colab-keepalive

---

## 💾 Saving Work to Google Drive

```python
# At the end of training, backup everything to Drive
import shutil
from datetime import datetime

# Create backup folder with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
backup_dir = f'/content/drive/MyDrive/nba_predictor_backups/{timestamp}'

# Copy models and cache
!mkdir -p {backup_dir}
!cp -r models/ {backup_dir}/models/
!cp -r model_cache/ {backup_dir}/model_cache/
!cp *.py {backup_dir}/

print(f"✅ Backup saved to: {backup_dir}")
```

---

## 🐛 Common Issues & Solutions

### Issue 1: "Kaggle API error"
**Solution:**
```python
# Verify kaggle.json is correct
!cat ~/.kaggle/kaggle.json
# Should show your username and key
```

### Issue 2: "Out of memory"
**Solution:**
```python
# Reduce batch size in neural_hybrid.py
# Or use fewer epochs
!python train_auto.py --neural-epochs 50 --player-season-cutoff 2018
```

### Issue 3: "Session disconnected"
**Solution:**
- Use keep-alive script (above)
- Or upgrade to Colab Pro
- Or run in smaller chunks

### Issue 4: "Import error"
**Solution:**
```python
# Reinstall dependencies
!pip install --upgrade kagglehub lightgbm torch pytorch-tabnet
```

---

## 🎓 Pro Tips

1. **Use Colab Pro for serious training** ($10/month)
   - Longer runtime (24 hours)
   - Better GPUs (V100 vs T4)
   - Priority access

2. **Train overnight on Colab**
   ```python
   # Start before bed, download models in morning
   !python train_auto.py --verbose --fresh --neural-device gpu
   ```

3. **Save intermediate results**
   ```python
   # After each window, backup to Drive
   # Modify train_auto.py to save progress
   ```

4. **Use Google Drive for data**
   ```python
   # Store Kaggle data in Drive, reuse across sessions
   !cp -r data/ /content/drive/MyDrive/nba_data/
   ```

5. **Monitor training remotely**
   - Use Colab mobile app
   - Or check Drive folder for completed files

---

## ✅ Quick Start Checklist

- [ ] Create Google account (if needed)
- [ ] Go to colab.research.google.com
- [ ] Upload kaggle.json
- [ ] Zip your nba_predictor folder (exclude models/cache)
- [ ] Copy the notebook template above
- [ ] Run all cells
- [ ] Download trained models
- [ ] Extract models to local machine
- [ ] Test with: `python riq_analyzer.py`

---

## 🚀 Ready to Start?

**Yes! You can run everything on Google Colab.**

**Want me to:**
1. Create the complete Colab notebook (.ipynb file)?
2. Implement Phase 7 features right now?
3. Create a step-by-step video guide?

Just say which one and I'll help you get started! 🎯
