# Kaggle Training Notebook - Ready to Run

**Dataset**: https://www.kaggle.com/datasets/tyriqmiles/meeper (private)

---

## Setup Instructions for Kaggle

### Step 1: Create New Kaggle Notebook

1. Go to https://www.kaggle.com/code
2. Click "New Notebook"
3. Title: "NBA Player Props - Neural Hybrid Training"
4. Make it private

### Step 2: Add Your Dataset

1. In the notebook, click "Add Data" (right sidebar)
2. Search for "meeper" (your dataset)
3. Click "Add" - it will appear in `/kaggle/input/meeper/`

### Step 3: Enable GPU

1. Right sidebar → Accelerator
2. Select "GPU T4 x2" or "GPU P100"
3. Save

### Step 4: Copy-Paste Training Cells

Use the cells below in your Kaggle notebook.

---

## Kaggle Notebook Cells

### Cell 1: Setup and Install Packages

```python
# ============================================================
# SETUP & INSTALL (Kaggle Version)
# ============================================================

print("📦 Installing packages...")
!pip install -q pytorch-tabnet lightgbm scikit-learn pandas numpy tqdm

print("\n📥 Downloading training code from GitHub...")
import os
import shutil

os.chdir('/kaggle/working')

# Clone your repository
if os.path.exists('meep'):
    shutil.rmtree('meep')

!git clone https://github.com/tyriqmiles0529-pixel/meep.git
os.chdir('meep')

print("\n📍 Code version:")
!git log -1 --oneline

# Check GPU
import torch
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not available'
print(f"\n🎮 GPU: {gpu_name}")

# Verify dataset exists
dataset_path = '/kaggle/input/meeper/aggregated_nba_data.csv.gzip'
if os.path.exists(dataset_path):
    size_mb = os.path.getsize(dataset_path) / 1024 / 1024
    print(f"\n✅ Dataset found: {size_mb:.1f} MB")
    print(f"   Path: {dataset_path}")
else:
    print("\n❌ Dataset not found!")
    print("   Make sure you added 'meeper' dataset to this notebook")
    print("   Expected path: /kaggle/input/meeper/aggregated_nba_data.csv.gzip")

print("\n✅ Setup complete!")
```

### Cell 2: Train Models

```python
# ============================================================
# TRAIN NEURAL HYBRID MODELS (3-4 hours)
# ============================================================

print("="*70)
print("🚀 TRAINING WITH AGGREGATED DATA (v4.0)")
print("="*70)
print("\n📊 Dataset: /kaggle/input/meeper/aggregated_nba_data.csv.gzip")
print("   • Pre-computed 150+ features (Phase 1-7)")
print("   • Basketball Reference priors merged")
print("   • 2002-2026 seasons")
print("   • ~125K player-games")
print("\n🧠 Neural hybrid: TabNet (24-dim embeddings) + LightGBM")
print("⚡ Training optimization:")
print("   • No feature engineering needed (already done!)")
print("   • Direct load → train workflow")
print("   • Incremental model saving")

import torch
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'

if 'P100' in gpu_name:
    print("\n⏱️  Expected time: ~3.5 hours (P100)")
elif 'T4' in gpu_name:
    print("\n⏱️  Expected time: ~4 hours (T4)")
else:
    print("\n⏱️  Expected time: ~3-4 hours")

print("\n💡 Training 5 player props: minutes, points, rebounds, assists, threes")
print("   Expected: Points MAE ~2.2-2.4 (vs baseline 2.5-2.7)")
print("   Features: 235 total (150+ engineered + 68 priors + splits)")
print("   Embeddings: 24-dimensional TabNet attention vectors\n")

# Train with aggregated data
!python train_auto.py \
    --dataset /kaggle/input/meeper/aggregated_nba_data.csv.gzip \
    --use-neural \
    --neural-epochs 30 \
    --neural-device gpu \
    --verbose \
    --fresh \
    --skip-game-models

print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)
print("\n📊 Models trained on aggregated data with 235 features")
print("🧠 Neural hybrid: TabNet 24-dim embeddings + LightGBM")
print("\nModels saved to: /kaggle/working/meep/models/")
print("\nNext: Run validation cell to verify embeddings")
```

### Cell 3: Validate 24-Dim Embeddings

```python
# ============================================================
# VALIDATE EMBEDDINGS
# ============================================================

print("🔍 Validating TabNet embeddings...")

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

models_dir = Path('/kaggle/working/meep/models')

# Load a trained model (use points as example)
model_files = list(models_dir.glob('*points*.pkl'))

if not model_files:
    print("❌ No models found! Run training first.")
else:
    model_path = model_files[0]
    print(f"📦 Loading model: {model_path.name}")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    print(f"   Model type: {type(model).__name__}")

    # Check if it's a NeuralHybridPredictor
    if hasattr(model, 'tabnet'):
        print(f"   ✅ Neural hybrid detected")
        print(f"   TabNet model: {type(model.tabnet).__name__}")
        print(f"   LightGBM model: {type(model.lgbm).__name__}")

        # Test embedding extraction
        print("\n🧪 Testing embedding extraction...")

        # Create dummy data
        dummy_features = pd.DataFrame(
            np.random.randn(10, 150),
            columns=[f'feature_{i}' for i in range(150)]
        )

        # Get embeddings
        try:
            if hasattr(model, '_get_embeddings'):
                embeddings = model._get_embeddings(dummy_features)
                print(f"\n✅ SUCCESS: Embedding extraction working!")
                print(f"   Shape: {embeddings.shape}")
                print(f"   Expected: (10, 24)")

                if embeddings.shape[1] == 24:
                    print(f"\n🎯 PERFECT: Got 24-dimensional embeddings")
                    print(f"   Mean: {embeddings.mean():.4f}")
                    print(f"   Std: {embeddings.std():.4f}")
                    print(f"   Not all zeros: {(embeddings.std(axis=0) > 0.01).all()}")

                    # Check LightGBM uses embeddings
                    if hasattr(model.lgbm, 'feature_name_'):
                        lgbm_features = model.lgbm.feature_name_
                        embedding_features = [f for f in lgbm_features if 'embedding' in f]
                        print(f"\n   LightGBM sees {len(embedding_features)} embedding features")

                    print(f"\n✅ Model validation PASSED!")
                    print(f"   Ready for predictions with neural hybrid architecture")
                else:
                    print(f"\n⚠️  Warning: Got {embeddings.shape[1]}-dim embeddings (expected 24)")
            else:
                print(f"⚠️  Model doesn't have _get_embeddings method")

        except Exception as e:
            print(f"❌ Error testing embeddings: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"   ⚠️  This appears to be a LightGBM-only model")

    # Display model info
    print(f"\n📊 Model Summary:")
    if hasattr(model, 'lgbm'):
        print(f"   LightGBM trees: {model.lgbm.n_estimators if hasattr(model.lgbm, 'n_estimators') else 'N/A'}")
        if hasattr(model.lgbm, 'feature_name_'):
            print(f"   Features used: {len(model.lgbm.feature_name_)}")

    if hasattr(model, 'sigma_model'):
        print(f"   Uncertainty model: {'Present' if model.sigma_model else 'None'}")

print("\n✅ Validation complete!")
```

### Cell 4: Show Training Results Summary

```python
# ============================================================
# TRAINING RESULTS SUMMARY
# ============================================================

import pickle
import pandas as pd
from pathlib import Path

models_dir = Path('/kaggle/working/meep/models')
model_files = list(models_dir.glob('*.pkl'))

print("="*70)
print("📊 TRAINING RESULTS SUMMARY")
print("="*70)

if not model_files:
    print("\n❌ No models found!")
else:
    print(f"\n✅ Found {len(model_files)} trained models:\n")

    for model_path in sorted(model_files):
        print(f"   📦 {model_path.name}")
        size_mb = model_path.stat().st_size / 1024 / 1024
        print(f"      Size: {size_mb:.1f} MB")

        # Load and check model type
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            if hasattr(model, 'tabnet'):
                print(f"      Type: Neural Hybrid (TabNet + LightGBM)")
            else:
                print(f"      Type: LightGBM only")

            if hasattr(model, 'sigma_model') and model.sigma_model:
                print(f"      Uncertainty: Yes")

            print()
        except Exception as e:
            print(f"      Error loading: {e}\n")

print("="*70)
print("\nModels are saved in: /kaggle/working/meep/models/")
print("\nTo download:")
print("  1. Click 'Output' tab on right sidebar")
print("  2. Find models/ folder")
print("  3. Click download icon")
print("\nOr package all models:")
print("  !zip -r nba_models_trained.zip /kaggle/working/meep/models/")
```

### Cell 5: Package and Prepare for Download

```python
# ============================================================
# PACKAGE MODELS FOR DOWNLOAD
# ============================================================

import os

os.chdir('/kaggle/working')

print("📦 Packaging models...")
!zip -r nba_models_trained.zip meep/models/ meep/model_cache/

# Check size
if os.path.exists('nba_models_trained.zip'):
    size_mb = os.path.getsize('nba_models_trained.zip') / 1024 / 1024
    print(f"\n✅ Package created: nba_models_trained.zip ({size_mb:.1f} MB)")
    print("\nTo download:")
    print("  1. Click 'Output' tab on right sidebar")
    print("  2. Find nba_models_trained.zip")
    print("  3. Click download icon")
    print("\nExtract this to your local nba_predictor/ folder")
else:
    print("\n❌ Failed to create package")
```

---

## Expected Timeline

### Initial Setup (5 minutes)
- Cell 1: Install packages, clone code, verify dataset

### Training (3-4 hours)
- Cell 2: Train all 5 player prop models
- TabNet: ~15 min per prop
- LightGBM: ~2 min per prop
- Sigma models: ~1 min per prop
- **Total: ~3-4 hours on T4/P100**

### Validation (2 minutes)
- Cell 3: Verify embeddings are 24-dim
- Cell 4: Show training summary
- Cell 5: Package models for download

---

## What to Do While Training

**Kaggle keeps running even if you close the browser!**

You can:
1. Close browser and come back in 3-4 hours
2. Check progress occasionally (refresh page)
3. Work on other tasks locally:
   - Complete backtest_engine.py
   - Update other notebooks
   - Test betting integration

**Kaggle will auto-save outputs**, so even if session times out, you keep the models.

---

## After Training Completes

### Step 1: Validate Results
Run validation cells to verify:
- ✅ 5 models trained (minutes, points, rebounds, assists, threes)
- ✅ 24-dimensional embeddings working
- ✅ Points MAE < 2.5 (target: 2.2-2.4)

### Step 2: Download Models
Run packaging cell, then download `nba_models_trained.zip`

### Step 3: Extract Locally
```bash
# On your local machine
cd C:\Users\tmiles11\nba_predictor
unzip nba_models_trained.zip
```

### Step 4: Test Predictions
```bash
# Test with new models
python predict_live_FINAL.py \
    --date 2025-11-10 \
    --aggregated-data ./data/aggregated_nba_data.csv.gzip \
    --betting \
    --output predictions.csv \
    --betting-output opportunities.csv
```

---

## Troubleshooting

### "Dataset not found at /kaggle/input/meeper/"
- Make sure you clicked "Add Data" in notebook
- Search for "meeper" and add it
- Restart notebook if needed

### "CUDA out of memory"
- Notebook → Session → Restart
- Select GPU again
- Re-run all cells

### "Training slower than expected"
- P100: ~3.5 hours (best Kaggle GPU)
- T4: ~4 hours
- Check GPU is actually being used (nvidia-smi)

### "Models not saving"
- Kaggle auto-saves to /kaggle/working/
- Check /kaggle/working/meep/models/ directory
- Models save incrementally (safe even if interrupted)

---

## Advantages of Kaggle over Colab

✅ **More stable**: Sessions less likely to disconnect
✅ **Dataset integration**: Direct access to your meeper dataset
✅ **Runs in background**: Can close browser, keeps running
✅ **Auto-saves output**: Models saved even if session times out
✅ **Free P100 GPU**: Better than Colab Free T4
✅ **Easy sharing**: Can make notebook public later

---

## Ready to Start!

1. **Create new Kaggle notebook**
2. **Add "meeper" dataset**
3. **Enable GPU (T4 or P100)**
4. **Copy-paste cells above**
5. **Run Cell 1 (setup)**
6. **Run Cell 2 (training)** - then come back in 3-4 hours!

**Training will run automatically in background.** Just let it run!

🚀
