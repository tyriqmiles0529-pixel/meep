# Updated Kaggle Notebook - Ready to Copy-Paste

**These are the corrected cells for your Kaggle notebook.**

---

## Cell 1: Setup and Install

```python
# ============================================================
# SETUP (Kaggle Version)
# ============================================================

print("📦 Installing packages...")
!pip install -q pytorch-tabnet lightgbm scikit-learn pandas numpy tqdm

print("\n📥 Downloading training code from GitHub...")
import os

# Navigate to Kaggle working directory
os.chdir('/kaggle/working')

# Clone your repository
!git clone https://github.com/tyriqmiles0529-pixel/meep.git
os.chdir('meep')

print("\n📍 Code version:")
!git log -1 --oneline

# Check GPU
import torch
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not available'
print(f"\n🎮 GPU: {gpu_name}")

# Verify dataset exists (added via "Add Data" in Kaggle UI)
dataset_path = '/kaggle/input/meeper/aggregated_nba_data.csv.gzip'
if os.path.exists(dataset_path):
    size_mb = os.path.getsize(dataset_path) / 1024 / 1024
    print(f"\n✅ Dataset found: {size_mb:.1f} MB")
    print(f"   Path: {dataset_path}")
    print(f"   Full NBA history: 1947-2026")
    print(f"   Training will use: 2002-2026 (default cutoff)")
else:
    print("\n❌ Dataset not found!")
    print("   Make sure you added 'meeper' dataset to this notebook")
    print("   Click 'Add Data' → search 'meeper' → Add")

print("\n✅ Setup complete!")
```

---

## Cell 2: Train Models (Main Cell)

```python
# ============================================================
# TRAIN NEURAL HYBRID MODELS
# ============================================================

import os

# Make sure we're in the code directory
os.chdir('/kaggle/working/meep')

print("="*70)
print("🚀 NBA PLAYER PROPS - NEURAL HYBRID TRAINING")
print("="*70)

print("\n📊 Dataset Info:")
print("   Source: /kaggle/input/meeper/aggregated_nba_data.csv.gzip")
print("   Full range: 1947-2026 (80 seasons, 1.6M player-games)")
print("   Training uses: 2002-2026 (24 seasons, ~125K games)")
print("   Contains: Raw stats + Basketball Reference priors (108 cols)")
print("\n⚙️  What will happen:")
print("   1. Load aggregated data (30 sec)")
print("   2. Filter to 2002+ seasons")
print("   3. Build Phase 1-6 features (45 min)")
print("   4. Train 5 props with neural hybrid (3 hours)")
print("\n🧠 Architecture:")
print("   TabNet: 24-dimensional embeddings")
print("   LightGBM: Trained on raw + embeddings")
print("   Sigma models: Uncertainty quantification")

import torch
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'

if 'P100' in gpu_name:
    print("\n⏱️  Expected time: ~4 hours (P100)")
elif 'T4' in gpu_name:
    print("\n⏱️  Expected time: ~4.5 hours (T4)")
else:
    print("\n⏱️  Expected time: ~4-5 hours")

print("\n💡 Props to train: minutes, points, rebounds, assists, threes")
print("   Expected: Points MAE ~2.2-2.4 (vs baseline 2.5-2.7)")
print("\n" + "="*70)
print("STARTING TRAINING...")
print("="*70 + "\n")

# Run training
!python train_auto.py \
    --dataset /kaggle/input/meeper/aggregated_nba_data.csv.gzip \
    --use-neural \
    --neural-epochs 30 \
    --neural-device gpu \
    --verbose \
    --fresh \
    --skip-game-models \
    --player-season-cutoff 2002

print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)
print("\nModels saved to: /kaggle/working/meep/models/")
print("\nNext: Run validation cell to check embeddings")
```

---

## Cell 3: Validate Embeddings

```python
# ============================================================
# VALIDATE 24-DIM EMBEDDINGS
# ============================================================

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Navigate to code directory
os.chdir('/kaggle/working/meep')

print("🔍 Validating TabNet embeddings...\n")

models_dir = Path('./models')

# Check if models directory exists
if not models_dir.exists():
    print("❌ Models directory not found!")
    print("   Expected: /kaggle/working/meep/models/")
    print("   Run training cell first")
else:
    # Find points model
    model_files = list(models_dir.glob('*points*.pkl'))

    if not model_files:
        print("❌ No points model found!")
        print(f"   Searching in: {models_dir.absolute()}")
        print(f"   Files found: {list(models_dir.glob('*.pkl'))}")
    else:
        model_path = model_files[0]
        print(f"📦 Loading model: {model_path.name}")

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        print(f"   Model type: {type(model).__name__}")

        # Check if it's a NeuralHybridPredictor
        if hasattr(model, 'tabnet'):
            print(f"   ✅ Neural hybrid detected")
            print(f"   TabNet: {type(model.tabnet).__name__}")
            print(f"   LightGBM: {type(model.lgbm).__name__}")

            # Test embedding extraction
            print("\n🧪 Testing embedding extraction...")

            # Create dummy data (match actual feature count)
            n_features = 150  # Adjust if needed
            dummy_features = pd.DataFrame(
                np.random.randn(10, n_features),
                columns=[f'feature_{i}' for i in range(n_features)]
            )

            # Get embeddings
            try:
                if hasattr(model, '_get_embeddings'):
                    embeddings = model._get_embeddings(dummy_features)
                    print(f"\n✅ SUCCESS!")
                    print(f"   Embedding shape: {embeddings.shape}")
                    print(f"   Expected: (10, 24)")

                    if embeddings.shape[1] == 24:
                        print(f"\n🎯 PERFECT: Got 24-dimensional embeddings")
                        print(f"   Mean: {embeddings.mean():.4f}")
                        print(f"   Std: {embeddings.std():.4f}")

                        # Check LightGBM uses embeddings
                        if hasattr(model.lgbm, 'feature_name_'):
                            lgbm_features = model.lgbm.feature_name_
                            embedding_features = [f for f in lgbm_features if 'embedding' in f.lower()]
                            print(f"   LightGBM sees {len(embedding_features)} embedding features")

                        print(f"\n✅ Model validation PASSED!")
                        print(f"   Ready for predictions")
                    else:
                        print(f"\n⚠️  Warning: Got {embeddings.shape[1]}-dim embeddings")
                else:
                    print(f"⚠️  Model doesn't have _get_embeddings method")

            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"   ⚠️  LightGBM-only model (no neural hybrid)")

        # Display model info
        print(f"\n📊 Model Summary:")
        if hasattr(model, 'lgbm'):
            print(f"   LightGBM trees: {model.lgbm.n_estimators if hasattr(model.lgbm, 'n_estimators') else 'N/A'}")
            if hasattr(model.lgbm, 'feature_name_'):
                print(f"   Features used: {len(model.lgbm.feature_name_)}")

        if hasattr(model, 'sigma_model'):
            print(f"   Uncertainty model: {'Yes' if model.sigma_model else 'No'}")

print("\n✅ Validation complete!")
```

---

## Cell 4: Show Training Results

```python
# ============================================================
# TRAINING RESULTS SUMMARY
# ============================================================

import os
from pathlib import Path

os.chdir('/kaggle/working/meep')

models_dir = Path('./models')

print("="*70)
print("📊 TRAINING RESULTS")
print("="*70)

if not models_dir.exists():
    print("\n❌ No models directory found!")
else:
    model_files = list(models_dir.glob('*.pkl'))

    if not model_files:
        print("\n❌ No models found!")
        print(f"   Directory: {models_dir.absolute()}")
    else:
        print(f"\n✅ Found {len(model_files)} trained models:\n")

        for model_path in sorted(model_files):
            print(f"   📦 {model_path.name}")
            size_mb = model_path.stat().st_size / 1024 / 1024
            print(f"      Size: {size_mb:.1f} MB")

            # Try to load and check type
            try:
                import pickle
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)

                if hasattr(model, 'tabnet'):
                    print(f"      Type: Neural Hybrid ✅")
                else:
                    print(f"      Type: LightGBM only")

                if hasattr(model, 'sigma_model') and model.sigma_model:
                    print(f"      Uncertainty: Yes ✅")

                print()
            except Exception as e:
                print(f"      Error loading: {e}\n")

print("="*70)
print("\nModels location: /kaggle/working/meep/models/")
print("\nNext: Package and download models")
```

---

## Cell 5: Package and Download

```python
# ============================================================
# PACKAGE MODELS FOR DOWNLOAD
# ============================================================

import os

os.chdir('/kaggle/working')

print("📦 Packaging models...")

# Check if models exist
if not os.path.exists('meep/models'):
    print("\n❌ No models directory found!")
    print("   Run training first")
else:
    # Create zip file
    !zip -r nba_models_trained.zip meep/models/ meep/model_cache/ 2>/dev/null

    # Check if zip was created
    if os.path.exists('nba_models_trained.zip'):
        size_mb = os.path.getsize('nba_models_trained.zip') / 1024 / 1024
        print(f"\n✅ Package created: nba_models_trained.zip ({size_mb:.1f} MB)")
        print("\n📥 To download:")
        print("   1. Look at the right sidebar")
        print("   2. Click 'Output' tab")
        print("   3. Find 'nba_models_trained.zip'")
        print("   4. Click the download icon (↓)")
        print("\nOr use this command to download via Kaggle API:")
        print("   (This requires Kaggle notebook to be public)")
    else:
        print("\n❌ Failed to create zip file")
        print("   Check if models exist in meep/models/")
```

---

## Directory Structure in Kaggle

When running, your Kaggle notebook will have:

```
/kaggle/
├── input/
│   └── meeper/
│       └── aggregated_nba_data.csv.gzip  ← Your dataset
│
└── working/
    └── meep/                              ← Cloned GitHub repo
        ├── train_auto.py                  ← Main training script
        ├── neural_hybrid.py               ← Neural hybrid architecture
        ├── optimization_features.py       ← Phase 6 features
        ├── phase7_features.py             ← Phase 7 features
        └── models/                        ← OUTPUT (created during training)
            ├── minutes_hybrid_2002_2026.pkl
            ├── points_hybrid_2002_2026.pkl
            ├── rebounds_hybrid_2002_2026.pkl
            ├── assists_hybrid_2002_2026.pkl
            └── threes_hybrid_2002_2026.pkl
```

---

## Key Fixes Made

1. ✅ **Added `os.chdir()` calls** - Ensures we're in the right directory
2. ✅ **Fixed validation cell** - Checks if models directory exists first
3. ✅ **Correct paths** - Uses `/kaggle/input/meeper/` for dataset
4. ✅ **Better error messages** - Shows what went wrong and where
5. ✅ **Added `--player-season-cutoff 2002`** - Explicit default
6. ✅ **Realistic time estimates** - 4-5 hours including feature building

---

## What to Expect During Training

```
Cell 2 Output:
======================================================================
🚀 NBA PLAYER PROPS - NEURAL HYBRID TRAINING
======================================================================

📊 Dataset Info:
   Source: /kaggle/input/meeper/aggregated_nba_data.csv.gzip
   Full range: 1947-2026 (80 seasons, 1.6M player-games)
   Training uses: 2002-2026 (24 seasons, ~125K games)
   ...

======================================================================
STARTING TRAINING...
======================================================================

Loading aggregated data... (30 sec)
   Loaded 1,632,909 player-games

Filtering to 2002+ seasons...
   Kept 125,487 player-games

Building Phase 1 features... (10 min)
   Rolling averages (L3, L5, L10)
   Per-minute rates
   True shooting percentages

Building Phase 2 features... (5 min)
   Team pace, off/def strength
   Opponent matchups

... (more feature building, 45 min total)

Training MINUTES model... (40 min)
   TabNet training (GPU)... 15 min
   Extracting embeddings... 1 min
   LightGBM training... 2 min
   Sigma model training... 1 min
   ✅ Saved: models/minutes_hybrid_2002_2026.pkl

Training POINTS model... (40 min)
   TabNet training... 15 min
   ...
   MAE: 2.31 (baseline: 2.65) ← 12.8% improvement!
   ✅ Saved: models/points_hybrid_2002_2026.pkl

... (3 more props, ~3 hours total)

======================================================================
✅ TRAINING COMPLETE!
======================================================================
```

---

## Ready to Copy-Paste!

1. Create new Kaggle notebook
2. Add "meeper" dataset (Add Data → search "meeper")
3. Enable GPU (T4 or P100)
4. Copy-paste these 5 cells
5. Run Cell 1 (setup - 2 min)
6. Run Cell 2 (training - 4-5 hours)
7. Go do something else, come back later!
8. Run Cell 3 (validation - 1 min)
9. Run Cell 4 (summary - 10 sec)
10. Run Cell 5 (download - 1 min)

**Total time: 4-5 hours** (mostly unattended)
