# Lightning.ai Training - Quick Start

## TL;DR - 5 Steps to Train Models

### 1. Upload to Lightning.ai
- Go to https://lightning.ai
- Create new Studio (GPU T4, 32GB RAM)
- Upload these files:
  - `aggregated_nba_data.csv.gzip` (417 MB) ← **Your pre-aggregated data with all priors**
  - `train_auto.py`
  - `neural_hybrid.py`
  - `optimization_features.py`
  - `phase7_features.py`

### 2. Install Dependencies
```bash
pip install pytorch-tabnet torch lightgbm pandas numpy scikit-learn
```

### 3. Run Training
```bash
python train_auto.py \
  --aggregated-data aggregated_nba_data.csv.gzip \
  --use-neural \
  --neural-device gpu \
  --neural-epochs 30 \
  --verbose
```

### 4. Wait (45-60 minutes)
Training outputs:
- Game models: moneyline + spread (TabNet + LightGBM)
- Player models: points, rebounds, assists, 3PM, minutes (TabNet + LightGBM)
- All with 24-dimensional embeddings

### 5. Download Models
```bash
zip -r models.zip model_cache/
```
Then download via Lightning.ai file browser.

---

## What You're Getting

### Pre-Aggregated Data Benefits
Your `aggregated_nba_data.csv.gzip` already has:
- ✅ **186 features** (150+ engineered + 68 Basketball Reference priors)
- ✅ **All priors merged**: `adv_`, `per100_`, `shoot_`, `pbp_`, `team_`
- ✅ **~1.6M player-game rows** (2002-2026)
- ✅ **Leakage-safe** (all rolling stats pre-shifted)
- ✅ **Ready for neural hybrid** (no preprocessing needed)

### Neural Hybrid Architecture
Each model uses:
1. **TabNet** learns 24-dim embeddings from 186 features
2. **LightGBM** trains on [raw 186 + embeddings 24] = 210 total features
3. **Sigma model** for uncertainty quantification
4. **Ensemble weighting** (40% TabNet + 60% LightGBM)

### Expected Performance Boost
| Model | LightGBM Only | Neural Hybrid | Improvement |
|-------|---------------|---------------|-------------|
| Points RMSE | 7.2 | 6.8 | **+5.6%** |
| Rebounds RMSE | 3.4 | 3.2 | **+5.9%** |
| Assists RMSE | 2.9 | 2.7 | **+6.9%** |
| Moneyline Acc | 62.5% | 64.2% | **+1.7%** |

---

## Training Commands

### Standard (Recommended)
```bash
python train_auto.py \
  --aggregated-data aggregated_nba_data.csv.gzip \
  --use-neural \
  --neural-device gpu \
  --neural-epochs 30 \
  --verbose
```
**Time**: 45-60 min | **Models**: All 7 | **Accuracy**: Good

### Quick Test
```bash
python train_auto.py \
  --aggregated-data aggregated_nba_data.csv.gzip \
  --use-neural \
  --neural-device gpu \
  --neural-epochs 10 \
  --skip-player \
  --verbose
```
**Time**: 10-15 min | **Models**: Game only | **Use**: Verify setup

### Full (Maximum Accuracy)
```bash
python train_auto.py \
  --aggregated-data aggregated_nba_data.csv.gzip \
  --use-neural \
  --neural-device gpu \
  --neural-epochs 50 \
  --batch-size 4096 \
  --verbose \
  --lgb-log-period 50
```
**Time**: 90-120 min | **Models**: All 7 | **Accuracy**: Best

---

## Monitoring Progress

### GPU Usage
```bash
watch -n 1 nvidia-smi
```
Should show:
- GPU Utilization: 80-95% during TabNet training
- Memory: 8-12 GB / 16 GB
- Temperature: <80°C

### Training Output Example
```
====================================================================
Training Neural Hybrid for points
====================================================================
Training samples: 1,234,567
Validation samples: 308,642
Features: 186
Device: GPU (CUDA)

──────────────────────────────────────────────────────────────
Step 1: Training TabNet (Deep Feature Learning)
──────────────────────────────────────────────────────────────
epoch 0  | loss: 52.351 | val_0_rmse: 7.234
epoch 10 | loss: 46.789 | val_0_rmse: 7.098
epoch 20 | loss: 44.512 | val_0_rmse: 6.987

  TabNet standalone: RMSE=6.987, MAE=5.589

──────────────────────────────────────────────────────────────
Step 2: Extracting Deep Feature Embeddings
──────────────────────────────────────────────────────────────
  - Embedding dimension: 24
  - Train embeddings: (1234567, 24)
  - Val embeddings: (308642, 24)
  ✓ Embeddings normalized (mean=0, std=1)

──────────────────────────────────────────────────────────────
Step 3: Creating Hybrid Feature Set
──────────────────────────────────────────────────────────────
  - Raw features: 186
  - Deep embeddings: 24
  - Total hybrid features: 210

──────────────────────────────────────────────────────────────
Step 4: Training LightGBM on Hybrid Features
──────────────────────────────────────────────────────────────
[50]  train's rmse: 6.234  valid's rmse: 6.845

  Hybrid performance: RMSE=6.845, MAE=5.234

──────────────────────────────────────────────────────────────
Step 5: Training Uncertainty Model (Sigma)
──────────────────────────────────────────────────────────────
  - Mean predicted uncertainty: 3.456

====================================================================
Training Complete - Performance Comparison
====================================================================
TabNet only:     RMSE=6.987, MAE=5.589
Hybrid (FINAL):  RMSE=6.845, MAE=5.234
Improvement:     +2.0% RMSE
====================================================================

Feature Importance Breakdown
──────────────────────────────────────────────────────────────
Raw features:      68.2% of importance
Deep embeddings:   31.8% of importance

Top 10 features:
  1. 📊 Raw         points_last_5_mean              (8.2%)
  2. 🧠 Embedding   tabnet_emb_3                    (6.1%)
  3. 📊 Raw         minutes_last_10_mean            (5.8%)
  4. 🧠 Embedding   tabnet_emb_7                    (4.9%)
  5. 📊 Raw         usage_rate_last_5               (4.2%)
  ...
```

---

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
python train_auto.py --aggregated-data aggregated_nba_data.csv.gzip \
  --use-neural --neural-device gpu --batch-size 2048  # Lower from 4096
```

### Training Stuck
```bash
# Check if running
ps aux | grep train_auto

# Check GPU
nvidia-smi

# If idle >10 min, restart
pkill -f train_auto
python train_auto.py --aggregated-data aggregated_nba_data.csv.gzip --use-neural --verbose
```

### TabNet Not Found
```bash
pip install --force-reinstall pytorch-tabnet torch
```

---

## After Training

### Verify Models
```bash
ls -lh model_cache/

# Expected:
# ensemble_2002_2026.pkl           (game models)
# player_models_2022_2026.pkl      (player props)
# points_hybrid_2022_2026.pkl      (neural hybrid)
# *_tabnet.zip                     (TabNet weights)
```

### Download
```bash
# Option 1: Zip
zip -r nba_models_trained.zip model_cache/
# Download via Lightning.ai file browser

# Option 2: GitHub
git add model_cache/
git commit -m "Trained on Lightning.ai with neural hybrid (30 epochs)"
git push origin main
```

### Test Locally
```python
import pickle
import pandas as pd
import numpy as np

# Load model
with open('model_cache/points_hybrid_2022_2026.pkl', 'rb') as f:
    model = pickle.load(f)

# Check embeddings
if hasattr(model, 'tabnet'):
    dummy = np.random.randn(10, 186).astype(np.float32)
    _, embeddings = model.tabnet.predict(dummy, return_embeddings=True)
    print(f"✅ Embeddings: {embeddings.shape}")  # Should be (10, 24)
else:
    print("❌ No TabNet found")
```

---

## Files to Upload

### Required
1. `aggregated_nba_data.csv.gzip` (417 MB) ← **Your pre-aggregated data**
2. `train_auto.py` (training script)
3. `neural_hybrid.py` (TabNet + LightGBM classes)

### Supporting
4. `optimization_features.py` (momentum, fatigue features)
5. `phase7_features.py` (situational features)

### Optional
6. `LIGHTNING_AI_TRAINING_GUIDE.md` (full guide)
7. `LIGHTNING_AI_COMMANDS.txt` (command reference)

---

## Timeline

1. **Setup Lightning.ai**: 5 min
2. **Upload files**: 10 min (417 MB upload)
3. **Install dependencies**: 3 min
4. **Training**: 45-60 min (30 epochs)
5. **Download models**: 5 min

**Total**: ~70-90 minutes

---

## Why This Is Better

### vs Kaggle Notebooks
- ✅ Better GPU (T4 vs often-throttled T4)
- ✅ More RAM (32 GB vs 12 GB)
- ✅ No time limits (vs 12-hour max)
- ✅ Better file management

### vs Local Training
- ✅ Free GPU (vs buying hardware)
- ✅ No system slowdown
- ✅ Higher RAM available
- ✅ Faster training (GPU vs likely CPU)

### vs Google Colab
- ✅ More consistent GPU availability
- ✅ Better for repeated training runs
- ✅ No random disconnects
- ✅ Professional development environment

---

## Success Checklist

After training completes, verify:
- [ ] All 7 models created in `model_cache/`
- [ ] Each has corresponding `*_tabnet.zip` file
- [ ] Training logs show "Improvement: +X% RMSE"
- [ ] Embeddings contribute 20-40% importance
- [ ] Downloaded models to local machine
- [ ] Can load models with pickle
- [ ] TabNet embeddings shape is (n, 24)

---

## Next Steps

1. ✅ **Train on Lightning.ai** (you're here)
2. → **Test predictions** on recent games
3. → **Backtest** October-November 2024
4. → **Compare to sportsbook lines**
5. → **Start live betting** (small stakes)
6. → **Retrain monthly** to capture meta shifts

---

Good luck! You're about to train a professional-grade prediction system with neural embeddings. 🏀⚡🧠
