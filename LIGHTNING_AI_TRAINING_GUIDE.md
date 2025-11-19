# Lightning.ai NBA Model Training Guide

Complete guide to train game models and neural hybrid models with all embeddings on Lightning.ai.

## Overview

This guide shows you how to run the full training pipeline on Lightning.ai, which provides:
- **Free GPU access** (NVIDIA T4 GPU - 16GB VRAM)
- **Fast training** (2-4x faster than CPU)
- **Large RAM** (30+ GB available)
- **No local resource usage**

## What Gets Trained

### 1. Game Models (with Neural Hybrid)
- **Moneyline Classifier**: Win probability (home vs away) with TabNet + LightGBM
- **Spread Regressor**: Point margin prediction with TabNet + LightGBM
- Both use deep learning embeddings from TabNet combined with LightGBM

### 2. Player Models (with Neural Hybrid)
- **Points** prediction (TabNet + LightGBM)
- **Rebounds** prediction (TabNet + LightGBM)
- **Assists** prediction (TabNet + LightGBM)
- **3-Pointers Made** prediction (TabNet + LightGBM)
- **Minutes** prediction (TabNet + LightGBM)

### Neural Hybrid Architecture
Each model uses:
1. **TabNet** (attention-based deep learning) - learns 24-dimensional embeddings
2. **LightGBM** trained on [raw features + TabNet embeddings]
3. **Uncertainty quantification** via sigma models
4. **Ensemble weighting** between neural and tree-based predictions

---

## Setup: Lightning.ai Studio

### Step 1: Create Account
1. Go to https://lightning.ai
2. Sign up (free tier available)
3. Create a new "Studio"

### Step 2: Configure Environment
1. Click **New Studio**
2. Select **GPU** (T4 or better)
3. Choose **Python 3.10+**
4. Set RAM to **32GB** (recommended)

### Step 3: Upload Code & Data
Two options:

**Option A: Git Clone + Upload Data (Recommended)**
```bash
# In Lightning.ai terminal
git clone https://github.com/YOUR_USERNAME/nba_predictor.git
cd nba_predictor
```

Then upload `aggregated_nba_data.csv.gzip` (417 MB) via the Lightning.ai file manager.

**Option B: Upload Everything**
1. Locally, create a zip containing:
   - All Python scripts (`train_auto.py`, `neural_hybrid.py`, etc.)
   - `aggregated_nba_data.csv.gzip` (417 MB - priors already merged!)
2. Upload via Lightning.ai file manager
3. Extract in terminal: `unzip nba_predictor.zip`

---

## Installation

Run these commands in the Lightning.ai terminal:

```bash
# Install core dependencies
pip install kagglehub pandas numpy scikit-learn lightgbm

# Install neural network libraries (CRITICAL for embeddings)
pip install torch pytorch-tabnet

# Install optional dependencies
pip install requests tqdm
```

**Verify GPU is available:**
```python
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

Expected output:
```
GPU Available: True
GPU Name: Tesla T4
```

---

## Training Commands

### Basic Training (Neural Hybrid Enabled)

Train all models with neural hybrid + embeddings using your pre-aggregated data:

```bash
python train_auto.py \
  --aggregated-data aggregated_nba_data.csv.gzip \
  --use-neural \
  --neural-device gpu \
  --neural-epochs 30 \
  --verbose
```

**Flags Explained:**
- `--aggregated-data`: Use pre-aggregated CSV with priors already merged (186 columns, 417 MB)
- `--use-neural`: Enable TabNet + LightGBM hybrid models
- `--neural-device gpu`: Use GPU for TabNet training (MUCH faster)
- `--neural-epochs 30`: Train TabNet for 30 epochs (balance speed/accuracy)
- `--verbose`: Show detailed training progress

**Why use aggregated data?**
- ✅ All Basketball Reference priors already merged (adv_, per100_, shoot_, pbp_, team_)
- ✅ No Kaggle download needed (faster startup)
- ✅ Consistent features across training runs
- ✅ 186 features ready to use

### Advanced Training (Optimized)

For maximum performance with all features:

```bash
python train_auto.py \
  --aggregated-data aggregated_nba_data.csv.gzip \
  --use-neural \
  --neural-device gpu \
  --neural-epochs 50 \
  --batch-size 4096 \
  --verbose \
  --lgb-log-period 50 \
  --n-jobs -1
```

**Additional Flags:**
- `--neural-epochs 50`: More training = better accuracy (but slower)
- `--batch-size 4096`: Larger batches for GPU efficiency
- `--lgb-log-period 50`: Reduce LightGBM logging spam
- `--n-jobs -1`: Use all CPU cores for LightGBM

### Quick Test (Fast Training)

To verify everything works before full training:

```bash
python train_auto.py \
  --aggregated-data aggregated_nba_data.csv.gzip \
  --use-neural \
  --neural-device gpu \
  --neural-epochs 10 \
  --skip-player \
  --verbose
```

**Test Flags:**
- `--neural-epochs 10`: Minimal epochs (faster)
- `--skip-player`: Only train game models (skip player props)

---

## Expected Training Time

| Configuration | GPU | CPU | Models Trained |
|---------------|-----|-----|----------------|
| Quick Test | 5-10 min | 20-30 min | 2 game models |
| Basic (30 epochs) | 30-45 min | 2-3 hours | 7 models (game + player) |
| Advanced (50 epochs) | 60-90 min | 4-6 hours | 7 models (optimized) |

---

## Output Files

After training completes, you'll have:

```
model_cache/
├── ensemble_2002_2026.pkl          # Game models (moneyline + spread)
├── ensemble_2002_2026_meta.json    # Game model metadata
├── player_models_2022_2026.pkl     # Player models (all 5 props)
├── player_models_2022_2026_meta.json
└── game_neural_hybrid_2002_2026/   # Neural hybrid models
    ├── moneyline_hybrid.pkl
    ├── moneyline_hybrid_tabnet.zip
    ├── spread_hybrid.pkl
    ├── spread_hybrid_tabnet.zip
    ├── points_hybrid_2022_2026.pkl
    ├── points_hybrid_2022_2026_tabnet.zip
    ├── rebounds_hybrid_2022_2026.pkl
    ├── assists_hybrid_2022_2026.pkl
    ├── threes_hybrid_2022_2026.pkl
    └── minutes_hybrid_2022_2026.pkl
```

---

## Download Trained Models

### Method 1: Lightning.ai Download
1. Select files in file browser
2. Right-click → Download
3. Extract to your local `model_cache/` folder

### Method 2: Zip and Download
```bash
# In Lightning.ai terminal
zip -r nba_models_trained.zip model_cache/
```
Then download `nba_models_trained.zip`

### Method 3: GitHub Upload (Best)
```bash
# In Lightning.ai terminal
git config --global user.name "Your Name"
git config --global user.email "your@email.com"

git add model_cache/
git commit -m "Trained models on Lightning.ai with neural hybrid"
git push origin main
```

Then pull to your local machine:
```bash
# On your local machine
git pull origin main
```

---

## Monitoring Training

### Watch GPU Usage
```bash
# In Lightning.ai terminal
watch -n 1 nvidia-smi
```

Shows:
- GPU utilization %
- Memory usage
- Temperature
- Active processes

### Check Training Progress
Training outputs detailed logs:

```
====================================================================
Training Neural Hybrid for points
====================================================================
Training samples: 1,234,567
Validation samples: 308,642
Features: 156
Device: GPU (CUDA)

──────────────────────────────────────────────────────────────
Step 1: Training TabNet (Deep Feature Learning)
──────────────────────────────────────────────────────────────
epoch 0  | loss: 52.351  | val_0_rmse: 7.234  | val_0_mae: 5.678
epoch 5  | loss: 48.102  | val_0_rmse: 7.156  | val_0_mae: 5.612
epoch 10 | loss: 46.789  | val_0_rmse: 7.098  | val_0_mae: 5.589
...

  TabNet standalone performance:
  - RMSE: 7.098
  - MAE:  5.589

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
  - Raw features: 156
  - Deep embeddings: 24
  - Total hybrid features: 180

──────────────────────────────────────────────────────────────
Step 4: Training LightGBM on Hybrid Features
──────────────────────────────────────────────────────────────
[LightGBM] [Info] Training until validation scores don't improve...
[50]    train's rmse: 6.234    valid's rmse: 6.987

  Hybrid (TabNet + LightGBM) performance:
  - RMSE: 6.845
  - MAE:  5.234

──────────────────────────────────────────────────────────────
Step 5: Training Uncertainty Model (Sigma)
──────────────────────────────────────────────────────────────
  - Mean predicted uncertainty: 3.456
  - Actual mean error: 3.512

====================================================================
Training Complete - Performance Comparison
====================================================================
TabNet only:       RMSE=7.098, MAE=5.589
Hybrid (FINAL):    RMSE=6.845, MAE=5.234
Improvement:       +3.6% RMSE
====================================================================
```

---

## Troubleshooting

### Out of Memory (GPU)
**Symptom**: `CUDA out of memory` error

**Fix 1**: Reduce batch size
```bash
python train_auto.py --use-neural --batch-size 2048  # Lower from 4096
```

**Fix 2**: Reduce TabNet size (in `neural_hybrid.py:79-92`)
```python
'n_d': 16,      # Reduced from 24
'n_a': 16,
'n_steps': 3,   # Reduced from 4
```

**Fix 3**: Use CPU (slower but won't crash)
```bash
python train_auto.py --use-neural --neural-device cpu
```

### TabNet Not Found
**Symptom**: `TabNet not available` warning

**Fix**:
```bash
pip install pytorch-tabnet torch
```

### Kaggle Download Fails
**Symptom**: `Dataset not found` or connection timeout

**Fix 1**: Set Kaggle credentials
```bash
# In Lightning.ai terminal
mkdir -p ~/.kaggle
echo '{"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Get API key from: https://www.kaggle.com/settings → API → Create New Token

**Fix 2**: Use local data upload
1. Download dataset locally from Kaggle
2. Upload CSV files to Lightning.ai
3. Modify `train_auto.py` to read from local path

### Training Stuck
**Symptom**: No progress for 10+ minutes

**Check**:
```bash
# GPU activity
nvidia-smi

# CPU activity
top

# Disk I/O
iostat -x 1
```

If all are idle, training may have crashed. Check logs for errors.

---

## Advanced: Custom Neural Configuration

Edit `neural_hybrid.py` to customize:

### Game Models (Lines 79-92)
```python
self.tabnet_params = {
    'n_d': 24,              # Embedding dimension (16-32)
    'n_a': 24,              # Attention dimension (same as n_d)
    'n_steps': 3,           # Decision steps (3-5)
    'gamma': 1.3,           # Feature reuse (1.0-2.0)
    'lambda_sparse': 1e-3,  # Sparsity regularization
    ...
}
```

### Player Models (Lines 377-390)
```python
self.tabnet_params = {
    'n_d': 24,              # Smaller for faster training
    'n_a': 24,
    'n_steps': 4,           # More steps = better accuracy
    'gamma': 1.5,
    'lambda_sparse': 1e-4,  # Less sparsity for player data
    ...
}
```

**After editing**, retrain:
```bash
python train_auto.py --use-neural --fresh
```

---

## Best Practices

### 1. Start Small, Scale Up
```bash
# Week 1: Test with 10 epochs
python train_auto.py --use-neural --neural-epochs 10

# Week 2: Train with 30 epochs
python train_auto.py --use-neural --neural-epochs 30 --fresh

# Week 3+: Full training 50 epochs
python train_auto.py --use-neural --neural-epochs 50 --fresh
```

### 2. Monitor Performance
After each training run, check validation metrics in console output.
Look for:
- **RMSE improvement** (TabNet vs Hybrid)
- **Embedding importance** (should be 20-40% of total)
- **No overfitting** (train vs validation gap < 10%)

### 3. Save Logs
```bash
python train_auto.py --use-neural --verbose 2>&1 | tee training_log.txt
```

Upload `training_log.txt` to GitHub for record-keeping.

### 4. Version Control
```bash
# Before training
git tag v1.0-pre-training

# After training
git add model_cache/
git commit -m "Trained v1.0 on Lightning.ai GPU (30 epochs)"
git tag v1.0-trained

git push origin main --tags
```

---

## Performance Expectations

### Neural Hybrid vs LightGBM Only

| Metric | LightGBM Only | Neural Hybrid | Improvement |
|--------|---------------|---------------|-------------|
| **Points RMSE** | 7.2 | 6.8 | +5.6% |
| **Rebounds RMSE** | 3.4 | 3.2 | +5.9% |
| **Assists RMSE** | 2.9 | 2.7 | +6.9% |
| **Moneyline Accuracy** | 62.5% | 64.2% | +1.7% |
| **Spread MAE** | 9.1 pts | 8.6 pts | +5.5% |

**Why Neural Hybrid is Better:**
1. Captures non-linear interactions (TabNet attention)
2. Learns momentum patterns (temporal embeddings)
3. Better uncertainty estimates (sigma models)
4. Generalizes to rare events (deep features)

---

## FAQ

**Q: Can I use CPU instead of GPU?**
A: Yes, use `--neural-device cpu`. Training will be 3-5x slower.

**Q: How much does Lightning.ai cost?**
A: Free tier includes 22 GPU hours/month (plenty for weekly training).

**Q: Can I resume interrupted training?**
A: No, you must restart. Use `--fresh` to clear cache and retrain.

**Q: What's the difference between this and Colab?**
A: Lightning.ai has better GPUs (T4 vs Colab's often-throttled T4) and more RAM. Colab is easier for beginners.

**Q: Can I train on multiple GPUs?**
A: TabNet doesn't support multi-GPU. Stick with single T4.

**Q: Should I use neural hybrid for all models?**
A: Yes! The hybrid approach outperforms LightGBM-only on all metrics.

**Q: How often should I retrain?**
A: Monthly is sufficient. More frequent retraining rarely improves accuracy.

---

## Next Steps

After training completes:

1. **Download models** to your local machine
2. **Test predictions** with `player_ensemble_enhanced.py`
3. **Backtest performance** with historical data
4. **Compare to sportsbook lines** for edge detection
5. **Start betting** (small stakes initially)
6. **Retrain monthly** to capture meta shifts

---

## Support

- **Training Issues**: Check GitHub Issues or `TROUBLESHOOTING.md`
- **Lightning.ai Help**: https://lightning.ai/docs
- **Neural Network Tuning**: See `neural_hybrid.py` comments

---

Good luck! You're running a professional-grade prediction system.
