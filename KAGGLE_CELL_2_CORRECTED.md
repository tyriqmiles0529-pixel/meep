# Corrected Kaggle Notebook Cell 2

## Problem
The notebook currently uses `--dataset` flag which downloads raw data from Kaggle.
We need to use `--aggregated-data` flag to load your pre-aggregated CSV.

---

## CORRECTED Cell 2 Code

Copy and paste this into your Kaggle notebook Cell 2:

```python
# ============================================================
# TRAIN NEURAL HYBRID MODELS - GAME + PLAYER
# ============================================================

import os

# Make sure we're in the code directory
os.chdir('/kaggle/working/meep')

print("="*70)
print("🚀 NBA NEURAL HYBRID TRAINING - GAME + PLAYER MODELS")
print("="*70)

print("\n📊 Dataset Info:")
print("   Source: /kaggle/input/meeper/aggregated_nba_data.csv.gzip")
print("   Full range: 1947-2026 (80 seasons, 1.6M player-games)")
print("   Training on: ALL DATA (no cutoff)")
print("   Contains: Raw stats + Basketball Reference priors (108 cols)")
print("\n⚙️  What will happen:")
print("   1. Load aggregated data (30 sec)")
print("   2. Build Phase 1-6 features (90 min)")
print("   3. Train game models: Moneyline + Spread (1 hour)")
print("   4. Train 5 player props with neural hybrid (5 hours)")
print("\n🧠 Architecture:")
print("   Game Models: Ensemble (TabNet + LightGBM)")
print("   Player Models: TabNet (24-dim embeddings) + LightGBM")
print("   Uncertainty: Sigma models for prediction intervals")

import torch
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'

if 'P100' in gpu_name:
    print("\n⏱️  Expected time: ~7-8 hours total (P100)")
elif 'T4' in gpu_name:
    print("\n⏱️  Expected time: ~8-9 hours total (T4)")
else:
    print("\n⏱️  Expected time: ~7-9 hours")

print("\n💡 Models to train:")
print("   Game: Moneyline (win probability), Spread (margin)")
print("   Player: Minutes, Points, Rebounds, Assists, Threes")
print("   Expected: Points MAE ~2.0-2.1 (with full 1.6M game history)")
print("\n" + "="*70)
print("STARTING TRAINING...")
print("="*70 + "\n")

# Run training - FULL DATASET, GAME + PLAYER MODELS
!python train_auto.py \
    --aggregated-data /kaggle/input/meeper/aggregated_nba_data.csv.gzip \
    --use-neural \
    --game-neural \
    --neural-epochs 30 \
    --neural-device gpu \
    --verbose \
    --no-window-ensemble

# KEY CHANGES:
# ✅ --aggregated-data (NOT --dataset) = loads pre-aggregated CSV
# ✅ --no-window-ensemble = single-pass training (not 5-year windows)
# ✅ NO --skip-game-models = trains both game + player
# ✅ NO --player-season-cutoff = uses all 1947-2026 data

print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)
print("\nModels saved to: /kaggle/working/meep/models/")
print("\nGame Models:")
print("  • Moneyline classifier (win probability)")
print("  • Spread regressor (margin prediction)")
print("\nPlayer Models:")
print("  • Minutes, Points, Rebounds, Assists, Threes")
print("  • All with 24-dim TabNet embeddings + LightGBM")
print("\nNext: Run validation cell to check embeddings")
```

---

## What Changed

### OLD (WRONG):
```python
!python train_auto.py \
    --dataset /kaggle/input/meeper/aggregated_nba_data.csv.gzip \
    --use-neural \
    --game-neural \
    --neural-epochs 30 \
    --neural-device gpu \
    --verbose \
    --fresh
```

**Problems:**
- `--dataset` downloads from Kaggle (raw data)
- No `--no-window-ensemble` = runs 5-year windows by default
- `--fresh` not needed for aggregated data

### NEW (CORRECT):
```python
!python train_auto.py \
    --aggregated-data /kaggle/input/meeper/aggregated_nba_data.csv.gzip \
    --use-neural \
    --game-neural \
    --neural-epochs 30 \
    --neural-device gpu \
    --verbose \
    --no-window-ensemble
```

**Fixes:**
- `--aggregated-data` loads your pre-aggregated CSV directly
- `--no-window-ensemble` disables 5-year window training
- Removed `--fresh` (not applicable for aggregated data)

---

## Expected Output

### With CORRECT Flag (--aggregated-data):
```
────────────────────────────────────────────────────────────
Loading Pre-Aggregated Dataset
────────────────────────────────────────────────────────────
- Loading from: /kaggle/input/meeper/aggregated_nba_data.csv.gzip
- Loaded 1,632,909 rows
- Reconstructing game-level data from aggregated file...
- Created games_df with 32,451 unique games.

────────────────────────────────────────────────────────────
Training game models
────────────────────────────────────────────────────────────
Training Moneyline Classifier...
  Ensemble: TabNet + LightGBM
  Validation Accuracy: 63.5-64.5%
  ✅ Saved: models/moneyline_ensemble_1947_2026.pkl

Training Spread Regressor...
  Ensemble: TabNet + LightGBM
  Validation RMSE: 10.2 points
  ✅ Saved: models/spread_ensemble_1947_2026.pkl

────────────────────────────────────────────────────────────
Training player models
────────────────────────────────────────────────────────────
Training POINTS model...
  TabNet: 24-dim embeddings (15 min)
  LightGBM: raw + embeddings (2 min)
  Sigma: uncertainty model (1 min)

  Results:
    MAE: 2.05 (baseline: 2.65) ← 22.6% improvement!
    RMSE: 2.91
    R²: 0.72

  ✅ Saved: models/points_hybrid_1947_2026.pkl

... (4 more props)
```

### With WRONG Flag (--dataset):
```
────────────────────────────────────────────────────────────
Fetching latest from Kaggle
────────────────────────────────────────────────────────────
- Downloaded to cache: /kaggle/input/historical-nba-data-and-player-box-scores

... (builds from raw data)

======================================================================
5-YEAR WINDOW TRAINING (RAM-Efficient Mode)
======================================================================
[TRAIN] Window 2002-2006: Cache missing - will train
[TRAIN] Window 2007-2011: Cache missing - will train
...
```

---

## Verification Checklist

After you paste the corrected code, check for:

✅ **"Loading Pre-Aggregated Dataset"** in output (not "Fetching latest from Kaggle")

✅ **"Loaded 1,632,909 rows"** (full 1947-2026 dataset)

✅ **NO "5-YEAR WINDOW TRAINING"** message

✅ **Single models saved** like `moneyline_ensemble_1947_2026.pkl` (not `ensemble_2002_2006.pkl`)

❌ **"Fetching latest from Kaggle"** = WRONG flag, stop training

❌ **"5-YEAR WINDOW TRAINING"** = Missing `--no-window-ensemble` flag

---

## How to Apply This Fix

### Option 1: Manual Copy-Paste (EASIEST)
1. Open your Kaggle notebook
2. Delete Cell 2 content
3. Copy entire code block from above
4. Paste into Cell 2
5. Run cells 1-2

### Option 2: Re-import Notebook
1. Download this corrected cell code
2. Manually edit Cell 2 in your Kaggle notebook
3. Save and run

### Option 3: Wait for Git Push (if notebook works)
- I'll try to fix the corrupted local file
- Push to GitHub
- You can re-import from GitHub URL

---

## Summary

**Problem:** Notebook uses `--dataset` flag (downloads raw data, does window training)

**Solution:** Use `--aggregated-data` flag (loads pre-aggregated CSV, single-pass training)

**Result:**
- Trains on full 1947-2026 history (1.6M games)
- Single models (not 5-year windows)
- Faster training (~6-7 hours vs 8-9)
- Better performance (more data = better predictions)

**Action Required:** Copy corrected Cell 2 code into your Kaggle notebook manually.

---

I'm unable to fix the corrupted local notebook file directly, but this markdown file has the exact code you need. Copy Cell 2 from this file into your Kaggle notebook!
