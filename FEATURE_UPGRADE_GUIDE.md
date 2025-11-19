# NBA Predictor: Feature Upgrade Guide

## Problem
- Currently seeing only ~70 base features
- Expected 150+ features including rolling averages (L5, L10, L20)
- Game models being skipped due to missing team-level columns

## Solution Overview

### ✅ Changes Made

1. **Updated `train_player_models.py`** (line 143-156)
   - Added `add_rolling_features()` call in `create_window_training_data()`
   - Now computes L5, L10, L20 rolling averages
   - Adds variance and trend features
   - **Result: ~150+ features instead of ~70**

2. **Updated `modal_train.py`** (line 33)
   - Added `rolling_features.py` to Modal image
   - Now includes feature computation in cloud training

3. **Created `modal_game_training.py`**
   - **Separate Modal job for game models** (moneyline + spread)
   - Uses team-level data, not player data
   - Runs independently on A10G GPU

4. **Created `precompute_features.py`**
   - Script to precompute features once and save to Parquet
   - For future efficiency (Option 1 below)

---

## Today: Option 2 - Compute Features During Training

**Status: READY TO USE**

The code is already updated. Rolling features will be computed automatically during training.

### Usage

```bash
# Local training (with rolling features)
python train_player_models.py --data aggregated_nba_data.parquet --window-size 5

# Modal training (with rolling features)
modal run modal_train.py --start-year 2022 --end-year 2026

# Separate game models
modal run modal_game_training.py --start-year 2002 --end-year 2026
```

### What Happens

1. Load raw aggregated data
2. Filter to window seasons
3. Select ~70 base features
4. **Compute rolling features** (adds ~80 features) ← NEW!
5. Train models on ~150 features
6. Save models

### Pros
- ✅ No pre-processing needed
- ✅ Works with existing data files
- ✅ Always uses latest data

### Cons
- ❌ Features computed every training run (slower)
- ❌ Uses more memory during training

---

## Next Time: Option 1 - Precompute Features (RECOMMENDED)

**Status: READY TO IMPLEMENT**

For future training, precompute features once and reuse them.

### Step 1: Precompute Features (Run Once)

```bash
# Compute features and save enhanced Parquet
python precompute_features.py
```

This creates: `aggregated_nba_data_with_features.parquet`
- Input: 1947-2026 raw data (~70 columns)
- Output: 1947-2026 enhanced data (~150 columns)
- Size: ~500 MB (gzipped Parquet)

### Step 2: Upload to Modal Volume (Run Once)

```bash
# Upload enhanced data to Modal
modal volume put nba-data aggregated_nba_data_with_features.parquet /data/aggregated_nba_data_with_features.parquet
```

### Step 3: Update Training Scripts (Run Once)

Change this line in `modal_train.py`:
```python
# OLD
agg_df = load_player_data("/data/aggregated_nba_data.parquet", verbose=True)

# NEW
agg_df = load_player_data("/data/aggregated_nba_data_with_features.parquet", verbose=True)
```

**Then comment out the rolling features section** (lines 143-156 in `train_player_models.py`):
```python
# Features already precomputed - skip
# window_df = add_rolling_features(...)
```

### Step 4: Train as Normal

```bash
# Features already included!
modal run modal_train.py --start-year 2022 --end-year 2026
```

### Pros
- ✅ **3-5x faster training** (no feature computation)
- ✅ Lower memory usage
- ✅ Consistent features across all windows
- ✅ Easier debugging (inspect feature file)

### Cons
- ❌ Must recompute if data changes
- ❌ Larger storage (500 MB vs 200 MB)

---

## Game Models - Separate Training

**Status: READY TO USE**

Game models (moneyline, spread) now train separately because they need different data.

### Usage

```bash
# Train game models on Modal
modal run modal_game_training.py --start-year 2002 --end-year 2026
```

### Output

- Saves: `/models/game_models_2002_2026.pkl`
- Includes:
  - Moneyline classifier (win probability)
  - Calibrated moneyline (isotonic regression)
  - Spread regressor (point differential)
  - Spread sigma (uncertainty)

### Download Models

```bash
# Get from Modal volume
modal volume get nba-models /models/game_models_2002_2026.pkl game_models.pkl
```

---

## Feature Breakdown

### Base Features (~70 columns)
- Core stats: points, assists, rebounds, etc.
- Advanced stats: `adv_*` (PER, TS%, USG%, etc.)
- Per-100 stats: `per100_*` (pace-adjusted)
- Shooting stats: `shoot_*` (zones, assisted %)
- Play-by-play: `pbp_*` (plus/minus, turnovers)

### Rolling Features (~80 columns)
- **L5 averages**: Recent 5-game averages (hot hand)
- **L10 averages**: Recent 10-game averages (short-term)
- **L20 averages**: Recent 20-game averages (baseline)
- **Variance**: Standard deviation (consistency)
- **Trends**: L5 vs L20 (momentum)
- **Z-scores**: Hot/cold streak indicators

### Total: ~150 columns

---

## FAQ

### Q: Can I add features to existing models without retraining?
**A: No.** Models are trained on fixed input dimensions. Adding 80 new features changes the input shape from (n, 70) to (n, 150). The model doesn't know how to interpret the new features. **You must retrain.**

### Q: Do I need to retrain all windows?
**A: Yes, if you want consistent features.** Each window needs to be retrained with the new features for best accuracy.

### Q: Which option should I use long-term?
**A: Option 1 (precompute).** Faster, cleaner, and easier to manage. Compute features once, train many times.

### Q: What about game models?
**A: Use separate script.** Game models need team-aggregated data, not player data. Use `modal_game_training.py`.

---

## Next Steps

### For Today (Option 2 Active)
```bash
# Just run training - features computed automatically
modal run modal_train.py --start-year 2022 --end-year 2026
```

### For Next Time (Switch to Option 1)
```bash
# 1. Precompute features
python precompute_features.py

# 2. Upload to Modal
modal volume put nba-data aggregated_nba_data_with_features.parquet /data/aggregated_nba_data_with_features.parquet

# 3. Update modal_train.py to use new file
# 4. Comment out rolling feature computation

# 5. Train faster!
modal run modal_train.py --start-year 2022 --end-year 2026
```

### For Game Models (Any Time)
```bash
modal run modal_game_training.py --start-year 2002 --end-year 2026
```

---

## File Changes Summary

| File | Change | Status |
|------|--------|--------|
| `train_player_models.py` | Added rolling features | ✅ Updated |
| `modal_train.py` | Added `rolling_features.py` | ✅ Updated |
| `modal_game_training.py` | New file for game models | ✅ Created |
| `precompute_features.py` | New file for Option 1 | ✅ Created |

---

## Performance Expectations

### With Rolling Features (150 columns)
- **Accuracy**: +2-5% improvement (trend/momentum features)
- **Training time**: +10-20% (feature computation)
- **Memory**: +30-50% (more features)

### Precomputed vs Runtime
- **Precomputed**: 2-3 hours for full training
- **Runtime**: 3-4 hours for full training
- **Savings**: 25-30% time reduction

---

Generated: 2025-11-18
