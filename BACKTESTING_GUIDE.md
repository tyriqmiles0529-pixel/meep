# Backtesting Guide - 2024-2025 Season Validation

## Overview

Since the current season is **2025-2026**, the **2024-2025 season** is the most recent complete season. We can use it to validate model accuracy on unseen data.

## Two Approaches

### Approach 1: Use Existing Models (Quick)

If your current training ends before 2024, you can backtest immediately:

```bash
# Download models from current training
py -3.12 -m modal volume get nba-models / model_cache

# Run backtest
python backtest_2024_2025.py
```

**What it does**:
- Loads all models that end before 2024
- Tests on complete 2024-2025 season
- Shows MAE for each prop
- Calculates ensemble performance (averaging all windows)

### Approach 2: Train Fresh Models (Recommended)

Train models specifically excluding 2024-2025 for clean validation:

```bash
# 1. Train models on 1947-2024 only (excludes 2024-2025)
py -3.12 -m modal run modal_train_exclude_2024.py --parallel 3

# 2. Download backtest models
py -3.12 -m modal volume get nba-models-backtest / model_cache_backtest

# 3. Run backtest
python backtest_2024_2025.py
```

## What Gets Validated

**Backtesting Script Tests**:
- Points prediction
- Rebounds prediction
- Assists prediction
- Three-pointers prediction
- Minutes prediction

**Metrics**:
- MAE (Mean Absolute Error) per window
- Ensemble MAE (averaging all windows)
- RMSE (Root Mean Squared Error)

## Expected Output

```
======================================================================
BACKTEST: 2024-2025 SEASON
======================================================================
Training data: 1947-2023 (excluding 2024-2025)
Test data: 2024-2025 complete season
======================================================================

📦 Loading data...
  Test (2024-2025): 450,123 player-games

🤖 Loading models...
  ✓ Loaded 1947-1949
  ✓ Loaded 1950-1952
  ...
  Skipping 2022-2024 (contains test data)

======================================================================
BACKTESTING ON 2024-2025 SEASON
======================================================================

Testing window 1947-1949...
  points: MAE = 0.234
  rebounds: MAE = 0.456
  assists: MAE = 0.189
  threes: MAE = 0.123
  minutes: MAE = 1.234

Testing window 1950-1952...
  points: MAE = 0.221
  ...

======================================================================
AGGREGATE RESULTS (Ensemble of all windows)
======================================================================

Ensemble Performance (averaging all windows):
  POINTS    : MAE = 3.245, RMSE = 4.567
  REBOUNDS  : MAE = 2.134, RMSE = 3.012
  ASSISTS   : MAE = 1.987, RMSE = 2.765
  THREES    : MAE = 0.987, RMSE = 1.234
  MINUTES   : MAE = 5.678, RMSE = 7.890

✓ Results saved to: backtest_results_2024_2025.json

======================================================================
BACKTEST COMPLETE
======================================================================
Windows tested: 24
Test samples: 450,123
Results saved: backtest_results_2024_2025.json
```

## Interpreting Results

### Good MAE Values
- **Points**: 3-5 MAE (excellent), 5-8 MAE (good)
- **Rebounds**: 2-3 MAE (excellent), 3-5 MAE (good)
- **Assists**: 1-2 MAE (excellent), 2-3 MAE (good)
- **Threes**: 0.8-1.2 MAE (excellent), 1.2-1.8 MAE (good)
- **Minutes**: 5-8 MAE (excellent), 8-12 MAE (good)

### What to Look For

✅ **Ensemble better than individual windows** - Shows windows complement each other
✅ **Consistent MAE across windows** - Model is stable over time
✅ **Lower MAE on recent windows** - Captures modern NBA trends

⚠️ **High MAE on recent windows** - Model struggles with current playstyle
⚠️ **Large variance across windows** - Model is unstable

## Using Backtest Results

### 1. Model Selection
Only use windows with MAE below threshold:
```python
# Filter to good windows
good_windows = [w for w, results in backtest_results.items()
                if results['points']['mae'] < 4.0]
```

### 2. Weighted Ensemble
Weight windows by inverse MAE:
```python
weights = {window: 1/results['points']['mae']
           for window, results in backtest_results.items()}
```

### 3. Recency Weighting
Combine backtest MAE with recency:
```python
weight = (1 / mae) * (window_end_year / 2024)
```

## Next Steps After Backtesting

1. **If MAE is good** (< 5 for points):
   - Use these models for 2025-2026 predictions
   - Deploy to production

2. **If MAE is mediocre** (5-8 for points):
   - Retrain with more epochs (increase from 15 to 25)
   - Add more advanced features
   - Tune hyperparameters

3. **If MAE is poor** (> 8 for points):
   - Check feature engineering
   - Verify data quality
   - Consider different model architecture

## Files Created

- `backtest_2024_2025.py` - Backtesting script
- `modal_train_exclude_2024.py` - Train models excluding 2024-2025
- `backtest_results_2024_2025.json` - Results output
- `BACKTESTING_GUIDE.md` - This guide

## Important Notes

⚠️ **Season Definition**: 2024-2025 season = games from Oct 2024 - June 2025 = year column = 2025

⚠️ **Data Leakage**: Any model trained on 2024 or 2025 data is contaminated for backtesting

⚠️ **Hybrid Models**: The backtest script handles both old single-task models and new hybrid multi-task models

## Troubleshooting

### "No 2024-2025 data found"
Your Parquet file doesn't have 2024-2025 data yet. Wait for the season to complete.

### "No valid models found (all contain 2024+ data)"
Current training includes 2024-2025. Use `modal_train_exclude_2024.py` to train clean models.

### "TypeError: predict() missing arguments"
Model is an old format. The script handles both formats automatically.
