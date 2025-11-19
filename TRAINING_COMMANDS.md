# NBA Model Training Commands

## Quick Start (Fresh Training)
```bash
# 1. Clear Modal volume cache (removes placeholder models)
py -3.12 -m modal run modal_clear_cache.py

# 2. Start training from scratch
py -3.12 -m modal run modal_train.py
```

## All Training Commands

### 0. Clear Modal Cache (If Needed)
```bash
py -3.12 -m modal run modal_clear_cache.py
```
Run this if you want to:
- Remove placeholder models from previous training
- Force re-training of all windows
- Start completely fresh

### 1. Train All Windows (Sequential)
```bash
py -3.12 -m modal run modal_train.py
```
- Trains all 27 windows one at a time
- Skips windows that are already cached
- Most reliable for first run

### 2. Train Specific Window
```bash
py -3.12 -m modal run modal_train.py --window-start 2022 --window-end 2024
```
Use this to:
- Test training on recent data first
- Re-train a specific window
- Debug training issues

### 3. Train Multiple Windows in Parallel
```bash
py -3.12 -m modal run modal_train.py --parallel 3
```
- Trains 3 windows simultaneously
- Faster but costs 3x the GPU hours
- Good for production runs after testing

## Download Models After Training

```bash
# Download all models to local model_cache/
py -3.12 -m modal volume get nba-models / model_cache
```

## Verify Models

```bash
# Check models have actual trained objects (not None)
python verify_models.py
```

## Monitor Training

All output appears in your **local terminal**. You'll see:
```
======================================================================
NBA MODEL TRAINING ON MODAL (PLAYER + GAME)
======================================================================
Running from: Your laptop
Executing on: Modal cloud (A10G GPU + 64GB RAM)
Models: Player props + Game outcomes (both with neural hybrids)
======================================================================

======================================================================
TRAINING WINDOW: 2022-2024
GPU: A10G | RAM: 64GB | Running on Modal
======================================================================

Loading data from Parquet file...
✓ Loaded 12,345,678 rows from aggregated_nba_data.parquet
  Season range: 1947-2026
  Columns: 186

Creating window training data...
  • Filtered aggregated data for window: 123,456 rows
  • Optimized data: 123,456 rows, 85 columns
  • Memory usage: 245.3 MB

======================================================================
TRAINING PLAYER MODELS: 2022-2024
======================================================================
Training data: 123,456 rows

  Training points model...
  ✓ points model trained

  Training rebounds model...
  ✓ rebounds model trained

  Training assists model...
  ✓ assists model trained

  Training threes model...
  ✓ threes model trained

  Training minutes model...
  ✓ minutes model trained

✓ Training complete for 2022-2024

🎯 Training GAME models (moneyline + spread)...
   ✓ Game models trained: 2,468 games

✅ Window 2022-2024 COMPLETE!
   Player models: /models/player_models_2022_2024.pkl
   Player metadata: /models/player_models_2022_2024_meta.json
   Game models: /models/game_models_2022_2024.pkl
   Game metadata: /models/game_models_2022_2024_meta.json
```

## Expected Training Time

### Per Window
- **Data loading**: 30-60 seconds
- **Player model training**: 10-20 minutes (5 props × 2-4 min each)
- **Game model training**: 2-5 minutes
- **Total per window**: 15-30 minutes

### All 27 Windows
- **Sequential**: 7-14 hours
- **Parallel (3x)**: 2.5-5 hours

## Expected Costs

### Modal Pricing
- **A10G GPU**: $1.10/hour
- **64GB RAM**: included
- **Storage**: Free (Modal volumes)

### Total Cost Estimate
- **Sequential training**: $8-16
- **Parallel (3x) training**: $8-16 (same total cost, just faster)

## Troubleshooting

### "No models found in cache"
- Normal! First run trains all windows from scratch
- After first run, cached windows will be skipped

### "File not found: aggregated_nba_data.parquet"
Upload Parquet first:
```bash
py -3.12 -m modal run modal_upload_parquet.py
```

### "Modal authentication failed"
Login first:
```bash
py -3.12 -m modal setup
```

### "CUDA out of memory"
Reduce batch size in modal_train.py line 136:
```python
batch_size=4096  # instead of 8192
```

## What Gets Saved

### Player Models (per window)
- `player_models_XXXX_YYYY.pkl`: 5 trained models (points, rebounds, assists, threes, minutes)
- `player_models_XXXX_YYYY_meta.json`: Training metrics and metadata

### Game Models (per window)
- `game_models_XXXX_YYYY.pkl`: 4 trained models (moneyline, spread, calibration, sigma)
- `game_models_XXXX_YYYY_meta.json`: Training metrics and metadata

### Total Files
- **27 windows** × **2 model types** × **2 files** = **108 files**
- **Size**: ~150-250 MB per window = **4-7 GB total**

## After Training

### 1. Download Models
```bash
py -3.12 -m modal volume get nba-models / model_cache
```

### 2. Verify Training
```bash
python verify_models.py
```

### 3. Create Ensemble Predictor
Combine all 27 windows into a single predictor for live 2025-2026 season predictions.

### 4. Backtest on 2024-2025
Validate model accuracy on complete 2024-2025 season data.

## Ready to Train

When you're ready, just run:
```bash
py -3.12 -m modal run modal_train.py
```

Everything is configured. The script will:
- ✅ Load 12.3M rows with 186 features
- ✅ Train 27 windows (3-year rolling windows)
- ✅ Train 5 player models per window (points, rebounds, assists, threes, minutes)
- ✅ Train 4 game models per window (moneyline, spread, calibration, sigma)
- ✅ Use TabNet + LightGBM hybrid with neural embeddings
- ✅ Save all models to Modal volume
- ✅ Report progress to your terminal

No compromises. Full feature set. Neural embeddings. Both player and game models.
