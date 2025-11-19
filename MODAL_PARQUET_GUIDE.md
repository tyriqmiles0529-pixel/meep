# Modal Training with Parquet - Simple Guide

## Why Parquet Instead of CSVs?

The CSV merging was failing because:
- Basketball Reference CSVs have a `player_id` column that's just a row index (1, 2, 3...)
- We need to merge on player NAMES, but name matching is unreliable
- Column name mismatches everywhere (`gameId` vs `game_id`, etc.)

**Solution**: Use the pre-aggregated Parquet file you already have!
- All 9 CSVs already merged ✅
- 180 columns with all features ✅
- 1947-2026 full historical data ✅
- Advanced stats, embeddings, temporal patterns ✅

## What's in the Parquet File?

From your training output, the Parquet has:
- **1,635,258 rows** (game-level player stats)
- **180 columns** including:
  - Base box scores (points, rebounds, assists, etc.)
  - Advanced metrics (PER, BPM, VORP, TS%, USG%)
  - Pace-adjusted stats (Per 100 Poss)
  - Plus/minus, turnovers (Play-By-Play)
  - Shooting zones and percentages
  - Rolling averages and momentum features
  - Player biographical data
  - Team context
- **Year range: 1947-2026** ✅

## Step-by-Step: Upload and Train

### 1. Upload Parquet to Modal
```bash
py -3.12 -m modal run modal_upload_parquet.py
```

This will:
- Mount your local `aggregated_nba_data.parquet`
- Copy it to Modal volume at `/data/aggregated_nba_data.parquet`
- Modal volume persists across runs (you only need to upload once!)

### 2. Train Single Window (Test)
```bash
py -3.12 -m modal run modal_train.py --window-start 2022 --window-end 2024
```

This will:
- Load Parquet from Modal volume (fast!)
- Train TabNet + LightGBM for 2022-2024
- Use A10G GPU + 64GB RAM (no limits!)
- Save model to `/models/player_models_2022_2024.pkl`
- All output in your local terminal ✅

### 3. Train All Windows (Full Training)
```bash
py -3.12 -m modal run modal_train.py --parallel 3
```

This will:
- Discover all seasons (1947-2026)
- Create 3-year windows (~27 windows total)
- Train 3 windows in parallel on separate GPUs
- Process entire history with full features!

### 4. Download Trained Models
```bash
py -3.12 -m modal volume get nba-models .
```

This downloads all models from Modal volume to your local machine.

## What Modal Gives You

### vs Kaggle Notebook
- **Time limit**: Unlimited (vs 9 hours)
- **RAM**: 64GB (vs 30GB)
- **GPU**: A10G dedicated (vs shared T4)
- **Parallel**: 3+ GPUs at once (vs 1 GPU)
- **Storage**: Persistent volumes (vs temporary)

### Cost
- A10G GPU: ~$1.10/hour
- Training 3 windows in parallel: ~$3.30/hour
- All ~27 windows: Estimated ~$10-15 total

## Key Files Updated

1. **modal_upload_parquet.py** - Uploads Parquet to Modal
2. **modal_train.py** - Now uses `/data/aggregated_nba_data.parquet`
3. **shared/data_loading.py** - Auto-detects Parquet vs CSV

## What You Get

After training, you'll have:
- Player models for every 3-year window from 1947-2026
- Models capture era evolution (rule changes, 3-point line, pace changes)
- TabNet embeddings learn player archetypes across history
- LightGBM ensemble for robust predictions
- Full temporal patterns from 79 years of NBA data!

Ready to train! 🚀
