# Refactored Training Architecture

## Overview

The training pipeline has been refactored into **modular, independent scripts** to prevent cascading failures and reduce RAM usage.

## New Structure

```
nba_predictor/
├── shared/
│   ├── data_loading.py       # Reusable data loading (79 optimized columns)
│   └── __init__.py
├── train_player_models.py    # Player props training (points, rebounds, assists, 3PM, minutes)
├── train_game_models.py       # Game outcome training (coming soon)
└── train_auto.py             # Legacy monolithic script (still works)
```

## Benefits

✅ **Independent Failures**: Game models save even if player training crashes
✅ **Less RAM**: Only load what you need (player OR game data)
✅ **Easier Debugging**: Smaller, focused scripts
✅ **Parallel Training**: Run on different machines simultaneously
✅ **Faster Iteration**: Test player models without waiting for game models

## Usage

### Player Model Training

Train player prop models independently:

```bash
# Full dataset (1947-2026, all 27 windows)
python train_player_models.py \
  --aggregated-data aggregated_nba_data.parquet \
  --window-size 3 \
  --neural-epochs 12

# Modern era only (2010+, 6 windows)
python train_player_models.py \
  --aggregated-data aggregated_nba_data.parquet \
  --min-year 2010 \
  --window-size 3 \
  --neural-epochs 12

# Skip cached windows (for incremental training)
python train_player_models.py \
  --aggregated-data aggregated_nba_data.parquet \
  --skip-cached

# Force retrain all windows
python train_player_models.py \
  --aggregated-data aggregated_nba_data.parquet \
  --force-retrain
```

### On Kaggle

```python
# In Kaggle notebook
!git clone https://github.com/tyriqmiles0529-pixel/meep.git
%cd meep

# Train only player models (saves RAM, faster)
!python train_player_models.py \
  --aggregated-data /kaggle/input/meepers/aggregated_nba_data.parquet \
  --window-size 3 \
  --neural-epochs 12 \
  --verbose
```

## Key Features

### Data Loading (`shared/data_loading.py`)

- **Memory Optimized**: Loads 79 columns instead of 186 (saves 10+ GB)
- **Chunked Loading**: Processes row groups one at a time (prevents 2.5GB malloc errors)
- **Year Filtering**: Optional `--min-year` and `--max-year` filters
- **Dtype Optimization**: Automatic downcasting to reduce memory

### Player Training (`train_player_models.py`)

- **Windowed Ensemble**: 3-year rolling windows for temporal accuracy
- **Independent Caching**: Each window cached separately
- **Crash Recovery**: Failed windows don't block others
- **Full Year Range**: Trains on 1947-2026 (or filtered range)

Example windows for full dataset:
- Window 1: 1947-1949
- Window 2: 1950-1952
- ...
- Window 26: 2022-2024
- Window 27: 2025-2026 (current season)

### Column Optimization

**Removed (39 redundant columns):**
- Metadata duplicates: `adv_lg`, `per100_age`, `pbp_pos`, etc.
- Cumulative stats: `adv_ows`, `adv_dws`, `adv_ws`
- Redundant percentages: `fieldGoalsPercentage`, `per100_fg_percent`

**Kept (79 high-value columns):**
- 18 base stats: IDs, dates, raw box score
- 4 high-value basic: Rebound splits, fouls, win
- 17 advanced rates: PER, TS%, BPM, VORP, usage%, etc.
- 22 per-100 stats: Pace-adjusted metrics
- 7 shooting stats: Avg distance, 3P zones, assisted rates
- 11 PBP stats: Plus/minus, turnovers, fouls, playmaking

## Migration from train_auto.py

### Before (Monolithic)
```bash
python train_auto.py \
  --aggregated-data data.parquet \
  --hybrid-player \
  --neural-epochs 12
```
- Trains game AND player models
- Game models lost if player crashes
- High RAM usage (loads everything)

### After (Modular)
```bash
# Train player models only
python train_player_models.py \
  --aggregated-data data.parquet \
  --neural-epochs 12

# Train game models only (coming soon)
python train_game_models.py \
  --teams-path data/TeamStatistics.csv
```
- Independent failures
- Lower RAM per script
- Easier to debug

## Output

### Cache Structure
```
model_cache/
├── player_models_1947_1949.pkl        # Window 1 models
├── player_models_1947_1949_meta.json  # Window 1 metadata
├── player_models_1950_1952.pkl        # Window 2 models
├── player_models_1950_1952_meta.json  # Window 2 metadata
└── ...
```

### Metadata Example
```json
{
  "window": "2022-2024",
  "seasons": [2022, 2023, 2024],
  "train_rows": 1540266,
  "metrics": {
    "window": "2022-2024",
    "train_rows": 1540266,
    "neural_epochs": 12
  }
}
```

## Next Steps

1. Extract actual training logic from `train_auto.py` into `train_player_models.py`
2. Create `train_game_models.py` for game outcome predictions
3. Create `predict.py` for unified inference (loads both game + player models)

## Questions About 2004+ Window Start

If you see windows starting from 2004-2006 instead of 1947-1949, check:

1. **Kaggle dataset**: Does `/kaggle/input/meepers/aggregated_nba_data.parquet` have full 1947-2026 data?
2. **Cached models**: Are windows 1947-2002 already cached in `model_cache/`?
3. **Memory limits**: Is `--memory-limit` flag being used (filters to 2002+)?

Verify full range:
```python
import pyarrow.parquet as pq
df = pq.read_table('aggregated_nba_data.parquet', columns=['season']).to_pandas()
print(f"Year range: {df['season'].min()}-{df['season'].max()}")
```
