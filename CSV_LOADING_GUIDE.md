# CSV Loading Guide - Bypass Parquet Issues

## Problem

If you're experiencing issues with the aggregated Parquet file (missing years, filtered data, etc.), you can now **load directly from the raw Kaggle CSV files**.

## Solution

The refactored training script now supports **both Parquet AND CSV** formats with auto-detection.

## Usage on Kaggle

### Option A: Use Aggregated Parquet (Recommended if working)

```bash
python train_player_models.py \
  --data /kaggle/input/meepers/aggregated_nba_data.parquet \
  --window-size 3 \
  --neural-epochs 12
```

### Option B: Use Raw CSV (Guaranteed Full Data)

```bash
python train_player_models.py \
  --data /kaggle/input/historical-nba-data-and-player-box-scores/PlayerStatistics.csv \
  --window-size 3 \
  --neural-epochs 12
```

## Auto-Detection

The script automatically detects the format based on file extension:
- `.parquet` → Uses optimized Parquet loader (79 columns, row-group chunking)
- `.csv` → Uses CSV loader (all columns, dtype optimization)

## Verification

Check that you're getting the full year range:

```python
from shared.data_loading import load_player_data, get_season_range

# Test Parquet
df1 = load_player_data('/kaggle/input/meepers/aggregated_nba_data.parquet')
print(f"Parquet range: {get_season_range(df1)}")

# Test CSV
df2 = load_player_data('/kaggle/input/historical-nba-data-and-player-box-scores/PlayerStatistics.csv')
print(f"CSV range: {get_season_range(df2)}")
```

Expected output:
```
Parquet range: (1947, 2026)  # or (2004, 2026) if filtered
CSV range: (1947, 2026)      # always full range
```

## When to Use CSV vs Parquet

### Use CSV When:
- ✅ Parquet file has missing years (e.g., only 2004-2026)
- ✅ You want guaranteed full historical data (1947-2026)
- ✅ Debugging data issues
- ✅ First time training to ensure full coverage

### Use Parquet When:
- ✅ File contains full 1947-2026 range
- ✅ You want faster loading (columnar format)
- ✅ You want optimized 79 columns (vs 100+ in CSV)
- ✅ Lower memory usage during loading

## Updated Kaggle Notebook

Cell 3 in `KAGGLE_TRAINING_REFACTORED.ipynb` now supports both:

```python
# Option A: Parquet
DATA_PATH = "/kaggle/input/meepers/aggregated_nba_data.parquet"

# Option B: CSV (use this if Parquet has issues)
# DATA_PATH = "/kaggle/input/historical-nba-data-and-player-box-scores/PlayerStatistics.csv"

agg_df = load_player_data(DATA_PATH, verbose=True)
```

## Benefits

| Feature | Parquet | CSV |
|---------|---------|-----|
| Full 1947-2026 range | ⚠️ Depends on file | ✅ Guaranteed |
| Loading speed | ✅ Fast (columnar) | ⚠️ Slower |
| Memory usage | ✅ Lower (79 cols) | ⚠️ Higher (all cols) |
| Advanced stats | ✅ Pre-computed | ❌ Need to merge |
| Debugging | ⚠️ Opaque binary | ✅ Human-readable |

## Recommendation

**Start with CSV** for your first full training run to ensure you have all 27 windows (1947-2026). Once confirmed working, you can switch to Parquet for faster iterations.

```bash
# First run: Use CSV to ensure full data
python train_player_models.py \
  --data /kaggle/input/historical-nba-data-and-player-box-scores/PlayerStatistics.csv \
  --window-size 3

# Later runs: Use Parquet for speed (if it has full data)
python train_player_models.py \
  --data /kaggle/input/meepers/aggregated_nba_data.parquet \
  --window-size 3 \
  --skip-cached  # Skip already trained windows
```
