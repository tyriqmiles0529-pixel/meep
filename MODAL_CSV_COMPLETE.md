# Modal CSV Training Setup - COMPLETE

## What We Just Completed

Fixed CSV aggregation to properly merge all 9 CSV files from both Kaggle datasets:

### Datasets
1. **eoinamoore/historical-nba-data-and-player-box-scores**
   - PlayerStatistics.csv (1947-2026 game-level box scores)
   - Players.csv (biographical data)
   - TeamStatistics.csv (team context)
   - Games.csv (game metadata)

2. **sumitrodatta/nba-aba-baa-stats**
   - Player Advanced.csv (PER, BPM, VORP, etc.)
   - Player Per 100 Poss.csv (pace-adjusted stats)
   - Player Play-By-Play.csv (plus/minus, turnovers)
   - Player Shooting.csv (shooting zones, percentages)
   - Team Summaries.csv (team aggregates)

### Key Fixes

1. **Created Season Column** (shared/csv_aggregation.py:70-80)
   - Extracts season from gameDate using NBA season logic
   - Oct-Dec games belong to next year's season
   - Now PlayerStatistics has `season` column for merging

2. **Created Player Name Column** (shared/csv_aggregation.py:82-87)
   - Concatenates firstName + lastName
   - Basketball Reference CSVs use player names (not IDs)
   - Now PlayerStatistics has `Player` column for merging

3. **Auto-Detection of Merge Keys**
   - Main dataframe prefers `Player` + `season` (lines 124-140)
   - Each Basketball Reference CSV auto-detects its columns
   - Advanced stats (lines 150-176): Detects `Player`/`Season`, renames to match
   - Per 100 Poss (lines 204-232): Same pattern
   - Play-By-Play (lines 258-286): Same pattern
   - Shooting (lines 312-340): Same pattern
   - Players.csv (lines 369-390): Detects `player_id`/`personId`

4. **Created Unified Load Function** (shared/data_loading.py:120-159)
   - `load_player_data()` auto-detects input type
   - Parquet file → loads Parquet
   - CSV file → loads single CSV
   - Directory → aggregates all 9 CSVs
   - Modal will use directory path: `/data/csv_dir`

5. **Updated Modal Training Script** (modal_train.py:61-66)
   - Changed from `/data/csv_dir/PlayerStatistics.csv` (single file)
   - To `/data/csv_dir` (entire directory)
   - Now loads all 9 CSVs with full feature set

## What's Already Set Up

1. ✅ Modal CLI installed and authenticated
2. ✅ Kaggle credentials stored as Modal secrets
3. ✅ modal_upload_data.py created and tested
4. ✅ All 9 CSVs uploaded to Modal volume `nba-data` in `/data/csv_dir/`
5. ✅ modal_train.py created with A10G GPU, 64GB RAM
6. ✅ CSV aggregation logic complete with robust column detection

## Ready to Train on Modal!

### Test Single Window
```bash
py -3.12 -m modal run modal_train.py --window-start 2022 --window-end 2024
```

This will:
1. Load all 9 CSVs from Modal volume
2. Merge them using auto-detected player names and seasons
3. Train TabNet + LightGBM hybrid models for 2022-2024
4. Save to Modal volume at `/models/player_models_2022_2024.pkl`
5. All output appears in your local terminal!

### Train All Windows (Parallel)
```bash
py -3.12 -m modal run modal_train.py --parallel 3
```

This will:
1. Discover all uncached windows
2. Train 3 windows in parallel on separate A10G GPUs
3. Process ~27 windows total (1947-2026 in 3-year chunks)
4. Save all models to Modal volume

### Download Models
```bash
py -3.12 -m modal volume get nba-models .
```

## Why This Works

**Before**: PlayerStatistics used `personId`, Basketball Reference used `Player` names
- Merge failed with KeyError: 'personId' not in Advanced.csv

**After**:
- Create `Player` column in PlayerStatistics (firstName + lastName)
- Create `season` column from gameDate (NBA season logic)
- Auto-detect merge keys in each CSV
- Rename columns to match before merging
- Merge on [`Player`, `season`] successfully!

**Result**: Full 1947-2026 historical data with all advanced stats:
- Game-level box scores (PlayerStatistics)
- Advanced metrics (PER, BPM, VORP)
- Pace-adjusted stats (Per 100 Poss)
- Plus/minus, turnovers (Play-By-Play)
- Shooting zones and percentages
- Player biographical data
- Team context

All 9 CSVs merged into one comprehensive dataset! 🎉
