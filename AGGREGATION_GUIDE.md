# 🚀 Pre-Aggregation Guide - Eliminate Training Merge Time

## Problem You're Solving

**BEFORE (slow training):**
```
Training starts
├─ Load PlayerStatistics.csv (1.6M rows)
├─ Load Advanced.csv (33K rows)
├─ Fuzzy match player names... 5-8 minutes
├─ Load Per 100 Poss.csv
├─ Fuzzy match player names... 3-5 minutes
├─ Load Shooting.csv
├─ Fuzzy match... 2-3 minutes
├─ Load Team priors
├─ Fuzzy match teams... 2-3 minutes
└─ TOTAL MERGE TIME: **10-20 minutes**

Then training actually starts...
```

**AFTER (fast training):**
```
Pre-aggregation (ONE TIME):
├─ Run create_aggregated_dataset.py
├─ Merge everything once
├─ Save aggregated_nba_data.csv.gzip
└─ Time: 5-10 minutes (ONE TIME ONLY)

Training (every time):
├─ Load aggregated_nba_data.csv.gzip
└─ Start training immediately
    TIME SAVED: 10-20 minutes per training run!
```

---

## One-Time Setup

### Step 1: Create Aggregated Dataset

```bash
# Run this ONCE to create the pre-merged dataset
python create_aggregated_dataset.py \
  --player-csv PlayerStatistics.csv \
  --priors-dir priors_data \
  --output aggregated_nba_data.csv \
  --compression gzip
```

**Output:**
- `aggregated_nba_data.csv.gzip` (~500-800 MB)
- Contains ALL merges pre-done
- Ready for instant training

**Time:** 5-10 minutes (one-time cost)

---

### Step 2: Upload to Colab

Instead of uploading:
- ❌ PlayerStatistics.csv (303 MB)
- ❌ priors_data.zip (4.6 MB)
- ❌ Total: 308 MB + merge time

Upload:
- ✅ aggregated_nba_data.csv.gzip (~500 MB)
- ✅ Total: 500 MB, **NO merge time**

**Net benefit:**
- Slightly larger file (~200 MB more)
- **Saves 10-20 minutes EVERY training run**
- Worth it after 2-3 training runs

---

## Monthly Update Workflow

### Option A: Re-aggregate from Scratch (Simplest)

```bash
# 1. Download latest Kaggle dataset
kagglehub dataset download eoinamoore/historical-nba-data-and-player-box-scores

# 2. Re-create aggregated dataset
python create_aggregated_dataset.py --output aggregated_nba_data_nov2024.csv

# 3. Upload new file to Colab
# 4. Train on updated data
```

**Pros:**
- Simple, clean
- Guaranteed consistency
- All new games included

**Cons:**
- Need to re-upload ~500 MB each month
- 5-10 min aggregation time

---

### Option B: Incremental Append (Advanced)

```python
# 1. Load existing aggregated data
df_existing = pd.read_csv('aggregated_nba_data.csv.gzip')

# 2. Get new games from NBA API
from nba_api.stats.endpoints import LeagueGameLog
new_games = LeagueGameLog(season='2024-25',
                          date_from_nullable='2024-11-01')

# 3. Merge priors for new games only
new_games_merged = merge_player_priors(new_games, df_adv, df_per100, ...)

# 4. Append to existing
df_updated = pd.concat([df_existing, new_games_merged])

# 5. Save
df_updated.to_csv('aggregated_nba_data.csv.gzip', compression='gzip')
```

**Pros:**
- Faster (only process new games)
- Smaller file transfers

**Cons:**
- More complex
- Risk of duplicates

---

## What Gets Aggregated

The script merges 7 CSV files into 1:

```
aggregated_nba_data.csv.gzip
│
├─ PlayerStatistics.csv (1946-2025, 1.6M rows)
│  └─ gameDate, personId, points, rebounds, assists, etc.
│
├─ Advanced.csv (1947-2025, season-level)
│  └─ adv_per, adv_ts_percent, adv_usg_percent, adv_ws, adv_bpm, adv_vorp
│
├─ Per 100 Poss.csv (1974-2025, season-level)
│  └─ per100_pts, per100_trb, per100_ast, per100_stl, per100_blk
│
├─ Player Shooting.csv (1997-2025, season-level)
│  └─ shoot_fg_pct_00_03, shoot_fg_pct_03_10, shoot_3p_pct
│
├─ Player Play By Play.csv (1997-2025, season-level)
│  └─ pbp_on_court_plus_minus, pbp_bad_pass_tov, pbp_lost_ball_tov
│
├─ Team Summaries.csv (1947-2025, season-level)
│  └─ team_w, team_l, team_srs, team_pace, team_ortg, team_drtg
│
└─ Team Abbrev.csv (1947-2025, season-level)
   └─ team abbreviation mappings
```

**Result:** Each game row has:
- Game stats (points, rebounds, etc.)
- Player season context (PER, Usage%, efficiency)
- Team season context (pace, ratings, record)
- Shooting splits (if available for that era)
- Play-by-play metrics (if available)

---

## Column Count Estimate

**Before aggregation:**
- PlayerStatistics.csv: ~35 columns

**After aggregation:**
- PlayerStatistics columns: 35
- Advanced stats: ~28 columns (adv_*)
- Per 100 Poss: ~32 columns (per100_*)
- Shooting: ~30 columns (shoot_*)
- Play-by-Play: ~24 columns (pbp_*)
- Team context: ~29 columns (team_*)
- **TOTAL: ~178 columns**

All prefixed to avoid conflicts:
- `adv_per` = Player Efficiency Rating from Advanced.csv
- `per100_pts` = Points per 100 possessions
- `shoot_3p_pct` = 3-point percentage from Shooting.csv
- `pbp_on_court_plus_minus` = On-court +/- from Play-by-Play
- `team_pace` = Team pace from Team Summaries

---

## Benefits

### 1. **Massive Time Savings**
- **Per training run**: Save 10-20 minutes
- **10 training runs**: Save 100-200 minutes (1.5-3 hours!)
- **Monthly retrains**: Save hours over time

### 2. **Consistency**
- Merge logic runs ONCE
- Same data for all training runs
- No merge bugs during training

### 3. **Simplicity**
- Training code simpler (just load CSV)
- No fuzzy matching logic needed
- Easier debugging

### 4. **Reproducibility**
- Version aggregated datasets
- `aggregated_nba_data_nov2024.csv.gzip`
- `aggregated_nba_data_dec2024.csv.gzip`
- Easy rollback if needed

---

## Next Steps

1. ✅ **Create aggregated dataset** (running now...)
2. ⏳ **Upload to Colab** (replace PlayerStatistics.csv + priors)
3. ⏳ **Update train_auto.py** to use aggregated data
4. ⏳ **Test training** (should be 10-20 min faster!)
5. ⏳ **Set up monthly re-aggregation** workflow

---

## File Sizes

| File | Size | Upload Time |
|------|------|-------------|
| PlayerStatistics.csv | 303 MB | ~2-3 min |
| priors_data.zip | 4.6 MB | ~10 sec |
| **OLD TOTAL** | **308 MB** | **~3 min** |
| | | |
| aggregated_nba_data.csv.gzip | ~500 MB | ~4-5 min |
| **NEW TOTAL** | **500 MB** | **~5 min** |

**Trade-off:**
- +2 min upload time
- **-10 to -20 min training time**
- **Net savings: 8-18 minutes per run**

Worth it!
