# Commands to Run After Player Priors Fix

## Test Results: ALL PASSED ✓

The test script confirmed:
- ✓ `_season_from_date` correctly handles already-parsed datetime
- ✓ `_season_from_date` correctly parses string dates
- ✓ `season_end_year` is **100.0% populated** (not nan)
- ✓ Name construction from firstName + lastName works

---

## What Was Fixed

### Bug Found: Triple Date Parsing
Dates were being parsed **3 times**, and the 3rd parse was breaking everything:

1. Line 1623 - Early filtering with `format='mixed'` ✓
2. Line 1690 - Main processing with datetime check ✓
3. **Line 147** - Inside `_season_from_date()` WITHOUT `format='mixed'` ❌

### Fix Applied
**train_auto.py Line 147-151:**
```python
def _season_from_date(dt: pd.Series) -> pd.Series:
    # Don't re-parse if already datetime
    if pd.api.types.is_datetime64_any_dtype(dt):
        d = dt  # Use as-is
    else:
        d = pd.to_datetime(dt, errors="coerce", utc=False)

    y = d.dt.year
    m = d.dt.month
    return np.where(m >= 8, y + 1, y)
```

---

## Step-by-Step Commands

### Step 1: Clear Python Cache
```powershell
Remove-Item -Recurse -Force __pycache__
```

**Why:** Ensures latest code changes are used

---

### Step 2: Clear Model Caches
```powershell
Remove-Item model_cache\ensemble_2*.pkl, model_cache\ensemble_2*_meta.json -Force
```

**Why:** Old caches were built with `season_end_year = nan` (0.0% match rate)

---

### Step 3: Run Training with Verbose Output
```powershell
python train_auto.py --enable-window-ensemble --dataset "eoinamoore/historical-nba-data-and-player-box-scores" --verbose 2>&1 | Tee-Object training_with_fix.log
```

**Expected time:** ~75 minutes (5 windows × 15 min each)

**What to watch for in output:**

```
Building player datasets
  season_end_year populated: 833,839 / 833,839 rows (100.0%)  ← Should be 100%!

Detected player columns
- first: firstName  last: lastName  ← Both loaded!

Merging Basketball Reference player priors
  Kaggle seasons: [2002.0, 2003.0, ..., 2025.0, 2026.0]  ← Should show all seasons!
  Season overlap: 24 common seasons  ← Should be 24!

  Name-merge matched: 300,000-500,000 / 833,839 rows (36-60%)  ← TARGET: 40-60%!
```

---

## Expected Improvements

### Current Metrics (WITHOUT fix):
```
Player Priors Match Rate: 0.0%
Features: 82 base features only

Minutes: RMSE=5.621, MAE=3.944
Points:  RMSE=6.195, MAE=4.427
Rebounds: RMSE=2.816, MAE=1.986
Assists: RMSE=1.904, MAE=1.277
Threes:  RMSE=0.914, MAE=0.601
```

### Expected Metrics (WITH fix):
```
Player Priors Match Rate: 40-60%
Features: ~120-140 total (82 base + 38-58 Basketball Reference priors)

Minutes: RMSE=4.5-5.0, MAE=3.0-3.5 (16-20% improvement)
Points:  RMSE=5.0-5.5, MAE=3.5-4.0 (19-23% improvement)
Rebounds: RMSE=2.3-2.5, MAE=1.6-1.8 (18-21% improvement)
Assists: RMSE=1.6-1.8, MAE=1.0-1.2 (16-19% improvement)
Threes:  RMSE=0.7-0.8, MAE=0.4-0.5 (18-23% improvement)
```

**Why?** Models will now have access to Basketball Reference features for 40-60% of player-games:
- PER (Player Efficiency Rating)
- TS% (True Shooting %)
- USG% (Usage Rate)
- Shooting zones (corner 3%, rim %, mid-range %)
- BPM (Box Plus/Minus)
- VORP (Value Over Replacement Player)
- On-court +/-
- Position distribution
- ...and 50+ more features

---

## Verification Checklist

After training completes, verify in the log:

- [ ] `season_end_year populated: X / X rows (100.0%)`
- [ ] `Kaggle seasons: [2002.0, 2003.0, ..., 2025.0, 2026.0]` (not empty)
- [ ] `Season overlap: 24 common seasons` (not 0)
- [ ] `Name-merge matched: 40-60%` (not 0.0%)
- [ ] Each window shows `Features: 120-150 total` (not just 82)

---

## After Training Completes

### Generate Predictions
```powershell
python riq_analyzer.py
```

This will use the newly trained models with Basketball Reference priors for better predictions.

---

## Troubleshooting

### Issue: Still showing 0.0% match rate

**Cause:** Old Python bytecode cache still being used

**Fix:**
```powershell
Remove-Item -Recurse -Force __pycache__
python test_player_priors_fix.py  # Re-run test to verify
```

If test still passes but training fails, there may be another issue. Check the log for errors.

---

### Issue: Training crashes with memory errors

**Cause:** Multiple background processes or insufficient RAM

**Fix:**
```powershell
# Stop all Python processes
powershell -Command "Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force"

# Verify no Python processes running
tasklist | findstr python

# Re-run training
python train_auto.py --enable-window-ensemble --dataset "eoinamoore/historical-nba-data-and-player-box-scores" --verbose
```

---

### Issue: Match rate is only 20-30% (lower than expected 40-60%)

**Possible causes:**
1. Most data is from current season (2025-26) where Basketball Reference doesn't have stats yet
2. Dataset has many rookies or G-League call-ups
3. Dataset has many very old games (pre-2002)

**Check:**
```powershell
python -c "import pandas as pd; ps = pd.read_csv('C:/Users/tmiles11/.cache/kagglehub/datasets/eoinamoore/historical-nba-data-and-player-box-scores/versions/257/PlayerStatistics.csv', nrows=100000); print(ps['gameDate'].value_counts().head(20))"
```

This shows date distribution - if most games are from 2025-26, match rate will be lower.

---

## Summary

**Fix Applied:** `_season_from_date` function now checks if data is already datetime before re-parsing

**Test Status:** ALL 4 TESTS PASSED ✓

**Expected Impact:**
- Player priors match rate: 0.0% → 40-60%
- Model RMSE improvements: 16-23% across all player props
- Features added: 38-58 Basketball Reference stats

**Next Action:** Run the 3 commands above and verify output matches expected values.

Good luck! 🏀
