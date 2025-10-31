# Cache Management Guide

## Understanding the Cache System

Your training pipeline uses two cache directories:

### 1. model_cache/ - Window Ensemble Caches
```
model_cache/
├── ensemble_2002_2006.pkl (5.7 MB)
├── ensemble_2002_2006_meta.json
├── ensemble_2007_2011.pkl (6.1 MB)
├── ensemble_2007_2011_meta.json
├── ensemble_2012_2016.pkl (5.8 MB)
├── ensemble_2012_2016_meta.json
├── ensemble_2017_2021.pkl (5.7 MB)
├── ensemble_2017_2021_meta.json
├── ensemble_2022_2026.pkl (4.9 MB)
└── ensemble_2022_2026_meta.json
```

**What's cached:**
- Trained ensemble models for each 5-year window
- Ridge, Elo, Four Factors, LightGBM base models
- Meta-learner weights for combining predictions
- Refit history (coefficients over time)

**When to clear:**
- After fixing player name matching (to include 68 new Basketball Reference features)
- After changing feature engineering logic
- After updating dataset (new games added)
- When you want to re-optimize from scratch

### 2. models/ - Final Ensemble Models
```
models/
├── ridge_model_enhanced.pkl
├── elo_model_enhanced.pkl
├── four_factors_model_enhanced.pkl
├── lgb_model_enhanced.pkl
├── ensemble_meta_learner_enhanced.pkl
└── ensemble_results_enhanced.json
```

**What's cached:**
- Final ensemble trained on ALL 2002-2026 data
- Used by riq_analyzer.py for live predictions
- Generated AFTER window ensembles complete

**When to clear:**
- After all window caches are cleared and retrained
- When making predictions on completely new data structure

---

## Commands to Clear Caches

### Option 1: Clear ONLY 5-Year Window Caches (Recommended)
This forces retraining of each window with the new player name matching fix:

```bash
# Windows (PowerShell):
Remove-Item model_cache\ensemble_2*.pkl, model_cache\ensemble_2*_meta.json -Force

# Linux/Mac:
rm -f model_cache/ensemble_2*.pkl model_cache/ensemble_2*_meta.json
```

**What happens next:**
- Training will rebuild each window from scratch
- With the firstName + lastName fix, windows will now include Basketball Reference priors
- Each window will take ~10-15 minutes to retrain
- Total retraining time: ~50-75 minutes (5 windows)

---

### Option 2: Clear ALL Caches (Full Reset)
This clears both window caches AND final ensemble models:

```bash
# Windows (PowerShell):
Remove-Item model_cache\*.pkl, model_cache\*.json, models\*_enhanced.pkl, models\*_enhanced.json -Force

# Linux/Mac:
rm -f model_cache/*.pkl model_cache/*.json models/*_enhanced.pkl models/*_enhanced.json
```

**What happens next:**
- Training rebuilds all 5 windows (50-75 minutes)
- Training rebuilds final ensemble on full dataset (10-15 minutes)
- Total retraining time: ~60-90 minutes

---

### Option 3: Clear ONLY Final Ensemble (Keep Windows)
Useful if windows are already trained with good features, but you want to re-combine them:

```bash
# Windows (PowerShell):
Remove-Item models\*_enhanced.pkl, models\*_enhanced.json -Force

# Linux/Mac:
rm -f models/*_enhanced.pkl models/*_enhanced.json
```

**What happens next:**
- Skips window retraining (uses cached windows)
- Only retrains final ensemble on full dataset (~10-15 minutes)

---

### Option 4: Clear Specific Window (Granular Control)
To retrain just one window (e.g., 2022-2026 current season):

```bash
# Windows (PowerShell):
Remove-Item model_cache\ensemble_2022_2026.pkl, model_cache\ensemble_2022_2026_meta.json -Force

# Linux/Mac:
rm -f model_cache/ensemble_2022_2026.pkl model_cache/ensemble_2022_2026_meta.json
```

**What happens next:**
- Only the 2022-2026 window retrains (~10-15 minutes)
- Other windows use cached models
- Useful for updating current season without full retrain

---

## Recommended Workflow After Player Name Fix

Since you just fixed the firstName + lastName loading issue, here's the recommended approach:

### Step 1: Clear Window Caches
```bash
# PowerShell:
Remove-Item model_cache\ensemble_2*.pkl, model_cache\ensemble_2*_meta.json -Force

# Linux/Mac:
rm -f model_cache/ensemble_2*.pkl model_cache/ensemble_2*_meta.json
```

### Step 2: Run Training with Verbose Output
```bash
python train_auto.py --enable-window-ensemble --dataset "eoinamoore/historical-nba-data-and-player-box-scores" --verbose
```

### Step 3: Verify Improved Match Rate
Look for this in the output:
```
Merging Basketball Reference player priors
  Filtered priors from 185,226 to 35,489 rows (seasons 2002-2026)
  DEBUG - Raw Kaggle names: ['LeBron James', 'Stephen Curry', ...]  ← Full names now!
  Name overlap (sample up to 5k): 500-2000 common normalized names  ← Much better!

Player priors matched by ID for 250,000 rows
Player priors matched by name for 400,000 rows (80%+ match rate!)  ← TARGET!
```

### Step 4: Check Window Training
Each window should now show improved metrics:
```
Training window 2002-2006...
  Merged player priors: 80,000 / 100,000 player-games (80% match!)  ← Good!
  Features: 150 total (82 base + 68 Basketball Reference priors)  ← Priors included!

✓ Enhanced Ensembler: Logloss = 0.6524  ← Should improve with priors
```

---

## Cache Validation (When to Trust Cached Models)

### Cached models are VALID when:
- ✅ No code changes to feature engineering
- ✅ No changes to player priors matching logic
- ✅ No new games added to dataset
- ✅ Training completes successfully
- ✅ Meta.json shows expected feature count

### Cached models are INVALID when:
- ❌ Player name matching was broken (like before firstName + lastName fix)
- ❌ Feature engineering changed (new rolling windows, new stats)
- ❌ Dataset updated (new season games added)
- ❌ Training crashed mid-window
- ❌ Memory errors during prior matching

### How to Check Cache Validity
```bash
# Check what features were cached in meta.json:
cat model_cache/ensemble_2022_2026_meta.json

# Look for:
{
  "window": "2022-2026",
  "n_games": 5681,
  "n_features": 150,  ← Should be ~150 if priors are included, ~82 if not
  "timestamp": "2024-10-30T16:07:23"
}
```

If `n_features` is ~82 instead of ~150, the cache was built WITHOUT Basketball Reference priors and should be cleared.

---

## Automatic Cache Invalidation

The training script automatically invalidates caches when:
1. Dataset path changes
2. Window boundaries change (e.g., 5-year → 10-year windows)
3. Feature set changes (detected by column count mismatch)

**Manual invalidation needed when:**
- Fix bugs in feature engineering (like firstName + lastName fix)
- Improve name matching logic
- Change normalization functions

---

## Disk Space Management

Current cache sizes:
```
model_cache/: ~28 MB (5 windows × 5.6 MB average)
models/: ~15 MB (final ensemble models)
TOTAL: ~43 MB
```

**To check cache size:**
```bash
# PowerShell:
Get-ChildItem model_cache, models -Recurse | Measure-Object -Property Length -Sum

# Linux/Mac:
du -sh model_cache/ models/
```

**Safe to delete:**
- Old window caches (ensemble_1947_1951.pkl, ensemble_2026_2026.pkl)
- Backup files (*_backup.pkl)
- JSON metadata files (can be regenerated)

**DO NOT delete during training:**
- Partially written .pkl files (training may crash)

---

## Summary Commands

**Quick reference for cache management:**

```bash
# Clear all 5-year window caches (RECOMMENDED after firstName + lastName fix):
rm -f model_cache/ensemble_2*.pkl model_cache/ensemble_2*_meta.json

# Clear everything and start fresh:
rm -f model_cache/*.pkl model_cache/*.json models/*_enhanced.*

# Clear only current season window:
rm -f model_cache/ensemble_2022_2026.*

# Clear only final ensemble (keep windows):
rm -f models/*_enhanced.*

# Check cache sizes:
du -sh model_cache/ models/

# Retrain with verbose output:
python train_auto.py --enable-window-ensemble --dataset "eoinamoore/historical-nba-data-and-player-box-scores" --verbose
```

---

## Expected Timeline After Clearing Caches

**Full retrain (5 windows + final ensemble):**
- Window 2002-2006: ~12 minutes
- Window 2007-2011: ~14 minutes
- Window 2012-2016: ~13 minutes
- Window 2017-2021: ~12 minutes
- Window 2022-2026: ~10 minutes
- Final ensemble: ~15 minutes
- **TOTAL: ~76 minutes**

**Benefits after retrain:**
- 80%+ player-game matches (vs 0.7% before)
- 68 additional Basketball Reference features per matched game
- Better predictions from improved context
- Models learn patterns like "high USG% player + low minutes = blowout"

---

## After Training Completes

Once training finishes with cleared caches:

1. **Verify match rate in logs:**
   ```
   Player priors matched by name for 400,000+ rows (80%+ match rate)
   ```

2. **Check final ensemble performance:**
   ```
   ✓ COMPLETE: Ridge + Elo + 4F + LGB ensemble ready
   Logloss: 0.6520 (should improve with priors vs 0.6624 before)
   ```

3. **Models are ready for predictions:**
   ```bash
   python riq_analyzer.py  # Use enhanced models for betting analysis
   ```

4. **Caches are now VALID** - future training runs will use them unless you clear again
