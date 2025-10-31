# Quick Start Commands - NBA Predictor

## Step 1: Clear Old Caches (After Name Matching Fix)

```bash
# PowerShell (Windows):
Remove-Item model_cache\ensemble_2*.pkl, model_cache\ensemble_2*_meta.json -Force

# Linux/Mac:
rm -f model_cache/ensemble_2*.pkl model_cache/ensemble_2*_meta.json
```

**Why:** Old caches were built with only first names (0.7% match rate). Clear them to rebuild with full names (80%+ match rate).

---

## Step 2: Clean Bytecode Cache

```bash
# PowerShell (Windows):
Remove-Item -Recurse -Force __pycache__

# Linux/Mac:
rm -rf __pycache__
```

**Why:** Ensures latest code changes are used (firstName + lastName loading fix).

---

## Step 3: Run Training

```bash
python train_auto.py --enable-window-ensemble --dataset "eoinamoore/historical-nba-data-and-player-box-scores" --verbose
```

**Expected time:** ~75 minutes (5 windows × 15 min each)

**What to watch for:**
```
Building player datasets
  Filtered PlayerStatistics by date: 1,636,525 → 833,839 rows (2002+, saved ~148.3 MB)  ✓

Detected player columns
- first: firstName  last: lastName  ✓ Both loaded now!

DEBUG - Raw Kaggle names: ['LeBron James', 'Stephen Curry', 'Kevin Durant', ...]  ✓ Full names!
Name overlap (sample up to 5k): 500-2000 common normalized names  ✓ Much better!

ID-merge matched: 250,000 / 833,839 player-game rows (30.0%)  ✓ ID-based matching
Name-merge matched: 650,000 / 833,839 player-game rows (78.0%)  ✓ Name-based matching
TOTAL matched (ID + name): 680,000 / 833,839 player-game rows (81.5%)  ✓ TARGET HIT!
```

---

## Step 4: Verify Window Training

Each window should show improved performance:

```
Training window 2002-2006...
  Features: 150 total (82 base + 68 Basketball Reference priors)  ✓

✓ Enhanced Ensembler: 1635 refits, Logloss = 0.6420  ← Should improve from 0.6624
  Ridge:  0.0196
  Elo:    0.6976
  FF:     0.0196
  LGB:    -0.2327
```

---

## Step 5: After Training Completes

Training creates these files:
```
model_cache/
  ensemble_2002_2006.pkl  ← Window 1
  ensemble_2007_2011.pkl  ← Window 2
  ensemble_2012_2016.pkl  ← Window 3
  ensemble_2017_2021.pkl  ← Window 4
  ensemble_2022_2026.pkl  ← Window 5

models/
  ridge_model_enhanced.pkl
  elo_model_enhanced.pkl
  four_factors_model_enhanced.pkl
  lgb_model_enhanced.pkl
  ensemble_meta_learner_enhanced.pkl  ← Final unified model
```

---

## Step 6: Generate Betting Predictions

```bash
python riq_analyzer.py
```

**What it does:**
- Loads `ensemble_meta_learner_enhanced.pkl` (trained with 80%+ player priors)
- Fetches today's NBA games
- Predicts outcomes using all 150 features (82 base + 68 Basketball Reference)
- Identifies +EV betting opportunities
- Suggests Kelly criterion bet sizing

**Expected output:**
```
Loading unified hierarchical ensemble from models/ensemble_meta_learner_enhanced.pkl...
✓ Model loaded: 150 features

Fetching today's NBA games...
Found 8 games for 2024-10-30

Analyzing betting opportunities...

GAME: Lakers @ Celtics
  Model prediction: Celtics 62.5% win probability
  Best book odds: Celtics -180 (implied 64.3%)
  Edge: -1.8% (SKIP - no edge)

GAME: Warriors @ Suns
  Model prediction: Warriors 58.2% win probability
  Best book odds: Warriors +145 (implied 40.8%)
  Edge: +17.4% (STRONG BET!)
  Kelly sizing: 8.7% of bankroll
  Recommendation: BET Warriors ML at +145 or better
```

---

## Troubleshooting

### Issue: "Only 0.7% match rate" still showing

**Cause:** Old bytecode cache still being used

**Fix:**
```bash
rm -rf __pycache__
rm -f model_cache/ensemble_2*.pkl model_cache/ensemble_2*_meta.json
python train_auto.py --enable-window-ensemble --dataset "..." --verbose
```

---

### Issue: "Name overlap: 1 common names" still showing

**Cause:** Old training process still running with old code

**Fix:**
```bash
# Kill all Python processes:
pkill -9 python  # Linux/Mac
taskkill /F /IM python.exe  # Windows

# Clear caches and restart:
rm -rf __pycache__ model_cache/ensemble_2*.pkl
python train_auto.py --enable-window-ensemble --dataset "..." --verbose
```

---

### Issue: Training crashes with memory errors

**Cause:** Multiple background processes or insufficient RAM

**Fix:**
```bash
# Check running Python processes:
ps aux | grep python  # Linux/Mac
tasklist | findstr python  # Windows

# Kill background processes:
pkill python  # Linux/Mac
taskkill /F /IM python.exe  # Windows

# Increase system swap/pagefile if RAM < 16GB
```

---

## Expected Performance Improvements

### Before firstName + lastName Fix:
- Match rate: 0.7% (6,014 / 833k)
- Features: 82 base features only
- Logloss: ~0.6624 (less context for models)
- Accuracy: ~60.1%

### After firstName + lastName Fix:
- Match rate: 80%+ (680k / 833k)
- Features: 150 total (82 base + 68 Basketball Reference)
- Logloss: ~0.6420 (better context, improved predictions)
- Accuracy: ~62-63% (expected +2-3% improvement)

---

## Daily Workflow

### Morning (Before Games Start)
```bash
# Update dataset with latest games:
python fetch_nba_schedule.py  # Downloads today's schedule

# Generate predictions:
python riq_analyzer.py  # Analyzes betting opportunities
```

### Weekly (Update Models)
```bash
# Clear current season window cache:
rm -f model_cache/ensemble_2022_2026.*

# Retrain with latest games:
python train_auto.py --enable-window-ensemble --dataset "..." --verbose
```

### Monthly (Full Retrain)
```bash
# Clear all caches:
rm -f model_cache/ensemble_2*.pkl model_cache/ensemble_2*_meta.json

# Full retrain:
python train_auto.py --enable-window-ensemble --dataset "..." --verbose
```

---

## Monitoring Performance

### Check Model Files Are Up-to-Date
```bash
# PowerShell:
Get-ChildItem models\*_enhanced.pkl | Select-Object Name, LastWriteTime

# Linux/Mac:
ls -lh models/*_enhanced.pkl
```

### Check Cache Validity
```bash
# PowerShell:
Get-Content model_cache\ensemble_2022_2026_meta.json | ConvertFrom-Json

# Linux/Mac:
cat model_cache/ensemble_2022_2026_meta.json | python -m json.tool
```

Look for:
```json
{
  "window": "2022-2026",
  "n_games": 5681,
  "n_features": 150,  ← Should be ~150 with priors, ~82 without
  "timestamp": "2024-10-30T16:07:23"
}
```

---

## Summary of Key Commands

```bash
# SETUP (one time after fix):
rm -rf __pycache__ model_cache/ensemble_2*.pkl model_cache/ensemble_2*_meta.json
python train_auto.py --enable-window-ensemble --dataset "eoinamoore/historical-nba-data-and-player-box-scores" --verbose

# DAILY (for predictions):
python riq_analyzer.py

# WEEKLY (update current season):
rm -f model_cache/ensemble_2022_2026.*
python train_auto.py --enable-window-ensemble --dataset "..." --verbose

# MONTHLY (full retrain):
rm -f model_cache/ensemble_2*.pkl model_cache/ensemble_2*_meta.json
python train_auto.py --enable-window-ensemble --dataset "..." --verbose
```

Your models are now ready to achieve 80%+ player priors match rate with 68 additional Basketball Reference features!
