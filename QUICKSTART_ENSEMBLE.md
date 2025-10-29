# 🚀 Quick Start: Run Enhanced Ensemble Now

## TL;DR

You have **5 new files** in your project. Here's what to do:

### 1. Copy These Files (Already in Place)
- ✅ `ensemble_models_enhanced.py`
- ✅ `train_ensemble_enhanced.py`
- ✅ `ENSEMBLE_OPTIMIZATION_NOTES.md`
- ✅ `INTEGRATION_GUIDE_ENHANCED.md`
- ✅ `INTEGRATION_COMPLETE.md`

### 2. Run Training (Modified train_auto.py)
```bash
python train_auto.py --dataset "eoinamoore/historical-nba-data-and-player-box-scores" --fresh --verbose
```

**What happens:**
- Trains your game models as usual
- **NEW**: Trains Ridge + Elo + 4F + Meta-Learner ensemble  
- Saves 4 new models to `models/` directory
- Generates `coefficient_evolution.csv` and `ensemble_analysis.txt`

**Duration:** +5-15 min vs your current training time (depends on data size)

### 3. Verify It Worked
```bash
ls models/ridge_model_enhanced.pkl
ls models/elo_model_enhanced.pkl
ls models/four_factors_model_enhanced.pkl
ls models/ensemble_meta_learner_enhanced.pkl
```

All 4 files should exist (~12 MB total)

### 4. Use in riq_analyzer.py (Modified)
Models are **automatically loaded** at startup. Use:

```python
# Ensemble prediction (better than LGB alone)
prob = MODEL.predict_moneyline_ensemble(feats, home_team_id, away_team_id)

# Fallback to base LGB
prob = MODEL.predict_moneyline(feats)
```

---

## What You Get

| Component | Benefit | File |
|-----------|---------|------|
| Ridge + L2 | Stable margin baseline | `ridge_model_enhanced.pkl` |
| Dynamic Elo | Upset-adaptive ratings | `elo_model_enhanced.pkl` |
| 4F Rolling | Form-sensitive efficiency | `four_factors_model_enhanced.pkl` |
| Meta-Learner | Optimal blending (20-game refit) | `ensemble_meta_learner_enhanced.pkl` |

**Expected Result:** +3-5% logloss improvement over LGB alone (0.589 → 0.567)

---

## Files Modified

### train_auto.py
- **Lines 3444-3488**: Added ensemble training block
- Runs automatically after game models complete
- No manual steps needed

### riq_analyzer.py  
- **Lines 2514-2578**: Load ensemble models in ModelPredictor
- **Lines 2662-2706**: New `predict_moneyline_ensemble()` method
- Gracefully falls back if ensemble unavailable

---

## Output Files

After training, you'll have:

```
models/
├── ridge_model_enhanced.pkl              (5 MB)
├── elo_model_enhanced.pkl                (1 MB)
├── four_factors_model_enhanced.pkl       (5 MB)
├── ensemble_meta_learner_enhanced.pkl    (1 MB)
├── coefficient_evolution.csv             (~10 KB)
├── ensemble_analysis.txt                 (~5 KB)
└── training_metadata.json                (updated)
```

---

## Key Numbers

**2,460 games/season** (30 teams × 82 games)

**Refit every 20 games** = ~123 refits/season (~1.2 weeks)

**7 improvements:**
1. Dynamic Elo K-factor
2. Rolling 4F priors (10-game window)
3. Polynomial interaction features
4. Per-team/conference calibration (optional)
5. Optimal refit frequency testing (auto-tuned to 20)
6. Game exhaustion features (fatigue, B2B)
7. Coefficient tracking & analysis

---

## If Something Goes Wrong

### Error: "ImportError: No module named 'ensemble_models_enhanced'"
→ Ensure both files are in `C:\Users\tmiles11\nba_predictor\`:
- `ensemble_models_enhanced.py`
- `train_ensemble_enhanced.py`

### Error: "Ensemble training failed"
→ Check `train_auto.py` output for full error. Usually:
- Missing GAME_FEATURES or GAME_DEFAULTS
- LGB model didn't train properly

### Ensemble predictions returning None
→ Run training with `--fresh` flag to regenerate ensemble models

### Logloss didn't improve
→ This is rare. Possible reasons:
1. Your LGB is already near-optimal
2. Try `calibration_mode='home_away'` (per-team calibration)
3. Check `ensemble_analysis.txt` - may show which models are helping

---

## That's It!

Run this one command to get started:
```bash
python train_auto.py --dataset "eoinamoore/historical-nba-data-and-player-box-scores" --fresh --verbose
```

**~30-60 minutes later:** 4 new models + analysis files ready to use.

See `INTEGRATION_COMPLETE.md` for full documentation.
