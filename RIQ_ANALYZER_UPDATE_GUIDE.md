# RIQ Analyzer Neural Hybrid Update Guide

## Overview

This guide explains how to update `riq_analyzer.py` to work with the new neural hybrid models from commit `697f9e7`.

## What Changed

### Before (Old System)
- **Models**: Plain LightGBM regressors (`lgb.LGBMRegressor`)
- **Features**: 61 features (Phases 1-3 only)
  - Phase 1: Shot volume
  - Phase 2: Matchup context  
  - Phase 3: Advanced rates
- **No Priors**: No Basketball Reference integration
- **RMSE**: Points ~5.2, Assists ~2.1, Rebounds ~2.3

### After (New System)
- **Models**: NeuralHybridPredictor (TabNet + LightGBM + 24-dim embeddings)
- **Features**: ~150-218 features (Phases 1-7)
  - Phase 1: Shot volume (16 features)
  - Phase 2: Matchup context (4 features)
  - Phase 3: Advanced rates (3 features)
  - Phase 4: Home/away splits (4 features)
  - Phase 5: Position/matchup adjustments (10 features)
  - Phase 6: Momentum & optimization (24 features)
  - Phase 7: Basketball Reference priors (68 features, 49% match rate)
- **RMSE**: Expected ~4.5 points, ~1.8 assists, ~2.0 rebounds (12-15% improvement)

## Files Needed

### 1. Updated Code
Located in: `riq_analyzer_neural.py` (this contains the updated code sections)

### 2. Trained Models  
Located in: `models/` directory
- `points_model.pkl` - NeuralHybridPredictor for points
- `assists_model.pkl` - NeuralHybridPredictor for assists
- `rebounds_model.pkl` - NeuralHybridPredictor for rebounds
- `threes_model.pkl` - NeuralHybridPredictor for threes
- `minutes_model.pkl` - NeuralHybridPredictor for minutes
- `moneyline_model.pkl` - Game-level model
- `spread_model.pkl` - Game-level model

### 3. Basketball Reference Priors
Located in: `priors_data/` directory
- `player_priors.csv` - 68 career stat features per player
- Download: `priors_data.zip` (if not already present)

## Update Steps

### Step 1: Backup Current File
```bash
cp riq_analyzer.py riq_analyzer_backup.py
```

### Step 2: Replace ModelPredictor Class

**Location**: Lines ~2649-2966 in `riq_analyzer.py`

**What to Replace**:
```python
class ModelPredictor:
    def __init__(self):
        # OLD CODE - loads plain LightGBM models
        ...
```

**Replace With**: 
Copy the `ModelPredictor` class from `riq_analyzer_neural.py` (Section 1)

**Key Changes**:
- Handles `NeuralHybridPredictor` objects instead of plain LightGBM
- `.predict()` method works with ensemble models automatically
- No need to manually extract embeddings (model handles it internally)

### Step 3: Add Priors Loading Function

**Location**: After line ~366 (after `save_equity` function)

**What to Add**:
Copy the `load_priors_data()` function from `riq_analyzer_neural.py` (Section 2)

**Purpose**:
- Loads Basketball Reference player priors
- Returns DataFrame with ~68 career stat features
- Handles missing priors gracefully (returns None)

### Step 4: Replace Feature Engineering

**Location**: Lines ~2968-3109 (the `build_player_features` function)

**What to Replace**:
```python
def build_player_features(df_last: pd.DataFrame, df_curr: pd.DataFrame) -> pd.DataFrame:
    """
    Build features for ML prediction matching train_auto.py schema with ALL PHASES.
    Includes Phase 1 (shot volume), Phase 2 (matchup), Phase 3 (advanced rates).
    
    NOW PROPERLY COMPUTES PHASE FEATURES FROM ACTUAL GAME DATA!
    """
    # OLD CODE - only 61 features, no momentum/priors
    ...
```

**Replace With**:
Copy the `build_player_features_expanded()` function from `riq_analyzer_neural.py` (Section 3)
Then rename it to `build_player_features` (remove `_expanded` suffix)

**Key Changes**:
- **Phase 1**: Shot volume rolling stats (L3/L5/L10), per-minute rates, True Shooting %
- **Phase 2**: Matchup pace, defensive difficulty, offensive environment
- **Phase 3**: Usage rate, rebound rate, assist rate
- **Phase 4**: Home/away performance splits
- **Phase 5**: Position detection, matchup adjustments, starter probability
- **Phase 6**: Momentum (short/med/long), acceleration, hot/cold streaks, variance, ceiling/floor
- **Phase 7**: Basketball Reference priors (merged by player name)

### Step 5: Update analyze_player_prop

**Location**: Lines ~3172-3334 (the `analyze_player_prop` function)

**Changes Needed**:

#### 5a. Load Priors Once (at top of function)
```python
def analyze_player_prop(prop: dict, matchup_context: dict) -> Optional[dict]:
    # ADD THIS AT THE TOP:
    priors = load_priors_data()  # Load priors once
    
    # Existing validation code...
    if prop.get("odds") is not None and (prop["odds"] < MIN_ODDS or prop["odds"] > MAX_ODDS):
        return None
    ...
```

#### 5b. Update Feature Building Call
**Location**: Around line ~3245

**Old Code**:
```python
feats_row = build_player_features(df_last, df_curr)
```

**New Code**:
```python
feats_row = build_player_features(df_last, df_curr, 
                                   player_name=prop["player"],
                                   priors=priors)
```

## Testing

### Quick Test (Single Prop)
```bash
python riq_analyzer.py
```

**Expected Output**:
```
✓ Loaded points model
✓ Loaded assists model
✓ Loaded rebounds model
✓ Loaded threes model
✓ Loaded minutes model
✓ Loaded 5,427 player priors (68 features)
...
   ✓ Matched priors for LeBron James (68 features)
   [PHASE INTEGRATION] Features built:
     Phase 1 (Shot Volume): FGA_L5=15.2, rate_fga=0.42, TS%_L5=0.587
     Phase 2 (Matchup): pace_factor=1.03, def_difficulty=1.02
     Phase 3 (Advanced): usage=28.3%, reb_rate=14.2%, ast_rate=22.1%
```

### Full Analysis Run
```bash
python riq_analyzer.py
```

**Check For**:
- ✅ No "shape mismatch" errors
- ✅ Priors matched for ~49% of players
- ✅ Predictions use 150-218 features (shown in debug output)
- ✅ ML predictions complete successfully

## Troubleshooting

### Error: "Feature shape mismatch"

**Cause**: Model expects different number of features than provided

**Fix**:
1. Check `model.feature_names` vs `feats.columns`:
```python
print(f"Model expects: {len(model.feature_names)} features")
print(f"Provided: {len(feats.columns)} features")
print(f"Missing: {set(model.feature_names) - set(feats.columns)}")
```

2. Verify all phases are implemented in `build_player_features`

### Error: "priors_data not found"

**Cause**: Basketball Reference priors not downloaded

**Fix**:
1. Download `priors_data.zip` from Colab or project repo
2. Extract to project root: `C:\Users\tmiles11\nba_predictor\priors_data\`
3. Verify `player_priors.csv` exists in that directory

**Fallback**: Code will work without priors (uses ~80 features instead of ~150)

### Error: "NeuralHybridPredictor object has no attribute predict"

**Cause**: Model file contains old LightGBM model, not neural hybrid

**Fix**:
1. Check model file size:
```bash
ls -lh models/*.pkl
```
Neural hybrid models are ~2-5 MB, plain LightGBM is ~500KB

2. Retrain models using latest `train_auto.py`:
```bash
python train_auto.py --epochs 30 --skip-game-models
```

### Warning: "No priors match for PLAYER_NAME"

**Cause**: Player not in Basketball Reference database (rookie, two-way player, etc.)

**Effect**: Model uses NaN for prior features (trained to handle this)

**Action**: No action needed - this is expected for ~51% of players

## Performance Expectations

### Prediction Accuracy
- **Points**: 4.3-4.7 RMSE (was 5.2)
- **Assists**: 1.7-1.9 RMSE (was 2.1)  
- **Rebounds**: 1.9-2.1 RMSE (was 2.3)
- **Threes**: 0.9-1.1 RMSE (was 1.2)

### Runtime
- **Feature Engineering**: +15% slower (more features)
- **Model Prediction**: +25% slower (TabNet overhead)
- **Overall**: ~40% slower per prop (still <1s per player)

### Memory Usage
- **Per Prediction**: ~2 MB (was 500 KB)
- **Model Loading**: ~150 MB total (was 25 MB)

## Verification Checklist

After updating, verify:

- [ ] All 5 player models load successfully
- [ ] Priors load successfully (or graceful fallback)
- [ ] Feature engineering produces ~150-218 features
- [ ] No shape mismatch errors
- [ ] Predictions complete for test props
- [ ] Win probability predictions are reasonable (0.45-0.65 for most props)
- [ ] ELG scores are calculated correctly
- [ ] Output JSON contains all expected fields

## Rollback Plan

If issues occur:

```bash
# Restore backup
cp riq_analyzer_backup.py riq_analyzer.py

# Verify old version works
python riq_analyzer.py
```

## Next Steps

After successful update:

1. **Run Full Analysis**: Test on today's games
2. **Compare Predictions**: Old vs new model outputs
3. **Monitor Performance**: Track prediction accuracy over 1-2 weeks
4. **Tune Thresholds**: Adjust `MIN_WIN_PROBABILITY` if needed (currently 56%)

## Support

If you encounter issues not covered here:

1. Check `train_auto.py` for reference feature engineering (lines 1649-2500)
2. Verify neural_hybrid.py is up to date (commit 697f9e7)
3. Check model files are from latest training run
4. Review error logs for specific failure points

## References

- **Neural Hybrid Architecture**: `neural_hybrid.py`
- **Feature Engineering**: `train_auto.py` (lines 1649-2500)
- **Model Training**: `QUICKSTART.md`
- **Colab Setup**: `COLAB_COMPLETE_GUIDE.md`
