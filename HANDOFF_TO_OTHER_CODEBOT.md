# Handoff: RIQ Analyzer Neural Hybrid Update

## Task Summary

Update `riq_analyzer.py` to work with new neural hybrid models (commit 697f9e7).

## Files Created for You

### 1. `riq_analyzer_neural.py` (19 KB)
**Purpose**: Contains all updated code sections ready to copy-paste

**Sections**:
- Section 1: Updated `ModelPredictor` class (handles NeuralHybridPredictor)
- Section 2: New `load_priors_data()` function (loads Basketball Reference priors)
- Section 3: Expanded `build_player_features()` function (61 → 150-218 features)
- Section 4: Integration examples and usage instructions

### 2. `RIQ_ANALYZER_UPDATE_GUIDE.md` (9.4 KB)
**Purpose**: Detailed step-by-step instructions with troubleshooting

**Contents**:
- What changed (before/after comparison)
- File locations to update (line numbers)
- Step-by-step replacement instructions
- Testing checklist
- Troubleshooting guide (common errors + fixes)
- Performance expectations
- Rollback plan

### 3. `RIQ_UPDATE_SUMMARY.txt` (8.5 KB)
**Purpose**: Quick reference / cheat sheet

**Contents**:
- TL;DR summary of changes
- Feature expansion breakdown (Phases 1-7)
- Testing checklist
- Common errors with fixes
- Time estimate (2-4 hours)

## What You Need to Do

### Quick Version (30 seconds)

1. Open `RIQ_ANALYZER_UPDATE_GUIDE.md`
2. Follow Steps 1-5 (backup → replace 3 sections → update 1 function)
3. Test with `python riq_analyzer.py`
4. Verify no "shape mismatch" errors

### Detailed Version (2-4 hours)

1. **Backup Current File**
   ```bash
   cp riq_analyzer.py riq_analyzer_backup.py
   ```

2. **Open Both Files**
   - Left: `riq_analyzer.py` (file to update)
   - Right: `riq_analyzer_neural.py` (source of updates)

3. **Make 4 Changes** (all copy-paste):
   
   #### Change 1: Replace `ModelPredictor` class
   - **Location**: Lines 2649-2966 in `riq_analyzer.py`
   - **Source**: Section 1 in `riq_analyzer_neural.py`
   - **Why**: Handle neural hybrid models instead of plain LightGBM
   
   #### Change 2: Add `load_priors_data()` function
   - **Location**: After line 366 in `riq_analyzer.py`
   - **Source**: Section 2 in `riq_analyzer_neural.py`
   - **Why**: Load Basketball Reference player priors
   
   #### Change 3: Replace `build_player_features()` function
   - **Location**: Lines 2968-3109 in `riq_analyzer.py`
   - **Source**: Section 3 in `riq_analyzer_neural.py` (rename `build_player_features_expanded` → `build_player_features`)
   - **Why**: Expand from 61 → 150-218 features (add Phases 4-7)
   
   #### Change 4: Update `analyze_player_prop()` function
   - **Location**: Lines 3172-3334 in `riq_analyzer.py`
   - **Changes**:
     ```python
     # At top of function, add:
     priors = load_priors_data()
     
     # Line ~3245, change:
     # OLD: feats_row = build_player_features(df_last, df_curr)
     # NEW: feats_row = build_player_features(df_last, df_curr, 
     #                                         player_name=prop["player"],
     #                                         priors=priors)
     ```

4. **Test**
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
     Phase 1 (Shot Volume): FGA_L5=15.2, TS%_L5=0.587
     Phase 2 (Matchup): pace_factor=1.03
     Phase 3 (Advanced): usage=28.3%
   ```

5. **Verify**
   - [ ] No "shape mismatch" errors
   - [ ] Priors loaded successfully
   - [ ] Feature count shows ~150-218 (not 61)
   - [ ] Predictions complete
   - [ ] Win probabilities are reasonable (0.45-0.65)

## Key Changes Explained

### Feature Count: 61 → 150-218

**Old (61 features)**:
- 18 base context
- 12 rolling stats (pts/ast/reb/3pm × L3/L5/L10)
- 16 Phase 1 (shot volume)
- 4 Phase 2 (matchup)
- 3 Phase 3 (advanced rates)
- 4 home/away splits
- 1 minutes
- 3 miscellaneous

**New (150-218 features)**:
- Everything above (61 features)
- **+10 Phase 5**: Position detection, matchup adjustments, starter probability
- **+24 Phase 6**: Momentum, acceleration, hot/cold streaks, variance, ceiling/floor
- **+68 Phase 7**: Basketball Reference priors (49% match rate)

### Model Loading

**Old**:
```python
# Plain LightGBM
with open(model_path, 'rb') as f:
    model = pickle.load(f)  # lgb.LGBMRegressor object

prediction = model.predict(features)  # Direct prediction
```

**New**:
```python
# Neural Hybrid
with open(model_path, 'rb') as f:
    model = pickle.load(f)  # NeuralHybridPredictor object

prediction = model.predict(features)  # Ensemble prediction
# (TabNet extracts embeddings, LightGBM uses raw + embeddings, weighted average)
```

The interface is the same - just `.predict()` - but internally it runs:
1. TabNet: Deep learning on raw features → prediction + 24-dim embeddings
2. LightGBM: Gradient boosting on raw + embeddings → prediction
3. Ensemble: Weighted average (40% TabNet + 60% LightGBM)

## What If Something Goes Wrong?

### Rollback
```bash
cp riq_analyzer_backup.py riq_analyzer.py
python riq_analyzer.py  # Verify old version works
```

### Common Issues

**"Feature shape mismatch"**
- **Cause**: Missing features or wrong names
- **Fix**: Check all phases implemented in `build_player_features`
- **Debug**: Compare `model.feature_names` vs `feats.columns`

**"priors_data not found"**
- **Cause**: Priors directory missing
- **Fix**: Download `priors_data.zip` and extract
- **Fallback**: Code works without (uses ~80 features instead of 150)

**"No attribute 'predict'"**
- **Cause**: Old model files (plain LightGBM, not neural hybrid)
- **Fix**: Retrain with `python train_auto.py --epochs 30`

## References for Debugging

If you need to understand the feature engineering logic:

1. **Feature Engineering**: `train_auto.py` lines 1649-2500
   - Function: `build_players_from_playerstats`
   - This is the source of truth for what features models expect

2. **Model Architecture**: `neural_hybrid.py`
   - Class: `NeuralHybridPredictor`
   - Shows how TabNet + LightGBM ensemble works

3. **Training Schema**: `models/*.pkl`
   - Check `model.feature_names` attribute
   - Must match features in `build_player_features` exactly

## Time Estimate

- **Code Changes**: 1-2 hours (mostly copy-paste)
- **Testing**: 30-60 minutes
- **Debugging**: 30-60 minutes (if issues)
- **Total**: 2-4 hours

## Success Criteria

After update, you should see:

✅ Models load without errors  
✅ Priors load (or graceful fallback)  
✅ Feature engineering produces ~150-218 features  
✅ No shape mismatch errors  
✅ Predictions complete successfully  
✅ Win probabilities in reasonable range  
✅ Output JSON contains all expected fields  

## Questions?

Check these documents in order:
1. `RIQ_UPDATE_SUMMARY.txt` - Quick answers
2. `RIQ_ANALYZER_UPDATE_GUIDE.md` - Detailed steps
3. `riq_analyzer_neural.py` - Code reference
4. `train_auto.py` (lines 1649-2500) - Source of truth

## Final Notes

This is a **surgical update** - minimal changes to existing logic:
- 3 code sections to replace (ModelPredictor, load_priors, build_player_features)
- 1 function call to update (analyze_player_prop)
- All complex logic is pre-written in `riq_analyzer_neural.py`

The new models are **drop-in replacements**:
- Same `.predict()` interface
- Ensemble handled automatically internally
- No manual embedding extraction needed

Good luck! The files contain everything you need. 🚀

---

**Created**: 2025-11-07  
**Commit**: 697f9e7  
**Updated by**: Codebot Assistant
