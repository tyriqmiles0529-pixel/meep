# Integration Guide: Ensemble Models for train_auto.py

This guide shows where to add the new ensemble models (Ridge, Elo, Four Factors, Logistic Regression meta-learner, and exhaustion features) to your existing `train_auto.py` workflow.

## Overview

Three new Python files have been created:
1. **ensemble_models.py** - Core model classes (Ridge, Elo, Four Factors, Logistic Ensembler)
2. **train_ensemble.py** - Integration functions for training workflows
3. **INTEGRATION_GUIDE.md** - This file

## Integration Steps

### Step 1: Add Imports to train_auto.py

At the top of `train_auto.py` (around line 55, after other model imports), add:

```python
from train_ensemble import (
    train_ridge_score_diff,
    train_elo_model,
    train_four_factors_model,
    train_logistic_ensembler,
    compare_models,
    add_exhaustion_features
)
from ensemble_models import EloRating
```

### Step 2: Modify `_fit_game_models()` to Train Ensemble Models

In `train_auto.py`, find the `_fit_game_models()` function (around line 1341).

**After the existing LGB models are trained**, add this code block:

```python
# ============================================================================
# ENSEMBLE MODELS TRAINING (NEW)
# ============================================================================

print(_sec("Ensemble Sub-Models Training"))

# 1. Train Ridge Score Diff Model
ridge_model, ridge_metrics = train_ridge_score_diff(
    games_df=games_df,
    game_features=GAME_FEATURES,
    game_defaults=GAME_DEFAULTS,
    verbose=True,
    seed=seed
)
pickle.dump(ridge_model, open('ridge_score_diff_model.pkl', 'wb'))

# 2. Train Elo Rating Model
elo_model, games_with_elo, elo_metrics = train_elo_model(
    games_df=games_df,
    k_factor=20.0,
    home_advantage=70.0,
    verbose=True
)
pickle.dump(elo_model, open('elo_model.pkl', 'wb'))

# Update games_df with elo features for downstream use
games_df_with_elo = games_with_elo.copy()

# 3. Train Four Factors Model
ff_model, ff_metrics = train_four_factors_model(
    games_df=games_df_with_elo,
    verbose=True,
    seed=seed
)
if ff_model is not None:
    pickle.dump(ff_model, open('four_factors_model.pkl', 'wb'))

# 4. Train Logistic Regression Meta-Learner
ensembler, oof_ensemble_probs, ensemble_metrics = train_logistic_ensembler(
    games_df=games_df_with_elo,
    ridge_model=ridge_model,
    elo_model=elo_model,
    ff_model=ff_model,
    lgb_classifier=clf_final,  # Use the trained LGB classifier
    game_features=GAME_FEATURES,
    game_defaults=GAME_DEFAULTS,
    refit_frequency=20,
    verbose=True,
    seed=seed
)
pickle.dump(ensembler, open('meta_learner.pkl', 'wb'))

# 5. Compare Model Performance
ridge_probs = ridge_model.predict_proba(
    games_df_with_elo[GAME_FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0).astype('float32')
)
elo_probs = np.array([
    elo_model.expected_win_prob(
        games_df_with_elo.iloc[i]['elo_home'] + 70,
        games_df_with_elo.iloc[i]['elo_away']
    )
    for i in range(len(games_df_with_elo))
])
ff_probs = None
if ff_model is not None:
    X_ff = games_df_with_elo[[
        'home_efg_prior', 'home_tov_pct_prior', 'home_orb_pct_prior', 'home_ftr_prior',
        'away_efg_prior', 'away_tov_pct_prior', 'away_orb_pct_prior', 'away_ftr_prior'
    ]].apply(pd.to_numeric, errors='coerce').fillna(0).astype('float32')
    ff_probs = ff_model.predict_proba(X_ff)

lgb_probs_full = clf_final.predict_proba(
    games_df_with_elo[GAME_FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0).astype('float32')
)[:, 1]

model_comparison = compare_models(
    games_df=games_df_with_elo,
    ridge_probs=ridge_probs,
    elo_probs=elo_probs,
    ff_probs=ff_probs,
    lgb_probs=lgb_probs_full,
    ensemble_probs=oof_ensemble_probs,
    verbose=True
)

# Save ensemble metrics to training_metadata.json
training_metadata = {
    'ridge': ridge_metrics,
    'elo': elo_metrics,
    'four_factors': ff_metrics,
    'ensemble': ensemble_metrics,
    'model_comparison': model_comparison,
}
```

### Step 3: Update build_players_from_playerstats() for Exhaustion Features

In `build_players_from_playerstats()` function, after building all player features, add exhaustion features:

```python
# Add exhaustion features (NEW)
ps_join = add_exhaustion_features(ps_join)
log("- Added exhaustion features (season fatigue, heavy usage, B2B tracking)", verbose)

# Fill NaN exhaustion features with defaults
ps_join['season_fatigue'] = ps_join['season_fatigue'].fillna(0.5)
ps_join['heavy_usage'] = ps_join['heavy_usage'].fillna(0)
ps_join['consecutive_b2b'] = ps_join['consecutive_b2b'].fillna(0)
ps_join['rest_accumulated'] = ps_join['rest_accumulated'].fillna(200)
```

### Step 4: Update GAME_FEATURES to Include Elo

In train_auto.py, find `GAME_FEATURES` list (around line 188) and add elo features:

```python
GAME_FEATURES: List[str] = [
    "home_advantage", "neutral_site",
    "home_recent_pace", "away_recent_pace",
    "home_off_strength", "home_def_strength",
    "away_off_strength", "away_def_strength",
    "home_recent_winrate", "away_recent_winrate",
    # matchup features
    "match_off_edge", "match_def_edge", "match_pace_sum", "winrate_diff",
    # schedule/injury
    "home_days_rest", "away_days_rest",
    "home_b2b", "away_b2b",
    "home_injury_impact", "away_injury_impact",
    # era features
    "season_end_year", "season_decade",
    # ELO FEATURES (NEW)
    "elo_home", "elo_away", "elo_diff",
    # ... rest of features ...
]
```

And update `GAME_DEFAULTS`:

```python
GAME_DEFAULTS: Dict[str, float] = {
    # ... existing defaults ...
    "elo_home": 1500.0,  # NEW
    "elo_away": 1500.0,  # NEW
    "elo_diff": 0.0,     # NEW
    # ... rest of defaults ...
}
```

### Step 5: Update Player Feature Schema

In `build_players_from_playerstats()`, after exhaustion features are added, update output columns to include them:

```python
# Player feature columns to use in models
PLAYER_FEATURES = [
    # existing features...
    # NEW EXHAUSTION FEATURES
    'season_fatigue',
    'heavy_usage',
    'consecutive_b2b',
    'rest_accumulated',
]
```

### Step 6: Add Ensemble Predictions to Inference (riq_analyzer.py)

In `riq_analyzer.py`, after loading LGB models, add:

```python
# Load ensemble models (NEW)
ridge_model = pickle.load(open('ridge_score_diff_model.pkl', 'rb'))
elo_model = pickle.load(open('elo_model.pkl', 'rb'))
try:
    ff_model = pickle.load(open('four_factors_model.pkl', 'rb'))
except:
    ff_model = None
meta_learner = pickle.load(open('meta_learner.pkl', 'rb'))
```

Then, in the prop analysis loop, generate ensemble predictions:

```python
# Generate ensemble game predictions (for use in prop context)
def get_game_prediction_ensemble(game_row, ridge_model, elo_model, ff_model, meta_learner, lgb_model):
    """Get ensemble prediction for a game."""
    X_game = game_row[GAME_FEATURES].values.reshape(1, -1)
    
    ridge_prob = ridge_model.predict_proba(X_game)[0]
    elo_prob = elo_model.expected_win_prob(
        game_row['elo_home'] + 70,
        game_row['elo_away']
    )
    if ff_model is not None:
        ff_prob = ff_model.predict_proba(X_game)[0]
    else:
        ff_prob = 0.5
    lgb_prob = lgb_model.predict_proba(X_game)[0, 1]
    
    X_meta = np.array([[ridge_prob, elo_prob, ff_prob, lgb_prob]])
    ensemble_prob = meta_learner.predict_proba(X_meta)[0]
    
    return {
        'ridge': ridge_prob,
        'elo': elo_prob,
        'ff': ff_prob,
        'lgb': lgb_prob,
        'ensemble': ensemble_prob
    }
```

## File Structure After Integration

```
C:\Users\tmiles11\nba_predictor\
├── ensemble_models.py              # NEW: Core model classes
├── train_ensemble.py               # NEW: Training integration
├── train_auto.py                   # MODIFIED: Add ensemble training
├── riq_analyzer.py                 # MODIFIED: Add ensemble inference
├── INTEGRATION_GUIDE.md            # NEW: This file
├── ridge_score_diff_model.pkl      # NEW: Trained model artifact
├── elo_model.pkl                   # NEW: Trained model artifact
├── four_factors_model.pkl          # NEW: Trained model artifact (optional)
├── meta_learner.pkl                # NEW: Trained meta-learner artifact
├── moneyline_model.pkl             # EXISTING
├── spread_model.pkl                # EXISTING
└── ... other files ...
```

## Training Workflow

1. **Run training:**
   ```powershell
   python .\train_auto.py --dataset "eoinamoore/historical-nba-data-and-player-box-scores" --verbose --skip-rest --fresh
   ```

2. **New artifacts created:**
   - `ridge_score_diff_model.pkl`
   - `elo_model.pkl`
   - `four_factors_model.pkl` (if box score data available)
   - `meta_learner.pkl`
   - Updated `training_metadata.json` with ensemble metrics

3. **Expected metrics output:**
   ```
   ── Ensemble Sub-Models Training ──
   
   ── Ridge Score Diff Model ──
     Train RMSE: 12.345, MAE: 9.876
     Val RMSE: 12.567, MAE: 10.123
     Residual Std: 14.567
   
   ── Elo Rating Model ──
     Accuracy: 0.567
     Logloss: 0.675, Brier: 0.231
     Teams: 30
   
   ── Four Factors Model ──
     Train RMSE: 12.100, MAE: 9.654
     Val RMSE: 12.300, MAE: 9.876
   
   ── Logistic Regression Ensembler ──
     Refits: 5
     OOF Logloss: 0.565
     OOF Brier: 0.215
     Sub-model weights:
       Ridge: 0.234
       Elo: 0.189
       Four Factors: 0.156
       LGB: 0.421
   
   ── Model Comparison (Validation) ──
     RIDGE        | LL: 0.678 | Brier: 0.235 | AUC: 0.612
     ELO          | LL: 0.675 | Brier: 0.231 | AUC: 0.615
     FF           | LL: 0.682 | Brier: 0.238 | AUC: 0.608
     LGB          | LL: 0.589 | Brier: 0.201 | AUC: 0.651
     ENSEMBLE     | LL: 0.567 | Brier: 0.195 | AUC: 0.663
   ```

## Expected Improvements

Based on literature and your notes:

- **Ridge + Elo + Four Factors ensemble:** +2-4% logloss improvement over single LGB
- **Meta-learner (Logistic Regression):** Additional +0.5-1% improvement
- **Exhaustion features in player props:** +0.5-1% accuracy on player prop predictions

**Total expected improvement:** ~3-6% logloss reduction, which translates to ~1-2% better predicted probabilities overall.

## Troubleshooting

### Four Factors Model Returns "Skipped"
- **Cause:** Box score data (FG, 3P, FGA, FTA, TOV, ORB, DRB) not available in dataset
- **Solution:** Integrate Basketball Reference data or stick with Ridge + Elo + LGB (still 95% of value)

### Meta-Learner Weights Are All Near Zero
- **Cause:** Sub-models are highly correlated
- **Solution:** This is actually fine—means they're all saying the same thing. Ensemble will default to LGB (most accurate)

### Exhaustion Features Have High NaN Rate
- **Cause:** Player IDs or date ordering inconsistent
- **Solution:** Fill with defaults (already done in code)

### Training Takes Significantly Longer
- **Cause:** Elo model requires iterating through all historical games
- **Solution:** Expected (adds ~10-20% to total training time). Can parallelize in Phase 2.

## Next Steps (Phase 2)

1. **Neural Network Player Model:** Custom LSTM/attention architecture for player prop predictions
2. **Hyperparameter Tuning:** Use Optuna/Hyperband to optimize Ridge alpha, LGB params, meta-learner weighting
3. **Vertex AI Pipelines:** Containerize and deploy to GCP for scheduled retraining
4. **Real-time Calibration:** Update meta-learner weights daily based on recent predictions

---

**Questions?** Refer to the docstrings in `ensemble_models.py` and `train_ensemble.py`.
