# ML Model Integration with RIQ MEEPING MACHINE

## Training Workflow

### Step 1: Get the Data
```bash
# Follow KAGGLE_SETUP.md to authenticate
python explore_dataset.py
```

### Step 2: Train Models
```bash
python train_prop_model.py
```

This will:
- Load historical player box scores
- Create features (rolling averages, opponent stats, situational)
- Train LightGBM models for PTS, AST, REB, 3PM
- Save trained models to `models/` directory

### Step 3: Integration

The trained models replace heuristic projections in the analyzer:

**Before (Heuristic):**
```python
# In nba_prop_analyzer_fixed.py
projection = _ewma(values, half_life=5.0) * pace * defense
```

**After (ML):**
```python
# Load trained model
with open('models/points_model.pkl', 'rb') as f:
    model_info = pickle.load(f)
    model = model_info['model']

# Create features
features = {
    'points_avg_3g': recent_3g_avg,
    'points_avg_10g': recent_10g_avg,
    'points_std_5g': std_5g,
    'is_home': 1 if home else 0,
    'rest_days': rest,
    'opp_points_allowed_10g': opp_def,
    # ... all features
}

# Predict
projection = model.predict([list(features.values())])[0]
```

## Model Features

### Rolling Statistics (per stat)
- `{stat}_avg_3g` - 3-game rolling average
- `{stat}_avg_5g` - 5-game rolling average
- `{stat}_avg_10g` - 10-game rolling average
- `{stat}_std_3g` - 3-game rolling std dev
- `{stat}_std_5g` - 5-game rolling std dev
- `{stat}_max_3g` - 3-game rolling max
- `{stat}_trend` - (3g_avg - 10g_avg) / 10g_avg

### Opponent Features
- `opp_{stat}_allowed_10g` - Opponent's 10-game avg allowed

### Situational Features
- `is_home` - Home game indicator (0/1)
- `rest_days` - Days since last game (0-7)
- `is_b2b` - Back-to-back game indicator
- `game_num` - Game number in season

## Expected Performance

Based on typical NBA prop models:

| Stat | Expected MAE | Notes |
|------|--------------|-------|
| PTS  | 4-6 points   | Most predictable |
| AST  | 1.5-2.5 assists | Moderate variance |
| REB  | 2-3 rebounds | Depends on matchup |
| 3PM  | 0.8-1.2 threes | High variance |

## Backtesting

Once models are trained, backtest with historical odds:

```python
# Simulate betting with ML projections
results = []
for date in test_dates:
    # Get props for this date
    props = get_historical_props(date)

    # Generate ML projections
    for prop in props:
        projection = model.predict(features)
        p_win = calculate_tail_probability(projection, line)

        # Use ELG framework
        p_samples = sample_beta_posterior(p_win, n_eff)
        f, p_c, _, _ = dynamic_fractional_kelly(p_samples, odds)

        if f > 0:
            # Record bet
            outcome = actual_stat > line if prop.pick == 'over' else actual_stat <= line
            results.append({
                'date': date,
                'prop': prop,
                'projection': projection,
                'actual': actual_stat,
                'won': outcome,
                'stake': bankroll * f,
                'profit': (bankroll * f * odds) if outcome else -bankroll * f
            })

# Calculate metrics
sharpe = calculate_sharpe_ratio(results)
max_dd = calculate_max_drawdown(results)
roi = sum(r['profit']) / sum(r['stake'])
```

## Next: Calibration

After training, calibrate probabilities:

```python
from sklearn.calibration import calibration_curve

# Get predicted probabilities
y_prob = model.predict_proba(X_test)[:, 1]  # For classification

# Check calibration
fraction_of_positives, mean_predicted_value = calibration_curve(
    y_true, y_prob, n_bins=10
)

# If poorly calibrated, use isotonic regression
from sklearn.isotonic import IsotonicRegression
calibrator = IsotonicRegression(out_of_bounds='clip')
calibrated_probs = calibrator.fit_transform(y_prob, y_true)
```

## Model Deployment

Once validated:

1. **Save models:**
   ```python
   # In train_prop_model.py (already done)
   pickle.dump(model_info, open('models/points_model.pkl', 'wb'))
   ```

2. **Update analyzer:**
   ```python
   # Add to nba_prop_analyzer_fixed.py
   ML_MODE = os.getenv("USE_ML_MODELS", "false").lower() == "true"

   if ML_MODE:
       projection = ml_predict(features)  # Use ML
   else:
       projection = _ewma(values) * pace * defense  # Use heuristic
   ```

3. **Track model versions:**
   ```python
   # models/model_registry.json
   {
       "points": {
           "version": "v1.0",
           "trained_date": "2025-10-24",
           "test_mae": 4.82,
           "features": ["points_avg_3g", "points_avg_10g", ...]
       }
   }
   ```

## Future: MLflow Integration

For production ML ops:

```python
import mlflow
import mlflow.lightgbm

with mlflow.start_run():
    # Log parameters
    mlflow.log_params(params)

    # Train model
    model = lgb.train(params, train_data)

    # Log metrics
    mlflow.log_metrics({"test_mae": test_mae, "test_rmse": test_rmse})

    # Log model
    mlflow.lightgbm.log_model(model, "model")

    # Register model
    mlflow.register_model(f"runs:/{run_id}/model", "nba_points_model")
```

Then in production:
```python
model = mlflow.lightgbm.load_model("models:/nba_points_model/production")
```

---

## Summary

**Current:** Heuristic EWMA projections
**Next:** LightGBM trained on historical data
**Future:** Real-time feature engineering + calibrated probabilities + MLflow deployment

The ELG/Kelly framework stays the same - we're just improving the input projections!
