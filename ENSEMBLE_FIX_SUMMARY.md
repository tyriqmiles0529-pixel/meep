# Ensemble Fix Summary

## The Problem

Initial TRUE backtest showed **ALL ensemble approaches performed WORSE than simple rolling average baseline** (-0.4% to -0.8%).

## Root Cause Analysis

### What We Discovered

Inspected `train_ensemble_players.py` lines 127-145 and found:

```python
baseline = hist_df[stat_col].mean()

# 1. Ridge prediction
ridge_pred = baseline  # ❌ Just baseline!

# 2. LightGBM prediction
lgbm_pred = baseline  # ❌ Just baseline!

# 3. Player Elo prediction
elo_pred = player_elo.get_prediction(str(player_id), baseline)  # ✅ Only this worked

# 4. Rolling average
rolling_avg = baseline  # ❌ Just baseline!

# 5. Team matchup adjustment
matchup_pred = baseline  # ❌ Just baseline!
```

**The meta-learner was trying to learn from:**
```
[baseline, baseline, elo_pred, baseline, baseline]
```

**No wonder it added no value!** 4 out of 5 inputs were identical.

## The Fix

### Changed `train_ensemble_players.py` to generate diverse signals:

#### 1. Ridge → Weighted Recent + Trend
```python
# Weighted average (recent games weighted more)
weights = np.arange(1, len(recent_stats_clean) + 1)
weights = weights / weights.sum()
ridge_pred = np.dot(recent_stats_clean, weights)

# Add trend component (half weight on slope)
if len(recent_stats_clean) >= 3:
    slope = np.polyfit(x, recent_stats_clean, 1)[0]
    ridge_pred = ridge_pred + slope * 0.5
```

#### 2. LightGBM → Exponential Moving Average
```python
# EMA gives more weight to recent games
alpha = 0.3
lgbm_pred = recent_stats_clean[-1]  # Start with most recent
for val in reversed(recent_stats_clean[:-1]):
    lgbm_pred = alpha * val + (1 - alpha) * lgbm_pred
```

#### 3. Player Elo → (Unchanged, already working)
```python
elo_pred = player_elo.get_prediction(str(player_id), baseline)
```

#### 4. Rolling Average → Simple Mean
```python
rolling_avg = np.mean(recent_stats_clean)
```

#### 5. Matchup → Variance-Adjusted
```python
# More volatile players get adjusted toward recent form
recent_std = np.std(recent_stats_clean)
cv = recent_std / max(baseline, 0.1)  # Coefficient of variation
matchup_pred = (cv * recent_stats_clean[-1] + (1 - cv) * baseline) / (1 + cv)
```

## Test Results

### Before Fix:
- Baseline RMSE: 6.846
- Ensemble RMSE: 6.855
- **Improvement: -0.1% (WORSE!)**

### After Fix (2017-2021 window, points):
- Baseline RMSE: 7.163
- Ensemble RMSE: 7.006
- **Improvement: +2.2% (BETTER!)**

### Learned Weights:
```
Ridge (weighted + trend):  2.735 ⭐ Highest!
Rolling Average:           1.442
Matchup (variance-adj):    1.428
LightGBM (EMA):            0.627
Player Elo:                0.292
```

The meta-learner heavily favors Ridge (weighted recent + trend) because it captures both recent form AND momentum.

## Next Steps

1. **Clear all cached ensembles:**
   ```bash
   python clear_ensemble_cache.py
   ```

2. **Retrain all window ensembles:**
   ```bash
   python train_auto.py
   ```

3. **Run comprehensive backtest:**
   ```bash
   python true_backtest_all_approaches.py
   ```

4. **Compare results:**
   - Baseline (rolling average)
   - Individual windows (2002-2006, 2007-2011, 2012-2016, 2017-2021, 2022-2024)
   - Cherry-pick (best window per stat)
   - Enhanced selector (context-aware selection)

## Expected Outcome

With fixed base predictions providing diverse signals:
- **Each window should beat baseline by ~2-3%**
- **Cherry-picking might achieve ~3-5%** (best window per stat)
- **Enhanced selector might achieve ~4-6%** (context-aware + top windows)

The key insight: **Diverse signals allow the meta-learner to learn meaningful weights**, combining strengths of different prediction approaches.
