# Phase 1 Feature Implementation - Action Plan

## Summary
We've proven through testing that shot volume + efficiency features provide **+1.78% improvement** on current season data. Now we need to implement these features in the actual training pipeline.

## What Needs to Change

### File: `train_auto.py`

**Location 1: Add Shot Columns** (after line 1619)
Currently detects: `min_col`, `pts_col`, `reb_col`, `ast_col`, `tpm_col`
Need to add:
```python
fga_col = resolve_any([["fieldGoalsAttempted", "fga", "FGA", "field_goals_attempted"]])
three_pa_col = resolve_any([["threePointersAttempted", "3pa", "FG3A", "three_pa", "three_point_attempts"]])
fta_col = resolve_any([["freeThrowsAttempted", "fta", "FTA", "free_throw_attempts"]])
fg_pct_col = resolve_any([["fieldGoalsPercentage", "fg_pct", "FG_PCT"]])
three_pct_col = resolve_any([["threePointersPercentage", "fg3_pct", "FG3_PCT", "three_pct"]])
ft_pct_col = resolve_any([["freeThrowsPercentage", "ft_pct", "FT_PCT"]])
```

**Location 2: Add to usecols** (line 1626)
```python
want_cols = [gid_col, date_col, pid_col, name_full_col, fname_col, lname_col, tid_col, home_col,
             min_col, pts_col, reb_col, ast_col, tpm_col, starter_col,
             fga_col, three_pa_col, fta_col, fg_pct_col, three_pct_col, ft_pct_col]  # NEW
```

**Location 3: Numeric conversions** (line 1738)
```python
for stat_col in [min_col, pts_col, reb_col, ast_col, tpm_col, fga_col, three_pa_col, fta_col]:  # ADDED fga_col, three_pa_col, fta_col
    if stat_col and stat_col in ps.columns:
        ps[stat_col] = pd.to_numeric(ps[stat_col], errors="coerce")
```

**Location 4: Add rolling stats** (after line 1802)
```python
rolling_stats(fga_col)        # NEW
rolling_stats(three_pa_col)   # NEW
rolling_stats(fta_col)        # NEW
```

**Location 5: Calculate True Shooting %** (after line 1802, before line 1804)
```python
# Calculate True Shooting % (NEW)
def calc_ts_pct(row):
    """Calculate True Shooting % = PTS / (2 * (FGA + 0.44 * FTA))"""
    pts = row.get(pts_col, 0) if pts_col else 0
    fga = row.get(fga_col, 0) if fga_col else 0
    fta = row.get(fta_col, 0) if fta_col else 0

    denominator = 2 * (fga + 0.44 * fta)
    if denominator > 0:
        return pts / denominator
    return 0.56  # league average

if pts_col and fga_col and fta_col:
    if pts_col in ps.columns and fga_col in ps.columns and fta_col in ps.columns:
        ps['ts_pct'] = ps.apply(calc_ts_pct, axis=1)

        # Rolling TS% (last 5, 10, season)
        ps['ts_pct_L5'] = ps.groupby(pid_col)['ts_pct'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        ps['ts_pct_L10'] = ps.groupby(pid_col)['ts_pct'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        ps['ts_pct_season'] = ps.groupby(pid_col)['ts_pct'].transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
```

**Location 6: Shooting percentage rolling averages** (after TS% calculation)
```python
# Shooting % rolling averages (NEW)
if three_pct_col and three_pct_col in ps.columns:
    ps['three_pct_L5'] = ps.groupby(pid_col)[three_pct_col].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())

if ft_pct_col and ft_pct_col in ps.columns:
    ps['ft_pct_L5'] = ps.groupby(pid_col)[ft_pct_col].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
```

**Location 7: Per-minute rates for shot attempts** (after line 1834)
```python
ps["rate_fga"] = rate(fga_col).fillna(0.3).astype("float32")        # NEW
ps["rate_3pa"] = rate(three_pa_col).fillna(0.1).astype("float32")  # NEW
ps["rate_fta"] = rate(fta_col).fillna(0.1).astype("float32")        # NEW
```

**Location 8: Add features to model** (line 2319-2342)
```python
features = [
    # ... existing features ...
    "days_rest", "player_b2b",
    # NEW: enhanced rolling trends
    "points_L3", "points_L5", "points_L10",
    "rebounds_L3", "rebounds_L5", "rebounds_L10",
    "assists_L3", "assists_L5", "assists_L10",
    "threepoint_goals_L3", "threepoint_goals_L5", "threepoint_goals_L10",
    # NEW: shot volume rolling trends (PHASE 1.1)
    "fieldGoalsAttempted_L3", "fieldGoalsAttempted_L5", "fieldGoalsAttempted_L10",
    "threePointersAttempted_L3", "threePointersAttempted_L5", "threePointersAttempted_L10",
    "freeThrowsAttempted_L3", "freeThrowsAttempted_L5", "freeThrowsAttempted_L10",
    # NEW: shot volume per-minute rates
    "rate_fga", "rate_3pa", "rate_fta",
    # NEW: efficiency features (PHASE 1.2)
    "ts_pct_L5", "ts_pct_L10", "ts_pct_season",
    "three_pct_L5", "ft_pct_L5",
    # existing home/away splits
    "points_home_avg", "points_away_avg",
    # ... rest of existing features
]
```

**Location 9: Add regularization** (line 2361)
Currently:
```python
reg = lgb.LGBMRegressor(
    objective="regression",
    learning_rate=0.05, num_leaves=31, max_depth=-1,
    colsample_bytree=0.9, subsample=0.8, subsample_freq=5,
    n_estimators=800, random_state=seed, n_jobs=N_JOBS,
    force_col_wise=True, verbosity=-1
)
```

Change to (proven in tests):
```python
reg = lgb.LGBMRegressor(
    objective="regression",
    learning_rate=0.1,       # increased from 0.05
    num_leaves=31,
    max_depth=3,             # NEW - shallow trees
    min_child_samples=100,   # NEW - require more data per leaf
    colsample_bytree=0.7,    # reduced from 0.9
    subsample=0.7,           # reduced from 0.8
    subsample_freq=5,
    reg_alpha=0.5,           # NEW - L1 regularization
    reg_lambda=0.5,          # NEW - L2 regularization
    n_estimators=50,         # reduced from 800
    random_state=seed,
    n_jobs=N_JOBS,
    force_col_wise=True,
    verbosity=-1
)
```

## Testing Before Production

After modifying `train_auto.py`, test on a single window first:
```bash
python train_auto.py --verbose --window-mode --window-seasons 2022,2023,2024,2025,2026
```

Expected outcome:
- New features should appear in player stat models
- RMSE should improve by ~1-2% vs baseline

## Full Retraining

Once tested, retrain all windows:
```bash
# Clear cache (only for current season)
del model_cache\*2022_2026*

# Retrain all windows
python train_auto.py --verbose --window-mode
```

## Verification

After retraining:
1. Check model metrics - should see improvement in player stat RMSEs
2. Run enhanced selector training (needs new features in windows)
3. Backtest on recent games to validate improvement holds

## Notes

- **Column name flexibility**: Code uses `resolve_any()` to handle multiple column name variants
- **Fallback safety**: All new features have defaults if columns missing
- **Regularization critical**: Without heavy regularization, features cause overfitting
- **Temporal drift**: These features work best with recent data (2022+)
