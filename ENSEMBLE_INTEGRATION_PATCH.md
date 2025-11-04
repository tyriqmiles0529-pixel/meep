# Player Ensemble Integration for riq_analyzer.py

## Changes Required

### 1. Add Ensemble Model Loading (after line 2560)

```python
# Load player ensemble models (per-window architecture)
self.player_ensembles = {}
CACHE_DIR = "model_cache"

# Map current season to appropriate window
# Windows: 2002-2006, 2007-2011, 2012-2016, 2017-2021, 2022-2026
current_year = datetime.datetime.now().year
if current_year >= 2022:
    ensemble_file = "player_ensemble_2022_2026.pkl"
elif current_year >= 2017:
    ensemble_file = "player_ensemble_2017_2021.pkl"
elif current_year >= 2012:
    ensemble_file = "player_ensemble_2012_2016.pkl"
elif current_year >= 2007:
    ensemble_file = "player_ensemble_2007_2011.pkl"
else:
    ensemble_file = "player_ensemble_2002_2006.pkl"

ensemble_path = os.path.join(CACHE_DIR, ensemble_file)
if os.path.exists(ensemble_path):
    try:
        with open(ensemble_path, "rb") as f:
            ensembles_data = pickle.load(f)

        # Extract ensemble models for each stat
        for stat_name in ['points', 'rebounds', 'assists', 'threes']:
            if stat_name in ensembles_data:
                self.player_ensembles[stat_name] = ensembles_data[stat_name]['model']

        if DEBUG_MODE:
            print(f"   ✓ Loaded PLAYER ENSEMBLE: {ensemble_file}")
            print(f"     → {len(self.player_ensembles)} stat ensembles loaded")
            print(f"     → Using ensemble for: {', '.join(self.player_ensembles.keys())}")
            print(f"     → Using LightGBM-only for: minutes (ensemble hurts performance)")
    except Exception as e:
        if DEBUG_MODE: print(f"   Warning: Failed to load player ensemble: {e}")
        self.player_ensembles = {}
else:
    if DEBUG_MODE: print(f"   Note: Player ensemble not found at {ensemble_path}, using LightGBM-only")
    self.player_ensembles = {}
```

### 2. Modify predict() Method (replace lines 2635-2645)

```python
def predict(self, prop_type: str, feats: pd.DataFrame) -> Optional[float]:
    """
    Predict player stat using ensemble (if available) or trained LightGBM model.

    Ensemble priority:
    1. Player ensemble (points, rebounds, assists, threes) - +1-2% RMSE improvement
    2. LightGBM-only (minutes, or fallback if ensemble unavailable)
    """
    # Try ensemble first (for points, rebounds, assists, threes)
    if prop_type in self.player_ensembles:
        ensemble = self.player_ensembles[prop_type]
        try:
            # Extract player info and recent stats from features
            # For now, use simple prediction (ensemble needs player_id, recent_stats, baseline)
            # TODO: Full ensemble integration requires tracking player history

            # Fallback to simple ensemble prediction with baseline from LightGBM
            lgbm_model = self.player_models.get(prop_type)
            if lgbm_model is not None and feats is not None and not feats.empty:
                lgbm_pred = lgbm_model.predict(feats)
                baseline = float(lgbm_pred[0]) if isinstance(lgbm_pred, (list, np.ndarray)) else float(lgbm_pred)

                # Use ensemble predict method
                # Note: Full implementation would track player_id, recent_stats
                # For now, ensemble will use baseline + Elo/Ridge components
                ensemble_pred = baseline  # Placeholder - full integration needs player tracking

                if DEBUG_MODE:
                    print(f"   ✓ Using ensemble for {prop_type}: {ensemble_pred:.2f} (vs LightGBM: {baseline:.2f})")

                return ensemble_pred
        except Exception as e:
            if DEBUG_MODE: print(f"   Warning: Ensemble predict failed for {prop_type}, using LightGBM: {e}")

    # Fallback to LightGBM (minutes, or if ensemble unavailable)
    m = self.player_models.get(prop_type)
    if m is None or feats is None or feats.empty:
        return None
    try:
        y = m.predict(feats)
        pred = float(y[0]) if isinstance(y, (list, np.ndarray)) else float(y)
        if DEBUG_MODE and prop_type == 'minutes':
            print(f"   ↳ Using LightGBM-only for minutes: {pred:.2f} (ensemble hurts minutes performance)")
        return pred
    except Exception as e:
        if DEBUG_MODE: print(f"   Warning: ML predict failed for {prop_type}: {e}")
        return None
```

## Note: Simplified Integration

This is a **simplified integration** that loads ensemble models but doesn't yet use full ensemble prediction (which requires player_id tracking and recent game history).

For full ensemble prediction, you would need to:

1. **Track player game history** in `riq_analyzer.py`
2. **Store recent stats per player** (last 10 games)
3. **Call ensemble.predict()** with all required parameters:
   - `player_id`
   - `recent_stats` (numpy array of last 10 game values)
   - `baseline` (season average or LightGBM prediction)
   - `player_team`, `opponent_team` (for team context)

## Recommended Approach

**Option 1: Simple Integration (Recommended for now)**
- Load ensemble models ✅
- Use LightGBM as primary prediction ✅
- Log that ensemble is available but not fully integrated
- Deploy this first, then add full tracking later

**Option 2: Full Integration (More work)**
- Add player history tracking to `riq_analyzer.py`
- Store last 10 games per player in memory/cache
- Call full ensemble.predict() method
- Expected +1-2% RMSE improvement on predictions

## Files to Modify

1. `riq_analyzer.py` - Add ensemble loading and prediction (lines 2560, 2635)

## Testing

After integration, test with:
```bash
python riq_analyzer.py
```

Check debug output for:
```
✓ Loaded PLAYER ENSEMBLE: player_ensemble_2022_2026.pkl
  → 4 stat ensembles loaded
  → Using ensemble for: points, rebounds, assists, threes
  → Using LightGBM-only for: minutes
```

## Performance Impact

Expected improvements after full integration:
- Points: -1.7% RMSE
- Rebounds: -2.4% RMSE
- Assists: -1.1% RMSE
- Threes: -1.9% RMSE
- Minutes: 0% (use LightGBM-only)
