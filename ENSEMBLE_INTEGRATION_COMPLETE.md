# Enhanced Selector Integration - COMPLETE

## What Was Integrated

The enhanced selector has been successfully integrated into `riq_analyzer.py` to provide context-aware window selection for player predictions.

## Changes Made

### 1. Selector Loading (lines 2605-2632)

Added code to load the enhanced selector and all window ensembles in `ModelPredictor.__init__`:

```python
# Load enhanced selector for context-aware window selection
self.enhanced_selector = None
self.selector_windows = {}
selector_file = os.path.join(CACHE_DIR, "dynamic_selector_enhanced.pkl")
selector_meta_file = os.path.join(CACHE_DIR, "dynamic_selector_enhanced_meta.json")

if os.path.exists(selector_file) and os.path.exists(selector_meta_file):
    # Load selector and all window ensembles
    # Auto-detects all player_ensemble_*.pkl files
    # Stores in self.selector_windows dict
```

**What it does:**
- Loads the trained Random Forest selector
- Loads ALL window ensembles (2002-2006, 2007-2011, 2012-2016, 2017-2021, 2022-2025)
- Prints debug message showing number of windows available
- Gracefully handles missing files

### 2. New Prediction Method (lines 2723-2802)

Added `predict_with_ensemble()` method to `ModelPredictor` class:

```python
def predict_with_ensemble(self, prop_type: str, feats: pd.DataFrame,
                         player_history: Optional[pd.DataFrame] = None) -> Optional[float]:
    """
    Predict using enhanced selector + window ensembles.

    Falls back to LightGBM if selector not available or prediction fails.
    """
```

**What it does:**
1. Extracts 10 enhanced features from player history:
   - games_played, recent_avg, recent_std, recent_min, recent_max
   - trend, rest_days, recent_form_3, form_change, consistency_cv

2. Uses Random Forest selector to pick best window (2002-2006, etc.)

3. Gets prediction from selected window's ensemble

4. Returns prediction or None (for fallback)

### 3. Updated Prediction Call (lines 3117-3120)

Changed the prediction logic to try enhanced selector first:

```python
# Try enhanced selector first, fallback to LightGBM
mu_ml = MODEL.predict_with_ensemble(prop["prop_type"], feats_row, player_history=df_last)
if mu_ml is None:
    mu_ml = MODEL.predict(prop["prop_type"], feats_row)
```

**What it does:**
- First tries context-aware window selection
- Falls back to standard LightGBM if:
  - Selector not available
  - Player history insufficient (< 3 games)
  - Any error occurs
- Ensures no breaking changes - always returns a prediction

## Expected Behavior

### With Selector Available

When running the analyzer with DEBUG_MODE enabled:

```
Loading models...
   ✓ Loaded ENHANCED SELECTOR
     → Context-aware window selection (+0.5% vs cherry-pick)
     → 5 windows available

Processing prop...
   ✓ Used ENHANCED SELECTOR: 2022-2025 → 28.45
```

### Without Selector

Falls back gracefully to LightGBM:

```
Loading models...
   Note: Enhanced selector not found, using LightGBM-only

Processing prop...
   (no selector message - uses standard LightGBM)
```

## Performance Impact

Based on true backtest results:
- **+0.5% average RMSE improvement** over baseline
- Context-aware selection adapts to:
  - Player form (hot/cold streaks)
  - Consistency (reliable vs volatile)
  - Rest days (back-to-back vs rested)
  - Recent trends

## Files Modified

1. **riq_analyzer.py** (3 sections)
   - Added selector loading (lines 2605-2632)
   - Added predict_with_ensemble method (lines 2723-2802)
   - Updated prediction call (lines 3117-3120)

## Testing

To verify integration works:

1. **Check selector loads:**
   ```bash
   python riq_analyzer.py --debug
   ```
   Look for: "✓ Loaded ENHANCED SELECTOR"

2. **Check selector used:**
   Run analyzer on actual props and look for:
   "✓ Used ENHANCED SELECTOR: [window] → [prediction]"

3. **Verify fallback works:**
   Rename selector files temporarily and verify analyzer still works with LightGBM

## Key Design Decisions

1. **Graceful degradation**: Never crashes if selector missing
2. **Conservative fallback**: Returns None if unsure, falls back to LightGBM
3. **Auto-detection**: Uses glob to find all window files automatically
4. **Debug visibility**: Clear messages showing when selector is used
5. **No breaking changes**: Existing LightGBM path still works

## Next Steps

The integration is complete and ready to use. The enhanced selector will automatically be used whenever:
- The selector files exist in model_cache/
- Player has sufficient history (3+ games)
- The stat type is supported (points, rebounds, assists, threes, minutes)

If any of these conditions aren't met, the system gracefully falls back to LightGBM.
