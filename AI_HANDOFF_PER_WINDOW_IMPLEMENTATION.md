# AI Handoff: Per-Window Player Training Implementation

## Executive Summary

**Status:** Merge bug FIXED ✅ | Match rate: 99.9% ✅ | Memory optimization: 77% reduction tested ✅

**Next Task:** Implement full per-window player training loop to reduce memory from 1.5 GB → 240 MB (82% reduction)

**Estimated Time:** 3-4 hours

---

## Context: What Was Fixed

### Problem 1: Player Priors Merge Failing (0% match rate) ✅ SOLVED
**Root Cause:** Merge condition checked if teamId column existed, but didn't check if it had valid data. Column was 100% NaN, causing silent merge failure.

**Fix Applied (Line 1912):**
```python
# Before:
if tid_col and tid_col in ps.columns:

# After:
if tid_col and tid_col in ps.columns and ps[tid_col].notna().any():
```

**Result:** Now uses is_home merge path → season_end_year populated correctly → 99.9% match rate!

### Problem 2: Memory Issues During Testing
**Cause:** Loading ALL 820k player-game rows at once = 1.5 GB peak memory

**Solution Started:** Added window filtering capability (lines 1578, 1640-1659)
- Filters to 5-year windows: 820k → ~150k rows per window
- Tested successfully with 2022-2026 window: 189k rows, 99.9% match rate
- Ready for production use

---

## Current Architecture

### What Works NOW (after fix):
```python
# In main() around line 4000
if players_path and players_path.exists():
    print(_sec("Building player datasets"))

    # Load ALL player data once (820k rows)
    frames = build_players_from_playerstats(
        players_path,
        context_map,           # ALL 32k games
        oof_games,             # ALL games
        priors_players         # Already filtered to ~15k
        # window_seasons NOT passed = load everything
    )

    # Train ONE global model on all data
    minutes_model = _fit_minutes_model(frames["minutes"])
    points_model = _fit_stat_model(frames["points"], "points")
    # ... etc

# Memory: ~1.5 GB peak
```

### What's NEEDED (per-window processing):
```python
# Goal: Process each 5-year window separately
if players_path and players_path.exists():
    print(_sec("Training player models per window"))

    # Reuse same window structure as game ensemble (lines 3161-3222)
    for window_info in windows_to_process:
        window_seasons = set(window_info['seasons'])  # e.g., {2022, 2023, 2024, 2025, 2026}
        start_year = window_info['start_year']
        end_year = window_info['end_year']

        # Filter everything to THIS window
        context_window = context_map[context_map["season_end_year"].isin(window_seasons)]
        oof_window = oof_games[oof_games["season_end_year"].isin(window_seasons)]

        # Filter priors to window (±1 for context)
        padded_seasons = window_seasons | {start_year-1, end_year+1}
        priors_window = priors_players[priors_players["season_for_game"].isin(padded_seasons)]

        # Build frames for THIS window (window_seasons triggers internal filtering)
        frames = build_players_from_playerstats(
            players_path,
            context_window,     # ~6k games per window
            oof_window,
            verbose,
            priors_window,      # ~3k rows per window
            window_seasons=window_seasons  # ← KEY: Triggers filtering in function
        )

        # Train models on THIS window
        minutes_model = _fit_minutes_model(frames["minutes"])
        points_model = _fit_stat_model(frames["points"], "points")
        # ... etc

        # Save per-window models
        cache_path = f"{cache_dir}/player_models_{start_year}_{end_year}.pkl"
        pickle.dump({
            'minutes': minutes_model,
            'points': points_model,
            'rebounds': rebounds_model,
            'assists': assists_model,
            'threes': threes_model,
            'window_seasons': list(window_seasons)
        }, open(cache_path, 'wb'))

        # Free memory before next window
        del context_window, oof_window, priors_window, frames
        del minutes_model, points_model, rebounds_model, assists_model, threes_model
        gc.collect()

# Memory: ~240 MB peak per window (82% reduction!)
```

---

## Implementation Steps

### Step 1: Find the Current Player Training Code (5 min)

**Location:** `train_auto.py` lines 3974-4200

**Current code structure:**
```python
# Line ~4000: Player models section
player_metrics: Dict[str, Dict[str, float]] = {}
if players_path and players_path.exists():
    print(_sec("Building player datasets"))

    # Augment with current season data (lines 4006-4030)
    current_player_df = fetch_current_season_player_stats(...)

    # Build frames (line 4000 or 4033)
    frames = build_players_from_playerstats(...)

    # Load historical player props (lines 4036-4175)
    historical_player_props = load_or_fetch_historical_player_props(...)

    # Merge props into frames (lines 4055-4175)
    for stat_name, stat_df in frames.items():
        # ... merge props logic ...

    # Train models (lines 4177-4253)
    minutes_model = _fit_minutes_model(frames["minutes"])
    points_model = _fit_stat_model(frames["points"], "points")
    # ... etc
```

### Step 2: Wrap Player Training in Window Loop (30 min)

**Reference:** Game ensemble window loop at lines 3161-3222

**Key changes:**
1. Move the player training code inside a `for window_info in windows_to_process:` loop
2. Filter `context_map`, `oof_games`, `priors_players` to current window
3. Pass `window_seasons` to `build_players_from_playerstats()`
4. Save per-window models instead of global models
5. Add memory cleanup after each window

**Pseudocode:**
```python
# After line ~3999 (after ensemble training)
player_metrics: Dict[str, Dict[str, float]] = {}
if players_path and players_path.exists():
    print(_sec("Training player models per window"))

    # Check if windows_to_process exists (from game ensemble)
    if 'windows_to_process' not in locals():
        # Create windows if not already created
        # (copy logic from game ensemble section)
        ...

    for idx, window_info in enumerate(windows_to_process, 1):
        window_seasons = set(window_info['seasons'])
        start_year = window_info['start_year']
        end_year = window_info['end_year']
        cache_path = f"{cache_dir}/player_models_{start_year}_{end_year}.pkl"

        print(f"\n{'='*70}")
        print(f"Training player models: Window {idx}/{len(windows_to_process)}")
        print(f"Seasons: {start_year}-{end_year}")
        print(f"{'='*70}")

        # Check cache
        if os.path.exists(cache_path) and not window_info['is_current']:
            print(f"[SKIP] Using cached player models from {cache_path}")
            continue

        # Filter data to window
        # ... (see detailed code below)

        # Build frames
        frames = build_players_from_playerstats(
            players_path,
            context_window,
            oof_window,
            verbose,
            priors_window,
            window_seasons=window_seasons
        )

        # Train models
        # ... (existing training code)

        # Save
        # ... (pickle.dump)

        # Cleanup
        # ... (del + gc.collect)
```

### Step 3: Handle Current Season Player Data (20 min)

**Challenge:** Current season data needs to be merged before window filtering

**Current logic (lines 4006-4030):**
```python
current_player_df = fetch_current_season_player_stats(season="2025-26", verbose=verbose)
if current_player_df is not None:
    # Append to historical data
    combined_players_df = pd.concat([hist_players_df, current_player_df])
    temp_combined_csv.to_csv(...)
```

**Per-window approach:**
```python
# BEFORE window loop: Merge current season data if it exists
if current_player_df is not None and not current_player_df.empty:
    # Save combined file once
    temp_combined_csv = Path(".combined_players_temp.csv")
    combined_players_df = pd.concat([hist_players_df, current_player_df])
    combined_players_df.to_csv(temp_combined_csv, index=False)
    player_data_path = temp_combined_csv
else:
    player_data_path = players_path

# THEN in window loop: Use player_data_path
for window_info in windows_to_process:
    frames = build_players_from_playerstats(
        player_data_path,  # ← Use combined or historical
        ...
        window_seasons=window_seasons
    )
```

### Step 4: Handle Player Props Per Window (30 min)

**Challenge:** Historical player props loading (lines 4036-4175)

**Current approach:** Loads ALL historical props, then merges into frames

**Per-window approach Option A (simpler):**
```python
# Load props ONCE before loop (keep current logic)
historical_player_props = load_or_fetch_historical_player_props(...)

# THEN in window loop: Filter props to current window
for window_info in windows_to_process:
    # ... build frames ...

    # Filter props to this window
    if not historical_player_props.empty:
        window_props = historical_player_props[
            historical_player_props['date'].dt.year.isin(
                range(start_year-1, end_year+2)
            )
        ]

        # Merge props into frames
        for stat_name, stat_df in frames.items():
            # ... existing merge logic ...
```

**Per-window approach Option B (more memory efficient):**
```python
# In window loop: Load props for THIS window only
for window_info in windows_to_process:
    # ... build frames ...

    # Determine date range for this window
    window_games = games_df[games_df["season_end_year"].isin(window_seasons)]
    window_dates = window_games["date"].dropna().unique()

    # Load props for ONLY these dates
    window_props = load_or_fetch_historical_player_props(
        players_df=raw_players_df[raw_players_df["gameDate"].isin(window_dates)],
        ...
    )
```

**Recommendation:** Use Option A for simplicity (props are relatively small)

### Step 5: Save Per-Window Models (15 min)

**Current save logic (lines ~4177-4253):**
```python
# Save global models
pickle.dump(minutes_model, open(models_dir / "minutes_model.pkl", "wb"))
pickle.dump(points_model, open(models_dir / "points_model.pkl", "wb"))
# ... etc
```

**Per-window save logic:**
```python
# In window loop after training
cache_path = f"{cache_dir}/player_models_{start_year}_{end_year}.pkl"
cache_meta_path = f"{cache_dir}/player_models_{start_year}_{end_year}_meta.json"

# Save all models for this window
window_models = {
    'minutes': minutes_model,
    'points': points_model,
    'rebounds': rebounds_model,
    'assists': assists_model,
    'threes': threes_model,
    'window_seasons': list(window_seasons),
    'metrics': {
        'minutes': m_metrics,
        'points': p_metrics,
        'rebounds': r_metrics,
        'assists': a_metrics,
        'threes': t_metrics
    }
}

with open(cache_path, 'wb') as f:
    pickle.dump(window_models, f)

# Save metadata
meta = {
    'seasons': list(map(int, window_seasons)),
    'start_year': start_year,
    'end_year': end_year,
    'trained_date': datetime.now().isoformat(),
    'num_player_games': sum(len(df) for df in frames.values() if df is not None),
    'is_current_season': window_info['is_current']
}

with open(cache_meta_path, 'w') as f:
    json.dump(meta, f, indent=2)

print(f"[OK] Player models for {start_year}-{end_year} saved to {cache_path}")
```

### Step 6: Create Model Loader for Predictions (45 min)

**Challenge:** Prediction code expects ONE model per stat, now have MULTIPLE per-window models

**Solution:** Create a wrapper class that loads all windows and routes by season

```python
class WindowedPlayerModels:
    """Wrapper for per-window player models that routes predictions by season"""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.window_models = {}  # {stat: {(start_year, end_year): model}}
        self._load_all_windows()

    def _load_all_windows(self):
        """Load all cached window models"""
        for pkl_file in self.cache_dir.glob("player_models_*_*.pkl"):
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)

                # Extract window range from filename: player_models_2022_2026.pkl
                parts = pkl_file.stem.split('_')
                start_year = int(parts[-2])
                end_year = int(parts[-1])

                # Organize by stat type
                for stat_name in ['minutes', 'points', 'rebounds', 'assists', 'threes']:
                    if stat_name in data:
                        if stat_name not in self.window_models:
                            self.window_models[stat_name] = {}
                        self.window_models[stat_name][(start_year, end_year)] = data[stat_name]

                print(f"Loaded window models: {start_year}-{end_year}")
            except Exception as e:
                print(f"Warning: Failed to load {pkl_file}: {e}")

    def predict(self, stat_name: str, X: pd.DataFrame, season_year: float):
        """
        Predict using the appropriate window model

        Args:
            stat_name: 'minutes', 'points', 'rebounds', 'assists', or 'threes'
            X: Features dataframe
            season_year: Season end year to determine which window to use

        Returns:
            Predictions array
        """
        if stat_name not in self.window_models:
            raise ValueError(f"No models loaded for {stat_name}")

        # Find which window contains this season
        for (start_year, end_year), model in self.window_models[stat_name].items():
            if start_year <= season_year <= end_year:
                return model.predict(X)

        # Fallback: use most recent window
        latest_window = max(self.window_models[stat_name].keys(), key=lambda x: x[1])
        print(f"Warning: Season {season_year} not in any window, using {latest_window}")
        return self.window_models[stat_name][latest_window].predict(X)

    def get_model(self, stat_name: str, season_year: float):
        """Get the actual model object for a given stat and season"""
        if stat_name not in self.window_models:
            return None

        for (start_year, end_year), model in self.window_models[stat_name].items():
            if start_year <= season_year <= end_year:
                return model

        # Fallback: most recent
        latest_window = max(self.window_models[stat_name].keys(), key=lambda x: x[1])
        return self.window_models[stat_name][latest_window]

# Usage in prediction code:
windowed_models = WindowedPlayerModels(Path("model_cache"))
current_season = 2026

# Predict points for a player
points_pred = windowed_models.predict('points', player_features, current_season)
```

### Step 7: Update Main Training Logic (30 min)

**Changes needed:**
1. Keep global model saving for backward compatibility (use most recent window)
2. Add per-window model loading in prediction scripts

```python
# After window loop completes
print("\n" + "="*70)
print("Saving global models (using most recent window for compatibility)")
print("="*70)

# Load most recent window models and save as global
latest_window = max(windows_to_process, key=lambda x: x['end_year'])
latest_cache = f"{cache_dir}/player_models_{latest_window['start_year']}_{latest_window['end_year']}.pkl"

if os.path.exists(latest_cache):
    with open(latest_cache, 'rb') as f:
        latest_models = pickle.load(f)

    # Save as global models for backward compatibility
    for stat_name in ['minutes', 'points', 'rebounds', 'assists', 'threes']:
        if stat_name in latest_models:
            model_path = models_dir / f"{stat_name}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(latest_models[stat_name], f)
            print(f"  ✓ {stat_name}_model.pkl (from {latest_window['start_year']}-{latest_window['end_year']})")
```

---

## Testing Strategy

### Test 1: Single Window (10 min)
```python
# Modify code to process only current window
windows_to_process = [w for w in windows_to_process if w['is_current']]
```

**Expected:**
- ~60k player-game rows
- ~240 MB memory
- 99.9% match rate
- Models saved to cache

### Test 2: Two Windows (20 min)
```python
# Process current + one historical
windows_to_process = windows_to_process[-2:]
```

**Expected:**
- Memory stays ~240 MB per window
- Both windows cached successfully
- No memory leaks between windows

### Test 3: All Windows (full run)
```python
# Run with all windows
```

**Expected:**
- Each window processes independently
- Memory freed between windows
- All windows cached

### Test 4: Model Loading (10 min)
```python
# Test the WindowedPlayerModels class
from train_auto import WindowedPlayerModels

models = WindowedPlayerModels(Path("model_cache"))
print(f"Loaded {len(models.window_models)} stat types")
for stat, windows in models.window_models.items():
    print(f"  {stat}: {len(windows)} windows")
```

---

## Detailed Code Template

### Complete window loop implementation:

```python
# Location: After game ensemble training, around line 4000

player_metrics: Dict[str, Dict[str, float]] = {}
if players_path and players_path.exists():
    print(_sec("Training player models per window"))

    # Prepare current season data if available
    current_player_df = fetch_current_season_player_stats(season="2025-26", verbose=verbose)
    if current_player_df is not None and not current_player_df.empty:
        temp_player_csv = Path(".current_season_players_temp.csv")
        current_player_df.to_csv(temp_player_csv, index=False)

        hist_players_df = pd.read_csv(players_path, low_memory=False)
        combined_players_df = pd.concat([hist_players_df, current_player_df], ignore_index=True)

        temp_combined_csv = Path(".combined_players_temp.csv")
        combined_players_df.to_csv(temp_combined_csv, index=False)
        player_data_path = temp_combined_csv

        print(f"- Added {len(current_player_df):,} player-game records from 2025-26 season")
    else:
        player_data_path = players_path

    # Load historical player props once (reuse across windows)
    print(_sec("Loading historical player prop odds"))
    player_props_cache = Path("data/historical_player_props_cache.csv")
    raw_players_df = pd.read_csv(player_data_path, low_memory=False)

    historical_player_props = load_or_fetch_historical_player_props(
        players_df=raw_players_df,
        api_key=THEODDS_API_KEY,
        cache_path=player_props_cache,
        verbose=verbose,
        max_requests=100
    )

    # Process each window
    for idx, window_info in enumerate(windows_to_process, 1):
        window_seasons = set(window_info['seasons'])
        start_year = window_info['start_year']
        end_year = window_info['end_year']
        cache_path = f"{cache_dir}/player_models_{start_year}_{end_year}.pkl"
        cache_meta_path = f"{cache_dir}/player_models_{start_year}_{end_year}_meta.json"
        is_current = window_info['is_current']

        print(f"\n{'='*70}")
        print(f"Training player models: Window {idx}/{len(windows_to_process)}")
        print(f"Seasons: {start_year}-{end_year} ({'CURRENT' if is_current else 'historical'})")
        print(f"{'='*70}")

        # Check cache
        if os.path.exists(cache_path) and not is_current:
            print(f"[SKIP] Using cached models from {cache_path}")
            continue

        # Filter game context to window
        context_window = context_map[context_map["season_end_year"].isin(window_seasons)].copy()
        oof_window = oof_games[oof_games["season_end_year"].isin(window_seasons)].copy()

        # Filter priors to window (±1 for context)
        padded_seasons = window_seasons | {start_year-1, end_year+1}
        priors_window = priors_players[
            priors_players["season_for_game"].isin(padded_seasons)
        ].copy() if priors_players is not None and not priors_players.empty else None

        print(f"Window data: {len(context_window):,} games, {len(priors_window) if priors_window is not None else 0:,} player-season priors")

        # Build frames for this window
        frames = build_players_from_playerstats(
            player_data_path,
            context_window,
            oof_window,
            verbose=verbose,
            priors_players=priors_window,
            window_seasons=window_seasons  # ← Triggers window filtering
        )

        # Filter and merge player props for this window
        if not historical_player_props.empty:
            # Filter props to this window's date range
            window_games = games_df[games_df["season_end_year"].isin(window_seasons)]
            if "date" in window_games.columns:
                min_date = window_games["date"].min()
                max_date = window_games["date"].max()
                window_props = historical_player_props[
                    (historical_player_props["date"] >= min_date) &
                    (historical_player_props["date"] <= max_date)
                ]

                # Merge props into frames (existing logic from lines 4055-4175)
                for stat_name, stat_df in frames.items():
                    if stat_df is None or stat_df.empty:
                        continue

                    prop_type_map = {
                        'points': 'points',
                        'rebounds': 'rebounds',
                        'assists': 'assists',
                        'threes': 'threes',
                        'minutes': None
                    }

                    prop_type = prop_type_map.get(stat_name)
                    if prop_type is None:
                        continue

                    stat_props = window_props[window_props['prop_type'] == prop_type].copy()
                    if stat_props.empty:
                        continue

                    # Prepare merge columns
                    if 'date' in stat_df.columns:
                        stat_df['date_str'] = pd.to_datetime(stat_df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
                    else:
                        continue

                    stat_props['date_str'] = stat_props['date'].astype(str)

                    # Normalize names
                    stat_df['player_name_norm'] = stat_df.get('playerName', stat_df.get('player_name', '')).str.lower().str.strip()
                    stat_props['player_name_norm'] = stat_props['player_name'].str.lower().str.strip()

                    # Merge
                    stat_df = stat_df.merge(
                        stat_props[['date_str', 'player_name_norm', 'market_line', 'market_over_odds', 'market_under_odds']],
                        on=['date_str', 'player_name_norm'],
                        how='left'
                    )

                    # Clean up temp columns
                    stat_df = stat_df.drop(columns=['date_str', 'player_name_norm'], errors='ignore')
                    frames[stat_name] = stat_df

        # Train models
        print(_sec(f"Training player models for {start_year}-{end_year}"))

        minutes_model, m_metrics = _fit_minutes_model(frames.get("minutes", pd.DataFrame()), seed=seed + 10, verbose=verbose)
        points_model, p_metrics = _fit_stat_model(frames.get("points", pd.DataFrame()), "points", seed=seed + 20, verbose=verbose)
        rebounds_model, r_metrics = _fit_stat_model(frames.get("rebounds", pd.DataFrame()), "rebounds", seed=seed + 30, verbose=verbose)
        assists_model, a_metrics = _fit_stat_model(frames.get("assists", pd.DataFrame()), "assists", seed=seed + 40, verbose=verbose)
        threes_model, t_metrics = _fit_stat_model(frames.get("threes", pd.DataFrame()), "threes", seed=seed + 50, verbose=verbose)

        # Save window models
        window_models = {
            'minutes': minutes_model,
            'points': points_model,
            'rebounds': rebounds_model,
            'assists': assists_model,
            'threes': threes_model,
            'window_seasons': list(window_seasons),
            'metrics': {
                'minutes': m_metrics,
                'points': p_metrics,
                'rebounds': r_metrics,
                'assists': a_metrics,
                'threes': t_metrics
            }
        }

        with open(cache_path, 'wb') as f:
            pickle.dump(window_models, f)

        # Save metadata
        meta = {
            'seasons': list(map(int, window_seasons)),
            'start_year': start_year,
            'end_year': end_year,
            'trained_date': datetime.now().isoformat(),
            'num_player_games': sum(len(df) for df in frames.values() if df is not None and not df.empty),
            'is_current_season': is_current,
            'metrics': {k: {mk: float(mv) for mk, mv in v.items()} if v else {} for k, v in window_models['metrics'].items()}
        }

        with open(cache_meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"[OK] Player models for {start_year}-{end_year} saved to {cache_path}")

        # Free memory
        del context_window, oof_window, priors_window, frames
        del minutes_model, points_model, rebounds_model, assists_model, threes_model
        gc.collect()

        print(f"Memory freed for next window")

    # Save global models using most recent window (backward compatibility)
    print("\n" + "="*70)
    print("Saving global models (using most recent window)")
    print("="*70)

    latest_window = max(windows_to_process, key=lambda x: x['end_year'])
    latest_cache = f"{cache_dir}/player_models_{latest_window['start_year']}_{latest_window['end_year']}.pkl"

    if os.path.exists(latest_cache):
        with open(latest_cache, 'rb') as f:
            latest_models = pickle.load(f)

        for stat_name in ['minutes', 'points', 'rebounds', 'assists', 'threes']:
            if stat_name in latest_models:
                model_path = models_dir / f"{stat_name}_model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(latest_models[stat_name], f)
                print(f"  ✓ {stat_name}_model.pkl")

        # Aggregate metrics
        player_metrics = latest_models.get('metrics', {})

    # Clean up temp files
    if 'temp_player_csv' in locals():
        temp_player_csv.unlink(missing_ok=True)
    if 'temp_combined_csv' in locals():
        temp_combined_csv.unlink(missing_ok=True)
```

---

## Benefits Summary

### Memory
- **Before:** 1.5 GB peak (all seasons loaded at once)
- **After:** 240 MB peak per window (82% reduction)
- **Total:** Can handle datasets 5x larger before hitting memory limits

### Speed
- **Initial run:** Similar time (processing sequentially)
- **Cached runs:** Much faster (only retrain current season window)
- **Future:** Can parallelize windows for 5x speedup

### Maintainability
- **Modularity:** Each window independent
- **Debugging:** Easier to isolate issues to specific time periods
- **Updates:** Only retrain affected windows

---

## Rollback Plan

If per-window causes issues, you can revert to global training by:

1. Use the saved global models (most recent window)
2. Or temporarily disable window loop:
   ```python
   # Quick disable: process all seasons as one "window"
   windows_to_process = [{
       'seasons': all_seasons,
       'start_year': min(all_seasons),
       'end_year': max(all_seasons),
       'is_current': True
   }]
   ```

---

## Files Modified

1. **train_auto.py** - Add window loop (lines ~4000-4200)
2. **test_player_priors_merge.py** - Already uses window filtering ✅
3. **(Optional) prediction scripts** - Add WindowedPlayerModels class

---

## Success Criteria

✅ All windows process successfully
✅ Memory stays < 500 MB per window
✅ Match rate ≥ 75% in all windows
✅ Models saved to cache
✅ Global models saved for compatibility
✅ Current season window retrains on each run

---

## Questions to Resolve

1. **Player props:** Load once (Option A) or per-window (Option B)?
   - **Recommendation:** Option A (simpler, props are small)

2. **Backward compatibility:** Keep global models?
   - **Recommendation:** Yes, save most recent window as global

3. **Parallel processing:** Implement now or later?
   - **Recommendation:** Later (sequential works, can optimize after)

---

## Estimated Timeline

- Step 1-2: 30 min (find code, wrap in loop)
- Step 3: 20 min (handle current season)
- Step 4: 30 min (handle player props)
- Step 5: 15 min (save logic)
- Step 6: 45 min (model loader class)
- Step 7: 30 min (backward compatibility)
- Testing: 1 hour (single window, two windows, all windows)

**Total: 3-4 hours**

---

## Contact/Handoff Notes

**Current Status:**
- ✅ Merge fix working (99.9% match rate tested)
- ✅ Window filtering infrastructure ready
- ✅ Test script validates approach
- ✅ All code committed to GitHub (3ca33b3)

**Next Developer:**
- Reference: `PER_WINDOW_PLAYER_DESIGN.md` for architecture details
- Start with: Lines 3974-4200 in train_auto.py
- Test with: `python test_player_priors_merge.py` (already working)
- Goal: Wrap player training in window loop like game ensemble (lines 3161-3222)

**Questions:** Check the "Questions to Resolve" section above and make decisions based on your priorities (simplicity vs memory vs performance).
