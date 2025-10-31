# Per-Window Player Processing Design

## Current Problem

### Player Models Processing (Lines 3974-4007)
```python
# ❌ Loads ALL data once
frames = build_players_from_playerstats(
    players_path,           # ALL 820k player-game rows (2002-2026)
    context_map,            # ALL 32k games context
    oof_games,              # ALL games OOF predictions
    priors_players          # ALL 15k player-seasons priors
)
# Then trains ONE global model on everything
```

**Memory Impact:**
- PlayerStatistics: 820k rows × 46 cols = ~300 MB
- Priors: 15k rows × 68 cols = ~10 MB
- Merged data: 820k × (46 + 68 + 17) = ~1 GB
- **Total peak: ~1.5 GB just for player data!**

### Game Ensemble Models (Lines 3161-3222)
```python
# ✅ Already per-window
for window_info in windows_to_process:
    window_seasons = [2017, 2018, 2019, 2020, 2021]  # 5 years
    games_window = games_df[games_df["season_end_year"].isin(window_seasons)]
    # Train on just this window
    # Free memory
    del games_window
    gc.collect()
```

**Memory Impact:**
- Only ~3-5k games per window instead of 32k
- **~80% memory reduction per window**

## Proposed Solution: Per-Window Player Processing

### Architecture

```python
for window_info in windows_to_process:
    window_seasons = [2017, 2018, 2019, 2020, 2021]

    # 1. Filter player data to window (±1 for context)
    padded_seasons = set(window_seasons) | {min-1, max+1}
    player_window = player_df[player_df["season_end_year"].isin(padded_seasons)]
    # 820k rows → ~150k per window (82% reduction!)

    # 2. Filter game context to window
    context_window = context_map[context_map["season_end_year"].isin(padded_seasons)]
    # 32k games → ~6k per window (82% reduction!)

    # 3. Filter priors to window seasons
    priors_window = priors_players[priors_players["season_for_game"].isin(padded_seasons)]
    # 15k rows → ~3k per window (80% reduction!)

    # 4. Build frames for this window only
    frames_window = build_players_from_playerstats(
        player_window,
        context_window,
        oof_games_window,
        priors_window
    )

    # 5. Train models on window
    minutes_model = _fit_minutes_model(frames_window["minutes"])
    points_model = _fit_stat_model(frames_window["points"])
    # ... etc

    # 6. Save window models to cache
    pickle.dump({
        'minutes': minutes_model,
        'points': points_model,
        # ...
    }, f"{cache_dir}/player_models_{start_year}_{end_year}.pkl")

    # 7. Free memory
    del player_window, context_window, priors_window, frames_window
    gc.collect()
```

## Memory Savings Calculation

### Before (Current):
```
PlayerStatistics: 820k rows                  = 300 MB
Priors:           15k rows                   = 10 MB
Context:          32k games × 2 sides        = 20 MB
Merged:           820k rows × 131 cols       = 1000 MB
-----------------------------------------------------------
TOTAL PEAK:                                    1330 MB
```

### After (Per-Window):
```
Per window (5 years):
PlayerStatistics: 150k rows (18%)            = 55 MB
Priors:           3k rows (20%)              = 2 MB
Context:          6k games × 2 sides (19%)   = 4 MB
Merged:           150k rows × 131 cols (18%) = 180 MB
-----------------------------------------------------------
TOTAL PEAK PER WINDOW:                         241 MB

Memory savings: 1330 MB → 241 MB = 82% reduction! ✅
```

## Implementation Plan

### Step 1: Add window parameter to build_players_from_playerstats
```python
def build_players_from_playerstats(
    player_path: Path,
    games_context: pd.DataFrame,
    oof_games: pd.DataFrame,
    verbose: bool,
    priors_players: Optional[pd.DataFrame] = None,
    window_seasons: Optional[Set[int]] = None  # NEW
) -> Dict[str, pd.DataFrame]:
```

### Step 2: Filter early in function (after loading CSV)
```python
# After loading PlayerStatistics
ps = pd.read_csv(player_path, ...)

# NEW: Filter to window seasons immediately
if window_seasons is not None:
    ps["season_end_year"] = _season_from_date(ps[date_col])
    padded = window_seasons | {min(window_seasons)-1, max(window_seasons)+1}
    ps = ps[ps["season_end_year"].isin(padded)]
    log(f"  Filtered to window seasons: {len_before} → {len(ps)} rows")
```

### Step 3: Create player model window loop (like game ensemble)
```python
# After training game models, before "Building player datasets"
if players_path and players_path.exists():
    print(_sec("Training player models per window"))

    for window_info in windows_to_process:  # Reuse same windows!
        window_seasons = set(window_info['seasons'])
        start_year = window_info['start_year']
        end_year = window_info['end_year']

        print(f"\\n{'='*70}")
        print(f"Training player models for window: {start_year}-{end_year}")

        # Filter context to window
        context_window = context_map[
            context_map["season_end_year"].isin(window_seasons)
        ]

        # Filter OOF to window
        oof_window = oof_games[
            oof_games["season_end_year"].isin(window_seasons)
        ]

        # Filter priors to window
        padded_seasons = window_seasons | {start_year-1, end_year+1}
        priors_window = priors_players[
            priors_players["season_for_game"].isin(padded_seasons)
        ] if priors_players is not None else None

        # Build frames for this window
        frames = build_players_from_playerstats(
            players_path,
            context_window,
            oof_window,
            verbose=verbose,
            priors_players=priors_window,
            window_seasons=window_seasons  # Pass to function for CSV filtering
        )

        # Train models
        minutes_model = _fit_minutes_model(frames.get("minutes"))
        # ... etc

        # Save to window cache
        cache_path = f"{cache_dir}/player_models_{start_year}_{end_year}.pkl"
        pickle.dump({
            'minutes': minutes_model,
            'points': points_model,
            # ...
        }, open(cache_path, 'wb'))

        # Free memory
        del context_window, oof_window, priors_window, frames
        gc.collect()
```

## Benefits

1. **Memory**: 82% reduction per window
2. **Speed**: Faster merging (150k vs 820k rows)
3. **Parallel**: Could train windows in parallel later
4. **Cache**: Can reuse cached historical windows
5. **Current Season**: Only retrain current window when new data arrives

## Risks & Mitigations

### Risk 1: Need context from prior seasons for rolling features
**Mitigation**: Add ±1 season padding when filtering (already in design)

### Risk 2: Model loading complexity (5 windows × N models)
**Mitigation**: Create simple loader that loads all window models and routes predictions by season

### Risk 3: Different model quality per window
**Mitigation**: Use same hyperparameters across windows, ensemble predictions if needed

## Next Steps

1. Modify `build_players_from_playerstats()` to accept and use `window_seasons`
2. Create player model window loop similar to game ensemble loop
3. Update model loading to handle per-window models
4. Test memory usage and match rates per window
