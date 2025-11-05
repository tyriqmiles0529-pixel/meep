# Window-Aware Data Loading Refactor Plan

## Current Problem
- Loads ALL 25 years of data (820K player-games, 153K priors) = 2-3GB RAM
- Crashes during data loading phase
- Never gets to check which windows are cached
- Wasteful when 4/5 windows are already cached

## Solution: Load Data Per Window

### Architecture Change

**BEFORE (Current):**
```python
# Load ALL data
games_df = load_all_games()  # 32K games
players_df = load_all_players()  # 820K rows
priors = load_all_priors()  # 153K rows

# Then filter per window
for window in windows:
    window_games = games_df[filter]
    window_players = players_df[filter]
    train(window_games, window_players)
```

**AFTER (Optimized):**
```python
# Determine which windows need training
windows_needed = check_caches()  # e.g., only 2022-2026

# Load data ONLY for needed windows
for window in windows_needed:
    # Load window-specific data
    games_df = load_games_for_window(2022, 2026)  # 6K games
    players_df = load_players_for_window(2022, 2026)  # 165K rows  
    priors = load_priors_for_window(2022, 2026)  # 31K rows
    
    # Process THIS window
    process_phases(games_df, players_df, priors)
    train_window(window)
    save_cache(window)
    
    # Free memory
    del games_df, players_df, priors
    gc.collect()
```

### Memory Savings

| Approach | Games | Players | Priors | Total RAM | Status |
|----------|-------|---------|--------|-----------|--------|
| Current (all data) | 32K | 820K | 153K | 2-3GB | **CRASHES** |
| Per-window | 6.5K | 165K | 31K | 400-500MB | **WORKS** |
| Cached windows | 0 | 0 | 0 | 0MB | **INSTANT** |

**5x memory reduction per window!**

### Implementation Steps

#### 1. Create Window-Aware Data Loaders

```python
def load_window_data(start_year, end_year, args, verbose=False):
    """
    Load ONLY data for a specific 5-year window.
    
    Returns:
        games_df, players_df, priors_players, priors_teams
    """
    # Download Kaggle dataset (cached, instant)
    ds_root = kagglehub.dataset_download(args.dataset)
    
    # Load games and filter to window
    games_df = load_games(ds_root)
    games_df = games_df[
        (games_df['season_end_year'] >= start_year) &
        (games_df['season_end_year'] <= end_year)
    ]
    
    # Load players and filter to window  
    players_df = load_players(ds_root)
    players_df = players_df[
        (players_df['season_end_year'] >= start_year) &
        (players_df['season_end_year'] <= end_year)
    ]
    
    # Load priors and filter to window (±1 year for lag features)
    priors_players = load_priors(args.priors_dataset, 
                                  seasons=range(start_year-1, end_year+2))
    priors_teams = load_team_priors(args.priors_dataset,
                                    seasons=range(start_year-1, end_year+2))
    
    return games_df, players_df, priors_players, priors_teams
```

#### 2. Refactor Main Training Loop

```python
# Step 1: Determine which windows need training
windows_to_train = []
for window in all_windows:
    if window_needs_training(window):  # Check cache
        windows_to_train.append(window)

if not windows_to_train:
    print("✅ All windows cached!")
    return

# Step 2: Train only needed windows
for window in windows_to_train:
    print(f"Training window {window.start}-{window.end}")
    
    # Load ONLY this window's data
    games_df, players_df, priors_p, priors_t = load_window_data(
        window.start, window.end, args, verbose
    )
    
    # Process phases for THIS window
    players_df = add_phase_features(players_df, games_df, priors_p)
    
    # Train THIS window
    models = train_player_models(players_df, games_df)
    
    # Save to cache
    save_window_cache(window, models)
    
    # FREE MEMORY
    del games_df, players_df, priors_p, priors_t, models
    gc.collect()
    
    print(f"✅ Window {window.start}-{window.end} complete")
```

#### 3. Keep Game Ensemble Separate

Game ensembles need full historical data (for Elo continuity), so keep that separate:

```python
# Train game ensembles ONCE with full data
if not game_ensemble_cached():
    games_df_full = load_all_games()
    train_game_ensembles(games_df_full)
    del games_df_full
    gc.collect()

# Then train player windows with filtered data
for window in windows_to_train:
    # Per-window loading as above
    ...
```

### Files to Modify

1. **train_auto.py**
   - Create `load_window_data()` function
   - Refactor main() to check caches first
   - Move data loading inside window loop
   - Add memory cleanup after each window

2. **player_ensemble_enhanced.py** (if needed)
   - Ensure it works with window-specific data
   - No changes needed if already window-agnostic

### Testing Plan

1. **Test with all caches deleted**
   - Should train all 5 windows sequentially
   - Each window uses ~500MB RAM
   - Total time: ~2-3 hours

2. **Test with 4 windows cached**
   - Should skip cached windows
   - Only load data for 1 window (2022-2026)
   - Total time: ~25-30 minutes

3. **Test with all caches present**
   - Should skip all player training
   - Only load data for dynamic selector
   - Total time: ~5 minutes

### Expected Results

✅ **No more crashes** - 5x less memory per window
✅ **Faster reruns** - Skip cached windows entirely  
✅ **Scalable** - Can train on machines with 8GB RAM
✅ **Cleaner** - Window-centric architecture
✅ **Parallel-friendly** - Could distribute windows to different machines

### Rollout

1. Implement `load_window_data()` function
2. Test with single window
3. Refactor main loop
4. Test full pipeline
5. Document new behavior
6. Commit and push

## Success Criteria

- ✅ Training completes all 5 windows without crashing
- ✅ Memory usage stays under 1GB per window
- ✅ Cached windows skip data loading entirely
- ✅ Total training time: ~2-3 hours for all windows
- ✅ Rerun with cache: <30 minutes for current season only
