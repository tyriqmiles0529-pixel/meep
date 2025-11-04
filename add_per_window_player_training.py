"""
Script to replace all-at-once player training with per-window training
Replaces lines 4000-4253 in train_auto.py
"""

# Read the file
with open('train_auto.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the section to replace (lines 4000-4253, 0-indexed = 3999-4252)
before_section = lines[:3999]  # Everything before line 4000
after_section = lines[4253:]   # Everything after line 4253

# New per-window player training code
new_player_training = '''    # ========================================================================
    # PLAYER MODELS (Per-Window Training for Memory Optimization)
    # ========================================================================
    player_metrics: Dict[str, Dict[str, float]] = {}
    if players_path and players_path.exists():
        print(_sec("Training player models per window"))

        # Prepare current season data if available (merge once before loop)
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

        # Get windows from game ensemble (should already exist from game training)
        if 'windows_to_process' not in locals():
            # Create windows if not already defined
            all_seasons = sorted([int(s) for s in games_df["season_end_year"].dropna().unique()])
            max_year = int(games_df["season_end_year"].max())
            window_size = 5
            windows_to_process = []
            for i in range(0, len(all_seasons), window_size):
                window_seasons = all_seasons[i:i+window_size]
                start_year = int(window_seasons[0])
                end_year = int(window_seasons[-1])
                windows_to_process.append({
                    'seasons': window_seasons,
                    'start_year': start_year,
                    'end_year': end_year,
                    'is_current': max_year in window_seasons
                })

        # Process each window
        for idx, window_info in enumerate(windows_to_process, 1):
            window_seasons = set(window_info['seasons'])
            start_year = window_info['start_year']
            end_year = window_info['end_year']
            cache_path = f"{cache_dir}/player_models_{start_year}_{end_year}.pkl"
            cache_meta_path = f"{cache_dir}/player_models_{start_year}_{end_year}_meta.json"
            is_current = window_info['is_current']

            print(f"\\n{'='*70}")
            print(f"Training player models: Window {idx}/{len(windows_to_process)}")
            print(f"Seasons: {start_year}-{end_year} ({'CURRENT' if is_current else 'historical'})")
            print(f"{'='*70}")

            # Check cache (skip historical windows if cached)
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

            # Build frames for this window (window_seasons triggers internal filtering)
            frames = build_players_from_playerstats(
                player_data_path,
                context_window,
                oof_window,
                verbose=verbose,
                priors_players=priors_window,
                window_seasons=window_seasons
            )

            # Load historical player props for this window
            print(_sec(f"Loading player props for {start_year}-{end_year}"))
            player_props_cache = Path("data/historical_player_props_cache.csv")

            # Filter raw player data to window for prop fetching
            raw_players_df = pd.read_csv(player_data_path, low_memory=False)
            date_col = [c for c in raw_players_df.columns if 'date' in c.lower()][0] if any('date' in c.lower() for c in raw_players_df.columns) else None

            if date_col:
                raw_players_df[date_col] = pd.to_datetime(raw_players_df[date_col], errors="coerce", format='mixed', utc=True).dt.tz_convert(None)
                raw_players_df['season_end_year'] = _season_from_date(raw_players_df[date_col])
                raw_players_df_window = raw_players_df[raw_players_df["season_end_year"].isin(window_seasons)]
            else:
                raw_players_df_window = raw_players_df

            historical_player_props = load_or_fetch_historical_player_props(
                players_df=raw_players_df_window,
                api_key=THEODDS_API_KEY,
                cache_path=player_props_cache,
                verbose=verbose,
                max_requests=100
            )

            # Merge player props into frames
            if not historical_player_props.empty:
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

                    stat_props = historical_player_props[historical_player_props['prop_type'] == prop_type].copy()
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
                    before_merge = len(stat_df)
                    stat_df = stat_df.merge(
                        stat_props[['date_str', 'player_name_norm', 'market_line', 'market_over_odds', 'market_under_odds']],
                        on=['date_str', 'player_name_norm'],
                        how='left'
                    )

                    props_matched = stat_df['market_line'].notna().sum()
                    print(f"- {stat_name}: merged {props_matched:,} / {before_merge:,} prop odds ({props_matched/before_merge*100:.1f}%)")

                    # Clean up
                    stat_df = stat_df.drop(columns=['date_str', 'player_name_norm'], errors='ignore')
                    frames[stat_name] = stat_df

            # Era filter + weights per frame
            player_cut = _parse_season_cutoff(args.player_season_cutoff, kind="player")
            for k, df in list(frames.items()):
                if df is None or df.empty:
                    continue
                if "season_end_year" in df.columns:
                    before = len(df)
                    df = df[(df["season_end_year"].fillna(player_cut)) >= player_cut].reset_index(drop=True)
                    frames[k] = df
                    if verbose:
                        print(f"- {k}: filtered by season >= {player_cut}: {before:,} -> {len(df):,}")
                    # weights
                    df["sample_weight"] = _compute_sample_weights(
                        df["season_end_year"].to_numpy(dtype="float64"),
                        decay=args.decay, min_weight=args.min_weight, lockout_weight=args.lockout_weight
                    )
                    frames[k] = df
                else:
                    df["sample_weight"] = 1.0
                    frames[k] = df

            # Train models for this window
            print(_sec(f"Training models for {start_year}-{end_year}"))

            minutes_model, m_metrics = _fit_minutes_model(frames.get("minutes", pd.DataFrame()), seed=seed + 10, verbose=verbose)
            points_model, points_sigma_model, p_metrics = _fit_stat_model(frames.get("points", pd.DataFrame()), seed=seed + 20, verbose=verbose, name="points")
            rebounds_model, rebounds_sigma_model, r_metrics = _fit_stat_model(frames.get("rebounds", pd.DataFrame()), seed=seed + 30, verbose=verbose, name="rebounds")
            assists_model, assists_sigma_model, a_metrics = _fit_stat_model(frames.get("assists", pd.DataFrame()), seed=seed + 40, verbose=verbose, name="assists")
            threes_model, threes_sigma_model, t_metrics = _fit_stat_model(frames.get("threes", pd.DataFrame()), seed=seed + 50, verbose=verbose, name="threes")

            # Save per-window models
            window_models = {
                'minutes': minutes_model,
                'points': points_model,
                'rebounds': rebounds_model,
                'assists': assists_model,
                'threes': threes_model,
                'points_sigma': points_sigma_model,
                'rebounds_sigma': rebounds_sigma_model,
                'assists_sigma': assists_sigma_model,
                'threes_sigma': threes_sigma_model,
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

            # Free memory before next window
            del context_window, oof_window, priors_window, frames, raw_players_df, raw_players_df_window
            del minutes_model, points_model, rebounds_model, assists_model, threes_model
            del points_sigma_model, rebounds_sigma_model, assists_sigma_model, threes_sigma_model
            gc.collect()

            print(f"Memory freed for next window")

        # Save global models using most recent window (backward compatibility)
        print("\\n" + "="*70)
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

                    if f"{stat_name}_sigma" in latest_models and latest_models[f"{stat_name}_sigma"] is not None:
                        sigma_model_path = models_dir / f"{stat_name}_sigma_model.pkl"
                        with open(sigma_model_path, 'wb') as f:
                            pickle.dump(latest_models[f"{stat_name}_sigma"], f)
                        print(f"  ✓ {stat_name}_sigma_model.pkl")

            # Aggregate metrics from latest window
            player_metrics = latest_models.get('metrics', {})

        # Clean up temp files
        if 'temp_player_csv' in locals():
            temp_player_csv.unlink(missing_ok=True)
        if 'temp_combined_csv' in locals():
            temp_combined_csv.unlink(missing_ok=True)
    else:
        print(_sec("Player models"))
        print("- Skipped (PlayerStatistics.csv not found in Kaggle dataset).")
'''

# Combine all sections
new_file = before_section + [new_player_training] + after_section

# Write back
with open('train_auto.py', 'w', encoding='utf-8') as f:
    f.writelines(new_file)

print("[OK] Replaced all-at-once player training with per-window training")
print("Lines 4000-4253 replaced with per-window implementation")
print("Run: python -m py_compile train_auto.py")
