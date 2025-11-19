#!/usr/bin/env python
"""
NBA Model Training on Modal (Player + Game Models)

Trains both player and game prediction models on Modal's cloud infrastructure.
Run from your laptop:
    modal run modal_train.py

All output appears in your local terminal!
"""

import modal

# Create Modal app
app = modal.App("nba-training")

# Define volumes (persistent storage)
data_volume = modal.Volume.from_name("nba-data")
model_volume = modal.Volume.from_name("nba-models", create_if_missing=True)

# Define image with all dependencies and code
image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("modal_requirements.txt")
    .add_local_dir("shared", remote_path="/root/shared")
    .add_local_file("train_player_models.py", remote_path="/root/train_player_models.py")
    .add_local_file("train_auto.py", remote_path="/root/train_auto.py")
    .add_local_file("train_ensemble_enhanced.py", remote_path="/root/train_ensemble_enhanced.py")
    .add_local_file("neural_hybrid.py", remote_path="/root/neural_hybrid.py")
    .add_local_file("hybrid_multi_task.py", remote_path="/root/hybrid_multi_task.py")
    .add_local_file("optimization_features.py", remote_path="/root/optimization_features.py")
    .add_local_file("phase7_features.py", remote_path="/root/phase7_features.py")
    .add_local_file("rolling_features.py", remote_path="/root/rolling_features.py")
)


@app.function(
    image=image,
    gpu="A10G",  # NVIDIA A10G GPU ($1.10/hour)
    memory=65536,  # 64GB RAM (way more than Kaggle!)
    timeout=86400,  # 24 hours (no Kaggle limit!)
    volumes={
        "/data": data_volume,
        "/models": model_volume
    }
)
def train_one_window(window_start: int, window_end: int):
    """
    Train models for a single window.

    This runs on Modal's A10G GPU with 64GB RAM.
    Output appears in your local terminal!
    """
    import sys
    sys.path.insert(0, "/root")

    from shared.data_loading import load_player_data, get_year_column
    from train_player_models import create_window_training_data

    import pickle
    import json

    print(f"\n{'='*70}")
    print(f"TRAINING WINDOW: {window_start}-{window_end}")
    print(f"GPU: A10G | RAM: 64GB | Running on Modal")
    print(f"{'='*70}\n")

    # Load data from Parquet file (all data pre-merged, 1947-2026)
    print("Loading data from Parquet file...")
    agg_df = load_player_data(
        "/data/aggregated_nba_data.parquet",  # Pre-merged Parquet (1947-2026 full data)
        verbose=True
    )

    year_col = get_year_column(agg_df)
    window_seasons = list(range(window_start, window_end + 1))

    # Create window data
    print(f"\nCreating window training data...")
    window_df = create_window_training_data(
        agg_df,
        window_seasons,
        year_col,
        verbose=True
    )

    # Train player models
    from train_player_models import train_player_window

    print(f"\n🏀 Training PLAYER models (TabNet + LightGBM hybrid)...")
    player_result = train_player_window(
        window_df,
        window_start,
        window_end,
        neural_epochs=12,
        verbose=True
    )

    player_models = player_result['models']
    player_metrics = player_result['metrics']

    # Train game models
    print(f"\n🎯 Training GAME models (moneyline + spread)...")
    from train_auto import build_games_from_teamstats
    from pathlib import Path

    # Build games DataFrame from window data
    try:
        # Create temporary CSV for games construction
        temp_team_stats = "/tmp/team_stats_temp.csv"

        # Extract team-level data from window_df
        team_cols = ['gameId', 'gameDate', 'teamId', 'opponentTeamId', 'home',
                     'teamScore', 'opponentScore']
        available_cols = [c for c in team_cols if c in window_df.columns]

        if len(available_cols) >= 5:  # Need at least core columns
            window_df[available_cols].to_csv(temp_team_stats, index=False)

            games_df, games_long, team_map = build_games_from_teamstats(
                Path(temp_team_stats),
                verbose=True,
                skip_rest=True
            )

            # Train game models
            from train_auto import _fit_game_models

            ml_model, ml_calib, sp_model, sp_sigma, oof_df, game_metrics = _fit_game_models(
                games_df=games_df,
                seed=42,
                verbose=True,
                folds=5,
                lgb_log_period=0,
                sample_weights=None,
                use_neural=True,  # Use neural hybrid for games too
                neural_device='gpu',
                neural_epochs=12,
                batch_size=8192
            )

            game_models = {
                'moneyline': ml_model,
                'moneyline_calibrated': ml_calib,
                'spread': sp_model,
                'spread_sigma': sp_sigma
            }

            print(f"   ✓ Game models trained: {len(games_df):,} games")
        else:
            print(f"   ⚠ Insufficient columns for game models, skipping")
            game_models = None
            game_metrics = {}

    except Exception as e:
        print(f"   ⚠ Game model training failed: {e}")
        game_models = None
        game_metrics = {}

    # Save player models
    player_cache_path = f"/models/player_models_{window_start}_{window_end}.pkl"
    player_meta_path = f"/models/player_models_{window_start}_{window_end}_meta.json"

    with open(player_cache_path, 'wb') as f:
        pickle.dump(player_models, f)

    player_metadata = {
        'window': f'{window_start}-{window_end}',
        'seasons': window_seasons,
        'train_rows': len(window_df),
        'neural_epochs': 12,
        'columns': len(window_df.columns),
        'metrics': player_metrics
    }

    with open(player_meta_path, 'w') as f:
        json.dump(player_metadata, f, indent=2)

    # Save game models
    if game_models:
        game_cache_path = f"/models/game_models_{window_start}_{window_end}.pkl"
        game_meta_path = f"/models/game_models_{window_start}_{window_end}_meta.json"

        with open(game_cache_path, 'wb') as f:
            pickle.dump(game_models, f)

        game_metadata = {
            'window': f'{window_start}-{window_end}',
            'seasons': window_seasons,
            'train_rows': len(games_df) if 'games_df' in locals() else 0,
            'neural_epochs': 12,
            'metrics': game_metrics
        }

        with open(game_meta_path, 'w') as f:
            json.dump(game_metadata, f, indent=2)

    # Commit volume (persist changes)
    model_volume.commit()

    print(f"\n✅ Window {window_start}-{window_end} COMPLETE!")
    print(f"   Player models: {player_cache_path}")
    print(f"   Player metadata: {player_meta_path}")
    if game_models:
        print(f"   Game models: {game_cache_path}")
        print(f"   Game metadata: {game_meta_path}")

    return {
        'player_metadata': player_metadata,
        'game_metadata': game_metadata if game_models else None
    }


@app.function(
    image=image,
    volumes={
        "/data": data_volume,
        "/models": model_volume
    }
)
def get_windows_to_train():
    """Discover which windows need training"""
    import sys
    sys.path.insert(0, "/root")

    from shared.data_loading import load_player_data, get_year_column
    import os

    # Load data to discover all seasons
    print("Discovering seasons from data...")
    agg_df = load_player_data("/data/aggregated_nba_data.parquet", verbose=False)
    year_col = get_year_column(agg_df)
    all_seasons = sorted([int(s) for s in agg_df[year_col].dropna().unique()])

    # Create 3-year windows
    window_size = 3
    all_windows = []

    for i in range(0, len(all_seasons), window_size):
        window_seasons = all_seasons[i:i+window_size]
        if window_seasons:
            start = window_seasons[0]
            end = window_seasons[-1]

            # Check if already cached
            cache_path = f"/models/player_models_{start}_{end}.pkl"
            is_cached = os.path.exists(cache_path)

            all_windows.append({
                'start': start,
                'end': end,
                'cached': is_cached
            })

    # Filter to uncached windows
    uncached = [w for w in all_windows if not w['cached']]

    print(f"\n{'='*70}")
    print(f"TRAINING PLAN")
    print(f"{'='*70}")
    print(f"Total windows: {len(all_windows)}")
    print(f"Cached: {len(all_windows) - len(uncached)}")
    print(f"Need training: {len(uncached)}")

    if uncached:
        print(f"\nWindows to train:")
        for w in uncached[:10]:
            print(f"  - {w['start']}-{w['end']}")
        if len(uncached) > 10:
            print(f"  ... and {len(uncached) - 10} more")

    return uncached


@app.local_entrypoint()
def main(
    window_start: int = None,
    window_end: int = None,
    parallel: int = 1
):
    """
    Main entry point - runs on YOUR LAPTOP!

    Usage:
        # Train all uncached windows (one at a time)
        modal run modal_train.py

        # Train specific window
        modal run modal_train.py --window-start 2022 --window-end 2024

        # Train 3 windows in parallel
        modal run modal_train.py --parallel 3
    """
    print("="*70)
    print("NBA MODEL TRAINING ON MODAL (PLAYER + GAME)")
    print("="*70)
    print("Running from: Your laptop")
    print("Executing on: Modal cloud (A10G GPU + 64GB RAM)")
    print("Models: Player props + Game outcomes (both with neural hybrids)")
    print("="*70)

    # Get windows to train
    windows = get_windows_to_train.remote()

    if not windows:
        print("\n✅ ALL WINDOWS ALREADY TRAINED!")
        return

    # If specific window requested
    if window_start and window_end:
        print(f"\nTraining window: {window_start}-{window_end}")
        result = train_one_window.remote(window_start, window_end)
        print(f"\n✅ Training complete!")
        print(f"Result: {result}")
        return

    # Train all windows
    total = len(windows)

    if parallel > 1:
        print(f"\n🚀 Training {parallel} windows in parallel...")

        for i in range(0, total, parallel):
            batch = windows[i:i+parallel]
            print(f"\n📦 Batch {i//parallel + 1}/{(total + parallel - 1)//parallel}")

            # Launch parallel training
            futures = [
                train_one_window.spawn(w['start'], w['end'])
                for w in batch
            ]

            # Wait for batch
            results = [f.get() for f in futures]
            print(f"✅ Batch complete!")

    else:
        print(f"\n📦 Training {total} windows (one at a time)...")

        for idx, w in enumerate(windows, 1):
            print(f"\n[{idx}/{total}] Training {w['start']}-{w['end']}...")
            result = train_one_window.remote(w['start'], w['end'])
            print(f"✅ Window complete: {result}")

    print("\n" + "="*70)
    print("🎉 ALL TRAINING COMPLETE!")
    print("="*70)
    print("Models saved to Modal volume: nba-models")
    print("\nTo download models:")
    print("  modal volume get nba-models .")


if __name__ == "__main__":
    # This allows running with: python modal_train.py (locally for testing)
    # But normally you run: modal run modal_train.py
    pass
