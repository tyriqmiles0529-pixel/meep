#!/usr/bin/env python
"""
Train models EXCLUDING 2024-2025 season for backtesting.

This creates models on 1947-2024 data only, so we can validate on 2024-2025.
"""

import modal

# Copy the existing modal_train.py but with max_year filter

app = modal.App("nba-training-backtest")

# Volumes
data_volume = modal.Volume.from_name("nba-data")
model_volume = modal.Volume.from_name("nba-models-backtest", create_if_missing=True)

# Same image as modal_train.py
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
)


@app.function(
    image=image,
    gpu="A10G",
    memory=65536,
    timeout=86400,
    volumes={
        "/data": data_volume,
        "/models": model_volume
    }
)
def train_one_window(window_start: int, window_end: int):
    """Train models for a single window (excluding 2024-2025)"""
    import sys
    sys.path.insert(0, "/root")

    from shared.data_loading import load_player_data, get_year_column
    from train_player_models import create_window_training_data, train_player_window
    import pickle
    import json

    print(f"\n{'='*70}")
    print(f"TRAINING WINDOW: {window_start}-{window_end} (BACKTEST MODE)")
    print(f"GPU: A10G | RAM: 64GB | Running on Modal")
    print(f"Excludes: 2024-2025 season for validation")
    print(f"{'='*70}\n")

    # Load data with max_year filter (exclude 2024-2025)
    print("Loading data from Parquet file (excluding 2024-2025)...")
    agg_df = load_player_data(
        "/data/aggregated_nba_data.parquet",
        max_year=2024,  # Only data up to 2023-2024 season
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
    print(f"\n🏀 Training PLAYER models (TabNet + LightGBM hybrid)...")
    player_result = train_player_window(
        window_df,
        window_start,
        window_end,
        neural_epochs=15,
        verbose=True
    )

    player_models = player_result['models']
    player_metrics = player_result['metrics']

    # Save player models
    player_cache_path = f"/models/player_models_{window_start}_{window_end}.pkl"
    player_meta_path = f"/models/player_models_{window_start}_{window_end}_meta.json"

    with open(player_cache_path, 'wb') as f:
        pickle.dump(player_models, f)

    player_metadata = {
        'window': f'{window_start}-{window_end}',
        'seasons': window_seasons,
        'train_rows': len(window_df),
        'neural_epochs': 15,
        'columns': len(window_df.columns),
        'metrics': player_metrics,
        'backtest_mode': True,
        'excluded_season': '2024-2025'
    }

    with open(player_meta_path, 'w') as f:
        json.dump(player_metadata, f, indent=2)

    # Commit volume
    model_volume.commit()

    print(f"\n✅ Window {window_start}-{window_end} COMPLETE!")
    print(f"   Player models: {player_cache_path}")
    print(f"   Player metadata: {player_meta_path}")

    return {'player_metadata': player_metadata}


@app.function(
    image=image,
    volumes={
        "/data": data_volume,
        "/models": model_volume
    }
)
def get_windows_to_train():
    """Discover which windows need training (up to 2024 only)"""
    import sys
    sys.path.insert(0, "/root")

    from shared.data_loading import load_player_data, get_year_column
    import os

    print("Discovering seasons from data (excluding 2024-2025)...")
    agg_df = load_player_data(
        "/data/aggregated_nba_data.parquet",
        max_year=2024,  # Exclude 2024-2025
        verbose=False
    )
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
    print(f"TRAINING PLAN (BACKTEST MODE)")
    print(f"{'='*70}")
    print(f"Excluded season: 2024-2025 (for validation)")
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
    parallel: int = 3
):
    """
    Train models for backtesting (excludes 2024-2025 season).

    Usage:
        # Train all windows (excluding 2024-2025)
        py -3.12 -m modal run modal_train_exclude_2024.py --parallel 3

        # Train specific window
        py -3.12 -m modal run modal_train_exclude_2024.py --window-start 2019 --window-end 2021
    """
    print("="*70)
    print("NBA MODEL TRAINING FOR BACKTESTING")
    print("="*70)
    print("Training data: 1947-2024 (EXCLUDES 2024-2025)")
    print("Purpose: Validate on complete 2024-2025 season")
    print("Models saved to: nba-models-backtest volume")
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
    print("Models saved to Modal volume: nba-models-backtest")
    print("\nTo download models:")
    print("  py -3.12 -m modal volume get nba-models-backtest / model_cache_backtest")
    print("\nTo run backtest:")
    print("  python backtest_2024_2025.py")


if __name__ == "__main__":
    pass
