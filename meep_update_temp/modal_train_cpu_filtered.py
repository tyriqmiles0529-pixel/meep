#!/usr/bin/env python
"""
NBA Model Training on Modal (CPU + Filtered Windows)

Trains window models on CPU, excluding 2022-2024 and 2026 windows.
Usage: modal run modal_train_cpu_filtered.py
"""

import modal

# Create Modal app
app = modal.App("nba-training-cpu")

# Define volumes (persistent storage)
data_volume = modal.Volume.from_name("nba-data")
model_volume = modal.Volume.from_name("nba-models-cpu", create_if_missing=True)

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
    gpu=None,  # CPU only!
    memory=32768,  # 32GB RAM
    timeout=86400,  # 24 hours
    volumes={
        "/data": data_volume,
        "/models": model_volume
    }
)
def train_one_window_cpu(window_start: int, window_end: int):
    """
    Train models for a single window on CPU.
    """
    import sys
    sys.path.insert(0, "/root")

    from shared.data_loading import load_player_data, get_year_column
    from train_player_models import create_window_training_data

    import pickle
    import json

    print(f"\n{'='*70}")
    print(f"TRAINING CPU WINDOW: {window_start}-{window_end}")
    print(f"GPU: None (CPU) | RAM: 32GB | Running on Modal")
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
    from hybrid_multi_task import HybridMultiTaskPlayer

    print(f"\n🏀 Training PLAYER models (TabNet + LightGBM hybrid) on CPU...")
    
    # Temporarily patch HybridMultiTaskPlayer to force CPU
    original_init = HybridMultiTaskPlayer.__init__
    def cpu_init(self, use_gpu=False):
        print(f"  [DEBUG] Initializing HybridMultiTaskPlayer with use_gpu=False")
        original_init(self, use_gpu=False)  # Force CPU
    HybridMultiTaskPlayer.__init__ = cpu_init
    
    print(f"  [DEBUG] Starting train_player_window...")
    player_result = train_player_window(
        window_df,
        window_start,
        window_end,
        verbose=True
    )
    print(f"  [DEBUG] train_player_window completed successfully!")

    # Save models to CPU volume
    player_cache_path = f"/models/player_models_{window_start}_{window_end}.pkl"
    with open(player_cache_path, 'wb') as f:
        pickle.dump(player_result, f)

    print(f"\n✅ CPU Window {window_start}-{window_end} COMPLETE!")
    print(f"   Player models: {player_cache_path}")

    return {
        'player_cache_path': player_cache_path,
        'window_start': window_start,
        'window_end': window_end
    }


@app.function(
    image=image,
    volumes={
        "/data": data_volume,
        "/models": model_volume
    }
)
def get_filtered_windows():
    """Get windows excluding 2022-2024 and 2026 for proper backtesting"""
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
    filtered_windows = []

    for i in range(0, len(all_seasons), window_size):
        window_seasons = all_seasons[i:i+window_size]
        if window_seasons:
            start = window_seasons[0]
            end = window_seasons[-1]

            # Check if already cached in CPU volume
            cache_path = f"/models/player_models_{start}_{end}.pkl"
            is_cached = os.path.exists(cache_path)

            # EXCLUDE windows containing 2022-2024 or 2026
            window_years = set(range(start, end + 1))
            exclude_years = {2022, 2023, 2024, 2026}
            
            has_excluded = bool(window_years & exclude_years)
            
            all_windows.append({
                'start': start,
                'end': end,
                'cached': is_cached,
                'excluded': has_excluded
            })
            
            if not has_excluded:
                filtered_windows.append({
                    'start': start,
                    'end': end,
                    'cached': is_cached
                })

    # Filter to uncached windows
    uncached = [w for w in filtered_windows if not w['cached']]

    print(f"\n{'='*70}")
    print(f"CPU TRAINING PLAN (FILTERED)")
    print(f"{'='*70}")
    print(f"Total windows: {len(all_windows)}")
    print(f"Excluded (2022-2024, 2026): {len(all_windows) - len(filtered_windows)}")
    print(f"Available for training: {len(filtered_windows)}")
    print(f"Already cached: {len(filtered_windows) - len(uncached)}")
    print(f"Need training: {len(uncached)}")

    if uncached:
        print(f"\nWindows to train on CPU:")
        for w in uncached[:10]:
            print(f"  - {w['start']}-{w['end']}")
        if len(uncached) > 10:
            print(f"  ... and {len(uncached) - 10} more")

    return uncached


@app.local_entrypoint()
def main():
    """
    Main entry point - runs on YOUR LAPTOP!
    
    Usage:
        # Train all uncached filtered windows on CPU
        modal run modal_train_cpu_filtered.py
    """
    print(f"\n{'='*70}")
    print(f"NBA MODEL TRAINING - CPU + FILTERED WINDOWS")
    print(f"{'='*70}")
    print(f"✅ Training on CPU (no device conflicts)")
    print(f"❌ Excluding windows with 2022-2024, 2026")
    print(f"📊 Proper backtesting setup:")
    print(f"   - Meta-learner trained on 2023-2024")
    print(f"   - Backtest on 2024-2025")
    print(f"{'='*70}\n")

    # Get windows to train
    windows = get_filtered_windows.remote()
    
    if not windows:
        print("✅ All filtered windows are already cached!")
        return

    print(f"\n🚀 Starting CPU training for {len(windows)} windows...")
    
    # Train each window
    for i, window in enumerate(windows, 1):
        print(f"\n[{i}/{len(windows)}] Training window {window['start']}-{window['end']}...")
        
        try:
            result = train_one_window_cpu.remote(window['start'], window['end'])
            print(f"✅ Complete: {result['player_cache_path']}")
        except Exception as e:
            print(f"❌ Failed: {e}")
            continue

    print(f"\n{'='*70}")
    print(f"✅ CPU TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Models saved to: nba-models-cpu volume")
    print(f"Ready for backtesting:")
    print(f"1. Update meta-learner to use CPU models")
    print(f"2. Train on 2023-2024 data")
    print(f"3. Backtest on 2024-2025 data")


if __name__ == "__main__":
    main()
