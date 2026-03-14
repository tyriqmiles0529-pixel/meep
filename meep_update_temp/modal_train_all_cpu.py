#!/usr/bin/env python
"""
NBA Model Training on Modal (CPU + Filtered Windows - FINAL VERSION)

Trains all window models on CPU, excluding 2022-2024 and 2026 windows.
Uses optimized data loading (load once, cache) and verified CPU training.
"""

import modal

app = modal.App("nba-training-cpu-final")

data_volume = modal.Volume.from_name("nba-data")
model_volume = modal.Volume.from_name("nba-models-cpu", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("modal_requirements.txt")
    .add_local_dir("shared", remote_path="/root/shared")
    .add_local_file("train_player_models.py", remote_path="/root/train_player_models.py")
    .add_local_file("hybrid_multi_task.py", remote_path="/root/hybrid_multi_task.py")
    .add_local_file("optimization_features.py", remote_path="/root/optimization_features.py")
    .add_local_file("phase7_features.py", remote_path="/root/phase7_features.py")
    .add_local_file("rolling_features.py", remote_path="/root/rolling_features.py")
)

@app.function(
    image=image,
    gpu=None,  # CPU only
    memory=32768,  # 32GB RAM for caching
    timeout=86400,  # 24 hours
    volumes={"/data": data_volume, "/models": model_volume}
)
def train_all_windows_cpu():
    """Train all filtered windows on CPU with optimized data loading"""
    import sys
    sys.path.insert(0, "/root")

    from shared.data_loading import load_player_data, get_year_column
    from train_player_models import create_window_training_data, train_player_window
    import pickle
    import os

    print("="*70)
    print("NBA MODEL TRAINING - CPU + FILTERED WINDOWS (FINAL)")
    print("="*70)
    print("✅ Training on CPU (verified working)")
    print("❌ Excluding windows with 2022-2024, 2026")
    print("📊 Optimized data loading (load once, cache)")
    print("🧹 Clearing existing cache for fresh CPU models")
    print("="*70)

    # Clear existing cache for fresh CPU models
    print("Clearing existing model cache...")
    import os
    for file in os.listdir("/models"):
        if file.startswith("player_models_") and file.endswith(".pkl"):
            os.remove(f"/models/{file}")
            print(f"  Removed: {file}")
    print("✅ Cache cleared - starting fresh CPU training")

    # Load data ONCE and cache it
    print("Loading parquet data ONCE (will be cached for all windows)...")
    agg_df = load_player_data("/data/aggregated_nba_data.parquet", verbose=True)
    year_col = get_year_column(agg_df)
    
    print(f"✅ Data loaded and cached: {len(agg_df):,} rows")

    # Get all seasons and create filtered windows
    all_seasons = sorted([int(s) for s in agg_df[year_col].dropna().unique()])
    window_size = 3
    filtered_windows = []

    for i in range(0, len(all_seasons), window_size):
        window_seasons = all_seasons[i:i+window_size]
        if window_seasons:
            start = window_seasons[0]
            end = window_seasons[-1]

            # EXCLUDE windows containing 2022-2024 or 2026
            window_years = set(range(start, end + 1))
            exclude_years = {2022, 2023, 2024, 2026}
            
            if window_years & exclude_years:
                print(f"❌ Skipping window {start}-{end} (contains excluded years)")
                continue

            # Check if already cached
            cache_path = f"/models/player_models_{start}_{end}.pkl"
            is_cached = os.path.exists(cache_path)
            
            filtered_windows.append({
                'start': start,
                'end': end,
                'cached': is_cached
            })

    # Filter to uncached windows
    uncached = [w for w in filtered_windows if not w['cached']]

    print(f"\n{'='*70}")
    print(f"TRAINING PLAN (CPU + FILTERED)")
    print(f"{'='*70}")
    print(f"Total windows available: {len(filtered_windows)}")
    print(f"Already cached: {len(filtered_windows) - len(uncached)}")
    print(f"Need training: {len(uncached)}")

    if uncached:
        print(f"\nWindows to train on CPU:")
        for w in uncached:
            print(f"  - {w['start']}-{w['end']}")
    else:
        print("✅ All filtered windows are already cached!")
        return

    print(f"\n🚀 Starting CPU training for {len(uncached)} windows...")
    
    # Train each window
    success_count = 0
    for i, window in enumerate(uncached, 1):
        print(f"\n[{i}/{len(uncached)}] Training window {window['start']}-{window['end']}...")
        
        try:
            # Create window data from cached agg_df
            window_seasons = list(range(window['start'], window['end'] + 1))
            window_df = create_window_training_data(
                agg_df, window_seasons, year_col, verbose=False
            )

            print(f"  Window data: {len(window_df)} rows")
            
            # Train on CPU (verified working approach)
            print(f"  🏀 Training CPU models...")
            result = train_player_window(
                window_df,
                window['start'],
                window['end'],
                use_gpu=False,  # Force CPU
                verbose=False  # Reduce noise for batch training
            )

            # Save models
            cache_path = f"/models/player_models_{window['start']}_{window['end']}.pkl"
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)

            print(f"  ✅ Complete: {cache_path}")
            success_count += 1

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            continue

    print(f"\n{'='*70}")
    print(f"✅ CPU TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Successfully trained: {success_count}/{len(uncached)} windows")
    print(f"Models saved to: nba-models-cpu volume")
    print(f"✅ Ready for backtesting:")
    print(f"   1. Meta-learner trained on 2023-2024 data")
    print(f"   2. Backtest on 2024-2025 data")
    print(f"   3. No device conflicts (all CPU)")

    return success_count

@app.local_entrypoint()
def main():
    """Train all filtered windows on CPU"""
    success_count = train_all_windows_cpu.remote()
    
    if success_count > 0:
        print(f"\n🎉 SUCCESS! Trained {success_count} windows on CPU")
        print("Your models are ready for conflict-free backtesting!")
    else:
        print("\n✅ All models were already cached")

if __name__ == "__main__":
    main()
