#!/usr/bin/env python
"""
Test CPU training for one window to verify it works
"""

import modal

app = modal.App("test-cpu-training")

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
    memory=32768,  # More memory for caching
    timeout=7200,  # 2 hours
    volumes={"/data": data_volume, "/models": model_volume}
)
def test_one_window_cpu():
    """Test training one window on CPU with optimized data loading"""
    import sys
    sys.path.insert(0, "/root")

    from shared.data_loading import load_player_data, get_year_column
    from train_player_models import create_window_training_data, train_player_window
    import pickle
    import os

    print("="*70)
    print("TESTING CPU TRAINING - ONE WINDOW (OPTIMIZED)")
    print("="*70)
    
    # Load data ONCE and cache it
    print("Loading parquet data ONCE (will be cached for all windows)...")
    agg_df = load_player_data("/data/aggregated_nba_data.parquet", verbose=True)
    year_col = get_year_column(agg_df)
    
    print(f"✅ Data loaded and cached: {len(agg_df):,} rows")
    
    # Test with early window (smaller dataset)
    window_seasons = [1947, 1948, 1949]
    
    # Create window data (uses cached agg_df)
    print("Creating window data from cached data...")
    window_df = create_window_training_data(
        agg_df, window_seasons, year_col, verbose=True
    )

    print(f"Window data: {len(window_df)} rows, {len(window_df.columns)} columns")

    # Train on CPU with verbose output
    print("\n🏀 Training on CPU - should show epochs and progress...")
    result = train_player_window(
        window_df,
        1947, 1949,
        use_gpu=False,  # Force CPU
        verbose=True
    )

    # Save result
    cache_path = "/models/test_cpu_window_1947_1949.pkl"
    with open(cache_path, 'wb') as f:
        pickle.dump(result, f)

    print(f"\n✅ CPU training test successful!")
    print(f"Models saved: {cache_path}")
    print(f"✅ Data was loaded only once (optimized!)")
    return True

@app.local_entrypoint()
def main():
    """Test CPU training"""
    print("Testing CPU training for one window...")
    success = test_one_window_cpu.remote()
    
    if success:
        print("\n✅ SUCCESS! CPU training works")
        print("Ready to train all filtered windows")
    else:
        print("\n❌ FAILED! CPU training still has issues")

if __name__ == "__main__":
    main()
