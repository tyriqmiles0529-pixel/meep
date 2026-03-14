#!/usr/bin/env python
"""
Modal Split Training - One Window Per Function Call

Splits training into separate function calls to avoid timeout issues.
Each window gets its own 2-hour timeout.

Usage:
    modal run modal_split_training.py
"""

import modal

app = modal.App("nba-split-training")

# Volumes
data_volume = modal.Volume.from_name("nba-data")
model_volume = modal.Volume.from_name("nba-models-cpu")

# Image with all dependencies copied
image = (
    modal.Image.debian_slim()
    .pip_install([
        "pandas>=2.0.0",
        "numpy>=1.24.0", 
        "scikit-learn>=1.3.0",
        "lightgbm>=4.0.0",
        "pytorch-tabnet>=4.0.0",
        "torch>=2.0.0",
        "joblib>=1.3.0",
        "pyarrow>=12.0.0",
        "fastparquet>=2023.0.0"
    ])
    .add_local_dir("shared", remote_path="/root/shared")
    .add_local_file("train_player_models.py", remote_path="/root/train_player_models.py")
    .add_local_file("hybrid_multi_task.py", remote_path="/root/hybrid_multi_task.py")
    .add_local_file("optimization_features.py", remote_path="/root/optimization_features.py")
    .add_local_file("phase7_features.py", remote_path="/root/phase7_features.py")
    .add_local_file("rolling_features.py", remote_path="/root/rolling_features.py")
)

@app.function(
    image=image,
    gpu=None,
    memory=32768,
    timeout=7200,  # 2 hours per window
    volumes={"/data": data_volume, "/models": model_volume}
)
def train_single_window(start_year: int, end_year: int):
    """Train a single window with 2-hour timeout"""
    import sys
    sys.path.insert(0, "/root")
    
    from shared.data_loading import load_player_data, get_year_column
    from train_player_models import create_window_training_data, train_player_window
    import pickle
    import os
    import pandas as pd
    
    print(f"🏀 Training window {start_year}-{end_year}...")
    
    # Load data
    df = pd.read_parquet("/data/aggregated_nba_data.parquet")
    
    # Create window data
    year_col = get_year_column(df)
    window_seasons = list(range(start_year, end_year + 1))
    window_df = create_window_training_data(df, window_seasons, year_col, verbose=False)
    
    print(f"   📊 Data: {len(window_df):,} rows")
    
    # Train with hybrid architecture
    result = train_player_window(
        window_df, 
        start_year, 
        end_year,
        neural_epochs=15,
        verbose=True,
        use_multi_task=True,
        use_gpu=False
    )
    
    # Save model
    model_path = f"/models/player_models_{start_year}_{end_year}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(result, f)
    
    size_mb = os.path.getsize(model_path) / (1024*1024)
    print(f"   ✅ Saved: {model_path} ({size_mb:.1f} MB)")
    
    return {
        "window": f"{start_year}-{end_year}",
        "status": "completed",
        "size_mb": size_mb
    }

@app.local_entrypoint()
def main():
    """Train all missing windows one by one"""
    print("🚀 Starting Split Training (One Window Per Call)...")
    print("   Each window gets its own 2-hour timeout")
    print()
    
    # Get missing windows
    missing_windows = [(2007, 2009), (2010, 2012), (2013, 2015), (2016, 2018), (2019, 2021)]
    
    print(f"📋 Training {len(missing_windows)} windows:")
    for start, end in missing_windows:
        print(f"   ❌ {start}-{end}")
    
    print(f"\n🏀 Training windows sequentially...")
    
    completed = 0
    failed = 0
    
    for i, (start_year, end_year) in enumerate(missing_windows, 1):
        print(f"\n[{i}/{len(missing_windows)}] Starting {start_year}-{end_year}...")
        
        try:
            result = train_single_window.remote(start_year, end_year)
            print(f"   ✅ Completed: {result['window']} ({result['size_mb']:.1f} MB)")
            completed += 1
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            failed += 1
            continue
    
    print(f"\n" + "="*80)
    print(f"🏁 SPLIT TRAINING COMPLETE!")
    print(f"="*80)
    print(f"✅ Completed windows: {completed}")
    print(f"❌ Failed windows: {failed}")
    print(f"📊 Total: {completed}/{len(missing_windows)}")
    
    if completed == len(missing_windows):
        print(f"\n🎉 ALL 5 WINDOWS COMPLETED!")
        print(f"   Ready for meta-learner training!")
        next_step = "python train_meta_learner_v2.py"
    else:
        print(f"\n⚠ {failed} windows failed")
        next_step = "Debug failed windows"
    
    print(f"\n🚀 Next step: {next_step}")

if __name__ == "__main__":
    main()