#!/usr/bin/env python
"""
Train 2022-2024 Window ONLY

Fast script to train just the missing 2022-2024 window.
Based on the working retrain_2022_plus.py script.

Run: modal run train_2022_2024_only.py
"""

import modal

app = modal.App("train-2022-2024-only")

# Volumes
data_volume = modal.Volume.from_name("nba-data")
model_volume = modal.Volume.from_name("nba-models")

# Image with all dependencies
image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("modal_requirements.txt")
    .add_local_dir("shared", remote_path="/root/shared")
    .add_local_file("train_player_models.py", remote_path="/root/train_player_models.py")
    .add_local_file("hybrid_multi_task.py", remote_path="/root/hybrid_multi_task.py")
    .add_local_file("rolling_features.py", remote_path="/root/rolling_features.py")
)


@app.function(
    image=image,
    gpu="A10G",  # $1.10/hour
    memory=65536,  # 64GB RAM
    timeout=14400,  # 4 hours
    volumes={
        "/data": data_volume,
        "/models": model_volume
    }
)
def train_2022_2024_window():
    """Train 2022-2024 window with rolling features"""
    import sys
    import os
    from pathlib import Path

    sys.path.insert(0, "/root")
    os.chdir("/root")

    print("="*70)
    print(f"TRAINING WINDOW: 2022-2024")
    print("="*70)

    # Import training modules
    from train_player_models import create_window_training_data, train_hybrid_models
    from shared import download_and_process_data

    # Download and process data
    print("[*] Downloading and processing data...")
    agg_df = download_and_process_data()

    # Create 2022-2024 window
    window_start, window_end = 2022, 2024
    window_seasons = list(range(window_start, window_end + 1))
    window_df = create_window_training_data(
        agg_df,
        window_seasons,
        'season_year',
        verbose=True
    )

    print(f"\n[*] Window data shape: {window_df.shape}")
    print(f"[*] Features: {len([c for c in window_df.columns if c not in ['points', 'reboundsTotal', 'assists', 'threePointersMade', 'numMinutes']])}")

    # Train models for this window
    print(f"\n[*] Training models for 2022-2024...")
    models = train_hybrid_models(window_df, verbose=True)

    # Save models
    output_file = f"/models/player_models_{window_start}_{window_end}.pkl"
    import pickle
    with open(output_file, 'wb') as f:
        pickle.dump(models, f)

    print(f"\n✅ Models saved: {output_file}")
    return {
        "window": f"{window_start}-{window_end}",
        "output_file": output_file,
        "shape": window_df.shape,
        "features": len([c for c in window_df.columns if c not in ['points', 'reboundsTotal', 'assists', 'threePointersMade', 'numMinutes']])
    }


@app.local_entrypoint()
def main():
    """Main entry point"""
    print("="*70)
    print("TRAINING 2022-2024 WINDOW ONLY")
    print("="*70)
    print("This will train just the missing 2022-2024 window")
    print("Estimated time: ~4 hours")
    print("Estimated cost: ~$4.40 (4 hours × $1.10/hour)")
    print("="*70)

    # Train the window
    result = train_2022_2024_window.remote()
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE")
    print("="*70)
    print(f"Window: {result['window']}")
    print(f"Samples: {result['shape'][0]:,}")
    print(f"Features: {result['features']}")
    print(f"Saved: {result['output_file']}")
    print("="*70)
    
    print("\n🎯 Next Steps:")
    print("1. Retrain meta-learner with all 27 windows")
    print("2. Run backtesting on 2022-2023 seasons")
    print("3. Move to production integration")

if __name__ == "__main__":
    main()
