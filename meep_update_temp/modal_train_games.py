#!/usr/bin/env python
"""
NBA Game Model Training on Modal

Trains game prediction models on Modal's cloud infrastructure.
Run from your laptop:
    modal run modal_train_games.py
"""

import modal

# Create Modal app
app = modal.App("nba-game-training")

# Define volumes (persistent storage)
data_volume = modal.Volume.from_name("nba-data")
model_volume = modal.Volume.from_name("nba-models", create_if_missing=True)

# Define image with all dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        "pandas",
        "numpy",
        "pyarrow",
        "lightgbm",
        "pytorch-tabnet",
        "torch",
        "scikit-learn",
        "scipy"
    )
)


@app.function(
    image=image,
    gpu="A10G",  # NVIDIA A10G GPU ($1.10/hour)
    memory=65536,  # 64GB RAM
    timeout=86400,  # 24 hours
    volumes={
        "/data": data_volume,
        "/models": model_volume
    }
)
def train_game_models():
    """
    Train game models on Modal's A10G GPU with 64GB RAM.
    Output appears in your local terminal!
    """
    import sys
    sys.path.insert(0, "/root/nba_predictor")

    from shared.data_loading import load_player_data
    import pickle
    import json

    print(f"\n{'='*70}")
    print(f"TRAINING GAME MODELS")
    print(f"GPU: A10G | RAM: 64GB | Running on Modal")
    print(f"{'='*70}\n")

    # Load data from Modal volume (CSV directory with all 9 tables)
    print("Loading data from Modal volume...")
    agg_df = load_player_data(
        "/data/csv_dir/",  # Directory with all 9 CSVs
        verbose=True
    )

    # Import and run game model training
    from train_auto import train_game_models as train_games_func

    print(f"\n🏀 Training game models...")

    # Train game models
    models = train_games_func(agg_df, verbose=True)

    # Save to Modal volume
    cache_path = "/models/game_models.pkl"
    meta_path = "/models/game_models_meta.json"

    with open(cache_path, 'wb') as f:
        pickle.dump(models, f)

    metadata = {
        'type': 'game_models',
        'train_rows': len(agg_df),
        'columns': len(agg_df.columns)
    }

    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Commit volume (persist changes)
    model_volume.commit()

    print(f"\n✅ GAME MODELS COMPLETE!")
    print(f"   Saved to: {cache_path}")
    print(f"   Metadata: {meta_path}")

    return metadata


@app.local_entrypoint()
def main():
    """
    Main entry point - runs on YOUR LAPTOP!

    Usage:
        modal run modal_train_games.py
    """
    print("="*70)
    print("NBA GAME MODEL TRAINING ON MODAL")
    print("="*70)
    print("Running from: Your laptop")
    print("Executing on: Modal cloud (A10G GPU)")
    print("="*70)

    print(f"\nTraining game models...")
    result = train_game_models.remote()
    print(f"\n✅ Training complete!")
    print(f"Result: {result}")

    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE!")
    print("="*70)
    print("Models saved to Modal volume: nba-models")
    print("\nTo download models:")
    print("  modal volume get nba-models .")


if __name__ == "__main__":
    pass
