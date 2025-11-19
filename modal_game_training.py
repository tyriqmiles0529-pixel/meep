#!/usr/bin/env python
"""
NBA Game Models Training on Modal (Moneyline + Spread)

Trains game prediction models (moneyline, spread) on Modal's cloud infrastructure.
Separate from player models for better organization.

Run from your laptop:
    modal run modal_game_training.py --start-year 2002 --end-year 2026

All output appears in your local terminal!
"""

import modal

# Create Modal app
app = modal.App("nba-game-training")

# Define volumes (persistent storage)
data_volume = modal.Volume.from_name("nba-data")
model_volume = modal.Volume.from_name("nba-models", create_if_missing=True)

# Define image with all dependencies and code
image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("modal_requirements.txt")
    .add_local_dir("shared", remote_path="/root/shared")
    .add_local_file("train_auto.py", remote_path="/root/train_auto.py")
    .add_local_file("neural_hybrid.py", remote_path="/root/neural_hybrid.py")
)


@app.function(
    image=image,
    gpu="A10G",  # NVIDIA A10G GPU ($1.10/hour)
    memory=32768,  # 32GB RAM
    timeout=14400,  # 4 hours
    volumes={
        "/data": data_volume,
        "/models": model_volume
    }
)
def train_game_models(start_year: int = 2002, end_year: int = 2026):
    """
    Train game models (moneyline + spread) for specified year range.

    This runs on Modal's A10G GPU with 32GB RAM.
    Output appears in your local terminal!
    """
    import sys
    sys.path.insert(0, "/root")

    from shared.data_loading import load_player_data, get_year_column
    from train_auto import build_games_from_teamstats, _fit_game_models
    from pathlib import Path
    import pandas as pd
    import pickle
    import json

    print(f"\n{'='*70}")
    print(f"TRAINING GAME MODELS: {start_year}-{end_year}")
    print(f"GPU: A10G | RAM: 32GB | Running on Modal")
    print(f"{'='*70}\n")

    # Load data from Parquet file
    print("Loading data from Parquet file...")
    agg_df = load_player_data(
        "/data/aggregated_nba_data.parquet",
        verbose=True
    )

    year_col = get_year_column(agg_df)

    # Filter to specified year range
    seasons = list(range(start_year, end_year + 1))
    df = agg_df[agg_df[year_col].isin(seasons)].copy()

    print(f"\nFiltered to {start_year}-{end_year}: {len(df):,} rows")

    # Extract team-level data
    team_cols = ['gameId', 'gameDate', 'teamId', 'opponentTeamId', 'home',
                 'teamScore', 'opponentScore', year_col]
    available_cols = [c for c in team_cols if c in df.columns]

    if len(available_cols) < 5:
        raise ValueError(f"Missing required columns for game models. Found: {available_cols}")

    # Create temporary CSV for games construction
    temp_team_stats = "/tmp/team_stats.csv"
    df[available_cols].to_csv(temp_team_stats, index=False)

    print(f"\nBuilding games DataFrame from team stats...")
    games_df, games_long, team_map = build_games_from_teamstats(
        Path(temp_team_stats),
        verbose=True,
        skip_rest=True
    )

    print(f"  • Games DataFrame: {len(games_df):,} games")
    print(f"  • Features: {len(games_df.columns)} columns")

    # Train game models
    print(f"\n🎯 Training game models (moneyline + spread)...")
    ml_model, ml_calib, sp_model, sp_sigma, oof_df, metrics = _fit_game_models(
        games_df=games_df,
        seed=42,
        verbose=True,
        folds=5,
        lgb_log_period=0,
        sample_weights=None,
        use_neural=True,  # Use neural hybrid
        neural_device='gpu',
        neural_epochs=15,
        batch_size=8192
    )

    game_models = {
        'moneyline': ml_model,
        'moneyline_calibrated': ml_calib,
        'spread': sp_model,
        'spread_sigma': sp_sigma
    }

    print(f"\n✓ Game models trained successfully")

    # Save models
    game_cache_path = f"/models/game_models_{start_year}_{end_year}.pkl"
    game_meta_path = f"/models/game_models_{start_year}_{end_year}_meta.json"

    with open(game_cache_path, 'wb') as f:
        pickle.dump(game_models, f)

    game_metadata = {
        'years': f'{start_year}-{end_year}',
        'seasons': seasons,
        'train_rows': len(games_df),
        'neural_epochs': 15,
        'columns': len(games_df.columns),
        'metrics': metrics
    }

    with open(game_meta_path, 'w') as f:
        json.dump(game_metadata, f, indent=2)

    print(f"\n{'='*70}")
    print(f"GAME MODEL TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Models saved to: {game_cache_path}")
    print(f"Metadata saved to: {game_meta_path}")
    print(f"\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Commit volumes
    data_volume.commit()
    model_volume.commit()

    return {
        'models': game_models,
        'metrics': metrics,
        'metadata': game_metadata
    }


@app.local_entrypoint()
def main(start_year: int = 2002, end_year: int = 2026):
    """
    Main entry point - runs locally and launches Modal job.

    Usage:
        modal run modal_game_training.py
        modal run modal_game_training.py --start-year 2010 --end-year 2024
    """
    print(f"\n🚀 Launching game model training on Modal...")
    print(f"   Years: {start_year}-{end_year}")
    print(f"   GPU: A10G ($1.10/hour)")
    print(f"   RAM: 32GB")

    result = train_game_models.remote(start_year, end_year)

    print(f"\n✓ Training complete!")
    print(f"\nTo download models:")
    print(f"   modal volume get nba-models /models/game_models_{start_year}_{end_year}.pkl game_models.pkl")
