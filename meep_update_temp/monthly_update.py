"""
Monthly Model Update Script

This script handles incremental training for monthly model updates:
1. Sliding window training (last N years)
2. LightGBM warm start from previous model
3. TabNet retrain on recent data

Usage:
    python monthly_update.py --strategy hybrid --window-years 3
"""

import argparse
import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
from pathlib import Path
from datetime import datetime, timedelta
import os


def get_sliding_window_data(df, window_years=3, player_start_year=2022):
    """
    Get training data for sliding window approach.

    Args:
        df: Full dataset
        window_years: Number of recent years to include
        player_start_year: Earliest year with player data (Kaggle limitation)

    Returns:
        Filtered dataframe with sliding window
    """
    # Get current season
    if 'season' in df.columns:
        current_season = df['season'].max()
    elif 'GAME_DATE_EST' in df.columns:
        df['season'] = pd.to_datetime(df['GAME_DATE_EST']).dt.year
        current_season = df['season'].max()
    else:
        raise ValueError("No season or date column found")

    # Calculate cutoff
    cutoff_season = current_season - window_years

    print(f"\n[INFO] Sliding Window Configuration:")
    print(f"  Current season: {current_season}")
    print(f"  Window years: {window_years}")
    print(f"  Training on: {cutoff_season}-{current_season}")

    # Filter data
    window_df = df[df['season'] >= cutoff_season].copy()

    print(f"  Total samples: {len(df):,}")
    print(f"  Window samples: {len(window_df):,} ({len(window_df)/len(df)*100:.1f}%)")

    return window_df


def get_new_games_since(df, last_update_date):
    """
    Get only games played since last model update.

    Args:
        df: Full dataset
        last_update_date: Date of last update (str or datetime)

    Returns:
        Dataframe with only new games
    """
    if isinstance(last_update_date, str):
        last_update_date = pd.to_datetime(last_update_date)

    # Ensure date column exists
    if 'GAME_DATE_EST' in df.columns:
        df['GAME_DATE_DT'] = pd.to_datetime(df['GAME_DATE_EST'])
    elif 'GAME_DATE_DT' not in df.columns:
        raise ValueError("No date column found")

    new_games = df[df['GAME_DATE_DT'] > last_update_date].copy()

    print(f"\n[INFO] New Games Since Last Update:")
    print(f"  Last update: {last_update_date.date()}")
    print(f"  New games: {len(new_games):,}")
    if len(new_games) > 0:
        print(f"  Date range: {new_games['GAME_DATE_DT'].min().date()} to {new_games['GAME_DATE_DT'].max().date()}")

    return new_games


def warm_start_lightgbm(previous_model_path, new_data, new_labels, params, num_rounds=200):
    """
    Continue training LightGBM from previous model.

    Args:
        previous_model_path: Path to previous .pkl or .txt model
        new_data: New training data (DataFrame or numpy array)
        new_labels: New target labels
        params: LightGBM parameters
        num_rounds: Number of additional boosting rounds

    Returns:
        Updated LightGBM model
    """
    print(f"\n[INFO] LightGBM Warm Start:")
    print(f"  Loading model from: {previous_model_path}")

    # Load previous model
    if str(previous_model_path).endswith('.pkl'):
        with open(previous_model_path, 'rb') as f:
            previous_model = pickle.load(f)
    else:
        previous_model = lgb.Booster(model_file=previous_model_path)

    print(f"  Previous model trees: {previous_model.num_trees()}")

    # Create dataset for new data
    new_dataset = lgb.Dataset(new_data, label=new_labels)

    print(f"  New training samples: {len(new_labels):,}")
    print(f"  Additional boosting rounds: {num_rounds}")

    # Continue training
    updated_model = lgb.train(
        params,
        new_dataset,
        num_boost_round=num_rounds,
        init_model=previous_model,  # Warm start from previous
        valid_sets=[new_dataset],
        valid_names=['new_games'],
        verbose_eval=50
    )

    print(f"  Updated model trees: {updated_model.num_trees()}")
    print(f"  Added {updated_model.num_trees() - previous_model.num_trees()} new trees")

    return updated_model


def main():
    parser = argparse.ArgumentParser(description='Monthly Model Update')
    parser.add_argument('--strategy', type=str, default='hybrid',
                       choices=['full_retrain', 'sliding_window', 'hybrid', 'warm_start'],
                       help='Update strategy')
    parser.add_argument('--window-years', type=int, default=3,
                       help='Years for sliding window (default: 3)')
    parser.add_argument('--last-update', type=str, default=None,
                       help='Date of last update (YYYY-MM-DD)')
    parser.add_argument('--warm-start', action='store_true',
                       help='Use warm start for LightGBM')
    parser.add_argument('--previous-model', type=str, default=None,
                       help='Path to previous model for warm start')
    parser.add_argument('--dataset', type=str,
                       default='eoinamoore/historical-nba-data-and-player-box-scores',
                       help='Kaggle dataset')

    args = parser.parse_args()

    print("="*70)
    print("MONTHLY MODEL UPDATE")
    print("="*70)
    print(f"Strategy: {args.strategy}")
    print(f"Sliding window: {args.window_years} years")
    print(f"Warm start: {args.warm_start}")

    # Example workflow (integrate with your train_auto.py)
    print("\n[INFO] This is a template script.")
    print("[INFO] Integrate with your existing train_auto.py")
    print("\n[WORKFLOW]:")

    if args.strategy == 'full_retrain':
        print("1. Run: python train_auto.py --dataset [dataset] --fresh --enable-window-ensemble")
        print("   (Full retrain on all historical data)")

    elif args.strategy == 'sliding_window':
        print(f"1. Filter data to last {args.window_years} years")
        print("2. Run: python train_auto.py with filtered data")
        print("   (Faster training, focuses on recent meta)")

    elif args.strategy == 'warm_start':
        if not args.previous_model:
            print("[ERROR] --previous-model required for warm start")
            return
        print(f"1. Load previous model from: {args.previous_model}")
        print("2. Get new games since last update")
        print("3. Continue training with lgb.train(init_model=...)")
        print("   (Fastest update, builds on previous)")

    elif args.strategy == 'hybrid':
        print("1. TabNet: Retrain on sliding window (last 3 years)")
        print("2. LightGBM: Warm start from previous model + new games")
        print("3. Recalibration: Update on recent games")
        print("   (Best of both: speed + accuracy)")

    print("\n[NEXT STEPS]:")
    print("1. Integrate this logic into train_auto.py")
    print("2. Add --strategy and --window-years flags")
    print("3. Implement warm start for LightGBM models")
    print("4. Create version control for models")


if __name__ == "__main__":
    main()
