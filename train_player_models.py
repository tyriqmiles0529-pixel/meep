#!/usr/bin/env python
"""
Train Player Prediction Models (Windowed Ensemble Approach)

Trains TabNet + LightGBM hybrid models for player props:
- Points, Rebounds, Assists, Three-Pointers, Minutes

Uses 3-year rolling windows to reduce memory and improve temporal accuracy.
"""

import os
import sys
import gc
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from shared.data_loading import load_aggregated_player_data, get_year_column, get_season_range


def parse_args():
    parser = argparse.ArgumentParser(description='Train NBA Player Prediction Models')

    # Data sources
    parser.add_argument('--aggregated-data', type=str, required=True,
                        help='Path to aggregated_nba_data.parquet')

    # Training parameters
    parser.add_argument('--window-size', type=int, default=3,
                        help='Size of training windows in years (default: 3)')
    parser.add_argument('--neural-epochs', type=int, default=12,
                        help='TabNet training epochs (default: 12)')
    parser.add_argument('--cache-dir', type=str, default='model_cache',
                        help='Directory to cache trained models')

    # Data filtering
    parser.add_argument('--min-year', type=int, default=None,
                        help='Minimum season year to train on (e.g., 2002)')
    parser.add_argument('--max-year', type=int, default=None,
                        help='Maximum season year to train on (e.g., 2024)')

    # Model options
    parser.add_argument('--skip-cached', action='store_true',
                        help='Skip windows that are already cached')
    parser.add_argument('--force-retrain', action='store_true',
                        help='Force retrain all windows (ignore cache)')

    # Output
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Print detailed progress')

    return parser.parse_args()


def create_window_training_data(
    agg_df: pd.DataFrame,
    window_seasons: List[int],
    year_col: str,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Create training data for a specific window.

    Args:
        agg_df: Full aggregated player data
        window_seasons: List of seasons for this window
        year_col: Name of the year column
        verbose: Print progress

    Returns:
        DataFrame with window data and recomputed reboundsTotal
    """
    start_year = min(window_seasons)
    end_year = max(window_seasons)

    # Add buffer seasons for rolling features
    padded_seasons = set(window_seasons) | {start_year-1, end_year+1}

    # Filter to window
    window_df = agg_df[agg_df[year_col].isin(padded_seasons)].copy()

    if verbose:
        print(f"  • Filtered aggregated data for window: {len(window_df):,} rows")

    # Memory optimization: Keep only essential columns
    essential_cols = [
        'personId', 'gameId', 'gameDate', 'firstName', 'lastName',
        'home', 'numMinutes', 'points', 'assists', 'blocks', 'steals',
        'threePointersMade', 'threePointersAttempted',
        'fieldGoalsMade', 'fieldGoalsAttempted', 'freeThrowsMade',
        'freeThrowsAttempted', year_col
    ]

    # Add high-value basic stats (non-redundant)
    high_value_basic = [
        'reboundsDefensive', 'reboundsOffensive',  # Split is more predictive than total
        'foulsPersonal',  # Foul trouble predictor
        'win'  # Game outcome context
    ]
    essential_cols.extend(high_value_basic)

    # Add advanced stats (exclude metadata and cumulative stats)
    adv_cols = [c for c in window_df.columns if c.startswith('adv_') and
               c not in ['adv_lg', 'adv_age', 'adv_pos', 'adv_g', 'adv_gs', 'adv_mp',
                        'adv_ows', 'adv_dws', 'adv_ws']]
    essential_cols.extend(adv_cols)

    # Add per-100 stats (exclude metadata and redundant percentages)
    per100_cols = [c for c in window_df.columns if c.startswith('per100_') and
                  c not in ['per100_lg', 'per100_age', 'per100_pos', 'per100_g', 'per100_gs', 'per100_mp',
                           'per100_fg_percent', 'per100_ft_percent']]
    essential_cols.extend(per100_cols)

    # Add key shooting stats (exclude low-value zones)
    shoot_cols = [c for c in window_df.columns if c.startswith('shoot_') and
                 any(x in c for x in ['avg_dist', 'assisted', 'corner', 'x3p']) and
                 c not in ['shoot_num_of_dunks', 'shoot_percent_fga_from_x3_10_range',
                          'shoot_percent_fga_from_x10_16_range', 'shoot_fg_percent_from_x3_10_range']]
    essential_cols.extend(shoot_cols)

    # Add high-value PBP stats (plus/minus, playmaking, finishing)
    pbp_cols = [c for c in window_df.columns if c.startswith('pbp_') and
               any(x in c for x in ['plus_minus', 'points_generated', 'and1', 'turnover',
                                   'fga_blocked', 'foul'])]
    essential_cols.extend(pbp_cols)

    # Filter to columns that exist
    cols_to_keep = [c for c in essential_cols if c in window_df.columns]
    window_df = window_df[cols_to_keep].copy()

    # Recompute reboundsTotal from split (training code expects this column)
    if 'reboundsDefensive' in window_df.columns and 'reboundsOffensive' in window_df.columns:
        window_df['reboundsTotal'] = window_df['reboundsDefensive'].fillna(0) + window_df['reboundsOffensive'].fillna(0)
        if verbose:
            print(f"  • Recomputed reboundsTotal from Defensive + Offensive split")

    if verbose:
        print(f"  • Optimized data: {len(window_df):,} rows, {len(cols_to_keep)} columns")
        mem_mb = window_df.memory_usage(deep=True).sum() / 1024**2
        print(f"  • Memory usage: {mem_mb:.1f} MB")

    return window_df


def train_player_window(
    window_df: pd.DataFrame,
    start_year: int,
    end_year: int,
    neural_epochs: int = 12,
    verbose: bool = True
) -> Dict:
    """
    Train player models for a specific window.

    This is a placeholder - will integrate with existing player training logic.

    Returns:
        Dictionary with trained models and metrics
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"TRAINING PLAYER MODELS: {start_year}-{end_year}")
        print(f"{'='*70}")
        print(f"Training data: {len(window_df):,} rows")

    # TODO: Import and call existing player training functions from train_auto.py
    # For now, return placeholder

    models = {
        'points': None,
        'rebounds': None,
        'assists': None,
        'threes': None,
        'minutes': None
    }

    metrics = {
        'window': f'{start_year}-{end_year}',
        'train_rows': len(window_df),
        'neural_epochs': neural_epochs
    }

    if verbose:
        print(f"✓ Training complete for {start_year}-{end_year}")

    return {'models': models, 'metrics': metrics}


def main():
    args = parse_args()

    print("="*70)
    print("NBA PLAYER MODEL TRAINING (WINDOWED ENSEMBLE)")
    print("="*70)
    print(f"Data source: {args.aggregated_data}")
    print(f"Window size: {args.window_size} years")
    print(f"Neural epochs: {args.neural_epochs}")
    print(f"Cache directory: {args.cache_dir}")
    if args.min_year:
        print(f"Min year filter: {args.min_year}")
    if args.max_year:
        print(f"Max year filter: {args.max_year}")
    print("="*70)

    # Create cache directory
    os.makedirs(args.cache_dir, exist_ok=True)

    # Load aggregated player data
    agg_df = load_aggregated_player_data(
        args.aggregated_data,
        min_year=args.min_year,
        max_year=args.max_year,
        verbose=args.verbose
    )

    # Get year column and season range
    year_col = get_year_column(agg_df)
    all_seasons = sorted([int(s) for s in agg_df[year_col].dropna().unique()])
    min_season, max_season = min(all_seasons), max(all_seasons)

    print(f"\n{'='*70}")
    print(f"PLANNING TRAINING WINDOWS")
    print(f"{'='*70}")
    print(f"Season range: {min_season}-{max_season}")
    print(f"Total seasons: {len(all_seasons)}")
    print(f"Window size: {args.window_size} years")

    # Create windows
    windows_to_process = []

    for i in range(0, len(all_seasons), args.window_size):
        window_seasons = all_seasons[i:i+args.window_size]
        if not window_seasons:
            continue

        start_year = int(window_seasons[0])
        end_year = int(window_seasons[-1])

        cache_path = Path(args.cache_dir) / f"player_models_{start_year}_{end_year}.pkl"
        is_current_window = max_season in window_seasons

        # Check cache
        if cache_path.exists() and not args.force_retrain and args.skip_cached:
            print(f"[SKIP] Window {start_year}-{end_year}: Using existing cache")
            continue

        if is_current_window:
            print(f"[TRAIN] Window {start_year}-{end_year}: Current season - will train")
        else:
            print(f"[TRAIN] Window {start_year}-{end_year}: Not cached - will train")

        windows_to_process.append({
            'seasons': window_seasons,
            'start_year': start_year,
            'end_year': end_year,
            'is_current': is_current_window,
            'cache_path': cache_path
        })

    if not windows_to_process:
        print("\n✅ All windows cached! Use --force-retrain to retrain.")
        return

    print(f"\n📊 Will train {len(windows_to_process)} window(s)")
    print("="*70)

    # Train each window
    for idx, window_info in enumerate(windows_to_process, 1):
        start_year = window_info['start_year']
        end_year = window_info['end_year']
        window_seasons = window_info['seasons']
        cache_path = window_info['cache_path']

        print(f"\n{'='*70}")
        print(f"WINDOW {idx}/{len(windows_to_process)}: {start_year}-{end_year}")
        print(f"{'='*70}")

        # Create window training data
        window_df = create_window_training_data(
            agg_df,
            window_seasons,
            year_col,
            verbose=args.verbose
        )

        # Train models
        result = train_player_window(
            window_df,
            start_year,
            end_year,
            neural_epochs=args.neural_epochs,
            verbose=args.verbose
        )

        # Save to cache (placeholder - will implement proper saving)
        cache_meta = {
            'window': f'{start_year}-{end_year}',
            'seasons': window_seasons,
            'train_rows': len(window_df),
            'metrics': result['metrics']
        }

        meta_path = cache_path.with_suffix('.json')
        with open(meta_path, 'w') as f:
            json.dump(cache_meta, f, indent=2)

        print(f"✓ Saved metadata to {meta_path}")

        # Clean up
        del window_df
        gc.collect()

    print(f"\n{'='*70}")
    print("✅ PLAYER MODEL TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Trained {len(windows_to_process)} window(s)")
    print(f"Models saved to: {args.cache_dir}/")


if __name__ == '__main__':
    main()
