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
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from shared.data_loading import load_player_data, get_year_column, get_season_range


def parse_args():
    parser = argparse.ArgumentParser(description='Train NBA Player Prediction Models')

    # Data sources
    parser.add_argument('--data', '--aggregated-data', type=str, required=True, dest='data',
                        help='Path to data file (.parquet or .csv - auto-detected)')

    # Training parameters
    parser.add_argument('--window-size', type=int, default=3,
                        help='Size of training windows in years (default: 3)')
    parser.add_argument('--neural-epochs', type=int, default=12,
                        help='TabNet training epochs (default: 12)')
    parser.add_argument('--shared-epochs', type=int, default=6,
                        help='Epochs for shared stats (points, assists, rebounds) (default: 6)')
    parser.add_argument('--independent-epochs', type=int, default=8,
                        help='Epochs for independent props (minutes, threes) (default: 8)')
    parser.add_argument('--patience', type=int, default=3,
                        help='Early stopping patience (default: 3)')
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

    # Parallel training parameters
    parser.add_argument('--parallel-windows', type=int, default=4,
                        help='Number of windows to train in parallel (default: 4)')
    parser.add_argument('--no-parallel', action='store_true',
                        help='Disable parallel training (use sequential)')

    # Output
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Print detailed progress')

    return parser.parse_args()


def train_single_window_worker(args_tuple):
    """Worker function for parallel window training"""
    (data_path, window_info, neural_epochs, shared_epochs, 
     independent_epochs, patience, verbose, worker_id) = args_tuple
    
    # Set environment variables for this worker to limit CPU usage
    os.environ['OMP_NUM_THREADS'] = '2'  # Limit LightGBM threads
    os.environ['OPENBLAS_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    
    # Suppress warnings in worker processes too
    import warnings
    warnings.filterwarnings('ignore')
    
    try:
        # Load data in worker (isolated memory)
        agg_df = load_player_data(data_path)
        year_col = get_year_column(agg_df)
        
        start_year = window_info['start_year']
        end_year = window_info['end_year']
        window_seasons = window_info['seasons']
        
        if verbose:
            print(f"[Worker {worker_id}] Starting window {start_year}-{end_year}")
        
        # Create window training data
        window_df = create_window_training_data(
            agg_df,
            window_seasons,
            year_col,
            verbose=False  # Reduce verbosity in parallel mode
        )
        
        # Train models
        result = train_player_window(
            window_df,
            start_year,
            end_year,
            neural_epochs=neural_epochs,
            shared_epochs=shared_epochs,
            independent_epochs=independent_epochs,
            patience=patience,
            verbose=False
        )
        
        # Save models (not just metadata)
        cache_path = window_info['cache_path']
        import joblib
        joblib.dump(result['models'], cache_path)
        
        # Clean up worker memory
        del agg_df, window_df, result
        gc.collect()
        
        return {
            'success': True,
            'window_info': window_info,
            'worker_id': worker_id
        }
        
    except Exception as e:
        import traceback
        return {
            'success': False,
            'window_info': window_info,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'worker_id': worker_id
        }


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

    # Add rolling features (L5, L10, L20 averages + trends)
    from rolling_features import add_rolling_features

    if verbose:
        print(f"\n  Adding rolling features...")

    window_df = add_rolling_features(
        window_df,
        windows=[5, 10, 20],  # L5, L10, L20 rolling averages
        add_variance=True,    # Add std deviation for consistency
        add_trend=True,       # Add momentum indicators
        low_memory=False,     # Use full feature set
        verbose=verbose
    )

    if verbose:
        print(f"  • Final dataset: {len(window_df):,} rows, {len(window_df.columns)} columns")
        mem_mb = window_df.memory_usage(deep=True).sum() / 1024**2
        print(f"  • Memory usage: {mem_mb:.1f} MB")

    return window_df


def train_player_window(
    window_df: pd.DataFrame,
    start_year: int,
    end_year: int,
    neural_epochs: int = 12,
    shared_epochs: int = 6,
    independent_epochs: int = 8,
    patience: int = 3,
    verbose: bool = True,
    use_multi_task: bool = True,
    use_gpu: bool = False  # Add this parameter
) -> Dict:
    """
    Train player models for a specific window.

    Uses hybrid multi-task by default (3x faster):
    - Multi-task: Points, Assists, Rebounds (shared TabNet)
    - Single-task: Minutes, Threes (separate models)

    Returns:
        Dictionary with trained models and metrics
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"TRAINING PLAYER MODELS: {start_year}-{end_year}")
        if use_multi_task:
            print(f"MODE: Hybrid Multi-Task (3x faster)")
        print(f"{'='*70}")
        print(f"Training data: {len(window_df):,} rows")

    if use_multi_task:
        # HYBRID MULTI-TASK MODE (3x faster, better accuracy)
        from hybrid_multi_task import HybridMultiTaskPlayer

        # Prepare data
        feature_cols = [c for c in window_df.columns if c not in [
            'points', 'reboundsTotal', 'assists', 'threePointersMade', 'numMinutes',
            'personId', 'gameId', 'gameDate', 'firstName', 'lastName'
        ]]

        X = window_df[feature_cols].fillna(0)

        # Prepare targets
        y_dict = {
            'points': window_df['points'].fillna(0).values,
            'rebounds': window_df['reboundsTotal'].fillna(0).values,
            'assists': window_df['assists'].fillna(0).values,
            'threes': window_df['threePointersMade'].fillna(0).values,
            'minutes': window_df['numMinutes'].fillna(0).values
        }

        # Train/val split (80/20)
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        X_val = X.iloc[split_idx:]

        y_train_dict = {k: v[:split_idx] for k, v in y_dict.items()}
        y_val_dict = {k: v[split_idx:] for k, v in y_dict.items()}

        # Train hybrid multi-task model
        model = HybridMultiTaskPlayer(use_gpu=use_gpu)
        model.fit(
            X_train, y_train_dict,
            X_val, y_val_dict,
            correlated_epochs=shared_epochs,  # Use shared epochs for points/assists/rebounds
            independent_epochs=independent_epochs,  # Use independent epochs for minutes/threes
            patience=patience,  # Use configurable patience
            batch_size=8192
        )

        # Get predictions and metrics
        prop_metrics = {}
        for prop in ['points', 'rebounds', 'assists', 'threes', 'minutes']:
            y_pred = model.predict(X_val)[prop]
            mae = np.mean(np.abs(y_pred - y_val_dict[prop]))
            prop_metrics[prop] = {'mae': float(mae)}

            if verbose:
                print(f"  ✓ {prop}: MAE = {mae:.2f}")

        # Store model (all props in one object)
        models = {
            'multi_task_model': model,
            'points': model,  # For backward compatibility
            'rebounds': model,
            'assists': model,
            'threes': model,
            'minutes': model
        }

        metrics = {
            'window': f'{start_year}-{end_year}',
            'train_rows': len(window_df),
            'neural_epochs': neural_epochs,
            'mode': 'hybrid_multi_task',
            'prop_metrics': prop_metrics
        }

    else:
        # SINGLE-TASK MODE (fallback)
        from train_auto import train_player_model_enhanced

        props = {
            'points': 'points',
            'rebounds': 'reboundsTotal',
            'assists': 'assists',
            'threes': 'threePointersMade',
            'minutes': 'numMinutes'
        }

        models = {}
        prop_metrics = {}

        for prop_key, prop_col in props.items():
            if verbose:
                print(f"\n  Training {prop_key} model...")

            try:
                model, metrics = train_player_model_enhanced(
                    df=window_df,
                    prop_name=prop_col,
                    verbose=verbose,
                    neural_epochs=neural_epochs
                )
                models[prop_key] = model
                prop_metrics[prop_key] = metrics

                if verbose:
                    print(f"  ✓ {prop_key} model trained")

            except Exception as e:
                if verbose:
                    print(f"  ✗ {prop_key} model failed: {e}")
                models[prop_key] = None
                prop_metrics[prop_key] = {'error': str(e)}

        metrics = {
            'window': f'{start_year}-{end_year}',
            'train_rows': len(window_df),
            'neural_epochs': neural_epochs,
            'mode': 'single_task',
            'prop_metrics': prop_metrics
        }

    if verbose:
        print(f"\n✓ Training complete for {start_year}-{end_year}")

    return {'models': models, 'metrics': metrics}


def main():
    args = parse_args()

    print("="*70)
    print("NBA PLAYER MODEL TRAINING (WINDOWED ENSEMBLE)")
    print("="*70)
    print(f"Data source: {args.data}")
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

    # Load player data (auto-detects Parquet or CSV)
    agg_df = load_player_data(
        args.data,
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

    # Create rolling windows (overlapping 3-year periods)
    windows_to_process = []

    for i in range(0, len(all_seasons) - args.window_size + 1, 1):
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
    
    # Choose training mode
    if args.no_parallel or len(windows_to_process) == 1:
        print("🔄 Using sequential training")
        train_sequential(args, windows_to_process, agg_df, year_col)
    else:
        print(f"🚀 Using parallel training with {args.parallel_windows} workers")
        train_parallel(args, windows_to_process)


def train_sequential(args, windows_to_process, agg_df, year_col):
    """Sequential training (original implementation)"""
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
            shared_epochs=args.shared_epochs,
            independent_epochs=args.independent_epochs,
            patience=args.patience,
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


def train_parallel(args, windows_to_process):
    """Parallel training implementation"""
    completed_windows = 0
    failed_windows = []
    
    # Prepare arguments for workers
    worker_args = []
    for idx, window_info in enumerate(windows_to_process):
        worker_args.append((
            args.data,
            window_info,
            args.neural_epochs,
            args.shared_epochs,
            args.independent_epochs,
            args.patience,
            args.verbose,
            idx + 1
        ))
    
    # Use ProcessPoolExecutor for parallel training
    with ProcessPoolExecutor(max_workers=args.parallel_windows) as executor:
        # Submit all jobs
        future_to_window = {
            executor.submit(train_single_window_worker, args): window_info 
            for args, window_info in zip(worker_args, windows_to_process)
        }
        
        # Process completed jobs
        for future in as_completed(future_to_window):
            window_info = future_to_window[future]
            completed_windows += 1
            
            try:
                result = future.result()
                
                if result['success']:
                    start_year = result['window_info']['start_year']
                    end_year = result['window_info']['end_year']
                    worker_id = result['worker_id']
                    
                    print(f"✅ [{completed_windows}/{len(windows_to_process)}] "
                          f"Worker {worker_id}: Window {start_year}-{end_year} completed")
                    
                    # Save metadata
                    cache_meta = {
                        'window': f'{start_year}-{end_year}',
                        'seasons': result['window_info']['seasons'],
                        'metrics': result['result']['metrics']
                    }
                    
                    cache_path = result['window_info']['cache_path']
                    meta_path = cache_path.with_suffix('.json')
                    with open(meta_path, 'w') as f:
                        json.dump(cache_meta, f, indent=2)
                    
                    # Save actual models (placeholder - implement proper saving)
                    # models_path = cache_path.with_suffix('.pkl')
                    # joblib.dump(result['result']['models'], models_path)
                    
                else:
                    failed_windows.append(result['window_info'])
                    print(f"❌ Window failed: {result['error']}")
                    
            except Exception as e:
                failed_windows.append(window_info)
                print(f"❌ Worker exception: {e}")
    
    # Summary
    print(f"\n{'='*70}")
    print("✅ PARALLEL TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Completed: {completed_windows - len(failed_windows)}/{len(windows_to_process)} windows")
    if failed_windows:
        print(f"Failed: {len(failed_windows)} windows")
        for window in failed_windows:
            print(f"  - {window['start_year']}-{window['end_year']}")
    print(f"Models saved to: {args.cache_dir}/")


if __name__ == '__main__':
    main()
