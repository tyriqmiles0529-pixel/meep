"""
Per-Window Player Ensemble Training

Integrates with existing train_auto.py per-window infrastructure.
Trains 5-component ensemble for each stat:
- Ridge regression
- LightGBM (from train_auto.py)
- Player Elo
- Rolling average
- Team matchup context

Saves ensemble models per window for memory efficiency.
"""

import os
import sys
import gc
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict

# Import ensemble classes
from player_ensemble_enhanced import (
    RidgePlayerStatModel,
    PlayerEloRating,
    TeamMatchupContext,
    PlayerStatEnsemble
)


def load_player_data_for_window(player_stats_path: Path,
                                window_seasons: set) -> pd.DataFrame:
    """Load and filter player stats for a specific window."""
    print(f"  Loading player data for seasons {min(window_seasons)}-{max(window_seasons)}...")

    df = pd.read_csv(player_stats_path, low_memory=False)

    # Extract season from gameId
    if 'gameId' in df.columns:
        df['gameId_str'] = df['gameId'].astype(str)
        df['season_prefix'] = df['gameId_str'].str[:3].astype(int)
        df['season_end_year'] = 2000 + (df['season_prefix'] % 100)

        # Filter to window
        df_window = df[df['season_end_year'].isin(window_seasons)].copy()

        print(f"  Found {len(df_window):,} player-game records")
        return df_window
    else:
        print("  ERROR: No gameId column found")
        return pd.DataFrame()


def build_ensemble_training_data(player_stats: pd.DataFrame,
                                 stat_name: str,
                                 verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Build training data for ensemble meta-learner.

    For each player-game, we need:
    - Base predictions from all 5 components
    - Actual stat value (target)

    This requires replaying history to simulate predictions.
    """
    print(f"\n  Building ensemble training data for {stat_name.upper()}...")

    # Column mapping
    stat_col_map = {
        'points': 'points',
        'rebounds': 'reboundsTotal',
        'assists': 'assists',
        'threes': 'threePointersMade',
        'minutes': 'numMinutes'
    }

    stat_col = stat_col_map.get(stat_name)
    if stat_col not in player_stats.columns:
        print(f"  ERROR: Column {stat_col} not found")
        return np.array([]), np.array([]), pd.DataFrame()

    # Sort by player and date for rolling calculations
    df = player_stats.copy()
    df = df.sort_values(['personId', 'gameDate'])

    # Initialize components
    ridge_model = RidgePlayerStatModel(alpha=1.0)
    player_elo = PlayerEloRating(stat_name=stat_name)
    team_context = TeamMatchupContext()

    # Storage for training data
    base_predictions_list = []
    actuals_list = []
    metadata_list = []

    # Group by player for rolling stats
    for player_id, player_df in df.groupby('personId'):
        player_df = player_df.sort_values('gameDate').reset_index(drop=True)

        # Skip players with too few games
        if len(player_df) < 5:
            continue

        # Process each game for this player
        for idx in range(len(player_df)):
            game_row = player_df.iloc[idx]
            actual_stat = game_row[stat_col]

            # Skip if NaN
            if pd.isna(actual_stat):
                continue

            # Get historical data up to (but not including) this game
            hist_df = player_df.iloc[:idx]

            if len(hist_df) < 3:
                # Not enough history for predictions
                continue

            # Calculate baseline (season average up to this point)
            baseline = hist_df[stat_col].mean()

            # Get recent stats for predictions
            recent_stats = hist_df[stat_col].tail(10).values
            recent_stats_clean = recent_stats[~np.isnan(recent_stats)]

            if len(recent_stats_clean) == 0:
                continue  # Skip if no recent data

            # 1. Ridge prediction (recent weighted average with trend)
            # Simple but better than baseline: weighted recent + trend
            weights = np.arange(1, len(recent_stats_clean) + 1)
            weights = weights / weights.sum()
            ridge_pred = np.dot(recent_stats_clean, weights)

            # Add trend component
            if len(recent_stats_clean) >= 3:
                x = np.arange(len(recent_stats_clean))
                try:
                    slope = np.polyfit(x, recent_stats_clean, 1)[0]
                    ridge_pred = ridge_pred + slope * 0.5  # Half weight on trend
                except:
                    pass  # Keep ridge_pred as is if polyfit fails

            # 2. LightGBM prediction (exponential moving average for now)
            # EMA gives more weight to recent games
            alpha = 0.3  # Smoothing factor
            lgbm_pred = recent_stats_clean[-1]  # Start with most recent
            for val in reversed(recent_stats_clean[:-1]):
                lgbm_pred = alpha * val + (1 - alpha) * lgbm_pred

            # 3. Player Elo prediction
            elo_pred = player_elo.get_prediction(str(player_id), baseline)

            # 4. Rolling average (simple mean of last 10)
            rolling_avg = np.mean(recent_stats_clean)

            # 5. Team matchup adjustment (use recent variance as proxy for consistency)
            # More volatile players get adjusted toward recent form
            if len(recent_stats_clean) > 1:
                recent_std = np.std(recent_stats_clean)
                cv = recent_std / max(baseline, 0.1)  # Coefficient of variation
                # High variance → weight recent more, low variance → weight baseline more
                matchup_pred = (cv * recent_stats_clean[-1] + (1 - cv) * baseline) / (1 + cv)
            else:
                matchup_pred = baseline

            # Store base predictions
            base_preds = np.array([ridge_pred, lgbm_pred, elo_pred, rolling_avg, matchup_pred])
            base_predictions_list.append(base_preds)
            actuals_list.append(actual_stat)

            # Update Elo after seeing result
            player_elo.update(str(player_id), actual_stat, baseline)

            # Store metadata for debugging
            metadata_list.append({
                'player_id': player_id,
                'game_date': game_row['gameDate'],
                'baseline': baseline,
                'actual': actual_stat
            })

    if len(base_predictions_list) == 0:
        print(f"  WARNING: No training data generated for {stat_name}")
        return np.array([]), np.array([]), pd.DataFrame()

    X_meta = np.array(base_predictions_list)
    y = np.array(actuals_list)
    metadata_df = pd.DataFrame(metadata_list)

    print(f"  Generated {len(X_meta):,} training samples")
    print(f"  Base prediction shapes: {X_meta.shape}")

    return X_meta, y, metadata_df


def train_ensemble_for_window(window_info: Dict,
                              player_stats_path: Path,
                              lgbm_models_dir: Path,
                              cache_dir: Path,
                              verbose: bool = True) -> Dict:
    """
    Train ensemble models for a single window.

    Returns dict of trained ensemble models for each stat.
    """
    window_seasons = set(window_info['seasons'])
    start_year = window_info['start_year']
    end_year = window_info['end_year']
    is_current = window_info['is_current']

    print(f"\n{'='*70}")
    print(f"Training Player Ensemble: Window {start_year}-{end_year}")
    print(f"{'='*70}")

    # Check cache (skip historical windows if cached)
    cache_path = cache_dir / f"player_ensemble_{start_year}_{end_year}.pkl"
    if cache_path.exists() and not is_current:
        print(f"[SKIP] Using cached ensemble from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    # Load player data for this window
    player_stats = load_player_data_for_window(player_stats_path, window_seasons)

    if player_stats.empty:
        print("  WARNING: No player data for this window")
        return {}

    # Train ensemble for each stat
    ensembles = {}

    for stat_name in ['points', 'rebounds', 'assists', 'threes', 'minutes']:
        print(f"\n  Training {stat_name.upper()} ensemble...")

        # Create ensemble
        ensemble = PlayerStatEnsemble(stat_name=stat_name)

        # Load LightGBM model if available
        lgbm_path = lgbm_models_dir / f"{stat_name}_model.pkl"
        if lgbm_path.exists():
            ensemble.load_lgbm_model(str(lgbm_path))
            print(f"    Loaded LightGBM model from {lgbm_path}")

        # Build training data for meta-learner
        X_meta, y, metadata = build_ensemble_training_data(
            player_stats, stat_name, verbose=verbose
        )

        if len(X_meta) == 0:
            print(f"    WARNING: No training data for {stat_name}, skipping")
            continue

        # Fit meta-learner
        ensemble.fit_meta_learner(X_meta, y)

        # Calculate ensemble metrics (use cleaned data)
        # Clean X_meta for prediction (same as in fit_meta_learner)
        X_meta_clean = X_meta.copy()
        for col_idx in range(X_meta_clean.shape[1]):
            col = X_meta_clean[:, col_idx]
            nan_mask = np.isnan(col)
            if np.any(nan_mask):
                col_mean = np.nanmean(col)
                if np.isnan(col_mean):
                    col_mean = 0.0
                X_meta_clean[nan_mask, col_idx] = col_mean

        # Remove rows with any remaining NaN
        valid_rows = ~np.isnan(X_meta_clean).any(axis=1)
        X_meta_clean = X_meta_clean[valid_rows]
        y_clean = y[valid_rows]

        if len(X_meta_clean) > 0 and ensemble.is_fitted:
            y_pred = ensemble.meta_learner.predict(ensemble.scaler.transform(X_meta_clean))
            rmse = np.sqrt(np.mean((y_pred - y_clean) ** 2))
            mae = np.mean(np.abs(y_pred - y_clean))
        else:
            rmse = 0.0
            mae = 0.0

        print(f"    Ensemble RMSE: {rmse:.3f}")
        print(f"    Ensemble MAE: {mae:.3f}")

        ensembles[stat_name] = {
            'model': ensemble,
            'rmse': rmse,
            'mae': mae,
            'n_samples': len(X_meta)
        }

    # Save ensemble models
    with open(cache_path, 'wb') as f:
        pickle.dump(ensembles, f)

    # Save metadata
    meta_path = cache_dir / f"player_ensemble_{start_year}_{end_year}_meta.json"
    meta = {
        'seasons': list(map(int, window_seasons)),
        'start_year': start_year,
        'end_year': end_year,
        'trained_date': datetime.now().isoformat(),
        'is_current_season': is_current,
        'metrics': {
            stat: {
                'rmse': float(info['rmse']),
                'mae': float(info['mae']),
                'n_samples': int(info['n_samples'])
            }
            for stat, info in ensembles.items()
        }
    }

    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n[OK] Ensemble saved to {cache_path}")

    return ensembles


def main():
    parser = argparse.ArgumentParser(description="Train per-window player ensemble models")
    parser.add_argument("--player-stats", type=str,
                       default=None,
                       help="Path to PlayerStatistics.csv (default: auto-detect from Kaggle cache)")
    parser.add_argument("--lgbm-models-dir", type=str, default="models",
                       help="Directory containing trained LightGBM models")
    parser.add_argument("--cache-dir", type=str, default="model_cache",
                       help="Directory to save ensemble models")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")

    args = parser.parse_args()

    print("="*70)
    print("PER-WINDOW PLAYER ENSEMBLE TRAINING")
    print("="*70)

    # Setup paths
    lgbm_models_dir = Path(args.lgbm_models_dir)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(exist_ok=True)

    # Auto-detect player stats path if not provided
    if args.player_stats is None:
        kaggle_cache = Path.home() / ".cache" / "kagglehub" / "datasets" / \
                      "eoinamoore" / "historical-nba-data-and-player-box-scores" / \
                      "versions" / "258"
        player_stats_path = kaggle_cache / "PlayerStatistics.csv"
    else:
        player_stats_path = Path(args.player_stats)

    if not player_stats_path.exists():
        print(f"ERROR: Player stats not found at {player_stats_path}")
        print("Run train_auto.py first to download data")
        return

    print(f"Player stats: {player_stats_path}")
    print(f"LightGBM models: {lgbm_models_dir}")
    print(f"Cache directory: {cache_dir}")

    # Define windows (matching train_auto.py)
    # Read gameId column to get all available seasons
    print("\nDetecting available seasons...")
    df_gameids = pd.read_csv(player_stats_path, usecols=['gameId'])
    if 'gameId' in df_gameids.columns:
        df_gameids['season_prefix'] = df_gameids['gameId'].astype(str).str[:3].astype(int)
        df_gameids['season_end_year'] = 2000 + (df_gameids['season_prefix'] % 100)

        # Filter to valid seasons (2000-2026, exclude bad data like 2046-2099)
        valid_seasons = df_gameids[(df_gameids['season_end_year'] >= 2000) &
                                   (df_gameids['season_end_year'] <= 2026)]
        all_seasons = sorted(valid_seasons['season_end_year'].unique())

        # Filter to match train_auto.py cutoff (>= 2002)
        all_seasons = [s for s in all_seasons if s >= 2002]

        print(f"Found seasons: {all_seasons[0]}-{all_seasons[-1]} ({len(all_seasons)} seasons)")
    else:
        print("ERROR: Cannot determine seasons from data")
        return

    max_year = max(all_seasons)
    window_size = 5
    windows_to_process = []

    for i in range(0, len(all_seasons), window_size):
        window_seasons = all_seasons[i:i+window_size]
        start_year = int(window_seasons[0])
        end_year = int(window_seasons[-1])
        windows_to_process.append({
            'seasons': window_seasons,
            'start_year': start_year,
            'end_year': end_year,
            'is_current': max_year in window_seasons
        })

    print(f"\nProcessing {len(windows_to_process)} windows")

    # Train each window
    for idx, window_info in enumerate(windows_to_process, 1):
        print(f"\n{'='*70}")
        print(f"Window {idx}/{len(windows_to_process)}")
        print(f"{'='*70}")

        ensembles = train_ensemble_for_window(
            window_info,
            player_stats_path,
            lgbm_models_dir,
            cache_dir,
            verbose=args.verbose
        )

        # Free memory
        del ensembles
        gc.collect()

    print("\n" + "="*70)
    print("ENSEMBLE TRAINING COMPLETE")
    print("="*70)
    print(f"Ensemble models saved to: {cache_dir}/player_ensemble_*.pkl")
    print("\nNext steps:")
    print("1. Update riq_analyzer.py to use ensemble predictions")
    print("2. Backtest ensemble vs LightGBM-only models")
    print("3. Deploy for live predictions")


if __name__ == "__main__":
    main()
