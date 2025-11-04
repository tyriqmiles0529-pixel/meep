"""
Test training with meta-learned context features.

Trains ONLY the current season window (2022-2026) with expanded
meta-learner that learns context weights from data.
"""

import gc
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple
from collections import defaultdict

# Import meta-context ensemble
from player_ensemble_meta_context import (
    RidgePlayerStatModel,
    PlayerEloRating,
    TeamContextFeatures,
    PlayerStatEnsembleMetaContext
)


def load_player_data_for_window(player_stats_path: Path,
                                window_seasons: set) -> pd.DataFrame:
    """Load and filter player stats for a specific window."""
    print(f"  Loading player data for seasons {min(window_seasons)}-{max(window_seasons)}...")

    df = pd.read_csv(player_stats_path, low_memory=False)

    if 'gameId' in df.columns:
        df['gameId_str'] = df['gameId'].astype(str)
        df['season_prefix'] = df['gameId_str'].str[:3].astype(int)
        df['season_end_year'] = 2000 + (df['season_prefix'] % 100)

        df_window = df[df['season_end_year'].isin(window_seasons)].copy()

        print(f"  Found {len(df_window):,} player-game records")
        return df_window
    else:
        print("  ERROR: No gameId column found")
        return pd.DataFrame()


def build_meta_context_training_data(player_stats: pd.DataFrame,
                                     stat_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build training data with expanded features (15 total).

    Returns:
    - X_meta: (n_samples, 15) - 5 base predictions + 10 context features
    - y: (n_samples,) - actual stat values
    """
    print(f"\n  Building meta-context training data for {stat_name.upper()}...")

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
        return np.array([]), np.array([])

    df = player_stats.copy()
    df = df.sort_values(['personId', 'gameDate'])

    # Initialize components
    ridge_model = RidgePlayerStatModel(alpha=1.0)
    player_elo = PlayerEloRating(stat_name=stat_name)
    team_context = TeamContextFeatures()

    # Storage
    expanded_features_list = []
    actuals_list = []

    # Process each player
    for player_id, player_df in df.groupby('personId'):
        player_df = player_df.sort_values('gameDate').reset_index(drop=True)

        if len(player_df) < 5:
            continue

        for idx in range(len(player_df)):
            game_row = player_df.iloc[idx]
            actual_stat = game_row[stat_col]

            if pd.isna(actual_stat):
                continue

            # Historical data (before this game)
            hist_df = player_df.iloc[:idx]

            if len(hist_df) < 3:
                continue

            # Baseline
            baseline = hist_df[stat_col].mean()
            recent_stats = hist_df[stat_col].tail(10).values

            # Get player team and opponent
            player_team = game_row.get('teamTricode', 'UNK')
            opponent_team = game_row.get('opponentTeamTricode', 'UNK')

            # Base predictions
            ridge_pred = baseline if len(recent_stats) == 0 else np.mean(recent_stats)
            lgbm_pred = baseline  # Placeholder
            elo_pred = player_elo.get_prediction(str(player_id), baseline)
            rolling_avg = np.mean(recent_stats) if len(recent_stats) > 0 else baseline

            # Team context features (10 features)
            if player_team != 'UNK' and opponent_team != 'UNK':
                context_features = team_context.get_raw_features(player_team, opponent_team)
            else:
                context_features = np.zeros(10)

            # Combine: 5 base + 10 context = 15 total
            expanded_features = np.concatenate([
                np.array([ridge_pred, lgbm_pred, elo_pred, rolling_avg, baseline]),
                context_features
            ])

            expanded_features_list.append(expanded_features)
            actuals_list.append(actual_stat)

            # Update Elo
            player_elo.update(str(player_id), actual_stat, baseline)

            # Update team stats (simplified - would extract from game data)
            team_context.update_team_stats(player_team, {
                'pace': 100.0,  # Placeholder
                'ortg': 110.0,
                'drtg': 110.0,
                'ast_rate': 0.6,
                '3pa_rate': 0.35,
                'usage_gini': 0.3,
                'transition_pct': 0.15
            })

    if len(expanded_features_list) == 0:
        print(f"  WARNING: No training data generated")
        return np.array([]), np.array([])

    X_meta = np.array(expanded_features_list)
    y = np.array(actuals_list)

    print(f"  Generated {len(X_meta):,} training samples with 15 features each")

    return X_meta, y


def train_meta_context_ensemble(window_seasons: set,
                                player_stats_path: Path,
                                cache_dir: Path) -> Dict:
    """Train ensemble with meta-learned context for current season."""

    print(f"\n{'='*70}")
    print(f"Training Meta-Context Ensemble: {min(window_seasons)}-{max(window_seasons)}")
    print(f"{'='*70}")

    # Load data
    player_stats = load_player_data_for_window(player_stats_path, window_seasons)

    if player_stats.empty:
        print("  WARNING: No player data")
        return {}

    # Train ensemble for each stat
    ensembles = {}

    for stat_name in ['points', 'rebounds', 'assists', 'threes', 'minutes']:
        print(f"\n  Training {stat_name.upper()} meta-context ensemble...")

        # Create ensemble
        ensemble = PlayerStatEnsembleMetaContext(stat_name=stat_name)

        # Build training data with 15 features
        X_meta, y = build_meta_context_training_data(player_stats, stat_name)

        if len(X_meta) == 0:
            print(f"    WARNING: No training data for {stat_name}")
            continue

        # Fit meta-learner (will print learned weights)
        ensemble.fit_meta_learner(X_meta, y)

        # Calculate metrics
        if ensemble.is_fitted:
            # Clean data for prediction
            X_clean = X_meta.copy()
            for col_idx in range(X_clean.shape[1]):
                col = X_clean[:, col_idx]
                nan_mask = np.isnan(col)
                if np.any(nan_mask):
                    col_mean = np.nanmean(col)
                    if np.isnan(col_mean):
                        col_mean = 0.0
                    X_clean[nan_mask, col_idx] = col_mean

            valid_rows = ~np.isnan(X_clean).any(axis=1)
            X_clean = X_clean[valid_rows]
            y_clean = y[valid_rows]

            if len(X_clean) > 0:
                y_pred = ensemble.meta_learner.predict(ensemble.scaler.transform(X_clean))
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

    # Save
    cache_path = cache_dir / f"player_ensemble_meta_context_{min(window_seasons)}_{max(window_seasons)}.pkl"
    with open(cache_path, 'wb') as f:
        pickle.dump(ensembles, f)

    # Save metadata
    meta_path = cache_dir / f"player_ensemble_meta_context_{min(window_seasons)}_{max(window_seasons)}_meta.json"
    meta = {
        'seasons': list(map(int, window_seasons)),
        'start_year': int(min(window_seasons)),
        'end_year': int(max(window_seasons)),
        'trained_date': datetime.now().isoformat(),
        'method': 'meta_learned_context',
        'n_features': 15,
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

    print(f"\n[OK] Meta-context ensemble saved to {cache_path}")

    return ensembles


def main():
    print("="*70)
    print("META-LEARNED CONTEXT ENSEMBLE TEST")
    print("="*70)

    # Setup paths
    kaggle_cache = Path.home() / ".cache" / "kagglehub" / "datasets" / \
                  "eoinamoore" / "historical-nba-data-and-player-box-scores" / \
                  "versions" / "258"
    player_stats_path = kaggle_cache / "PlayerStatistics.csv"
    cache_dir = Path("model_cache")
    cache_dir.mkdir(exist_ok=True)

    if not player_stats_path.exists():
        print(f"ERROR: Player stats not found at {player_stats_path}")
        return

    # Test on current season window
    window_seasons = {2022, 2023, 2024, 2025, 2026}

    ensembles = train_meta_context_ensemble(
        window_seasons,
        player_stats_path,
        cache_dir
    )

    print("\n" + "="*70)
    print("META-CONTEXT TRAINING COMPLETE")
    print("="*70)

    for stat_name, info in ensembles.items():
        print(f"\n{stat_name.upper()}:")
        print(f"  RMSE: {info['rmse']:.3f}")
        print(f"  MAE: {info['mae']:.3f}")
        print(f"  Samples: {info['n_samples']:,}")

    print("\n💡 Next steps:")
    print("1. Compare meta-learned weights to baseline (ENSEMBLE_SUMMARY.md)")
    print("2. Check if RMSE improved beyond +0.5% to +2.4%")
    print("3. Analyze which context features got highest weights")


if __name__ == "__main__":
    main()
