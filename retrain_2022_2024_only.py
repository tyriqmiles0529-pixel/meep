"""
Retrain 2022-2024 Window ONLY

Just trains on 2022-2024 data (no 2025-2026).
Saves as player_ensemble_2022_2024.pkl (clean, no leakage).

Use backtest_all_windows.py to test all windows afterwards.
"""

import gc
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Tuple

# Import ensemble classes
from player_ensemble_enhanced import (
    RidgePlayerStatModel,
    PlayerEloRating,
    TeamMatchupContext,
    PlayerStatEnsemble
)

print("="*70)
print("RETRAIN 2022-2024 WINDOW (No Leakage)")
print("="*70)

# Paths
kaggle_cache = Path.home() / ".cache" / "kagglehub" / "datasets" / \
              "eoinamoore" / "historical-nba-data-and-player-box-scores" / \
              "versions" / "258"
player_stats_path = kaggle_cache / "PlayerStatistics.csv"
cache_dir = Path("model_cache")
cache_dir.mkdir(exist_ok=True)

if not player_stats_path.exists():
    print(f"ERROR: Player stats not found at {player_stats_path}")
    exit(1)

print(f"\nLoading data...")
df = pd.read_csv(player_stats_path, low_memory=False)

# Extract season
df['gameId_str'] = df['gameId'].astype(str)
df['season_prefix'] = df['gameId_str'].str[:3].astype(int)
df['season_end_year'] = 2000 + (df['season_prefix'] % 100)

# Train on 2022-2024 ONLY
df_train = df[(df['season_end_year'] >= 2022) & (df['season_end_year'] <= 2024)].copy()

print(f"\nTraining data:")
print(f"  Seasons: 2022-2024")
print(f"  Records: {len(df_train):,}")
print(f"  Players: {df_train['personId'].nunique():,}")

# Column mapping
stat_col_map = {
    'points': 'points',
    'rebounds': 'reboundsTotal',
    'assists': 'assists',
    'threes': 'threePointersMade',
    'minutes': 'numMinutes'
}


def build_ensemble_training_data(player_stats: pd.DataFrame,
                                 stat_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Build training data for ensemble"""
    print(f"\n  Building training data for {stat_name.upper()}...")

    stat_col = stat_col_map.get(stat_name)
    if stat_col not in player_stats.columns:
        print(f"  ERROR: Column {stat_col} not found")
        return np.array([]), np.array([])

    df = player_stats.copy()
    df = df.sort_values(['personId', 'gameDate'])

    # Initialize components
    ridge_model = RidgePlayerStatModel(alpha=1.0)
    player_elo = PlayerEloRating(stat_name=stat_name)
    team_context = TeamMatchupContext()

    base_predictions_list = []
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

            hist_df = player_df.iloc[:idx]

            if len(hist_df) < 3:
                continue

            baseline = hist_df[stat_col].mean()
            recent_stats = hist_df[stat_col].tail(10).values

            # Base predictions
            ridge_pred = baseline if len(recent_stats) == 0 else np.mean(recent_stats)
            lgbm_pred = baseline
            elo_pred = player_elo.get_prediction(str(player_id), baseline)
            rolling_avg = np.mean(recent_stats) if len(recent_stats) > 0 else baseline
            matchup_pred = baseline

            base_preds = np.array([ridge_pred, lgbm_pred, elo_pred, rolling_avg, matchup_pred])
            base_predictions_list.append(base_preds)
            actuals_list.append(actual_stat)

            # Update Elo
            player_elo.update(str(player_id), actual_stat, baseline)

    if len(base_predictions_list) == 0:
        print(f"  WARNING: No training data generated")
        return np.array([]), np.array([])

    X_meta = np.array(base_predictions_list)
    y = np.array(actuals_list)

    print(f"  Generated {len(X_meta):,} training samples")
    return X_meta, y


print("\n" + "="*70)
print("TRAINING ENSEMBLE")
print("="*70)

# Train ensemble for each stat
trained_ensembles = {}

for stat_name in ['points', 'rebounds', 'assists', 'threes', 'minutes']:
    print(f"\n{'='*70}")
    print(f"Training {stat_name.upper()}")
    print(f"{'='*70}")

    ensemble = PlayerStatEnsemble(stat_name=stat_name)

    # Build training data
    X_meta, y = build_ensemble_training_data(df_train, stat_name)

    if len(X_meta) == 0:
        print(f"  Skipping {stat_name} - no training data")
        continue

    # Fit meta-learner
    ensemble.fit_meta_learner(X_meta, y)

    # Calculate training metrics
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

    if len(X_clean) > 0 and ensemble.is_fitted:
        y_pred = ensemble.meta_learner.predict(ensemble.scaler.transform(X_clean))
        train_rmse = np.sqrt(np.mean((y_pred - y_clean) ** 2))
        train_mae = np.mean(np.abs(y_pred - y_clean))
    else:
        train_rmse = 0.0
        train_mae = 0.0

    print(f"  Training RMSE: {train_rmse:.3f}")
    print(f"  Training MAE: {train_mae:.3f}")

    trained_ensembles[stat_name] = {
        'model': ensemble,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'n_train_samples': len(X_meta)
    }

    # Clean up
    del X_meta, y
    gc.collect()

print("\n" + "="*70)
print("SAVING CLEAN ENSEMBLE")
print("="*70)

# Save retrained ensemble
output_file = cache_dir / "player_ensemble_2022_2024.pkl"
output_meta_file = cache_dir / "player_ensemble_2022_2024_meta.json"

with open(output_file, 'wb') as f:
    pickle.dump(trained_ensembles, f)

# Save metadata
meta_data = {
    'start_year': 2022,
    'end_year': 2024,
    'seasons': [2022, 2023, 2024],
    'trained_date': datetime.now().isoformat(),
    'metrics': {
        stat: {
            'rmse': trained_ensembles[stat]['train_rmse'],
            'mae': trained_ensembles[stat]['train_mae'],
            'n_samples': trained_ensembles[stat]['n_train_samples']
        }
        for stat in trained_ensembles.keys()
    }
}

with open(output_meta_file, 'w') as f:
    json.dump(meta_data, f, indent=2)

print(f"\n[SAVED] Clean ensemble: {output_file}")
print(f"[SAVED] Metadata: {output_meta_file}")
print(f"\nTraining complete!")
print(f"  Trained on: 2022-2024 (3 seasons)")
print(f"  NO data leakage - 2025 and 2026 excluded")
print(f"\nNext step: Run backtest_all_windows.py to test all 5 windows")
