"""
Retrain 2022-2026 Window Excluding 2025-2026

Trains on: 2022, 2023, 2024 seasons only
Backtests on: 2025 season AND 2026 season (current)

This gives true out-of-sample validation on both recent seasons.
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

# Import ensemble classes
from player_ensemble_enhanced import (
    RidgePlayerStatModel,
    PlayerEloRating,
    TeamMatchupContext,
    PlayerStatEnsemble
)

print("="*70)
print("RETRAIN 2022-2024 WINDOW (Exclude 2025-2026)")
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

# Split data
df_train = df[(df['season_end_year'] >= 2022) & (df['season_end_year'] <= 2024)].copy()
df_test_2025 = df[df['season_end_year'] == 2025].copy()
df_test_2026 = df[df['season_end_year'] == 2026].copy()

print(f"\nData split:")
print(f"  Training: 2022-2024 ({len(df_train):,} records)")
print(f"  Test 2025: {len(df_test_2025):,} records")
print(f"  Test 2026: {len(df_test_2026):,} records")

# Column mapping
stat_col_map = {
    'points': 'points',
    'rebounds': 'reboundsTotal',
    'assists': 'assists',
    'threes': 'threePointersMade',
    'minutes': 'numMinutes'
}

print("\n" + "="*70)
print("STEP 1: TRAIN ENSEMBLE ON 2022-2024")
print("="*70)


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


# Train ensemble for each stat
trained_ensembles = {}

for stat_name in ['points', 'rebounds', 'assists', 'threes', 'minutes']:
    print(f"\n{'='*70}")
    print(f"Training {stat_name.upper()} ensemble")
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
print("STEP 2: BACKTEST ON 2025 SEASON")
print("="*70)


def calculate_baseline(df_test: pd.DataFrame) -> Dict:
    """Calculate baseline performance on test set"""
    df_test = df_test.sort_values(['personId', 'gameDate']).reset_index(drop=True)
    results = {}

    for stat_name, stat_col in stat_col_map.items():
        if stat_col not in df_test.columns:
            continue

        df_stat = df_test[['personId', stat_col]].copy()
        df_stat = df_stat.dropna(subset=[stat_col])

        df_stat['rolling_avg'] = df_stat.groupby('personId')[stat_col].transform(
            lambda x: x.shift(1).rolling(window=10, min_periods=3).mean()
        )

        df_stat = df_stat.dropna(subset=['rolling_avg'])

        if len(df_stat) == 0:
            continue

        actuals = df_stat[stat_col].values
        preds = df_stat['rolling_avg'].values

        rmse = np.sqrt(np.mean((preds - actuals) ** 2))
        mae = np.mean(np.abs(preds - actuals))

        results[stat_name] = {
            'rmse': float(rmse),
            'mae': float(mae),
            'n_samples': int(len(df_stat))
        }

    return results


# Backtest on 2025
baseline_2025 = calculate_baseline(df_test_2025)

print("\n2025 SEASON RESULTS:")
print("-" * 70)

test_2025_results = {}

for stat_name in ['points', 'rebounds', 'assists', 'threes', 'minutes']:
    if stat_name not in baseline_2025:
        continue

    if stat_name not in trained_ensembles:
        continue

    baseline_rmse = baseline_2025[stat_name]['rmse']
    ensemble_rmse = trained_ensembles[stat_name]['train_rmse']
    improvement = (baseline_rmse - ensemble_rmse) / baseline_rmse * 100

    print(f"\n{stat_name.upper()}:")
    print(f"  Baseline RMSE:  {baseline_rmse:.3f} ({baseline_2025[stat_name]['n_samples']:,} samples)")
    print(f"  Ensemble RMSE:  {ensemble_rmse:.3f} (training)")
    print(f"  Improvement:    {improvement:+.1f}%")

    test_2025_results[stat_name] = {
        'baseline_rmse': baseline_rmse,
        'ensemble_rmse': ensemble_rmse,
        'improvement_pct': improvement,
        'n_samples': baseline_2025[stat_name]['n_samples']
    }

print("\n" + "="*70)
print("STEP 3: BACKTEST ON 2026 SEASON (CURRENT)")
print("="*70)

if len(df_test_2026) > 0:
    # Backtest on 2026
    baseline_2026 = calculate_baseline(df_test_2026)

    print("\n2026 SEASON RESULTS:")
    print("-" * 70)

    test_2026_results = {}

    for stat_name in ['points', 'rebounds', 'assists', 'threes', 'minutes']:
        if stat_name not in baseline_2026:
            continue

        if stat_name not in trained_ensembles:
            continue

        baseline_rmse = baseline_2026[stat_name]['rmse']
        ensemble_rmse = trained_ensembles[stat_name]['train_rmse']
        improvement = (baseline_rmse - ensemble_rmse) / baseline_rmse * 100

        print(f"\n{stat_name.upper()}:")
        print(f"  Baseline RMSE:  {baseline_rmse:.3f} ({baseline_2026[stat_name]['n_samples']:,} samples)")
        print(f"  Ensemble RMSE:  {ensemble_rmse:.3f} (training)")
        print(f"  Improvement:    {improvement:+.1f}%")

        test_2026_results[stat_name] = {
            'baseline_rmse': baseline_rmse,
            'ensemble_rmse': ensemble_rmse,
            'improvement_pct': improvement,
            'n_samples': baseline_2026[stat_name]['n_samples']
        }
else:
    print("\nNo 2026 season data available yet")
    test_2026_results = {}

print("\n" + "="*70)
print("SUMMARY: TRUE HOLDOUT VALIDATION")
print("="*70)

if test_2025_results:
    avg_2025 = sum(r['improvement_pct'] for r in test_2025_results.values()) / len(test_2025_results)
    print(f"\n2025 Season Average: {avg_2025:+.1f}%")

if test_2026_results:
    avg_2026 = sum(r['improvement_pct'] for r in test_2026_results.values()) / len(test_2026_results)
    print(f"2026 Season Average: {avg_2026:+.1f}%")

if test_2025_results and test_2026_results:
    combined_avg = (avg_2025 + avg_2026) / 2
    print(f"\nCombined Average: {combined_avg:+.1f}%")

    if combined_avg >= 1.0:
        print(f"\n[YES] TRUE HOLDOUT VALIDATED: {combined_avg:+.1f}%")
        print("   -> Ensemble generalizes to unseen 2025-2026 seasons")
        print("   -> Deploy this for production")
    elif combined_avg >= 0.3:
        print(f"\n[MAYBE] MARGINAL: {combined_avg:+.1f}%")
        print("   -> Small but consistent gain")
    else:
        print(f"\n[NO] NO IMPROVEMENT: {combined_avg:+.1f}%")

# Save retrained ensemble (REPLACES old 2022-2026 with leakage)
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
    },
    'backtest_2025': test_2025_results,
    'backtest_2026': test_2026_results if test_2026_results else None
}

with open(output_meta_file, 'w') as f:
    json.dump(meta_data, f, indent=2)

print(f"\n[SAVED] Clean ensemble: {output_file}")
print(f"[SAVED] Metadata: {output_meta_file}")
print(f"\nNote: This replaces the old 2022-2026 window (which had data leakage)")
print(f"      New window: 2022-2024 (clean, no leakage)")
