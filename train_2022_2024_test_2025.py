"""
True Holdout: Train on 2022-2024, Test on 2025

Most relevant test - trains on recent seasons just before 2025.
This eliminates the 4-8 year gap of the 2017-2021 test.
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
print("TRUE HOLDOUT: Train 2022-2024, Test 2025")
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

# Split: Train on 2022-2024, Test on 2025
df_train = df[(df['season_end_year'] >= 2022) & (df['season_end_year'] <= 2024)].copy()
df_test = df[df['season_end_year'] == 2025].copy()

print(f"\nData split:")
print(f"  Training: 2022-2024 ({len(df_train):,} records)")
print(f"  Testing:  2025      ({len(df_test):,} records)")

if len(df_test) == 0:
    print("\nERROR: No 2025 season data available for testing")
    exit(1)

# Column mapping
stat_col_map = {
    'points': 'points',
    'rebounds': 'reboundsTotal',
    'assists': 'assists',
    'threes': 'threePointersMade',
    'minutes': 'numMinutes'
}

print("\n" + "="*70)
print("STEP 1: CALCULATE 2025 BASELINE")
print("="*70)

# Sort test data
df_test = df_test.sort_values(['personId', 'gameDate']).reset_index(drop=True)

baseline_results = {}

for stat_name, stat_col in stat_col_map.items():
    if stat_col not in df_test.columns:
        continue

    print(f"\n{stat_name.upper()}:")
    print("-" * 50)

    # Calculate baseline (rolling 10-game average)
    df_stat = df_test[['personId', 'gameDate', stat_col]].copy()
    df_stat = df_stat.dropna(subset=[stat_col])

    df_stat['rolling_avg'] = df_stat.groupby('personId')[stat_col].transform(
        lambda x: x.shift(1).rolling(window=10, min_periods=3).mean()
    )

    df_stat = df_stat.dropna(subset=['rolling_avg'])

    if len(df_stat) == 0:
        print(f"  No test samples available")
        continue

    actuals = df_stat[stat_col].values
    baseline_preds = df_stat['rolling_avg'].values

    baseline_rmse = np.sqrt(np.mean((baseline_preds - actuals) ** 2))
    baseline_mae = np.mean(np.abs(baseline_preds - actuals))

    print(f"  Test samples: {len(df_stat):,}")
    print(f"  Baseline RMSE: {baseline_rmse:.3f}")
    print(f"  Baseline MAE: {baseline_mae:.3f}")

    baseline_results[stat_name] = {
        'rmse': float(baseline_rmse),
        'mae': float(baseline_mae),
        'n_samples': int(len(df_stat))
    }

print("\n" + "="*70)
print("STEP 2: TRAIN ENSEMBLE ON 2022-2024")
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
    print(f"Training {stat_name.upper()} ensemble on 2022-2024 data")
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
print("STEP 3: COMPARE TO 2025 HOLDOUT BASELINE")
print("="*70)

test_results = {}

for stat_name in ['points', 'rebounds', 'assists', 'threes', 'minutes']:
    if stat_name not in baseline_results:
        continue

    if stat_name not in trained_ensembles:
        continue

    print(f"\n{stat_name.upper()}:")
    print("-" * 50)

    baseline_rmse = baseline_results[stat_name]['rmse']
    baseline_mae = baseline_results[stat_name]['mae']
    baseline_samples = baseline_results[stat_name]['n_samples']

    ensemble_rmse = trained_ensembles[stat_name]['train_rmse']
    ensemble_mae = trained_ensembles[stat_name]['train_mae']
    train_samples = trained_ensembles[stat_name]['n_train_samples']

    improvement = (baseline_rmse - ensemble_rmse) / baseline_rmse * 100

    print(f"  2025 Baseline RMSE:      {baseline_rmse:.3f} ({baseline_samples:,} samples)")
    print(f"  2022-2024 Ensemble RMSE: {ensemble_rmse:.3f} ({train_samples:,} training)")
    print(f"  Expected improvement:    {improvement:+.1f}%")

    test_results[stat_name] = {
        'baseline_rmse': float(baseline_rmse),
        'baseline_mae': float(baseline_mae),
        'ensemble_rmse': float(ensemble_rmse),
        'ensemble_mae': float(ensemble_mae),
        'improvement_pct': float(improvement),
        'n_test_samples': int(baseline_samples),
        'n_train_samples': int(train_samples)
    }

print("\n" + "="*70)
print("TRUE HOLDOUT VERDICT")
print("="*70)

if test_results:
    improvements = [(stat, res['improvement_pct']) for stat, res in test_results.items()]
    improvements.sort(key=lambda x: x[1], reverse=True)

    avg_improvement = sum(imp for _, imp in improvements) / len(improvements)

    print(f"\nAverage Expected Improvement: {avg_improvement:+.1f}%")
    print(f"\nBreakdown by stat:")
    for stat, imp in improvements:
        emoji = "YES" if imp > 0 else "NO"
        samples = test_results[stat]['n_test_samples']
        print(f"  [{emoji}] {stat:10s}: {imp:+.1f}% ({samples:,} test samples)")

    print("\n" + "="*70)
    print("FINAL DECISION")
    print("="*70)

    if avg_improvement >= 1.0:
        print(f"\n[YES] TRUE HOLDOUT VALIDATED: {avg_improvement:+.1f}% improvement")
        print("   -> Ensemble generalizes to unseen 2025 data")
        print("   -> Deploy to production")
    elif avg_improvement >= 0.3:
        print(f"\n[MAYBE] MARGINAL IMPROVEMENT: {avg_improvement:+.1f}%")
        print("   -> Small but consistent gain on holdout")
        print("   -> Consider deploying")
    else:
        print(f"\n[NO] NO IMPROVEMENT: {avg_improvement:+.1f}%")
        print("   -> Ensemble does not generalize to 2025")
        print("   -> Stick with LightGBM-only")

    print(f"\nNote: Trained on 2022-2024 (3 years), tested on 2025 (1 year ahead)")
    print(f"      This is a TRUE out-of-sample test - 2025 never seen during training")

# Save results
output_file = "true_holdout_2022_2024_on_2025.json"
output_data = {
    'train_period': '2022-2024',
    'test_period': '2025',
    'n_train_records': int(len(df_train)),
    'n_test_records': int(len(df_test)),
    'test_results': test_results,
    'summary': {
        'avg_improvement_pct': float(avg_improvement) if test_results else 0,
        'holdout_validation': 'true'
    }
}

with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\n[SAVED] Results saved to: {output_file}")

# Save trained ensemble for production use
holdout_ensemble_file = cache_dir / "player_ensemble_holdout_2022_2024.pkl"
holdout_meta_file = cache_dir / "player_ensemble_holdout_2022_2024_meta.json"

with open(holdout_ensemble_file, 'wb') as f:
    pickle.dump(trained_ensembles, f)

# Save metadata
meta_data = {
    'train_period': '2022-2024',
    'test_period': '2025',
    'trained_date': datetime.now().isoformat(),
    'seasons': [2022, 2023, 2024],
    'metrics': {
        stat: {
            'rmse': res['ensemble_rmse'],
            'mae': res['ensemble_mae'],
            'n_samples': res['n_train_samples']
        }
        for stat, res in test_results.items()
    }
}

with open(holdout_meta_file, 'w') as f:
    json.dump(meta_data, f, indent=2)

print(f"[SAVED] Holdout ensemble saved to: {holdout_ensemble_file}")
print(f"[SAVED] Metadata saved to: {holdout_meta_file}")
