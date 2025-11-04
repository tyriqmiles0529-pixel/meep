"""
Quick Test: Does Fixed Ensemble Beat Baseline?

Tests the fixed ensemble training on a single window (2017-2021)
and compares to baseline on 2025 data.
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Import training functions
sys.path.insert(0, str(Path(__file__).parent))
from train_ensemble_players import (
    load_player_data_for_window,
    build_ensemble_training_data
)
from player_ensemble_enhanced import PlayerStatEnsemble

print("="*70)
print("TEST: Fixed Ensemble vs Baseline")
print("="*70)

# Paths
kaggle_cache = Path.home() / ".cache" / "kagglehub" / "datasets" / \
              "eoinamoore" / "historical-nba-data-and-player-box-scores" / \
              "versions" / "258"
player_stats_path = kaggle_cache / "PlayerStatistics.csv"
cache_dir = Path("model_cache")

if not player_stats_path.exists():
    print(f"ERROR: Player stats not found at {player_stats_path}")
    sys.exit(1)

print(f"\nLoading data...")
df = pd.read_csv(player_stats_path, low_memory=False)

# Extract season
df['gameId_str'] = df['gameId'].astype(str)
df['season_prefix'] = df['gameId_str'].str[:3].astype(int)
df['season_end_year'] = 2000 + (df['season_prefix'] % 100)

# Train on 2017-2021
print(f"\nTraining ensemble on 2017-2021...")
window_seasons = {2017, 2018, 2019, 2020, 2021}
df_train = df[df['season_end_year'].isin(window_seasons)].copy()
print(f"  Training samples: {len(df_train):,}")

# Test on 2025
df_test = df[df['season_end_year'] == 2025].copy()
df_test = df_test.sort_values(['personId', 'gameDate']).reset_index(drop=True)
print(f"  Test samples: {len(df_test):,}")

# Column mapping
stat_col_map = {
    'points': 'points',
    'rebounds': 'reboundsTotal',
    'assists': 'assists',
    'threes': 'threePointersMade',
    'minutes': 'numMinutes'
}

# Train ensemble for points only (quick test)
stat_name = 'points'
stat_col = stat_col_map[stat_name]

print(f"\n{'='*70}")
print(f"Training {stat_name.upper()} ensemble with fixed base predictions")
print(f"{'='*70}")

# Build training data with FIXED base predictions
X_meta, y_train, metadata = build_ensemble_training_data(
    df_train, stat_name, verbose=True
)

if len(X_meta) == 0:
    print(f"ERROR: No training data generated")
    sys.exit(1)

# Create and train ensemble
ensemble = PlayerStatEnsemble(stat_name=stat_name)
ensemble.fit_meta_learner(X_meta, y_train)

print(f"\n{'='*70}")
print(f"Backtest on 2025")
print(f"{'='*70}")

# Test ensemble on 2025
predictions_ensemble = []
predictions_baseline = []
actuals = []

sample_count = 0
max_samples = 2000

for player_id, player_df in df_test.groupby('personId'):
    if sample_count >= max_samples:
        break

    player_df = player_df.sort_values('gameDate').reset_index(drop=True)

    if len(player_df) < 5:
        continue

    for idx in range(3, len(player_df)):
        if sample_count >= max_samples:
            break

        game_row = player_df.iloc[idx]
        actual_stat = game_row[stat_col]

        if pd.isna(actual_stat):
            continue

        hist_df = player_df.iloc[:idx]
        recent_values = hist_df[stat_col].tail(10).values
        recent_values = recent_values[~np.isnan(recent_values)]

        if len(recent_values) < 3:
            continue

        # Baseline prediction (rolling average)
        baseline_pred = np.mean(recent_values)

        # Ensemble prediction (using same logic as training)
        baseline_val = hist_df[stat_col].mean()

        # 1. Ridge (weighted recent + trend)
        weights = np.arange(1, len(recent_values) + 1)
        weights = weights / weights.sum()
        ridge_pred = np.dot(recent_values, weights)
        if len(recent_values) >= 3:
            x = np.arange(len(recent_values))
            try:
                slope = np.polyfit(x, recent_values, 1)[0]
                ridge_pred = ridge_pred + slope * 0.5
            except:
                pass

        # 2. LightGBM (EMA)
        alpha = 0.3
        lgbm_pred = recent_values[-1]
        for val in reversed(recent_values[:-1]):
            lgbm_pred = alpha * val + (1 - alpha) * lgbm_pred

        # 3. Elo (returns baseline for new player)
        elo_pred = ensemble.player_elo.get_prediction(str(player_id), baseline_val)

        # 4. Rolling avg
        rolling_avg = np.mean(recent_values)

        # 5. Matchup
        if len(recent_values) > 1:
            recent_std = np.std(recent_values)
            cv = recent_std / max(baseline_val, 0.1)
            matchup_pred = (cv * recent_values[-1] + (1 - cv) * baseline_val) / (1 + cv)
        else:
            matchup_pred = baseline_val

        # Get ensemble prediction
        base_preds = np.array([ridge_pred, lgbm_pred, elo_pred, rolling_avg, matchup_pred])
        X_scaled = ensemble.scaler.transform(base_preds.reshape(1, -1))
        ensemble_pred = ensemble.meta_learner.predict(X_scaled)[0]

        # Update Elo
        ensemble.player_elo.update(str(player_id), actual_stat, baseline_val)

        predictions_ensemble.append(ensemble_pred)
        predictions_baseline.append(baseline_pred)
        actuals.append(actual_stat)
        sample_count += 1

if len(predictions_ensemble) == 0:
    print(f"ERROR: No predictions generated")
    sys.exit(1)

# Calculate metrics
predictions_ensemble = np.array(predictions_ensemble)
predictions_baseline = np.array(predictions_baseline)
actuals = np.array(actuals)

rmse_ensemble = np.sqrt(np.mean((predictions_ensemble - actuals) ** 2))
rmse_baseline = np.sqrt(np.mean((predictions_baseline - actuals) ** 2))

improvement = (rmse_baseline - rmse_ensemble) / rmse_baseline * 100

print(f"\nResults ({len(actuals):,} predictions):")
print(f"  Baseline RMSE:  {rmse_baseline:.3f}")
print(f"  Ensemble RMSE:  {rmse_ensemble:.3f}")
print(f"  Improvement:    {improvement:+.1f}%")

if improvement > 1.0:
    print(f"\n[YES] Fixed ensemble BEATS baseline by {improvement:.1f}%!")
    print(f"  The diverse base predictions provide real signal")
elif improvement > 0:
    print(f"\n[MARGINAL] Small improvement: {improvement:.1f}%")
    print(f"  Base predictions help slightly")
else:
    print(f"\n[NO] Ensemble still worse than baseline")
    print(f"  Even with diverse signals, ensemble adds noise")
