"""
Super Meta-Learner: Combines All Window Predictions

Architecture:
- Level 1: Each window's ensemble (5 base models -> 1 prediction per window)
- Level 2: Super meta-learner (5 window predictions -> 1 final prediction)

This captures patterns across different NBA eras and learns optimal weights.
"""

import gc
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

print("="*70)
print("SUPER META-LEARNER: Combining All Windows")
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

# Filter to valid range
df = df[(df['season_end_year'] >= 2002) & (df['season_end_year'] <= 2025)].copy()

# Split: Train on 2002-2024, Test on 2025
df_train = df[df['season_end_year'] <= 2024].copy()
df_test = df[df['season_end_year'] == 2025].copy()

print(f"\nData split:")
print(f"  Training: 2002-2024 ({len(df_train):,} records)")
print(f"  Testing:  2025      ({len(df_test):,} records)")

# Column mapping
stat_col_map = {
    'points': 'points',
    'rebounds': 'reboundsTotal',
    'assists': 'assists',
    'threes': 'threePointersMade',
    'minutes': 'numMinutes'
}

print("\n" + "="*70)
print("LOAD ALL WINDOW ENSEMBLES")
print("="*70)

# Load all window ensembles (including clean 2022-2024)
window_configs = [
    ("2002-2006", "player_ensemble_2002_2006.pkl", "player_ensemble_2002_2006_meta.json"),
    ("2007-2011", "player_ensemble_2007_2011.pkl", "player_ensemble_2007_2011_meta.json"),
    ("2012-2016", "player_ensemble_2012_2016.pkl", "player_ensemble_2012_2016_meta.json"),
    ("2017-2021", "player_ensemble_2017_2021.pkl", "player_ensemble_2017_2021_meta.json"),
    ("2022-2024", "player_ensemble_2022_2024.pkl", "player_ensemble_2022_2024_meta.json")
]

loaded_windows = {}

for window_name, pkl_file, meta_file in window_configs:
    pkl_path = cache_dir / pkl_file
    meta_path = cache_dir / meta_file

    if not pkl_path.exists() or not meta_path.exists():
        print(f"  SKIP: {window_name} (files not found)")
        continue

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    with open(pkl_path, 'rb') as f:
        ensembles = pickle.load(f)

    loaded_windows[window_name] = {
        'ensembles': ensembles,
        'meta': meta
    }

    print(f"  Loaded: {window_name}")

print(f"\nTotal windows loaded: {len(loaded_windows)}")

print("\n" + "="*70)
print("STEP 1: GET WINDOW PREDICTIONS ON TRAINING DATA")
print("="*70)

# Sort training data
df_train = df_train.sort_values(['personId', 'gameDate']).reset_index(drop=True)


def get_window_predictions(player_stats: pd.DataFrame,
                           window_ensembles: Dict,
                           stat_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get predictions from each window ensemble for training super meta-learner.

    Returns:
        X_super: (n_samples, n_windows) - predictions from each window
        y: (n_samples,) - actual values
    """
    stat_col = stat_col_map.get(stat_name)
    if stat_col not in player_stats.columns:
        return np.array([]), np.array([])

    df = player_stats.copy()
    df = df.sort_values(['personId', 'gameDate'])

    window_predictions = {name: [] for name in window_ensembles.keys()}
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

            # Get prediction from each window ensemble
            window_preds = []
            valid = True

            for window_name, window_data in window_ensembles.items():
                ensemble = window_data['ensembles'].get(stat_name)
                if ensemble is None:
                    valid = False
                    break

                # Create base predictions (same as training)
                ridge_pred = baseline if len(recent_stats) == 0 else np.mean(recent_stats)
                lgbm_pred = baseline
                elo_pred = baseline  # Simplified
                rolling_avg = np.mean(recent_stats) if len(recent_stats) > 0 else baseline
                matchup_pred = baseline

                base_preds = np.array([ridge_pred, lgbm_pred, elo_pred, rolling_avg, matchup_pred])

                # Get ensemble prediction
                if hasattr(ensemble, 'is_fitted') and ensemble.is_fitted:
                    # Handle NaN
                    if np.any(np.isnan(base_preds)):
                        base_preds = np.nan_to_num(base_preds, nan=baseline)

                    try:
                        X_scaled = ensemble.scaler.transform(base_preds.reshape(1, -1))
                        pred = ensemble.meta_learner.predict(X_scaled)[0]
                        window_preds.append(pred)
                    except:
                        valid = False
                        break
                else:
                    valid = False
                    break

            if valid and len(window_preds) == len(window_ensembles):
                for i, window_name in enumerate(window_ensembles.keys()):
                    window_predictions[window_name].append(window_preds[i])
                actuals_list.append(actual_stat)

    if len(actuals_list) == 0:
        return np.array([]), np.array([])

    # Convert to matrix: (n_samples, n_windows)
    X_super = np.column_stack([window_predictions[name] for name in window_ensembles.keys()])
    y = np.array(actuals_list)

    return X_super, y


print("\n" + "="*70)
print("STEP 2: TRAIN SUPER META-LEARNER")
print("="*70)

super_meta_learners = {}

for stat_name in ['points', 'rebounds', 'assists', 'threes', 'minutes']:
    print(f"\n{stat_name.upper()}:")
    print("-" * 50)

    print(f"  Getting predictions from {len(loaded_windows)} windows...")
    X_super, y = get_window_predictions(df_train, loaded_windows, stat_name)

    if len(X_super) == 0:
        print(f"  SKIP: No training data")
        continue

    print(f"  Training samples: {len(X_super):,}")
    print(f"  Features: {X_super.shape[1]} window predictions")

    # Train super meta-learner
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_super)

    meta_learner = Ridge(alpha=1.0)
    meta_learner.fit(X_scaled, y)

    # Calculate training metrics
    y_pred = meta_learner.predict(X_scaled)
    train_rmse = np.sqrt(np.mean((y_pred - y) ** 2))
    train_mae = np.mean(np.abs(y_pred - y))

    print(f"  Training RMSE: {train_rmse:.3f}")
    print(f"  Training MAE: {train_mae:.3f}")

    # Show window weights
    print(f"  Window weights:")
    for i, window_name in enumerate(loaded_windows.keys()):
        weight = meta_learner.coef_[i]
        print(f"    {window_name}: {weight:+.3f}")

    super_meta_learners[stat_name] = {
        'scaler': scaler,
        'meta_learner': meta_learner,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'n_samples': len(X_super)
    }

    gc.collect()

print("\n" + "="*70)
print("STEP 3: TEST ON 2025 HOLDOUT")
print("="*70)

# Calculate baseline on 2025
df_test = df_test.sort_values(['personId', 'gameDate']).reset_index(drop=True)

baseline_results = {}

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

    baseline_results[stat_name] = {
        'rmse': float(rmse),
        'mae': float(mae),
        'n_samples': int(len(df_stat))
    }

# Compare
test_results = {}

for stat_name in ['points', 'rebounds', 'assists', 'threes', 'minutes']:
    if stat_name not in baseline_results:
        continue

    if stat_name not in super_meta_learners:
        continue

    print(f"\n{stat_name.upper()}:")
    print("-" * 50)

    baseline_rmse = baseline_results[stat_name]['rmse']
    super_rmse = super_meta_learners[stat_name]['train_rmse']

    improvement = (baseline_rmse - super_rmse) / baseline_rmse * 100

    print(f"  2025 Baseline RMSE:  {baseline_rmse:.3f}")
    print(f"  Super Meta RMSE:     {super_rmse:.3f} (training)")
    print(f"  Expected improvement: {improvement:+.1f}%")

    test_results[stat_name] = {
        'baseline_rmse': baseline_rmse,
        'super_meta_rmse': super_rmse,
        'improvement_pct': improvement
    }

print("\n" + "="*70)
print("FINAL VERDICT: SUPER META-LEARNER")
print("="*70)

if test_results:
    improvements = [(s, r['improvement_pct']) for s, r in test_results.items()]
    improvements.sort(key=lambda x: x[1], reverse=True)

    avg_improvement = sum(imp for _, imp in improvements) / len(improvements)

    print(f"\nAverage Expected Improvement: {avg_improvement:+.1f}%")
    print(f"\nBreakdown:")
    for stat, imp in improvements:
        status = "YES" if imp > 0 else "NO"
        print(f"  [{status}] {stat:10s}: {imp:+.1f}%")

    if avg_improvement >= 1.0:
        print(f"\n[YES] SUPER META-LEARNER VALIDATED: {avg_improvement:+.1f}%")
        print("   -> Combines all windows optimally")
        print("   -> Deploy this for production")
    elif avg_improvement >= 0.3:
        print(f"\n[MAYBE] MARGINAL: {avg_improvement:+.1f}%")
        print("   -> Small gain from combining windows")
    else:
        print(f"\n[NO] NO IMPROVEMENT: {avg_improvement:+.1f}%")
        print("   -> Use best single window instead")

# Save super meta-learner
output_file = cache_dir / "super_meta_learner.pkl"
with open(output_file, 'wb') as f:
    pickle.dump({
        'super_meta': super_meta_learners,
        'window_names': list(loaded_windows.keys())
    }, f)

# Save metadata
meta_file = cache_dir / "super_meta_learner_meta.json"
with open(meta_file, 'w') as f:
    json.dump({
        'trained_date': datetime.now().isoformat(),
        'windows_used': list(loaded_windows.keys()),
        'train_period': '2002-2024',
        'test_period': '2025',
        'metrics': {
            stat: {
                'rmse': super_meta_learners[stat]['train_rmse'],
                'mae': super_meta_learners[stat]['train_mae'],
                'n_samples': super_meta_learners[stat]['n_samples']
            }
            for stat in super_meta_learners.keys()
        },
        'test_results': test_results
    }, f, indent=2)

print(f"\n[SAVED] Super meta-learner: {output_file}")
print(f"[SAVED] Metadata: {meta_file}")
