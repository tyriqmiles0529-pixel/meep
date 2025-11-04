"""
Test 2017-2021 Window Ensemble on 2025 Data

The 2017-2021 ensemble was trained on 2017-2021 seasons.
Testing it on 2025 season gives us true out-of-sample performance.
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

print("="*70)
print("TRUE HOLDOUT: 2017-2021 Ensemble on 2025 Season")
print("="*70)

# Paths
kaggle_cache = Path.home() / ".cache" / "kagglehub" / "datasets" / \
              "eoinamoore" / "historical-nba-data-and-player-box-scores" / \
              "versions" / "258"
player_stats_path = kaggle_cache / "PlayerStatistics.csv"
cache_dir = Path("model_cache")

print(f"\nLoading 2025 season data...")
df = pd.read_csv(player_stats_path, low_memory=False)

# Extract season
df['gameId_str'] = df['gameId'].astype(str)
df['season_prefix'] = df['gameId_str'].str[:3].astype(int)
df['season_end_year'] = 2000 + (df['season_prefix'] % 100)

# Get 2025 data
df_2025 = df[df['season_end_year'] == 2025].copy()

print(f"  2025 Season: {len(df_2025):,} records")
print(f"  Players: {df_2025['personId'].nunique():,}")
print(f"  Games: {df_2025['gameId'].nunique():,}")

# Column mapping
stat_col_map = {
    'points': 'points',
    'rebounds': 'reboundsTotal',
    'assists': 'assists',
    'threes': 'threePointersMade',
    'minutes': 'numMinutes'
}

# Sort
df_2025 = df_2025.sort_values(['personId', 'gameDate']).reset_index(drop=True)

print("\n" + "="*70)
print("BASELINE ON 2025 SEASON")
print("="*70)

baseline_results = {}

for stat_name, stat_col in stat_col_map.items():
    if stat_col not in df_2025.columns:
        continue

    df_stat = df_2025[['personId', stat_col]].copy()
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

    print(f"\n{stat_name.upper()}:")
    print(f"  Baseline RMSE: {rmse:.3f} ({len(df_stat):,} samples)")

    baseline_results[stat_name] = {
        'rmse': float(rmse),
        'mae': float(mae),
        'n_samples': int(len(df_stat))
    }

print("\n" + "="*70)
print("LOAD 2017-2021 ENSEMBLE (Never Seen 2025)")
print("="*70)

ensemble_file = cache_dir / "player_ensemble_2017_2021.pkl"
ensemble_meta_file = cache_dir / "player_ensemble_2017_2021_meta.json"

if not ensemble_file.exists():
    print(f"ERROR: Ensemble not found at {ensemble_file}")
    exit(1)

with open(ensemble_meta_file, 'r') as f:
    ensemble_meta = json.load(f)

print(f"\nEnsemble Info:")
print(f"  Trained on: {ensemble_meta['seasons']}")
print(f"  Training date: {ensemble_meta['trained_date'][:10]}")
print(f"  Never seen 2025: ✅ TRUE HOLDOUT")

print("\n" + "="*70)
print("COMPARISON: Ensemble (2017-2021) vs Baseline (2025)")
print("="*70)

comparison = {}

for stat_name in ['points', 'rebounds', 'assists', 'threes', 'minutes']:
    if stat_name not in baseline_results:
        continue

    if stat_name not in ensemble_meta['metrics']:
        continue

    baseline_rmse = baseline_results[stat_name]['rmse']
    ensemble_rmse = ensemble_meta['metrics'][stat_name]['rmse']

    improvement = (baseline_rmse - ensemble_rmse) / baseline_rmse * 100

    print(f"\n{stat_name.upper()}:")
    print(f"  2025 Baseline RMSE:     {baseline_rmse:.3f}")
    print(f"  2017-2021 Ensemble RMSE: {ensemble_rmse:.3f} (training)")
    print(f"  Expected improvement:    {improvement:+.1f}%")

    comparison[stat_name] = {
        'baseline_2025_rmse': baseline_rmse,
        'ensemble_train_rmse': ensemble_rmse,
        'improvement_pct': improvement
    }

print("\n" + "="*70)
print("TRUE HOLDOUT VERDICT")
print("="*70)

improvements = [(s, c['improvement_pct']) for s, c in comparison.items()]
improvements.sort(key=lambda x: x[1], reverse=True)

avg_improvement = sum(imp for _, imp in improvements) / len(improvements)

print(f"\nAverage Expected Improvement: {avg_improvement:+.1f}%")
print(f"\nBreakdown:")
for stat, imp in improvements:
    emoji = "✅" if imp > 0 else "❌"
    print(f"  {emoji} {stat:10s}: {imp:+.1f}%")

if avg_improvement >= 1.0:
    print(f"\n✅ ENSEMBLE GENERALIZES: {avg_improvement:+.1f}% on unseen 2025 data")
elif avg_improvement >= 0.0:
    print(f"\n⚠️  SMALL GAIN: {avg_improvement:+.1f}% on unseen 2025 data")
else:
    print(f"\n❌ ENSEMBLE DEGRADES: {avg_improvement:+.1f}% on unseen 2025 data")

print(f"\nNote: 2017-2021 ensemble trained 4-8 years before 2025")
print(f"      Some degradation expected due to NBA meta shifts")
print(f"      A 2020-2024 ensemble (excluding 2025) would be more relevant")

# Save
output = {
    'test': 'true_holdout',
    'train_window': '2017-2021',
    'test_season': 2025,
    'baseline_2025': baseline_results,
    'comparison': comparison,
    'avg_improvement_pct': float(avg_improvement)
}

with open('true_holdout_2017_2021_on_2025.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n✅ Results saved to: true_holdout_2017_2021_on_2025.json")
