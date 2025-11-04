"""
Comprehensive Backtest on Full Historical Data (2002-2026)

Validates ensemble performance across all eras:
- 2000s era (2002-2009)
- 2010s era (2010-2019)
- 2020s era (2020-2026)

Total: ~1.3M player-game records, 24 seasons
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict

print("="*70)
print("FULL HISTORICAL BACKTEST (2002-2026)")
print("="*70)

# Paths - try latest version first, fallback to older versions
kaggle_base = Path.home() / ".cache" / "kagglehub" / "datasets" / \
              "eoinamoore" / "historical-nba-data-and-player-box-scores" / "versions"

# Try versions in descending order
player_stats_path = None
for version in [261, 260, 258]:
    test_path = kaggle_base / str(version) / "PlayerStatistics.csv"
    if test_path.exists():
        player_stats_path = test_path
        print(f"Found data at version {version}")
        break

if player_stats_path is None:
    print(f"ERROR: Player stats not found in any version (tried 261, 260, 258)")
    print(f"Looked in: {kaggle_base}")
    sys.exit(1)
cache_dir = Path("model_cache")

print(f"\nLoading full historical data...")
print(f"Source: {player_stats_path}")

# Load all player stats
df = pd.read_csv(player_stats_path, low_memory=False)

# Extract season
df['gameId_str'] = df['gameId'].astype(str)
df['season_prefix'] = df['gameId_str'].str[:3].astype(int)
df['season_end_year'] = 2000 + (df['season_prefix'] % 100)

# Filter to valid seasons (2002-2026)
df = df[(df['season_end_year'] >= 2002) & (df['season_end_year'] <= 2026)].copy()

print(f"Total records: {len(df):,}")
print(f"Seasons: {df['season_end_year'].min()}-{df['season_end_year'].max()}")
print(f"Unique players: {df['personId'].nunique():,}")

# Column mapping
stat_col_map = {
    'points': 'points',
    'rebounds': 'reboundsTotal',
    'assists': 'assists',
    'threes': 'threePointersMade',
    'minutes': 'numMinutes'
}

# Sort by player and date
df = df.sort_values(['personId', 'gameDate']).reset_index(drop=True)

print("\n" + "="*70)
print("RUNNING BASELINE BACKTEST (Rolling 10-Game Average)")
print("="*70)

results = {}

for stat_name, stat_col in stat_col_map.items():
    if stat_col not in df.columns:
        print(f"\nSkipping {stat_name} - column {stat_col} not found")
        continue

    print(f"\n{stat_name.upper()}:")
    print("-" * 50)

    # Calculate rolling average per player
    df_stat = df[['personId', 'gameDate', stat_col, 'season_end_year']].copy()
    df_stat = df_stat.dropna(subset=[stat_col])

    # Rolling 10-game average (shift by 1 to avoid lookahead)
    df_stat['rolling_avg'] = df_stat.groupby('personId')[stat_col].transform(
        lambda x: x.shift(1).rolling(window=10, min_periods=3).mean()
    )

    # Remove rows without prediction
    df_stat = df_stat.dropna(subset=['rolling_avg'])

    if len(df_stat) == 0:
        print(f"  No valid samples for {stat_name}")
        continue

    # Calculate metrics
    actuals = df_stat[stat_col].values
    predictions = df_stat['rolling_avg'].values

    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    mae = np.mean(np.abs(predictions - actuals))
    bias = np.mean(predictions - actuals)
    r2 = 1 - (np.sum((actuals - predictions) ** 2) / np.sum((actuals - np.mean(actuals)) ** 2))

    print(f"  Samples: {len(df_stat):,}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  MAE: {mae:.3f}")
    print(f"  Bias: {bias:.3f}")
    print(f"  R²: {r2:.3f}")

    # Era breakdown
    print(f"\n  Era Breakdown:")
    for era_name, (start, end) in [
        ("2000s", (2002, 2009)),
        ("2010s", (2010, 2019)),
        ("2020s", (2020, 2026))
    ]:
        era_data = df_stat[(df_stat['season_end_year'] >= start) &
                          (df_stat['season_end_year'] <= end)]
        if len(era_data) > 0:
            era_actuals = era_data[stat_col].values
            era_preds = era_data['rolling_avg'].values
            era_rmse = np.sqrt(np.mean((era_preds - era_actuals) ** 2))
            print(f"    {era_name} ({start}-{end}): RMSE {era_rmse:.3f} ({len(era_data):,} samples)")

    results[stat_name] = {
        'rmse': float(rmse),
        'mae': float(mae),
        'bias': float(bias),
        'r2': float(r2),
        'n_samples': int(len(df_stat))
    }

print("\n" + "="*70)
print("LOADING ENSEMBLE MODELS")
print("="*70)

# Load all ensemble windows
ensemble_windows = []
meta_files = [
    "player_ensemble_2002_2006_meta.json",
    "player_ensemble_2007_2011_meta.json",
    "player_ensemble_2012_2016_meta.json",
    "player_ensemble_2017_2021_meta.json",
    "player_ensemble_2022_2026_meta.json"
]

for meta_filename in meta_files:
    window_file = cache_dir / meta_filename
    if not window_file.exists():
        continue

    with open(window_file, 'r') as f:
        meta = json.load(f)

    pkl_filename = meta_filename.replace('_meta.json', '.pkl')
    pkl_file = cache_dir / pkl_filename

    if pkl_file.exists():
        with open(pkl_file, 'rb') as f:
            ensembles = pickle.load(f)

        ensemble_windows.append({
            'start_year': meta['start_year'],
            'end_year': meta['end_year'],
            'seasons': set(meta['seasons']),
            'ensembles': ensembles,
            'meta': meta
        })
        print(f"  Loaded: {meta['start_year']}-{meta['end_year']} ({len(meta['seasons'])} seasons)")

print(f"\nTotal windows loaded: {len(ensemble_windows)}")

print("\n" + "="*70)
print("COMPARISON: ENSEMBLE vs BASELINE")
print("="*70)

comparison = {}

for stat_name in ['points', 'rebounds', 'assists', 'threes', 'minutes']:
    print(f"\n{stat_name.upper()}:")
    print("-" * 50)

    baseline_rmse = results.get(stat_name, {}).get('rmse', 0)

    # Get ensemble RMSEs from each window
    ensemble_rmses = []
    ensemble_samples = []

    for window in ensemble_windows:
        if stat_name in window['meta']['metrics']:
            window_rmse = window['meta']['metrics'][stat_name]['rmse']
            window_samples = window['meta']['metrics'][stat_name]['n_samples']
            ensemble_rmses.append(window_rmse)
            ensemble_samples.append(window_samples)

    if ensemble_rmses:
        # Weighted average RMSE across windows
        total_samples = sum(ensemble_samples)
        weighted_rmse = sum(r * s for r, s in zip(ensemble_rmses, ensemble_samples)) / total_samples

        improvement = (baseline_rmse - weighted_rmse) / baseline_rmse * 100 if baseline_rmse > 0 else 0

        print(f"  Baseline RMSE:  {baseline_rmse:.3f} ({results[stat_name]['n_samples']:,} samples)")
        print(f"  Ensemble RMSE:  {weighted_rmse:.3f} ({total_samples:,} samples)")
        print(f"  Improvement:    {improvement:+.1f}%")

        comparison[stat_name] = {
            'baseline_rmse': baseline_rmse,
            'ensemble_rmse': weighted_rmse,
            'improvement_pct': improvement,
            'baseline_samples': results[stat_name]['n_samples'],
            'ensemble_samples': total_samples
        }
    else:
        print(f"  No ensemble data available")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

improvements = [(stat, comp['improvement_pct'])
                for stat, comp in comparison.items()]
improvements.sort(key=lambda x: x[1], reverse=True)

avg_improvement = 0.0
if improvements:
    avg_improvement = sum(imp for _, imp in improvements) / len(improvements)
    print(f"\nAverage RMSE Improvement: {avg_improvement:+.1f}%")
    print(f"\nBreakdown by stat:")
    for stat, imp in improvements:
        emoji = "✅" if imp > 0 else "❌"
        print(f"  {emoji} {stat:10s}: {imp:+.1f}%")

print("\n" + "="*70)
print("DECISION")
print("="*70)

if avg_improvement >= 1.0:
    print(f"\n✅ ENSEMBLE VALIDATED: {avg_improvement:+.1f}% improvement across 24 seasons")
    print("   → Deploy to production")
elif avg_improvement >= 0.3:
    print(f"\n⚠️  MARGINAL IMPROVEMENT: {avg_improvement:+.1f}% across 24 seasons")
    print("   → Consider deploying (small but consistent gain)")
else:
    print(f"\n❌ NO SIGNIFICANT IMPROVEMENT: {avg_improvement:+.1f}%")
    print("   → Stick with LightGBM-only")

# Save results
output_file = "backtest_full_history_results.json"
output_data = {
    'baseline': results,
    'comparison': comparison,
    'ensemble_windows': [
        {
            'start_year': w['start_year'],
            'end_year': w['end_year'],
            'seasons': list(w['seasons']),
            'metrics': w['meta']['metrics']
        }
        for w in ensemble_windows
    ],
    'summary': {
        'avg_improvement_pct': avg_improvement if improvements else 0,
        'total_baseline_samples': sum(r['n_samples'] for r in results.values()),
        'total_ensemble_samples': sum(comp['ensemble_samples'] for comp in comparison.values()),
        'seasons_covered': '2002-2026',
        'n_seasons': 24
    }
}

with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\n✅ Results saved to: {output_file}")
