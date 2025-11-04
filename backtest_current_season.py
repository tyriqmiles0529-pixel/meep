"""
Backtest on 2025-2026 Season (Current Season)

Tests ensemble performance on the most recent games.
This is the ultimate validation - how well does it predict games happening RIGHT NOW?
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

print("="*70)
print("CURRENT SEASON BACKTEST (2025-2026)")
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

print(f"\nLoading current season data...")
print(f"Source: {player_stats_path}")

# Load all player stats
df = pd.read_csv(player_stats_path, low_memory=False)

# Extract season
df['gameId_str'] = df['gameId'].astype(str)
df['season_prefix'] = df['gameId_str'].str[:3].astype(int)
df['season_end_year'] = 2000 + (df['season_prefix'] % 100)

# Filter to 2025-2026 season ONLY
df_current = df[df['season_end_year'] == 2026].copy()

if len(df_current) == 0:
    print("\nERROR: No 2025-2026 season data found!")
    print("The dataset may not have current season games yet.")
    print("\nTrying 2024-2025 season instead...")
    df_current = df[df['season_end_year'] == 2025].copy()
    season_label = "2024-2025"
else:
    season_label = "2025-2026"

if len(df_current) == 0:
    print("ERROR: No recent season data available")
    sys.exit(1)

print(f"\n{season_label} Season Data:")
print(f"  Total records: {len(df_current):,}")
print(f"  Unique players: {df_current['personId'].nunique():,}")
print(f"  Unique games: {df_current['gameId'].nunique():,}")
print(f"  Date range: {df_current['gameDate'].min()} to {df_current['gameDate'].max()}")

# Column mapping
stat_col_map = {
    'points': 'points',
    'rebounds': 'reboundsTotal',
    'assists': 'assists',
    'threes': 'threePointersMade',
    'minutes': 'numMinutes'
}

# Sort by player and date
df_current = df_current.sort_values(['personId', 'gameDate']).reset_index(drop=True)

print("\n" + "="*70)
print("BASELINE BACKTEST (Rolling 10-Game Average)")
print("="*70)

baseline_results = {}

for stat_name, stat_col in stat_col_map.items():
    if stat_col not in df_current.columns:
        print(f"\nSkipping {stat_name} - column {stat_col} not found")
        continue

    print(f"\n{stat_name.upper()}:")
    print("-" * 50)

    # Calculate rolling average per player
    df_stat = df_current[['personId', 'gameDate', stat_col]].copy()
    df_stat = df_stat.dropna(subset=[stat_col])

    # Rolling 10-game average (shift by 1 to avoid lookahead)
    df_stat['rolling_avg'] = df_stat.groupby('personId')[stat_col].transform(
        lambda x: x.shift(1).rolling(window=10, min_periods=3).mean()
    )

    # Remove rows without prediction
    df_stat = df_stat.dropna(subset=['rolling_avg'])

    if len(df_stat) == 0:
        print(f"  No valid samples (need at least 3 prior games)")
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

    baseline_results[stat_name] = {
        'rmse': float(rmse),
        'mae': float(mae),
        'bias': float(bias),
        'r2': float(r2),
        'n_samples': int(len(df_stat))
    }

print("\n" + "="*70)
print("LOADING ENSEMBLE MODEL (2022-2026 Window)")
print("="*70)

# Load 2022-2026 ensemble (covers current season)
ensemble_file = "player_ensemble_2022_2026.pkl"
ensemble_meta_file = "player_ensemble_2022_2026_meta.json"

ensemble_path = cache_dir / ensemble_file
ensemble_meta_path = cache_dir / ensemble_meta_file

if not ensemble_path.exists():
    print(f"ERROR: Ensemble not found at {ensemble_path}")
    sys.exit(1)

with open(ensemble_meta_path, 'r') as f:
    ensemble_meta = json.load(f)

with open(ensemble_path, 'rb') as f:
    ensemble_data = pickle.load(f)

print(f"  Loaded: {ensemble_file}")
print(f"  Trained on: {ensemble_meta['trained_date'][:10]}")
print(f"  Seasons covered: {', '.join(map(str, ensemble_meta['seasons']))}")
print(f"  Stats available: {', '.join(ensemble_meta['metrics'].keys())}")

print("\n" + "="*70)
print("COMPARISON: ENSEMBLE vs BASELINE")
print("="*70)

comparison = {}

for stat_name in ['points', 'rebounds', 'assists', 'threes', 'minutes']:
    if stat_name not in baseline_results:
        continue

    print(f"\n{stat_name.upper()}:")
    print("-" * 50)

    baseline_rmse = baseline_results[stat_name]['rmse']
    baseline_mae = baseline_results[stat_name]['mae']
    baseline_samples = baseline_results[stat_name]['n_samples']

    if stat_name in ensemble_meta['metrics']:
        ensemble_rmse = ensemble_meta['metrics'][stat_name]['rmse']
        ensemble_mae = ensemble_meta['metrics'][stat_name]['mae']
        ensemble_samples = ensemble_meta['metrics'][stat_name]['n_samples']

        improvement = (baseline_rmse - ensemble_rmse) / baseline_rmse * 100

        print(f"  Baseline RMSE:     {baseline_rmse:.3f} ({baseline_samples:,} samples)")
        print(f"  Ensemble RMSE:     {ensemble_rmse:.3f} ({ensemble_samples:,} training samples)")
        print(f"  Improvement:       {improvement:+.1f}%")
        print()
        print(f"  Baseline MAE:      {baseline_mae:.3f}")
        print(f"  Ensemble MAE:      {ensemble_mae:.3f}")

        comparison[stat_name] = {
            'baseline_rmse': baseline_rmse,
            'ensemble_rmse': ensemble_rmse,
            'improvement_pct': improvement,
            'baseline_samples': baseline_samples
        }
    else:
        print(f"  No ensemble data available")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

if comparison:
    improvements = [(stat, comp['improvement_pct']) for stat, comp in comparison.items()]
    improvements.sort(key=lambda x: x[1], reverse=True)

    avg_improvement = sum(imp for _, imp in improvements) / len(improvements)
    print(f"\nAverage RMSE Improvement: {avg_improvement:+.1f}%")
    print(f"\nBreakdown by stat:")
    for stat, imp in improvements:
        emoji = "✅" if imp > 0 else "❌"
        print(f"  {emoji} {stat:10s}: {imp:+.1f}%")

    print("\n" + "="*70)
    print(f"CURRENT SEASON ({season_label}) VALIDATION")
    print("="*70)

    if avg_improvement >= 1.0:
        print(f"\n✅ ENSEMBLE PERFORMS WELL: {avg_improvement:+.1f}% improvement")
        print("   → Ensemble validated on current season games")
    elif avg_improvement >= 0.3:
        print(f"\n⚠️  MARGINAL IMPROVEMENT: {avg_improvement:+.1f}%")
        print("   → Ensemble shows small but positive gain")
    else:
        print(f"\n❌ NO IMPROVEMENT: {avg_improvement:+.1f}%")
        print("   → Ensemble not improving predictions on current games")

    print(f"\nNote: Baseline tested on {season_label} games")
    print(f"      Ensemble trained on 2022-2026 window (includes training data)")
    print(f"      True out-of-sample test would require 2026-2027 data")

# Save results
output_file = f"backtest_current_season_{season_label.replace('-', '_')}.json"
output_data = {
    'season': season_label,
    'baseline': baseline_results,
    'comparison': comparison,
    'ensemble_meta': {
        'window': '2022-2026',
        'trained_date': ensemble_meta['trained_date'],
        'metrics': ensemble_meta['metrics']
    },
    'summary': {
        'avg_improvement_pct': avg_improvement if comparison else 0,
        'n_games': int(df_current['gameId'].nunique()),
        'n_players': int(df_current['personId'].nunique())
    }
}

with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\n✅ Results saved to: {output_file}")
