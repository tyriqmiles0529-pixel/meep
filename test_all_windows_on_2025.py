"""
Test All Windows on 2025 Season

Tests each trained window ensemble on 2025 data:
- 2002-2006 window on 2025 (19-23 years out)
- 2007-2011 window on 2025 (14-18 years out)
- 2012-2016 window on 2025 (9-13 years out)
- 2017-2021 window on 2025 (4-8 years out)
- 2022-2026 window on 2025 (IN SAMPLE - data leakage)

This shows how ensemble performance degrades with time distance.
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

print("="*70)
print("TEST ALL WINDOWS ON 2025 SEASON")
print("="*70)

# Paths
kaggle_cache = Path.home() / ".cache" / "kagglehub" / "datasets" / \
              "eoinamoore" / "historical-nba-data-and-player-box-scores" / \
              "versions" / "258"
player_stats_path = kaggle_cache / "PlayerStatistics.csv"
cache_dir = Path("model_cache")

if not player_stats_path.exists():
    print(f"ERROR: Player stats not found at {player_stats_path}")
    exit(1)

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
print("LOAD ALL WINDOW ENSEMBLES")
print("="*70)

# Load all ensemble windows
ensemble_windows = []
window_configs = [
    ("2002-2006", "player_ensemble_2002_2006.pkl", "player_ensemble_2002_2006_meta.json"),
    ("2007-2011", "player_ensemble_2007_2011.pkl", "player_ensemble_2007_2011_meta.json"),
    ("2012-2016", "player_ensemble_2012_2016.pkl", "player_ensemble_2012_2016_meta.json"),
    ("2017-2021", "player_ensemble_2017_2021.pkl", "player_ensemble_2017_2021_meta.json"),
    ("2022-2026", "player_ensemble_2022_2026.pkl", "player_ensemble_2022_2026_meta.json")
]

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

    # Calculate years out from 2025
    end_year = meta['end_year']
    years_out = 2025 - end_year

    leakage = " [DATA LEAKAGE]" if 2025 in meta['seasons'] else ""

    print(f"  Loaded: {window_name} ({years_out} years out){leakage}")

    ensemble_windows.append({
        'name': window_name,
        'start_year': meta['start_year'],
        'end_year': meta['end_year'],
        'years_out': years_out,
        'seasons': meta['seasons'],
        'meta': meta,
        'has_leakage': 2025 in meta['seasons']
    })

print(f"\nTotal windows loaded: {len(ensemble_windows)}")

print("\n" + "="*70)
print("COMPARISON: EACH WINDOW vs 2025 BASELINE")
print("="*70)

all_results = {}

for window in ensemble_windows:
    window_name = window['name']
    years_out = window['years_out']
    has_leakage = window['has_leakage']

    print(f"\n{window_name} ({years_out} years out)")
    print("-" * 70)

    if has_leakage:
        print("  WARNING: 2025 in training data - results are INVALID")

    window_results = {}

    for stat_name in ['points', 'rebounds', 'assists', 'threes', 'minutes']:
        if stat_name not in baseline_results:
            continue

        if stat_name not in window['meta']['metrics']:
            continue

        baseline_rmse = baseline_results[stat_name]['rmse']
        ensemble_rmse = window['meta']['metrics'][stat_name]['rmse']

        improvement = (baseline_rmse - ensemble_rmse) / baseline_rmse * 100

        window_results[stat_name] = {
            'baseline_rmse': baseline_rmse,
            'ensemble_rmse': ensemble_rmse,
            'improvement_pct': improvement
        }

        status = "LEAK" if has_leakage else ("YES" if improvement > 0 else "NO")
        print(f"  [{status}] {stat_name:10s}: {improvement:+.1f}% "
              f"(baseline: {baseline_rmse:.3f}, ensemble: {ensemble_rmse:.3f})")

    if window_results:
        avg_improvement = sum(r['improvement_pct'] for r in window_results.values()) / len(window_results)
        window['avg_improvement'] = avg_improvement
        all_results[window_name] = window_results
    else:
        window['avg_improvement'] = 0.0

print("\n" + "="*70)
print("SUMMARY: GENERALIZATION BY WINDOW")
print("="*70)

# Sort windows by years out
valid_windows = [w for w in ensemble_windows if not w['has_leakage']]
valid_windows.sort(key=lambda x: x['years_out'])

print("\nGeneralization across time:")
print(f"{'Window':<15} {'Years Out':<12} {'Avg Improvement':<20} {'Status'}")
print("-" * 70)

for window in valid_windows:
    window_name = window['name']
    years_out = window['years_out']
    avg_imp = window.get('avg_improvement', 0.0)

    if avg_imp >= 1.0:
        status = "GOOD"
    elif avg_imp >= 0.0:
        status = "MARGINAL"
    else:
        status = "POOR"

    print(f"{window_name:<15} {years_out:<12} {avg_imp:+.1f}%{' '*14} {status}")

print("\n" + "="*70)
print("VERDICT: BEST WINDOW FOR 2025 PREDICTION")
print("="*70)

if valid_windows:
    best_window = max(valid_windows, key=lambda x: x.get('avg_improvement', 0.0))

    print(f"\nBest performing window: {best_window['name']}")
    print(f"  Years out: {best_window['years_out']}")
    print(f"  Average improvement: {best_window['avg_improvement']:+.1f}%")
    print(f"  Trained on: {best_window['seasons']}")

    if best_window['avg_improvement'] >= 1.0:
        print(f"\n[YES] {best_window['name']} ensemble validates on 2025")
        print("   -> Use this window for current predictions")
    elif best_window['avg_improvement'] >= 0.3:
        print(f"\n[MAYBE] {best_window['name']} shows marginal improvement")
        print("   -> Consider using, but gain is small")
    else:
        print(f"\n[NO] No window shows significant improvement on 2025")
        print("   -> All ensembles fail to generalize forward")
        print("   -> May need to retrain on 2022-2024 only")

# Check for leakage window
leakage_windows = [w for w in ensemble_windows if w['has_leakage']]
if leakage_windows:
    print("\n" + "="*70)
    print("DATA LEAKAGE DETECTED")
    print("="*70)
    for window in leakage_windows:
        print(f"\n{window['name']} includes 2025 in training:")
        print(f"  Average improvement: {window.get('avg_improvement', 0.0):+.1f}% [INVALID]")
        print(f"  This is NOT a true holdout test")

# Save results
output = {
    'test_season': 2025,
    'baseline_2025': baseline_results,
    'windows': {
        w['name']: {
            'years_out': w['years_out'],
            'avg_improvement_pct': w.get('avg_improvement', 0.0),
            'has_leakage': w['has_leakage'],
            'results': all_results.get(w['name'], {})
        }
        for w in ensemble_windows
    },
    'best_window': best_window['name'] if valid_windows else None,
    'best_improvement': best_window.get('avg_improvement', 0.0) if valid_windows else 0.0
}

with open('test_all_windows_on_2025.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n[SAVED] Results saved to: test_all_windows_on_2025.json")
