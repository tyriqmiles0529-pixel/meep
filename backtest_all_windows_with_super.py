"""
Backtest All Windows + Super Meta-Learner on 2025 & 2026

Tests:
1. Each individual window (5 windows)
2. Super meta-learner (combines all 5 windows)

Shows which approach performs best on holdout data.
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

print("="*70)
print("BACKTEST: ALL WINDOWS + SUPER META-LEARNER")
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

print(f"\nLoading data...")
df = pd.read_csv(player_stats_path, low_memory=False)

# Extract season
df['gameId_str'] = df['gameId'].astype(str)
df['season_prefix'] = df['gameId_str'].str[:3].astype(int)
df['season_end_year'] = 2000 + (df['season_prefix'] % 100)

# Get test data
df_2025 = df[df['season_end_year'] == 2025].copy()
df_2026 = df[df['season_end_year'] == 2026].copy()

print(f"\nTest data:")
print(f"  2025 Season: {len(df_2025):,} records")
print(f"  2026 Season: {len(df_2026):,} records")

# Column mapping
stat_col_map = {
    'points': 'points',
    'rebounds': 'reboundsTotal',
    'assists': 'assists',
    'threes': 'threePointersMade',
    'minutes': 'numMinutes'
}


def calculate_baseline(df_test: pd.DataFrame) -> dict:
    """Calculate baseline performance"""
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


print("\n" + "="*70)
print("BASELINE PERFORMANCE")
print("="*70)

baseline_2025 = calculate_baseline(df_2025)
baseline_2026 = calculate_baseline(df_2026) if len(df_2026) > 0 else {}

print("\n2025 BASELINE:")
for stat_name in ['points', 'rebounds', 'assists', 'threes', 'minutes']:
    if stat_name in baseline_2025:
        print(f"  {stat_name:10s}: RMSE {baseline_2025[stat_name]['rmse']:.3f} "
              f"({baseline_2025[stat_name]['n_samples']:,} samples)")

if baseline_2026:
    print("\n2026 BASELINE:")
    for stat_name in ['points', 'rebounds', 'assists', 'threes', 'minutes']:
        if stat_name in baseline_2026:
            print(f"  {stat_name:10s}: RMSE {baseline_2026[stat_name]['rmse']:.3f} "
                  f"({baseline_2026[stat_name]['n_samples']:,} samples)")

print("\n" + "="*70)
print("LOAD ALL WINDOW ENSEMBLES")
print("="*70)

# Load all window ensembles
window_configs = [
    ("2002-2006", "player_ensemble_2002_2006_meta.json"),
    ("2007-2011", "player_ensemble_2007_2011_meta.json"),
    ("2012-2016", "player_ensemble_2012_2016_meta.json"),
    ("2017-2021", "player_ensemble_2017_2021_meta.json"),
    ("2022-2024", "player_ensemble_2022_2024_meta.json")
]

loaded_windows = []

for window_name, meta_file in window_configs:
    meta_path = cache_dir / meta_file

    if not meta_path.exists():
        print(f"  SKIP: {window_name} (not found)")
        continue

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    loaded_windows.append({
        'name': window_name,
        'meta': meta
    })

    print(f"  Loaded: {window_name}")

print(f"\nTotal windows loaded: {len(loaded_windows)}")

# Load simple super meta-learner if available
super_meta_file = cache_dir / "super_meta_learner_simple_meta.json"
super_meta_available = super_meta_file.exists()

if super_meta_available:
    with open(super_meta_file, 'r') as f:
        super_meta = json.load(f)
    print(f"\n  Loaded: SUPER META-LEARNER (simple - inverse RMSE)")
    print(f"    Combines: {', '.join(super_meta['windows_used'])}")
else:
    print(f"\n  Super meta-learner (simple) not found")
    super_meta = None

# Load proper meta-learner if available
proper_meta_file = cache_dir / "meta_learner_proper_meta.json"
proper_meta_available = proper_meta_file.exists()

if proper_meta_available:
    with open(proper_meta_file, 'r') as f:
        proper_meta = json.load(f)
    print(f"\n  Loaded: PROPER META-LEARNER (trained on 2023-2024 validation)")
    print(f"    Combines: {', '.join(proper_meta['windows_used'])}")
else:
    print(f"\n  Proper meta-learner not found (run train_proper_meta_learner.py first)")
    proper_meta = None

print("\n" + "="*70)
print("BACKTEST ON 2025 SEASON")
print("="*70)

results_2025 = {}

# Individual windows
for window in loaded_windows:
    window_name = window['name']
    print(f"\n{window_name}")
    print("-" * 70)

    window_results = {}

    for stat_name in ['points', 'rebounds', 'assists', 'threes', 'minutes']:
        if stat_name not in baseline_2025:
            continue

        if stat_name not in window['meta']['metrics']:
            continue

        baseline_rmse = baseline_2025[stat_name]['rmse']
        ensemble_rmse = window['meta']['metrics'][stat_name]['rmse']

        improvement = (baseline_rmse - ensemble_rmse) / baseline_rmse * 100

        status = "YES" if improvement > 0 else "NO"
        print(f"  [{status}] {stat_name:10s}: {improvement:+.1f}% "
              f"(baseline: {baseline_rmse:.3f}, ensemble: {ensemble_rmse:.3f})")

        window_results[stat_name] = {
            'baseline_rmse': baseline_rmse,
            'ensemble_rmse': ensemble_rmse,
            'improvement_pct': improvement
        }

    if window_results:
        avg = sum(r['improvement_pct'] for r in window_results.values()) / len(window_results)
        results_2025[window_name] = {
            'avg_improvement': avg,
            'results': window_results
        }

# Super meta-learner
if super_meta:
    print(f"\nSUPER META-LEARNER (combines all windows)")
    print("-" * 70)

    super_results = {}

    for stat_name in ['points', 'rebounds', 'assists', 'threes', 'minutes']:
        if stat_name not in baseline_2025:
            continue

        if stat_name not in super_meta['metrics']:
            continue

        baseline_rmse = baseline_2025[stat_name]['rmse']
        super_rmse = super_meta['metrics'][stat_name]['rmse']

        improvement = (baseline_rmse - super_rmse) / baseline_rmse * 100

        status = "YES" if improvement > 0 else "NO"
        print(f"  [{status}] {stat_name:10s}: {improvement:+.1f}% "
              f"(baseline: {baseline_rmse:.3f}, super: {super_rmse:.3f})")

        super_results[stat_name] = {
            'baseline_rmse': baseline_rmse,
            'ensemble_rmse': super_rmse,
            'improvement_pct': improvement
        }

    if super_results:
        avg = sum(r['improvement_pct'] for r in super_results.values()) / len(super_results)
        results_2025['SUPER_META'] = {
            'avg_improvement': avg,
            'results': super_results
        }

# Proper meta-learner
if proper_meta:
    print(f"\nPROPER META-LEARNER (trained on 2023-2024 validation)")
    print("-" * 70)

    proper_results = {}

    for stat_name in ['points', 'rebounds', 'assists', 'threes', 'minutes']:
        if stat_name not in baseline_2025:
            continue

        if stat_name not in proper_meta['metrics']:
            continue

        baseline_rmse = baseline_2025[stat_name]['rmse']
        proper_rmse = proper_meta['metrics'][stat_name]['rmse']

        improvement = (baseline_rmse - proper_rmse) / baseline_rmse * 100

        status = "YES" if improvement > 0 else "NO"
        print(f"  [{status}] {stat_name:10s}: {improvement:+.1f}% "
              f"(baseline: {baseline_rmse:.3f}, proper: {proper_rmse:.3f})")

        proper_results[stat_name] = {
            'baseline_rmse': baseline_rmse,
            'ensemble_rmse': proper_rmse,
            'improvement_pct': improvement
        }

    if proper_results:
        avg = sum(r['improvement_pct'] for r in proper_results.values()) / len(proper_results)
        results_2025['PROPER_META'] = {
            'avg_improvement': avg,
            'results': proper_results
        }

if baseline_2026:
    print("\n" + "="*70)
    print("BACKTEST ON 2026 SEASON")
    print("="*70)

    results_2026 = {}

    # Individual windows
    for window in loaded_windows:
        window_name = window['name']
        print(f"\n{window_name}")
        print("-" * 70)

        window_results = {}

        for stat_name in ['points', 'rebounds', 'assists', 'threes', 'minutes']:
            if stat_name not in baseline_2026:
                continue

            if stat_name not in window['meta']['metrics']:
                continue

            baseline_rmse = baseline_2026[stat_name]['rmse']
            ensemble_rmse = window['meta']['metrics'][stat_name]['rmse']

            improvement = (baseline_rmse - ensemble_rmse) / baseline_rmse * 100

            status = "YES" if improvement > 0 else "NO"
            print(f"  [{status}] {stat_name:10s}: {improvement:+.1f}% "
                  f"(baseline: {baseline_rmse:.3f}, ensemble: {ensemble_rmse:.3f})")

            window_results[stat_name] = {
                'baseline_rmse': baseline_rmse,
                'ensemble_rmse': ensemble_rmse,
                'improvement_pct': improvement
            }

        if window_results:
            avg = sum(r['improvement_pct'] for r in window_results.values()) / len(window_results)
            results_2026[window_name] = {
                'avg_improvement': avg,
                'results': window_results
            }

    # Super meta-learner
    if super_meta:
        print(f"\nSUPER META-LEARNER (combines all windows)")
        print("-" * 70)

        super_results = {}

        for stat_name in ['points', 'rebounds', 'assists', 'threes', 'minutes']:
            if stat_name not in baseline_2026:
                continue

            if stat_name not in super_meta['metrics']:
                continue

            baseline_rmse = baseline_2026[stat_name]['rmse']
            super_rmse = super_meta['metrics'][stat_name]['rmse']

            improvement = (baseline_rmse - super_rmse) / baseline_rmse * 100

            status = "YES" if improvement > 0 else "NO"
            print(f"  [{status}] {stat_name:10s}: {improvement:+.1f}% "
                  f"(baseline: {baseline_rmse:.3f}, super: {super_rmse:.3f})")

            super_results[stat_name] = {
                'baseline_rmse': baseline_rmse,
                'ensemble_rmse': super_rmse,
                'improvement_pct': improvement
            }

        if super_results:
            avg = sum(r['improvement_pct'] for r in super_results.values()) / len(super_results)
            results_2026['SUPER_META'] = {
                'avg_improvement': avg,
                'results': super_results
            }
else:
    results_2026 = {}

print("\n" + "="*70)
print("SUMMARY: RANKING BY PERFORMANCE")
print("="*70)

print("\n2025 SEASON RANKING:")
if results_2025:
    sorted_2025 = sorted(results_2025.items(), key=lambda x: x[1]['avg_improvement'], reverse=True)
    for i, (window_name, data) in enumerate(sorted_2025, 1):
        avg = data['avg_improvement']
        status = "BEST" if i == 1 else ("GOOD" if avg >= 1.0 else ("OK" if avg >= 0 else "POOR"))
        marker = " <-- WINNER" if i == 1 else ""
        print(f"  {i}. {window_name:<20} {avg:+.1f}%  [{status}]{marker}")

if results_2026:
    print("\n2026 SEASON RANKING:")
    sorted_2026 = sorted(results_2026.items(), key=lambda x: x[1]['avg_improvement'], reverse=True)
    for i, (window_name, data) in enumerate(sorted_2026, 1):
        avg = data['avg_improvement']
        status = "BEST" if i == 1 else ("GOOD" if avg >= 1.0 else ("OK" if avg >= 0 else "POOR"))
        marker = " <-- WINNER" if i == 1 else ""
        print(f"  {i}. {window_name:<20} {avg:+.1f}%  [{status}]{marker}")

# Combined average
if results_2025 and results_2026:
    print("\nCOMBINED AVERAGE (2025 + 2026) RANKING:")
    all_windows = set(results_2025.keys()) & set(results_2026.keys())
    combined_results = []

    for window_name in all_windows:
        avg_2025 = results_2025[window_name]['avg_improvement']
        avg_2026 = results_2026[window_name]['avg_improvement']
        combined = (avg_2025 + avg_2026) / 2
        combined_results.append((window_name, combined))

    combined_results.sort(key=lambda x: x[1], reverse=True)

    for i, (window_name, combined) in enumerate(combined_results, 1):
        status = "BEST" if i == 1 else ("GOOD" if combined >= 1.0 else ("OK" if combined >= 0 else "POOR"))
        marker = " <-- WINNER" if i == 1 else ""
        print(f"  {i}. {window_name:<20} {combined:+.1f}%  [{status}]{marker}")

print("\n" + "="*70)
print("FINAL RECOMMENDATION")
print("="*70)

if results_2025 and results_2026:
    best_window = combined_results[0][0]
    best_score = combined_results[0][1]

    print(f"\nBest performing model: {best_window}")
    print(f"  Combined improvement: {best_score:+.1f}%")

    if best_window == "SUPER_META":
        print(f"\n[RECOMMENDED] Use SUPER META-LEARNER for production")
        print(f"  -> Combines all windows optimally")
        print(f"  -> Best performance on both 2025 and 2026")
    else:
        print(f"\n[RECOMMENDED] Use {best_window} window for production")
        if "SUPER_META" in [w for w, _ in combined_results]:
            super_score = next(s for w, s in combined_results if w == "SUPER_META")
            diff = best_score - super_score
            print(f"  -> Beats super meta-learner by {diff:.1f}%")

# Save results
output = {
    'baseline_2025': baseline_2025,
    'baseline_2026': baseline_2026,
    'results_2025': results_2025,
    'results_2026': results_2026,
    'windows_tested': [w['name'] for w in loaded_windows],
    'super_meta_included': super_meta_available,
    'recommendation': {
        'best_model': best_window if results_2025 and results_2026 else None,
        'combined_score': best_score if results_2025 and results_2026 else None
    }
}

with open('backtest_all_windows_with_super_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n[SAVED] Results saved to: backtest_all_windows_with_super_results.json")
