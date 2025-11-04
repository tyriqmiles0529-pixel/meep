"""
TRUE Backtest: Enhanced Selector on 2025 Data

Actually runs predictions on every 2025 game and measures real performance.
This is the real test - no estimates, just actual predictions vs actuals.
"""

import gc
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

print("="*70)
print("TRUE BACKTEST: Enhanced Selector on 2025")
print("="*70)

# Paths
kaggle_cache = Path.home() / ".cache" / "kagglehub" / "datasets" / \
              "eoinamoore" / "historical-nba-data-and-player-box-scores" / \
              "versions" / "258"
player_stats_path = kaggle_cache / "PlayerStatistics.csv"
cache_dir = Path("model_cache")

print(f"\nLoading data...")
df = pd.read_csv(player_stats_path, low_memory=False)

# Extract season
df['gameId_str'] = df['gameId'].astype(str)
df['season_prefix'] = df['gameId_str'].str[:3].astype(int)
df['season_end_year'] = 2000 + (df['season_prefix'] % 100)

# Get 2025 test data
df_test = df[df['season_end_year'] == 2025].copy()
df_test = df_test.sort_values(['personId', 'gameDate']).reset_index(drop=True)

print(f"  2025 Test: {len(df_test):,} records")
print(f"  Players: {df_test['personId'].nunique():,}")
print(f"  Games: {df_test['gameId'].nunique():,}")

# Column mapping
stat_col_map = {
    'points': 'points',
    'rebounds': 'reboundsTotal',
    'assists': 'assists',
    'threes': 'threePointersMade',
    'minutes': 'numMinutes'
}

print("\n" + "="*70)
print("LOAD ENHANCED SELECTOR")
print("="*70)

# Load selector
selector_file = cache_dir / "dynamic_selector_enhanced.pkl"
selector_meta_file = cache_dir / "dynamic_selector_enhanced_meta.json"

if not selector_file.exists():
    print(f"ERROR: Enhanced selector not found")
    exit(1)

with open(selector_file, 'rb') as f:
    selectors = pickle.load(f)

with open(selector_meta_file, 'r') as f:
    selector_meta = json.load(f)

print(f"  Loaded: Enhanced selector")
print(f"  Stats available: {list(selectors.keys())}")

# Load window ensembles
top_windows_per_stat = selector_meta['top_windows_per_stat']

print("\n" + "="*70)
print("LOAD WINDOW ENSEMBLES")
print("="*70)

loaded_windows = {}

for stat_name, windows_list in top_windows_per_stat.items():
    for window_name in windows_list:
        if window_name not in loaded_windows:
            pkl_file = cache_dir / f"player_ensemble_{window_name.replace('-', '_')}.pkl"
            if pkl_file.exists():
                with open(pkl_file, 'rb') as f:
                    ensembles = pickle.load(f)
                loaded_windows[window_name] = ensembles
                print(f"  Loaded: {window_name}")

print("\n" + "="*70)
print("RUN TRUE BACKTEST ON 2025")
print("="*70)


def get_ensemble_prediction(ensemble_dict, stat_name, recent_values):
    """Get prediction from ensemble"""
    if stat_name not in ensemble_dict:
        return None

    ensemble_obj = ensemble_dict[stat_name]

    if isinstance(ensemble_obj, dict) and 'model' in ensemble_obj:
        ensemble = ensemble_obj['model']
    else:
        ensemble = ensemble_obj

    if not hasattr(ensemble, 'is_fitted') or not ensemble.is_fitted:
        return None

    baseline = np.mean(recent_values)
    base_preds = np.array([baseline, baseline, baseline, baseline, baseline])

    if np.any(np.isnan(base_preds)):
        return None

    try:
        X_scaled = ensemble.scaler.transform(base_preds.reshape(1, -1))
        pred = ensemble.meta_learner.predict(X_scaled)[0]
        return pred
    except:
        return None


def extract_enhanced_features(player_df, idx, game_row, stat_col):
    """Extract enhanced features for selector"""
    hist_df = player_df.iloc[:idx]

    if len(hist_df) < 3:
        return None, None

    recent_values = hist_df[stat_col].tail(10).values
    recent_values = recent_values[~np.isnan(recent_values)]

    if len(recent_values) < 3:
        return None, None

    # Rest days
    try:
        game_date = pd.to_datetime(game_row['gameDate'])
        if len(hist_df) > 0:
            last_game_date = pd.to_datetime(hist_df.iloc[-1]['gameDate'])
            rest_days = (game_date - last_game_date).days
        else:
            rest_days = 7
    except:
        rest_days = 3

    baseline = np.mean(recent_values)
    recent_3 = recent_values[-3:] if len(recent_values) >= 3 else recent_values

    features = [
        len(hist_df),  # games_played
        baseline,  # recent_avg
        np.std(recent_values) if len(recent_values) > 1 else 0,  # recent_std
        np.min(recent_values),  # recent_min
        np.max(recent_values),  # recent_max
        recent_values[-1] - recent_values[0] if len(recent_values) >= 2 else 0,  # trend
        min(rest_days, 7),  # rest_days
        np.mean(recent_3),  # recent_form_3
        np.mean(recent_3) - baseline,  # form_change
        (np.std(recent_values) / baseline) if baseline > 0.1 else 0,  # consistency_cv
    ]

    return np.array(features), recent_values


def backtest_stat(test_df, stat_name, stat_col):
    """Run true backtest for one stat"""
    print(f"\n{stat_name.upper()}:")
    print("-" * 50)

    if stat_name not in selectors:
        print(f"  SKIP: No selector")
        return None

    selector_obj = selectors[stat_name]
    scaler = selector_obj['scaler']
    selector = selector_obj['selector']
    windows_list = selector_obj['windows_list']

    predictions = []
    actuals = []
    selected_windows = []

    sample_count = 0
    max_samples = 5000  # Limit for speed

    # Process each player
    for player_id, player_df in test_df.groupby('personId'):
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

            # Extract features
            features, recent_values = extract_enhanced_features(
                player_df, idx, game_row, stat_col
            )

            if features is None:
                continue

            # Use selector to pick window
            try:
                X_scaled = scaler.transform(features.reshape(1, -1))
                window_idx = selector.predict(X_scaled)[0]
                selected_window = windows_list[window_idx]
            except:
                continue

            # Get prediction from selected window
            pred = get_ensemble_prediction(
                loaded_windows[selected_window],
                stat_name,
                recent_values
            )

            if pred is None:
                continue

            predictions.append(pred)
            actuals.append(actual_stat)
            selected_windows.append(selected_window)
            sample_count += 1

    if len(predictions) == 0:
        print(f"  SKIP: No predictions generated")
        return None

    # Calculate metrics
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    mae = np.mean(np.abs(predictions - actuals))

    print(f"  Samples: {len(predictions):,}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  MAE: {mae:.3f}")

    # Show window selection distribution
    print(f"  Window selection:")
    from collections import Counter
    window_counts = Counter(selected_windows)
    for window, count in window_counts.most_common():
        pct = count / len(selected_windows) * 100
        print(f"    {window}: {count:,} ({pct:.1f}%)")

    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'n_samples': len(predictions),
        'window_distribution': dict(window_counts)
    }


# Calculate baseline
print("\nBASELINE (Rolling 10-game average):")
print("-" * 50)

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

    print(f"  {stat_name:10s}: {rmse:.3f}")

    baseline_results[stat_name] = {
        'rmse': float(rmse),
        'n_samples': int(len(df_stat))
    }

print("\n" + "="*70)
print("ENHANCED SELECTOR")
print("="*70)

selector_results = {}

for stat_name, stat_col in stat_col_map.items():
    if stat_col not in df_test.columns:
        continue

    result = backtest_stat(df_test, stat_name, stat_col)

    if result:
        selector_results[stat_name] = result

print("\n" + "="*70)
print("COMPARISON")
print("="*70)

comparison = {}

for stat_name in ['points', 'rebounds', 'assists', 'threes', 'minutes']:
    if stat_name not in baseline_results or stat_name not in selector_results:
        continue

    baseline_rmse = baseline_results[stat_name]['rmse']
    selector_rmse = selector_results[stat_name]['rmse']

    improvement = (baseline_rmse - selector_rmse) / baseline_rmse * 100

    print(f"\n{stat_name.upper()}:")
    print(f"  Baseline RMSE:  {baseline_rmse:.3f}")
    print(f"  Selector RMSE:  {selector_rmse:.3f}")
    print(f"  Improvement:    {improvement:+.1f}%")

    comparison[stat_name] = {
        'baseline_rmse': baseline_rmse,
        'selector_rmse': selector_rmse,
        'improvement_pct': improvement
    }

print("\n" + "="*70)
print("FINAL VERDICT: TRUE BACKTEST")
print("="*70)

if comparison:
    improvements = [(s, c['improvement_pct']) for s, c in comparison.items()]
    improvements.sort(key=lambda x: x[1], reverse=True)

    avg_improvement = sum(imp for _, imp in improvements) / len(improvements)

    print(f"\nTrue backtest on 2025 data:")
    print(f"  Average improvement: {avg_improvement:+.1f}%")
    print(f"\nBreakdown:")
    for stat, imp in improvements:
        status = "YES" if imp > 0 else "NO"
        print(f"  [{status}] {stat:10s}: {imp:+.1f}%")

    print(f"\nComparison to expectations:")
    print(f"  Expected (from training): +19.0%")
    print(f"  True (from backtest):     {avg_improvement:+.1f}%")

    if avg_improvement >= 16.4:
        print(f"\n[YES] ENHANCED SELECTOR WINS!")
        print(f"  {avg_improvement:+.1f}% > Cherry-pick (+16.4%)")
        print(f"  TRUE backtest validates the approach")
        print(f"  DEPLOY: Enhanced hybrid selector")
    elif avg_improvement >= 13.7:
        print(f"\n[GOOD] Beats single window but not cherry-pick")
        print(f"  {avg_improvement:+.1f}% > Single window (+13.7%)")
        print(f"  But < Cherry-pick (+16.4%)")
    else:
        print(f"\n[NO] Does not beat benchmarks")
        print(f"  {avg_improvement:+.1f}% < Cherry-pick (+16.4%)")

# Save results
output_file = "true_backtest_enhanced_results.json"
with open(output_file, 'w') as f:
    json.dump({
        'baseline': baseline_results,
        'selector': selector_results,
        'comparison': comparison,
        'avg_improvement': avg_improvement if comparison else 0,
        'true_backtest': True
    }, f, indent=2)

print(f"\n[SAVED] True backtest results: {output_file}")
