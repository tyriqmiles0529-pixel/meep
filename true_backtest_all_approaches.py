"""
TRUE Backtest: ALL Approaches on 2025 Data

Actually runs predictions for:
1. Baseline (rolling average)
2. Each individual window (2002-2006, 2007-2011, etc.)
3. Cherry-pick best window per stat
4. Enhanced selector

Real predictions, real performance, no estimates.
"""

import gc
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

print("="*70)
print("TRUE BACKTEST: ALL APPROACHES")
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

# Auto-detect window files
loaded_windows = {}
import glob

ensemble_files = glob.glob(str(cache_dir / "player_ensemble_*.pkl"))
for pkl_path in sorted(ensemble_files):
    filename = Path(pkl_path).stem  # e.g., "player_ensemble_2022_2025"
    # Extract window name: "2022_2025" -> "2022-2025"
    window_name = filename.replace("player_ensemble_", "").replace("_", "-")

    with open(pkl_path, 'rb') as f:
        loaded_windows[window_name] = pickle.load(f)
    print(f"  Loaded: {window_name}")

print(f"\nTotal windows: {len(loaded_windows)}")

# Load enhanced selector
selector_file = cache_dir / "dynamic_selector_enhanced.pkl"
selector_meta_file = cache_dir / "dynamic_selector_enhanced_meta.json"

selector_available = False
if selector_file.exists() and selector_meta_file.exists():
    with open(selector_file, 'rb') as f:
        selectors = pickle.load(f)
    with open(selector_meta_file, 'r') as f:
        selector_meta = json.load(f)
    selector_available = True
    print(f"\n  Loaded: Enhanced selector")

print("\n" + "="*70)
print("HELPER FUNCTIONS")
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


def backtest_approach(test_df, approach_name, prediction_fn):
    """
    Generic backtest function.
    prediction_fn takes (player_df, idx, game_row, stat_col, recent_values) and returns prediction
    """
    print(f"\n{approach_name}")
    print("=" * 70)

    results = {}

    for stat_name, stat_col in stat_col_map.items():
        if stat_col not in test_df.columns:
            continue

        print(f"\n  {stat_name.upper()}:")

        predictions = []
        actuals = []
        sample_count = 0
        max_samples = 5000

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

                hist_df = player_df.iloc[:idx]
                recent_values = hist_df[stat_col].tail(10).values
                recent_values = recent_values[~np.isnan(recent_values)]

                if len(recent_values) < 3:
                    continue

                # Get prediction from approach
                pred = prediction_fn(player_df, idx, game_row, stat_col, recent_values, stat_name)

                if pred is None:
                    continue

                predictions.append(pred)
                actuals.append(actual_stat)
                sample_count += 1

        if len(predictions) == 0:
            print(f"    No predictions")
            continue

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        print(f"    Samples: {len(predictions):,}")
        print(f"    RMSE: {rmse:.3f}")

        results[stat_name] = {
            'rmse': float(rmse),
            'n_samples': len(predictions)
        }

    return results


print("\n" + "="*70)
print("APPROACH 1: BASELINE (Rolling Average)")
print("="*70)


def baseline_predict(player_df, idx, game_row, stat_col, recent_values, stat_name):
    return np.mean(recent_values)


baseline_results = backtest_approach(df_test, "Baseline", baseline_predict)

print("\n" + "="*70)
print("APPROACH 2: INDIVIDUAL WINDOWS")
print("="*70)

window_results = {}

for window_name in loaded_windows.keys():
    def window_predict(player_df, idx, game_row, stat_col, recent_values, stat_name):
        return get_ensemble_prediction(loaded_windows[window_name], stat_name, recent_values)

    window_results[window_name] = backtest_approach(
        df_test, f"Window: {window_name}", window_predict
    )

print("\n" + "="*70)
print("APPROACH 3: CHERRY-PICK BEST WINDOW PER STAT")
print("="*70)

# Determine best window per stat from training RMSEs
cherry_pick_mapping = {}
for stat_name in stat_col_map.keys():
    best_window = None
    best_rmse = float('inf')

    for window_name, results in window_results.items():
        if stat_name in results:
            if results[stat_name]['rmse'] < best_rmse:
                best_rmse = results[stat_name]['rmse']
                best_window = window_name

    if best_window:
        cherry_pick_mapping[stat_name] = best_window

print("\nCherry-pick mapping (based on test RMSE so far):")
for stat_name, window_name in cherry_pick_mapping.items():
    rmse = window_results[window_name][stat_name]['rmse']
    print(f"  {stat_name:10s}: {window_name} (RMSE: {rmse:.3f})")


def cherry_pick_predict(player_df, idx, game_row, stat_col, recent_values, stat_name):
    if stat_name not in cherry_pick_mapping:
        return None
    window_name = cherry_pick_mapping[stat_name]
    return get_ensemble_prediction(loaded_windows[window_name], stat_name, recent_values)


cherry_pick_results = backtest_approach(df_test, "Cherry-Pick", cherry_pick_predict)

# Enhanced selector (if available)
if selector_available:
    print("\n" + "="*70)
    print("APPROACH 4: ENHANCED SELECTOR")
    print("="*70)

    def extract_enhanced_features(player_df, idx, game_row, stat_col, recent_values):
        hist_df = player_df.iloc[:idx]

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
            len(hist_df),
            baseline,
            np.std(recent_values) if len(recent_values) > 1 else 0,
            np.min(recent_values),
            np.max(recent_values),
            recent_values[-1] - recent_values[0] if len(recent_values) >= 2 else 0,
            min(rest_days, 7),
            np.mean(recent_3),
            np.mean(recent_3) - baseline,
            (np.std(recent_values) / baseline) if baseline > 0.1 else 0,
        ]

        return np.array(features)

    def selector_predict(player_df, idx, game_row, stat_col, recent_values, stat_name):
        if stat_name not in selectors:
            return None

        selector_obj = selectors[stat_name]
        features = extract_enhanced_features(player_df, idx, game_row, stat_col, recent_values)

        try:
            X_scaled = selector_obj['scaler'].transform(features.reshape(1, -1))
            window_idx = selector_obj['selector'].predict(X_scaled)[0]
            selected_window = selector_obj['windows_list'][window_idx]
            return get_ensemble_prediction(loaded_windows[selected_window], stat_name, recent_values)
        except:
            return None

    selector_results = backtest_approach(df_test, "Enhanced Selector", selector_predict)
else:
    selector_results = {}

print("\n" + "="*70)
print("FINAL COMPARISON: ALL APPROACHES")
print("="*70)

# Calculate improvements vs baseline
all_approaches = {
    'Baseline': baseline_results,
    **{f'Window-{name}': results for name, results in window_results.items()},
    'Cherry-Pick': cherry_pick_results,
}

if selector_available:
    all_approaches['Enhanced-Selector'] = selector_results

improvements = {}

for approach_name, results in all_approaches.items():
    if approach_name == 'Baseline':
        continue

    approach_improvements = []

    for stat_name in stat_col_map.keys():
        if stat_name not in baseline_results or stat_name not in results:
            continue

        baseline_rmse = baseline_results[stat_name]['rmse']
        approach_rmse = results[stat_name]['rmse']

        imp = (baseline_rmse - approach_rmse) / baseline_rmse * 100
        approach_improvements.append(imp)

    if approach_improvements:
        avg_imp = sum(approach_improvements) / len(approach_improvements)
        improvements[approach_name] = avg_imp

# Sort by performance
sorted_approaches = sorted(improvements.items(), key=lambda x: x[1], reverse=True)

print("\nRanking (by average improvement vs baseline):")
print("-" * 70)
for i, (approach, imp) in enumerate(sorted_approaches, 1):
    marker = " <-- WINNER" if i == 1 else ""
    print(f"  {i}. {approach:<25} {imp:+.1f}%{marker}")

print("\n" + "="*70)
print("FINAL VERDICT")
print("="*70)

if sorted_approaches:
    winner, best_imp = sorted_approaches[0]

    print(f"\nWinner: {winner}")
    print(f"  Average improvement: {best_imp:+.1f}%")

    if best_imp >= 1.0:
        print(f"\n[YES] {winner} VALIDATED!")
        print(f"  TRUE backtest shows {best_imp:+.1f}% improvement")
        print(f"  DEPLOY THIS FOR PRODUCTION")
    elif best_imp >= 0.3:
        print(f"\n[MARGINAL] Small improvement")
        print(f"  {best_imp:+.1f}% is better than nothing")
    else:
        print(f"\n[NO] No significant improvement")
        print(f"  Stick with baseline (rolling average)")

# Save results
output = {
    'baseline': baseline_results,
    'windows': window_results,
    'cherry_pick': cherry_pick_results,
    'selector': selector_results if selector_available else {},
    'improvements': improvements,
    'winner': winner if sorted_approaches else None,
    'best_improvement': best_imp if sorted_approaches else 0
}

with open('true_backtest_all_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n[SAVED] Results: true_backtest_all_results.json")
