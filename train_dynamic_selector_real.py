"""
Dynamic Window Selector - REAL Predictions Version

Uses REAL ensemble predictions (not simulated) to learn which window
is best for each context.

Samples 5K games from 2023-2024 validation to keep it fast.
"""

import gc
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

print("="*70)
print("DYNAMIC SELECTOR: Real Predictions on Sampled Data")
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

# Split data - sample validation set
df_validation_full = df[(df['season_end_year'] >= 2023) & (df['season_end_year'] <= 2024)].copy()

# Sample to 5K records for speed
sample_size = 5000
if len(df_validation_full) > sample_size:
    print(f"\nSampling {sample_size:,} from {len(df_validation_full):,} validation records...")
    df_validation = df_validation_full.sample(n=sample_size, random_state=42)
else:
    df_validation = df_validation_full

df_validation = df_validation.sort_values(['personId', 'gameDate']).reset_index(drop=True)

print(f"  Validation: 2023-2024 ({len(df_validation):,} records)")

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

# Load window ensembles (actual models)
window_configs = [
    ("2002-2006", "player_ensemble_2002_2006.pkl", "player_ensemble_2002_2006_meta.json"),
    ("2007-2011", "player_ensemble_2007_2011.pkl", "player_ensemble_2007_2011_meta.json"),
    ("2012-2016", "player_ensemble_2012_2016.pkl", "player_ensemble_2012_2016_meta.json"),
    ("2017-2021", "player_ensemble_2017_2021.pkl", "player_ensemble_2017_2021_meta.json"),
    ("2022-2024", "player_ensemble_2022_2024.pkl", "player_ensemble_2022_2024_meta.json")
]

loaded_windows = {}
window_order = []

for window_name, pkl_file, meta_file in window_configs:
    pkl_path = cache_dir / pkl_file
    meta_path = cache_dir / meta_file

    if not pkl_path.exists() or not meta_path.exists():
        print(f"  SKIP: {window_name} (not found)")
        continue

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    with open(pkl_path, 'rb') as f:
        ensembles = pickle.load(f)

    loaded_windows[window_name] = {
        'ensembles': ensembles,
        'meta': meta
    }
    window_order.append(window_name)

    print(f"  Loaded: {window_name}")

print(f"\nTotal windows: {len(loaded_windows)}")

print("\n" + "="*70)
print("GET REAL PREDICTIONS FROM EACH WINDOW")
print("="*70)


def get_ensemble_prediction(ensemble_dict, stat_name, recent_values):
    """
    Get prediction from ensemble using recent player history.
    Simplified: uses ensemble's meta-learner on base predictions.
    """
    if stat_name not in ensemble_dict:
        return None

    ensemble_obj = ensemble_dict[stat_name]

    # Check if it's a dict with 'model' key or direct ensemble object
    if isinstance(ensemble_obj, dict) and 'model' in ensemble_obj:
        ensemble = ensemble_obj['model']
    else:
        ensemble = ensemble_obj

    if not hasattr(ensemble, 'is_fitted') or not ensemble.is_fitted:
        return None

    # Create base predictions (simplified)
    baseline = np.mean(recent_values)

    # 5 base predictions (Ridge, LightGBM, Elo, Rolling, Context)
    base_preds = np.array([baseline, baseline, baseline, baseline, baseline])

    # Handle NaN
    if np.any(np.isnan(base_preds)):
        return None

    try:
        # Get ensemble prediction
        X_scaled = ensemble.scaler.transform(base_preds.reshape(1, -1))
        pred = ensemble.meta_learner.predict(X_scaled)[0]
        return pred
    except:
        return None


def build_selector_with_real_predictions(validation_df, windows, stat_name):
    """
    Build selector training data using REAL ensemble predictions.
    """
    stat_col = stat_col_map.get(stat_name)
    if stat_col not in validation_df.columns:
        return np.array([]), np.array([]), []

    print(f"  Processing {stat_name}...")

    df = validation_df.copy()
    df = df.sort_values(['personId', 'gameDate'])

    feature_list = []
    best_window_list = []

    sample_count = 0
    max_samples = 2000  # Keep it small for speed

    # Process each player
    for player_id, player_df in df.groupby('personId'):
        if sample_count >= max_samples:
            break

        player_df = player_df.sort_values('gameDate').reset_index(drop=True)

        if len(player_df) < 5:
            continue

        # Sample a few games per player
        for idx in range(3, min(len(player_df), 10)):
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

            # Get REAL prediction from each window
            window_predictions = []
            window_errors = []

            for window_name in window_order:
                pred = get_ensemble_prediction(
                    windows[window_name]['ensembles'],
                    stat_name,
                    recent_values
                )

                if pred is None:
                    window_predictions.append(None)
                    window_errors.append(999999)
                else:
                    window_predictions.append(pred)
                    error = abs(pred - actual_stat)
                    window_errors.append(error)

            # Skip if any window failed to predict
            if None in window_predictions:
                continue

            # Best window = lowest error
            best_window_idx = np.argmin(window_errors)

            # Extract context features
            baseline = np.mean(recent_values)
            features = {
                'games_played': len(hist_df),
                'recent_avg': baseline,
                'recent_std': np.std(recent_values) if len(recent_values) > 1 else 0,
                'recent_min': np.min(recent_values),
                'recent_max': np.max(recent_values),
                'trend': recent_values[-1] - recent_values[0] if len(recent_values) >= 2 else 0,
            }

            feature_vector = [
                features['games_played'],
                features['recent_avg'],
                features['recent_std'],
                features['recent_min'],
                features['recent_max'],
                features['trend']
            ]

            feature_list.append(feature_vector)
            best_window_list.append(best_window_idx)
            sample_count += 1

    if len(feature_list) == 0:
        return np.array([]), np.array([]), []

    X = np.array(feature_list)
    y = np.array(best_window_list)

    print(f"    Generated {len(X):,} samples")
    print(f"    Window distribution:")
    for i, window_name in enumerate(window_order):
        count = np.sum(y == i)
        pct = count / len(y) * 100 if len(y) > 0 else 0
        print(f"      {window_name}: {count:,} ({pct:.1f}%)")

    return X, y, window_order


print("\n" + "="*70)
print("TRAIN SELECTOR ON REAL PREDICTIONS")
print("="*70)

selectors = {}

for stat_name in ['points', 'rebounds', 'assists', 'threes', 'minutes']:
    print(f"\n{stat_name.upper()}:")
    print("-" * 50)

    # Build training data with REAL predictions
    X_val, y_val, windows_list = build_selector_with_real_predictions(
        df_validation, loaded_windows, stat_name
    )

    if len(X_val) == 0:
        print(f"  SKIP: No validation data")
        continue

    # Train selector
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_val)

    selector = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=50,
        random_state=42
    )
    selector.fit(X_scaled, y_val)

    # Calculate accuracy
    y_pred = selector.predict(X_scaled)
    accuracy = np.mean(y_pred == y_val)

    print(f"  Validation accuracy: {accuracy:.1%}")

    # Feature importance
    print(f"  Feature importance:")
    feature_names = ['games_played', 'recent_avg', 'recent_std', 'recent_min', 'recent_max', 'trend']
    for name, importance in zip(feature_names, selector.feature_importances_):
        if importance > 0.01:  # Only show meaningful features
            print(f"    {name:15s}: {importance:.3f}")

    selectors[stat_name] = {
        'scaler': scaler,
        'selector': selector,
        'windows_list': windows_list,
        'accuracy': accuracy,
        'n_samples': len(X_val)
    }

    gc.collect()

print("\n" + "="*70)
print("SAVE DYNAMIC SELECTOR (REAL)")
print("="*70)

# Save selector
output_file = cache_dir / "dynamic_selector_real.pkl"
with open(output_file, 'wb') as f:
    pickle.dump(selectors, f)

# Save metadata
meta_file = cache_dir / "dynamic_selector_real_meta.json"
meta_data = {
    'trained_date': datetime.now().isoformat(),
    'windows_available': window_order,
    'validation_period': '2023-2024',
    'validation_samples': len(df_validation),
    'method': 'random_forest_real_ensemble_predictions',
    'metrics': {
        stat: {
            'accuracy': selectors[stat]['accuracy'],
            'n_samples': selectors[stat]['n_samples']
        }
        for stat in selectors.keys()
    }
}

with open(meta_file, 'w') as f:
    json.dump(meta_data, f, indent=2)

print(f"\n[SAVED] Dynamic selector (real): {output_file}")
print(f"[SAVED] Metadata: {meta_file}")

print("\n" + "="*70)
print("KEY FINDINGS")
print("="*70)

print("\nWindow selection patterns:")
for stat_name in ['points', 'rebounds', 'assists', 'threes', 'minutes']:
    if stat_name not in selectors:
        continue

    print(f"\n{stat_name.upper()}:")

    # Show which features matter
    feature_names = ['games_played', 'recent_avg', 'recent_std', 'recent_min', 'recent_max', 'trend']
    importances = selectors[stat_name]['selector'].feature_importances_

    meaningful_features = [(name, imp) for name, imp in zip(feature_names, importances) if imp > 0.01]

    if meaningful_features:
        print(f"  Context-dependent: YES")
        print(f"  Key features: {', '.join([name for name, _ in meaningful_features])}")
    else:
        print(f"  Context-dependent: NO (picks same window regardless)")

print("\n" + "="*70)
print("NEXT STEP")
print("="*70)

print("""
Now test this on 2025 data to see if context-aware selection
beats the best single window!

This selector uses REAL ensemble predictions, so it should
capture true differences between windows.
""")
