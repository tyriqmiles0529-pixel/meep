"""
Test Shot Volume Features - Phase 1.1

Tests the impact of adding shot volume features (FGA, 3PA, FTA) to the points prediction model.
Compares baseline (no volume features) vs. enhanced (with volume features).
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import train_test_split

print("="*70)
print("SHOT VOLUME FEATURES TEST - PHASE 1.1")
print("="*70)

# Load data
kaggle_cache = Path.home() / ".cache" / "kagglehub" / "datasets" / \
              "eoinamoore" / "historical-nba-data-and-player-box-scores" / \
              "versions" / "258"
player_stats_path = kaggle_cache / "PlayerStatistics.csv"

print(f"\nLoading data from {player_stats_path}...")
df = pd.read_csv(player_stats_path, low_memory=False)

# Extract season
df['gameId_str'] = df['gameId'].astype(str)
df['season_prefix'] = df['gameId_str'].str[:3].astype(int)
df['season_end_year'] = 2000 + (df['season_prefix'] % 100)

# Use 2017-2021 for training, 2022 for testing (smaller, faster)
df_train = df[(df['season_end_year'] >= 2017) & (df['season_end_year'] <= 2021)].copy()
df_test = df[df['season_end_year'] == 2022].copy()

print(f"  Training: 2017-2021 ({len(df_train):,} records)")
print(f"  Testing:  2022      ({len(df_test):,} records)")

# Sort by player and date
df_train = df_train.sort_values(['personId', 'gameDate']).reset_index(drop=True)
df_test = df_test.sort_values(['personId', 'gameDate']).reset_index(drop=True)


def build_features_baseline(player_df, idx, target_stat='points'):
    """
    Build BASELINE features (no shot volume).
    Current approach: only recent averages and basic context.
    """
    hist_df = player_df.iloc[:idx]

    if len(hist_df) < 3:
        return None

    # Basic rolling averages (last 5 games)
    pts_last5 = hist_df['points'].tail(5).mean()
    mins_last5 = hist_df['numMinutes'].tail(5).mean()

    # Basic features
    features = {
        'games_played': len(hist_df),
        'pts_avg_last5': pts_last5,
        'mins_avg_last5': mins_last5,
        'pts_per_minute': pts_last5 / max(mins_last5, 1),
        'is_home': player_df.iloc[idx].get('home', 1),
    }

    return features


def build_features_enhanced(player_df, idx, target_stat='points'):
    """
    Build ENHANCED features (WITH shot volume).

    New features:
    - fga_avg_last5: Recent field goal attempts
    - fga_per_minute: Shot frequency
    - three_pa_avg_last5: Recent 3-point attempts
    - three_pa_per_minute: 3-point attempt frequency
    - fta_avg_last5: Recent free throw attempts
    """
    hist_df = player_df.iloc[:idx]

    if len(hist_df) < 3:
        return None

    # Basic rolling averages
    pts_last5 = hist_df['points'].tail(5).mean()
    mins_last5 = hist_df['numMinutes'].tail(5).mean()

    # NEW: Shot volume features
    fga_last5 = hist_df['fieldGoalsAttempted'].tail(5).mean()
    three_pa_last5 = hist_df['threePointersAttempted'].tail(5).mean()
    fta_last5 = hist_df['freeThrowsAttempted'].tail(5).mean()

    features = {
        # Baseline features
        'games_played': len(hist_df),
        'pts_avg_last5': pts_last5,
        'mins_avg_last5': mins_last5,
        'pts_per_minute': pts_last5 / max(mins_last5, 1),
        'is_home': player_df.iloc[idx].get('home', 1),

        # NEW: Shot volume features
        'fga_avg_last5': fga_last5,
        'fga_per_minute': fga_last5 / max(mins_last5, 1),
        'three_pa_avg_last5': three_pa_last5,
        'three_pa_per_minute': three_pa_last5 / max(mins_last5, 1),
        'fta_avg_last5': fta_last5,
    }

    return features


def build_dataset(df, feature_fn, target_stat='points', max_samples=10000):
    """Build training/test dataset using specified feature function."""
    X_list = []
    y_list = []
    sample_count = 0

    for player_id, player_df in df.groupby('personId'):
        if sample_count >= max_samples:
            break

        player_df = player_df.sort_values('gameDate').reset_index(drop=True)

        if len(player_df) < 5:
            continue

        for idx in range(3, len(player_df)):
            if sample_count >= max_samples:
                break

            game_row = player_df.iloc[idx]
            actual_stat = game_row[target_stat]

            if pd.isna(actual_stat):
                continue

            features = feature_fn(player_df, idx, target_stat)
            if features is None:
                continue

            X_list.append(features)
            y_list.append(actual_stat)
            sample_count += 1

    if len(X_list) == 0:
        return None, None

    X = pd.DataFrame(X_list)
    y = np.array(y_list)

    return X, y


print("\n" + "="*70)
print("BASELINE MODEL (No Shot Volume Features)")
print("="*70)

print("\nBuilding training data (baseline)...")
X_train_base, y_train_base = build_dataset(df_train, build_features_baseline, 'points', max_samples=10000)

print("\nBuilding test data (baseline)...")
X_test_base, y_test_base = build_dataset(df_test, build_features_baseline, 'points', max_samples=5000)

print(f"\n  Training samples: {len(X_train_base):,}")
print(f"  Test samples:     {len(X_test_base):,}")
print(f"  Features:         {len(X_train_base.columns)}")

# Train baseline model
print("\nTraining baseline LightGBM...")
model_base = lgb.LGBMRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    verbose=-1
)
model_base.fit(X_train_base, y_train_base)

# Evaluate
y_pred_base = model_base.predict(X_test_base)
rmse_base = np.sqrt(np.mean((y_pred_base - y_test_base) ** 2))
mae_base = np.mean(np.abs(y_pred_base - y_test_base))

print(f"\n  Test RMSE: {rmse_base:.3f}")
print(f"  Test MAE:  {mae_base:.3f}")


print("\n" + "="*70)
print("ENHANCED MODEL (With Shot Volume Features)")
print("="*70)

print("\nBuilding training data (enhanced)...")
X_train_enh, y_train_enh = build_dataset(df_train, build_features_enhanced, 'points', max_samples=10000)

print("\nBuilding test data (enhanced)...")
X_test_enh, y_test_enh = build_dataset(df_test, build_features_enhanced, 'points', max_samples=5000)

print(f"\n  Training samples: {len(X_train_enh):,}")
print(f"  Test samples:     {len(X_test_enh):,}")
print(f"  Features:         {len(X_train_enh.columns)}")
print(f"\n  NEW FEATURES:")
new_features = [f for f in X_train_enh.columns if f not in X_train_base.columns]
for feat in new_features:
    print(f"    - {feat}")

# Train enhanced model
print("\nTraining enhanced LightGBM...")
model_enh = lgb.LGBMRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    verbose=-1
)
model_enh.fit(X_train_enh, y_train_enh)

# Evaluate
y_pred_enh = model_enh.predict(X_test_enh)
rmse_enh = np.sqrt(np.mean((y_pred_enh - y_test_enh) ** 2))
mae_enh = np.mean(np.abs(y_pred_enh - y_test_enh))

print(f"\n  Test RMSE: {rmse_enh:.3f}")
print(f"  Test MAE:  {mae_enh:.3f}")


print("\n" + "="*70)
print("COMPARISON")
print("="*70)

rmse_improvement = (rmse_base - rmse_enh) / rmse_base * 100
mae_improvement = (mae_base - mae_enh) / mae_base * 100

print(f"\nBaseline RMSE:  {rmse_base:.3f}")
print(f"Enhanced RMSE:  {rmse_enh:.3f}")
print(f"Improvement:    {rmse_improvement:+.2f}%")
print(f"\nBaseline MAE:   {mae_base:.3f}")
print(f"Enhanced MAE:   {mae_enh:.3f}")
print(f"Improvement:    {mae_improvement:+.2f}%")


print("\n" + "="*70)
print("FEATURE IMPORTANCE (Enhanced Model)")
print("="*70)

feature_importance = pd.DataFrame({
    'feature': X_train_enh.columns,
    'importance': model_enh.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
for i, row in feature_importance.head(10).iterrows():
    is_new = " ⭐ NEW" if row['feature'] in new_features else ""
    print(f"  {row['feature']:25s}: {row['importance']:6.1f}{is_new}")


print("\n" + "="*70)
print("VERDICT")
print("="*70)

if rmse_improvement >= 1.5:
    print(f"\n✓ SUCCESS! Shot volume features provide +{rmse_improvement:.1f}% improvement")
    print(f"  This meets expectations for Phase 1.1 (+1.5-2.5%)")
    print(f"\n  RECOMMENDATION: Proceed with full Phase 1 implementation")
    print(f"  - Add these features to train_auto.py")
    print(f"  - Train all windows with shot volume features")
    print(f"  - Test on all prop types (3PM, REB, AST)")
elif rmse_improvement >= 0.5:
    print(f"\n✓ MODEST SUCCESS: +{rmse_improvement:.1f}% improvement")
    print(f"  Below expectations but still positive")
    print(f"\n  RECOMMENDATION: Continue to Phase 1.2 (efficiency features)")
    print(f"  - True Shooting % may provide additional lift")
else:
    print(f"\n✗ MINIMAL IMPACT: Only +{rmse_improvement:.1f}% improvement")
    print(f"\n  RECOMMENDATION: Investigate further")
    print(f"  - Check if FGA data quality is good")
    print(f"  - Try True Shooting % (accounts for efficiency)")
    print(f"  - May need matchup context to see full benefit")

print(f"\n" + "="*70)
