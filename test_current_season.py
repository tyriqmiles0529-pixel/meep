"""
Test on Current Season - 2024-2025

Train on early 2024-2025 games, test on later 2025 games.
This tests if features work when the game style hasn't changed.

If features work here but not on 2017-2021→2022, it confirms
the issue was temporal drift (game evolution), not bad features.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from datetime import datetime

print("="*70)
print("CURRENT SEASON TEST - 2024-2025 Data")
print("="*70)

# Load data
kaggle_cache = Path.home() / ".cache" / "kagglehub" / "datasets" / \
              "eoinamoore" / "historical-nba-data-and-player-box-scores" / \
              "versions" / "258"
player_stats_path = kaggle_cache / "PlayerStatistics.csv"

print(f"\nLoading data...")
df = pd.read_csv(player_stats_path, low_memory=False)

# Extract season
df['gameId_str'] = df['gameId'].astype(str)
df['season_prefix'] = df['gameId_str'].str[:3].astype(int)
df['season_end_year'] = 2000 + (df['season_prefix'] % 100)

# Get 2024-2025 season data
df_2024_2025 = df[(df['season_end_year'] == 2024) | (df['season_end_year'] == 2025)].copy()

print(f"  Total 2024-2025: {len(df_2024_2025):,} records")

# Parse dates to split train/test (handle mixed formats, convert to UTC)
df_2024_2025['gameDate_dt'] = pd.to_datetime(df_2024_2025['gameDate'], format='mixed', errors='coerce', utc=True)
df_2024_2025 = df_2024_2025.dropna(subset=['gameDate_dt'])  # Drop any rows with invalid dates
df_2024_2025 = df_2024_2025.sort_values('gameDate_dt').reset_index(drop=True)

# Split: First 70% for training, last 30% for testing
split_idx = int(len(df_2024_2025) * 0.7)
df_train = df_2024_2025.iloc[:split_idx].copy()
df_test = df_2024_2025.iloc[split_idx:].copy()

train_date_range = f"{df_train['gameDate_dt'].min().date()} to {df_train['gameDate_dt'].max().date()}"
test_date_range = f"{df_test['gameDate_dt'].min().date()} to {df_test['gameDate_dt'].max().date()}"

print(f"\n  Training: {len(df_train):,} records ({train_date_range})")
print(f"  Testing:  {len(df_test):,} records ({test_date_range})")

df_train = df_train.sort_values(['personId', 'gameDate']).reset_index(drop=True)
df_test = df_test.sort_values(['personId', 'gameDate']).reset_index(drop=True)


def build_features_baseline(player_df, idx):
    """Baseline: Only basic averages."""
    hist_df = player_df.iloc[:idx]
    if len(hist_df) < 3:
        return None

    pts_last5 = hist_df['points'].tail(5).mean()
    mins_last5 = hist_df['numMinutes'].tail(5).mean()

    return {
        'games_played': len(hist_df),
        'pts_avg_last5': pts_last5,
        'mins_avg_last5': mins_last5,
        'pts_per_minute': pts_last5 / max(mins_last5, 1),
        'is_home': player_df.iloc[idx].get('home', 1),
    }


def build_features_full(player_df, idx):
    """Full: Volume + Efficiency."""
    hist_df = player_df.iloc[:idx]
    if len(hist_df) < 3:
        return None

    # Calculate True Shooting %
    def calc_ts_pct(row):
        pts = row['points']
        fga = row['fieldGoalsAttempted']
        fta = row['freeThrowsAttempted']
        denominator = 2 * (fga + 0.44 * fta)
        if denominator > 0:
            return pts / denominator
        return 0.5

    hist_df = hist_df.copy()
    hist_df['ts_pct'] = hist_df.apply(calc_ts_pct, axis=1)

    # Stats
    pts_last5 = hist_df['points'].tail(5).mean()
    mins_last5 = hist_df['numMinutes'].tail(5).mean()
    fga_last5 = hist_df['fieldGoalsAttempted'].tail(5).mean()
    three_pa_last5 = hist_df['threePointersAttempted'].tail(5).mean()
    fta_last5 = hist_df['freeThrowsAttempted'].tail(5).mean()
    ts_pct_last5 = hist_df['ts_pct'].tail(5).mean()
    ts_pct_last10 = hist_df['ts_pct'].tail(10).mean()
    ts_pct_season = hist_df['ts_pct'].mean()
    three_pct_last5 = hist_df['threePointersPercentage'].tail(5).mean()
    ft_pct_last5 = hist_df['freeThrowsPercentage'].tail(5).mean()

    return {
        'games_played': len(hist_df),
        'pts_avg_last5': pts_last5,
        'mins_avg_last5': mins_last5,
        'pts_per_minute': pts_last5 / max(mins_last5, 1),
        'is_home': player_df.iloc[idx].get('home', 1),
        'fga_avg_last5': fga_last5,
        'fga_per_minute': fga_last5 / max(mins_last5, 1),
        'three_pa_avg_last5': three_pa_last5,
        'three_pa_per_minute': three_pa_last5 / max(mins_last5, 1),
        'fta_avg_last5': fta_last5,
        'ts_pct_last5': ts_pct_last5,
        'ts_pct_last10': ts_pct_last10,
        'ts_pct_season': ts_pct_season,
        'three_pct_last5': three_pct_last5,
        'ft_pct_last5': ft_pct_last5,
    }


def build_dataset(df, feature_fn, max_samples=5000):
    """Build dataset."""
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
            actual_stat = game_row['points']

            if pd.isna(actual_stat):
                continue

            features = feature_fn(player_df, idx)
            if features is None:
                continue

            X_list.append(features)
            y_list.append(actual_stat)
            sample_count += 1

    X = pd.DataFrame(X_list)
    y = np.array(y_list)
    return X, y


print("\n" + "="*70)
print("BASELINE MODEL (No Volume/Efficiency)")
print("="*70)

X_train_base, y_train_base = build_dataset(df_train, build_features_baseline, 5000)
X_test_base, y_test_base = build_dataset(df_test, build_features_baseline, 2000)

print(f"  Train: {len(X_train_base):,} samples")
print(f"  Test:  {len(X_test_base):,} samples")

# Use heavy regularization (best from previous test)
model_base = lgb.LGBMRegressor(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.1,
    min_child_samples=100,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.5,
    reg_lambda=0.5,
    random_state=42,
    verbose=-1
)
model_base.fit(X_train_base, y_train_base)

y_pred_train_base = model_base.predict(X_train_base)
y_pred_test_base = model_base.predict(X_test_base)

rmse_train_base = np.sqrt(np.mean((y_pred_train_base - y_train_base) ** 2))
rmse_test_base = np.sqrt(np.mean((y_pred_test_base - y_test_base) ** 2))

print(f"\n  Training RMSE: {rmse_train_base:.3f}")
print(f"  Test RMSE:     {rmse_test_base:.3f}")
print(f"  Overfit Gap:   {rmse_test_base - rmse_train_base:.3f}")


print("\n" + "="*70)
print("FULL MODEL (Volume + Efficiency)")
print("="*70)

X_train_full, y_train_full = build_dataset(df_train, build_features_full, 5000)
X_test_full, y_test_full = build_dataset(df_test, build_features_full, 2000)

print(f"  Train: {len(X_train_full):,} samples, {len(X_train_full.columns)} features")
print(f"  Test:  {len(X_test_full):,} samples")

new_features = [f for f in X_train_full.columns if f not in X_train_base.columns]
print(f"\n  NEW FEATURES ({len(new_features)}):")
for feat in new_features:
    print(f"    - {feat}")

model_full = lgb.LGBMRegressor(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.1,
    min_child_samples=100,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.5,
    reg_lambda=0.5,
    random_state=42,
    verbose=-1
)
model_full.fit(X_train_full, y_train_full)

y_pred_train_full = model_full.predict(X_train_full)
y_pred_test_full = model_full.predict(X_test_full)

rmse_train_full = np.sqrt(np.mean((y_pred_train_full - y_train_full) ** 2))
rmse_test_full = np.sqrt(np.mean((y_pred_test_full - y_test_full) ** 2))

print(f"\n  Training RMSE: {rmse_train_full:.3f}")
print(f"  Test RMSE:     {rmse_test_full:.3f}")
print(f"  Overfit Gap:   {rmse_test_full - rmse_train_full:.3f}")


print("\n" + "="*70)
print("COMPARISON")
print("="*70)

improvement = (rmse_test_base - rmse_test_full) / rmse_test_base * 100

print(f"\nBaseline (no features):     {rmse_test_base:.3f}")
print(f"Full (volume + efficiency): {rmse_test_full:.3f}")
print(f"Improvement:                {improvement:+.2f}%")


print("\n" + "="*70)
print("FEATURE IMPORTANCE (Full Model)")
print("="*70)

feature_importance = pd.DataFrame({
    'feature': X_train_full.columns,
    'importance': model_full.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Features:")
for i, row in feature_importance.head(10).iterrows():
    is_new = " ⭐" if row['feature'] in new_features else ""
    print(f"  {row['feature']:25s}: {row['importance']:6.1f}{is_new}")


print("\n" + "="*70)
print("VERDICT")
print("="*70)

if improvement >= 1.5:
    print(f"\n✓ SUCCESS! Features work on current season")
    print(f"  {improvement:+.1f}% improvement when training and testing on same era")
    print(f"\n  KEY FINDING:")
    print(f"  - Features ARE valuable (they work within same season)")
    print(f"  - Previous failure was due to temporal drift (2017-2021 → 2022)")
    print(f"\n  RECOMMENDATION:")
    print(f"  1. Use rolling training windows (most recent 2-3 years only)")
    print(f"  2. Retrain models frequently (monthly or quarterly)")
    print(f"  3. Add these features to production with recent data")
elif improvement >= 0.5:
    print(f"\n✓ MODEST SUCCESS: {improvement:+.1f}% improvement")
    print(f"  Better than 2017-2021 test, but still below target")
    print(f"\n  POSSIBLE REASONS:")
    print(f"  - Small sample size (early season)")
    print(f"  - Need more features (matchup context)")
else:
    print(f"\n✗ NO IMPROVEMENT: {improvement:+.1f}%")
    print(f"\n  Even on current season data, features don't help")
    print(f"  This suggests:")
    print(f"  - Features may not be as predictive as expected")
    print(f"  - Need different approach (matchup context, etc.)")

print(f"\n" + "="*70)
