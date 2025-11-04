"""
Test Volume + Efficiency Features - Phase 1.1 + 1.2

Previous test showed shot volume features are important but caused overfitting.
This test adds True Shooting % to provide efficiency context alongside volume.

Volume tells us OPPORTUNITY, Efficiency tells us SKILL.
Together they should predict points better.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path

print("="*70)
print("VOLUME + EFFICIENCY FEATURES TEST")
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

# Use 2017-2021 for training, 2022 for testing
df_train = df[(df['season_end_year'] >= 2017) & (df['season_end_year'] <= 2021)].copy()
df_test = df[df['season_end_year'] == 2022].copy()

print(f"  Training: 2017-2021 ({len(df_train):,} records)")
print(f"  Testing:  2022      ({len(df_test):,} records)")

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


def build_features_volume_only(player_df, idx):
    """Volume only: What we tested before (resulted in -1.3%)."""
    hist_df = player_df.iloc[:idx]
    if len(hist_df) < 3:
        return None

    pts_last5 = hist_df['points'].tail(5).mean()
    mins_last5 = hist_df['numMinutes'].tail(5).mean()
    fga_last5 = hist_df['fieldGoalsAttempted'].tail(5).mean()
    three_pa_last5 = hist_df['threePointersAttempted'].tail(5).mean()
    fta_last5 = hist_df['freeThrowsAttempted'].tail(5).mean()

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
    }


def build_features_volume_efficiency(player_df, idx):
    """
    Volume + Efficiency: The complete picture.

    NEW: True Shooting % (TS%)
    Formula: TS% = PTS / (2 * (FGA + 0.44 * FTA))

    This accounts for:
    - 2-point shots (worth 2)
    - 3-point shots (worth 3)
    - Free throws (worth 1, but ~0.44 factor for possessions)

    A player with 20 FGA and 50% TS will score ~20 points.
    A player with 20 FGA and 60% TS will score ~24 points.
    """
    hist_df = player_df.iloc[:idx]
    if len(hist_df) < 3:
        return None

    # Calculate True Shooting % for each game
    def calc_ts_pct(row):
        pts = row['points']
        fga = row['fieldGoalsAttempted']
        fta = row['freeThrowsAttempted']

        # True Shooting %
        denominator = 2 * (fga + 0.44 * fta)
        if denominator > 0:
            return pts / denominator
        return 0.5  # league average if no attempts

    hist_df = hist_df.copy()
    hist_df['ts_pct'] = hist_df.apply(calc_ts_pct, axis=1)

    # Basic stats
    pts_last5 = hist_df['points'].tail(5).mean()
    mins_last5 = hist_df['numMinutes'].tail(5).mean()
    fga_last5 = hist_df['fieldGoalsAttempted'].tail(5).mean()
    three_pa_last5 = hist_df['threePointersAttempted'].tail(5).mean()
    fta_last5 = hist_df['freeThrowsAttempted'].tail(5).mean()

    # NEW: Efficiency metrics
    ts_pct_last5 = hist_df['ts_pct'].tail(5).mean()
    ts_pct_last10 = hist_df['ts_pct'].tail(10).mean()
    ts_pct_season = hist_df['ts_pct'].mean()  # Season baseline

    # NEW: 3P shooting %
    three_pct_last5 = hist_df['threePointersPercentage'].tail(5).mean()

    # NEW: FT shooting %
    ft_pct_last5 = hist_df['freeThrowsPercentage'].tail(5).mean()

    return {
        # Baseline
        'games_played': len(hist_df),
        'pts_avg_last5': pts_last5,
        'mins_avg_last5': mins_last5,
        'pts_per_minute': pts_last5 / max(mins_last5, 1),
        'is_home': player_df.iloc[idx].get('home', 1),

        # Volume
        'fga_avg_last5': fga_last5,
        'fga_per_minute': fga_last5 / max(mins_last5, 1),
        'three_pa_avg_last5': three_pa_last5,
        'three_pa_per_minute': three_pa_last5 / max(mins_last5, 1),
        'fta_avg_last5': fta_last5,

        # NEW: Efficiency
        'ts_pct_last5': ts_pct_last5,
        'ts_pct_last10': ts_pct_last10,
        'ts_pct_season': ts_pct_season,
        'three_pct_last5': three_pct_last5,
        'ft_pct_last5': ft_pct_last5,
    }


def build_dataset(df, feature_fn, max_samples=10000):
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

    if len(X_list) == 0:
        return None, None

    X = pd.DataFrame(X_list)
    y = np.array(y_list)
    return X, y


print("\n" + "="*70)
print("MODEL 1: BASELINE (Basic Features Only)")
print("="*70)

X_train_1, y_train_1 = build_dataset(df_train, build_features_baseline, 10000)
X_test_1, y_test_1 = build_dataset(df_test, build_features_baseline, 5000)

print(f"  Features: {len(X_train_1.columns)}")

model_1 = lgb.LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42, verbose=-1)
model_1.fit(X_train_1, y_train_1)

y_pred_1 = model_1.predict(X_test_1)
rmse_1 = np.sqrt(np.mean((y_pred_1 - y_test_1) ** 2))
print(f"  Test RMSE: {rmse_1:.3f}")


print("\n" + "="*70)
print("MODEL 2: VOLUME ONLY (What We Tested Before)")
print("="*70)

X_train_2, y_train_2 = build_dataset(df_train, build_features_volume_only, 10000)
X_test_2, y_test_2 = build_dataset(df_test, build_features_volume_only, 5000)

print(f"  Features: {len(X_train_2.columns)}")

model_2 = lgb.LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42, verbose=-1)
model_2.fit(X_train_2, y_train_2)

y_pred_2 = model_2.predict(X_test_2)
rmse_2 = np.sqrt(np.mean((y_pred_2 - y_test_2) ** 2))
print(f"  Test RMSE: {rmse_2:.3f}")
print(f"  vs Baseline: {(rmse_1 - rmse_2) / rmse_1 * 100:+.2f}%")


print("\n" + "="*70)
print("MODEL 3: VOLUME + EFFICIENCY (Complete Picture)")
print("="*70)

X_train_3, y_train_3 = build_dataset(df_train, build_features_volume_efficiency, 10000)
X_test_3, y_test_3 = build_dataset(df_test, build_features_volume_efficiency, 5000)

print(f"  Features: {len(X_train_3.columns)}")
new_features = [f for f in X_train_3.columns if f not in X_train_2.columns]
print(f"\n  NEW EFFICIENCY FEATURES:")
for feat in new_features:
    print(f"    - {feat}")

model_3 = lgb.LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42, verbose=-1)
model_3.fit(X_train_3, y_train_3)

y_pred_3 = model_3.predict(X_test_3)
rmse_3 = np.sqrt(np.mean((y_pred_3 - y_test_3) ** 2))
print(f"\n  Test RMSE: {rmse_3:.3f}")
print(f"  vs Baseline: {(rmse_1 - rmse_3) / rmse_1 * 100:+.2f}%")


print("\n" + "="*70)
print("COMPARISON")
print("="*70)

print(f"\n1. Baseline (basic):            {rmse_1:.3f}")
print(f"2. Volume only:                  {rmse_2:.3f}  ({(rmse_1 - rmse_2) / rmse_1 * 100:+.1f}%)")
print(f"3. Volume + Efficiency:          {rmse_3:.3f}  ({(rmse_1 - rmse_3) / rmse_1 * 100:+.1f}%)")
print(f"\n   Volume + Efficiency vs Volume: {(rmse_2 - rmse_3) / rmse_2 * 100:+.1f}%")


print("\n" + "="*70)
print("FEATURE IMPORTANCE (Volume + Efficiency)")
print("="*70)

feature_importance = pd.DataFrame({
    'feature': X_train_3.columns,
    'importance': model_3.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 Features:")
for i, row in feature_importance.head(15).iterrows():
    is_volume = " [VOLUME]" if 'fga' in row['feature'] or 'fta' in row['feature'] or '3pa' in row['feature'] or 'three_pa' in row['feature'] else ""
    is_efficiency = " [EFFICIENCY]" if 'ts_pct' in row['feature'] or 'pct' in row['feature'] or 'percent' in row['feature'] else ""
    print(f"  {row['feature']:25s}: {row['importance']:6.1f}{is_volume}{is_efficiency}")


print("\n" + "="*70)
print("VERDICT")
print("="*70)

improvement = (rmse_1 - rmse_3) / rmse_1 * 100

if improvement >= 1.5:
    print(f"\n✓ SUCCESS! Volume + Efficiency = +{improvement:.1f}% improvement")
    print(f"  This meets Phase 1 expectations!")
    print(f"\n  NEXT STEPS:")
    print(f"  1. Add these features to train_auto.py")
    print(f"  2. Train all windows with volume + efficiency")
    print(f"  3. Test on other prop types (3PM, REB, AST)")
elif improvement >= 0.5:
    print(f"\n✓ MODEST SUCCESS: +{improvement:.1f}% improvement")
    print(f"  Below target but positive")
    print(f"\n  NEXT STEPS:")
    print(f"  1. Add matchup context (opponent defense)")
    print(f"  2. May need Phase 2 features to hit +2% target")
else:
    print(f"\n✗ NO IMPROVEMENT: {improvement:+.1f}%")
    print(f"\n  Volume + Efficiency didn't help.")
    print(f"  Likely need different approach or matchup context.")

print(f"\n" + "="*70)
