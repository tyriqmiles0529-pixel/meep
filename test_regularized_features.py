"""
Test Regularized Model with Volume + Efficiency

Previous tests showed features are important but cause overfitting.
This test adds regularization to LightGBM to prevent overfitting while
still benefiting from the new features.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path

print("="*70)
print("REGULARIZED MODEL TEST - Prevent Overfitting")
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


def build_features_full(player_df, idx):
    """Full feature set: Volume + Efficiency."""
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

    X = pd.DataFrame(X_list)
    y = np.array(y_list)
    return X, y


print("\nBuilding datasets...")
X_train, y_train = build_dataset(df_train, build_features_full, 10000)
X_test, y_test = build_dataset(df_test, build_features_full, 5000)

print(f"  Train: {len(X_train):,} samples, {len(X_train.columns)} features")
print(f"  Test:  {len(X_test):,} samples")


print("\n" + "="*70)
print("MODEL 1: Default LightGBM (What We've Been Using)")
print("="*70)

print("\nParameters:")
print("  n_estimators: 100")
print("  max_depth: 6")
print("  learning_rate: 0.05")
print("  (No regularization)")

model_default = lgb.LGBMRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    verbose=-1
)
model_default.fit(X_train, y_train)

y_pred_train_default = model_default.predict(X_train)
y_pred_test_default = model_default.predict(X_test)

rmse_train_default = np.sqrt(np.mean((y_pred_train_default - y_train) ** 2))
rmse_test_default = np.sqrt(np.mean((y_pred_test_default - y_test) ** 2))

print(f"\n  Training RMSE: {rmse_train_default:.3f}")
print(f"  Test RMSE:     {rmse_test_default:.3f}")
print(f"  Overfit Gap:   {rmse_test_default - rmse_train_default:.3f}")


print("\n" + "="*70)
print("MODEL 2: Regularized LightGBM (Prevent Overfitting)")
print("="*70)

print("\nParameters:")
print("  n_estimators: 100")
print("  max_depth: 4              (reduced from 6)")
print("  learning_rate: 0.05")
print("  min_child_samples: 50     (NEW - require more data per leaf)")
print("  subsample: 0.8            (NEW - use 80% of data per tree)")
print("  colsample_bytree: 0.8     (NEW - use 80% of features per tree)")
print("  reg_alpha: 0.1            (NEW - L1 regularization)")
print("  reg_lambda: 0.1           (NEW - L2 regularization)")

model_reg = lgb.LGBMRegressor(
    n_estimators=100,
    max_depth=4,              # Shallower trees
    learning_rate=0.05,
    min_child_samples=50,     # Require more samples per leaf
    subsample=0.8,            # Row subsampling
    colsample_bytree=0.8,     # Feature subsampling
    reg_alpha=0.1,            # L1 regularization
    reg_lambda=0.1,           # L2 regularization
    random_state=42,
    verbose=-1
)
model_reg.fit(X_train, y_train)

y_pred_train_reg = model_reg.predict(X_train)
y_pred_test_reg = model_reg.predict(X_test)

rmse_train_reg = np.sqrt(np.mean((y_pred_train_reg - y_train) ** 2))
rmse_test_reg = np.sqrt(np.mean((y_pred_test_reg - y_test) ** 2))

print(f"\n  Training RMSE: {rmse_train_reg:.3f}")
print(f"  Test RMSE:     {rmse_test_reg:.3f}")
print(f"  Overfit Gap:   {rmse_test_reg - rmse_train_reg:.3f}")


print("\n" + "="*70)
print("MODEL 3: Heavy Regularization (Max Generalization)")
print("="*70)

print("\nParameters:")
print("  n_estimators: 50          (fewer trees)")
print("  max_depth: 3              (very shallow)")
print("  learning_rate: 0.1        (higher = less precise fit)")
print("  min_child_samples: 100    (require lots of data per leaf)")
print("  subsample: 0.7")
print("  colsample_bytree: 0.7")
print("  reg_alpha: 0.5")
print("  reg_lambda: 0.5")

model_heavy = lgb.LGBMRegressor(
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
model_heavy.fit(X_train, y_train)

y_pred_train_heavy = model_heavy.predict(X_train)
y_pred_test_heavy = model_heavy.predict(X_test)

rmse_train_heavy = np.sqrt(np.mean((y_pred_train_heavy - y_train) ** 2))
rmse_test_heavy = np.sqrt(np.mean((y_pred_test_heavy - y_test) ** 2))

print(f"\n  Training RMSE: {rmse_train_heavy:.3f}")
print(f"  Test RMSE:     {rmse_test_heavy:.3f}")
print(f"  Overfit Gap:   {rmse_test_heavy - rmse_train_heavy:.3f}")


print("\n" + "="*70)
print("COMPARISON")
print("="*70)

print(f"\n{'Model':<30s} {'Train RMSE':<12s} {'Test RMSE':<12s} {'Gap':<8s}")
print("-" * 70)
print(f"{'1. Default':<30s} {rmse_train_default:>11.3f} {rmse_test_default:>11.3f} {rmse_test_default - rmse_train_default:>7.3f}")
print(f"{'2. Regularized':<30s} {rmse_train_reg:>11.3f} {rmse_test_reg:>11.3f} {rmse_test_reg - rmse_train_reg:>7.3f}")
print(f"{'3. Heavy Regularization':<30s} {rmse_train_heavy:>11.3f} {rmse_test_heavy:>11.3f} {rmse_test_heavy - rmse_train_heavy:>7.3f}")

# Find best test RMSE
best_test = min(rmse_test_default, rmse_test_reg, rmse_test_heavy)
best_model = ["Default", "Regularized", "Heavy Regularization"][
    [rmse_test_default, rmse_test_reg, rmse_test_heavy].index(best_test)
]

print(f"\n{'='*70}")
print(f"BEST MODEL: {best_model}")
print(f"  Test RMSE: {best_test:.3f}")
print(f"{'='*70}")


print("\n" + "="*70)
print("FEATURE IMPORTANCE (Regularized Model)")
print("="*70)

feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model_reg.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Features:")
for i, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']:25s}: {row['importance']:6.1f}")


print("\n" + "="*70)
print("VERDICT")
print("="*70)

# Compare best model to baseline from previous tests
baseline_rmse = 6.556  # From previous test

improvement = (baseline_rmse - best_test) / baseline_rmse * 100

print(f"\nBaseline (no volume/efficiency): {baseline_rmse:.3f}")
print(f"Best model (with features):      {best_test:.3f}")
print(f"Improvement:                      {improvement:+.2f}%")

if improvement >= 1.5:
    print(f"\n✓ SUCCESS! Regularization fixed overfitting")
    print(f"  Volume + Efficiency features now provide {improvement:+.1f}% improvement")
    print(f"\n  RECOMMENDATION:")
    print(f"  1. Use {best_model} hyperparameters for all models")
    print(f"  2. Add these features to train_auto.py")
    print(f"  3. Retrain all windows with regularization")
elif improvement >= 0.5:
    print(f"\n✓ MODEST SUCCESS: {improvement:+.1f}% improvement")
    print(f"  Regularization helped but not enough")
    print(f"\n  RECOMMENDATION:")
    print(f"  - Features may not generalize well to 2022")
    print(f"  - Try using more recent training data (2019-2021 only)")
    print(f"  - Or add matchup context features")
else:
    print(f"\n✗ STILL NO IMPROVEMENT: {improvement:+.1f}%")
    print(f"\n  The features don't generalize to 2022 data")
    print(f"  POSSIBLE CAUSES:")
    print(f"  1. Game changed significantly 2021→2022")
    print(f"  2. Features need matchup context (opponent defense)")
    print(f"  3. Need different training window (more recent)")

print(f"\n" + "="*70)
