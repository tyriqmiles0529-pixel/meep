#!/usr/bin/env python3
"""
Train ML models for NBA prop predictions

This script will:
1. Load historical NBA player box score data
2. Engineer features (rolling averages, matchup stats, etc.)
3. Train LightGBM models for PTS, AST, REB, 3PM
4. Calibrate probabilities
5. Save models for use in RIQ MEEPING MACHINE
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, log_loss
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import pickle
import json
from datetime import datetime, timedelta
from pathlib import Path

# Configuration
DATA_DIR = Path("data")  # Where Kaggle data is extracted
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

STATS_TO_MODEL = ["points", "assists", "rebounds", "threepoint_goals"]
LOOKBACK_GAMES = [3, 5, 10, 15]  # Rolling window sizes
MIN_GAMES_FOR_TRAINING = 10

print("=" * 70)
print("NBA PROP MODEL TRAINING PIPELINE")
print("=" * 70)

# ====== FEATURE ENGINEERING ======
def create_rolling_features(df: pd.DataFrame, stat: str, windows: list = [3, 5, 10]) -> pd.DataFrame:
    """
    Create rolling average features for a stat

    Args:
        df: DataFrame with player game logs (must have 'date' and stat column)
        stat: Stat to create features for (e.g., 'points')
        windows: List of window sizes

    Returns:
        DataFrame with added rolling features
    """
    df = df.sort_values(['player_id', 'date'])

    for window in windows:
        # Rolling mean
        df[f'{stat}_avg_{window}g'] = df.groupby('player_id')[stat].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        # Rolling std
        df[f'{stat}_std_{window}g'] = df.groupby('player_id')[stat].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).std()
        )
        # Rolling max
        df[f'{stat}_max_{window}g'] = df.groupby('player_id')[stat].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).max()
        )

    # Trend (3-game vs 10-game)
    if 3 in windows and 10 in windows:
        df[f'{stat}_trend'] = (
            df[f'{stat}_avg_3g'] - df[f'{stat}_avg_10g']
        ) / (df[f'{stat}_avg_10g'] + 1e-6)

    return df


def create_opponent_features(df: pd.DataFrame, stat: str) -> pd.DataFrame:
    """
    Create opponent defense features

    Args:
        df: DataFrame with game logs
        stat: Stat to create features for

    Returns:
        DataFrame with opponent features
    """
    # Opponent average stat allowed (last 10 games)
    df[f'opp_{stat}_allowed_10g'] = df.groupby(['opponent_id', 'date'])[stat].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean()
    )

    return df


def create_situational_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create situational features (home/away, rest days, etc.)
    """
    # Home/Away
    df['is_home'] = (df['home_away'] == 'home').astype(int)

    # Rest days (days since last game)
    df['rest_days'] = df.groupby('player_id')['date'].diff().dt.days
    df['rest_days'] = df['rest_days'].fillna(3).clip(0, 7)

    # Back-to-back games
    df['is_b2b'] = (df['rest_days'] == 1).astype(int)

    # Season progress (game number in season)
    df['game_num'] = df.groupby(['player_id', 'season']).cumcount() + 1

    return df


# ====== MODEL TRAINING ======
def train_stat_model(
    df: pd.DataFrame,
    stat: str,
    target_col: str = None,
    test_seasons: list = ['2023-2024']
) -> dict:
    """
    Train a LightGBM model for a specific stat

    Args:
        df: DataFrame with features
        stat: Stat name (e.g., 'points')
        target_col: Name of target column (defaults to stat)
        test_seasons: Seasons to hold out for testing

    Returns:
        dict with model, features, and metrics
    """
    if target_col is None:
        target_col = stat

    print(f"\n{'='*70}")
    print(f"Training model for: {stat.upper()}")
    print(f"{'='*70}")

    # Feature columns
    feature_cols = [col for col in df.columns if (
        col.endswith('_avg_3g') or
        col.endswith('_avg_5g') or
        col.endswith('_avg_10g') or
        col.endswith('_std_3g') or
        col.endswith('_std_5g') or
        col.endswith('_max_3g') or
        col.endswith('_trend') or
        col.startswith('opp_') or
        col in ['is_home', 'rest_days', 'is_b2b', 'game_num']
    )]

    # Split train/test by season
    train_df = df[~df['season'].isin(test_seasons)].copy()
    test_df = df[df['season'].isin(test_seasons)].copy()

    # Drop rows with missing target or insufficient games
    train_df = train_df[train_df[target_col].notna()]
    train_df = train_df[train_df['game_num'] >= MIN_GAMES_FOR_TRAINING]
    test_df = test_df[test_df[target_col].notna()]
    test_df = test_df[test_df['game_num'] >= MIN_GAMES_FOR_TRAINING]

    # Drop rows with NaN features
    train_df = train_df.dropna(subset=feature_cols)
    test_df = test_df.dropna(subset=feature_cols)

    print(f"Train: {len(train_df):,} games | Test: {len(test_df):,} games")
    print(f"Features: {len(feature_cols)}")

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # Train LightGBM
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[test_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(50)]
    )

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Metrics
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print(f"\n📊 Metrics:")
    print(f"   Train MAE: {train_mae:.2f} | RMSE: {train_rmse:.2f}")
    print(f"   Test MAE:  {test_mae:.2f} | RMSE: {test_rmse:.2f}")

    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)

    print(f"\n🔑 Top 10 Features:")
    for idx, row in importance.head(10).iterrows():
        print(f"   {row['feature']:<30s} {row['importance']:>10.0f}")

    return {
        'model': model,
        'features': feature_cols,
        'metrics': {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse
        },
        'importance': importance,
        'stat': stat
    }


# ====== MAIN PIPELINE ======
def main():
    print("\n📂 Looking for data in:", DATA_DIR.absolute())

    # Check if data directory exists
    if not DATA_DIR.exists():
        print(f"❌ Data directory not found: {DATA_DIR}")
        print("\n💡 Instructions:")
        print("1. Set up Kaggle credentials (see KAGGLE_SETUP.md)")
        print("2. Run: python explore_dataset.py")
        print("3. Or manually download and extract to ./data/")
        return

    # Look for CSV files
    csv_files = list(DATA_DIR.glob("*.csv"))
    if not csv_files:
        print(f"❌ No CSV files found in {DATA_DIR}")
        return

    print(f"✅ Found {len(csv_files)} CSV file(s):")
    for f in csv_files:
        print(f"   - {f.name}")

    # TODO: Load and process the actual dataset
    # This is a template - adjust based on actual data structure

    print("\n⚠️ Dataset-specific processing needed!")
    print("Once you run explore_dataset.py and see the structure,")
    print("I'll customize this script to match your data format.")

    # Example structure (customize based on actual data):
    # df = pd.read_csv(DATA_DIR / "player_box_scores.csv")
    # df['date'] = pd.to_datetime(df['date'])
    # df = df.sort_values(['player_id', 'date'])
    #
    # for stat in STATS_TO_MODEL:
    #     df = create_rolling_features(df, stat, LOOKBACK_GAMES)
    #     df = create_opponent_features(df, stat)
    #
    # df = create_situational_features(df)
    #
    # models = {}
    # for stat in STATS_TO_MODEL:
    #     result = train_stat_model(df, stat, test_seasons=['2023-2024'])
    #     models[stat] = result
    #
    #     # Save model
    #     model_path = MODEL_DIR / f"{stat}_model.pkl"
    #     with open(model_path, 'wb') as f:
    #         pickle.dump(result, f)
    #     print(f"✅ Saved: {model_path}")


if __name__ == "__main__":
    main()
    print("\n" + "=" * 70)
    print("🎉 Training pipeline ready!")
    print("=" * 70)
