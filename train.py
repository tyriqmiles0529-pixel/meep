#!/usr/bin/env python3
"""
AUTOMATED NBA PROP MODEL TRAINING

This script automatically:
1. Downloads Kaggle dataset (if needed)
2. Processes and cleans data
3. Engineers features
4. Trains LightGBM models for PTS, AST, REB, 3PM
5. Saves models for use in RIQ analyzer

Just run: python train_auto.py

Requirements:
- Kaggle credentials set up (run: python setup_kaggle.py)
- Dependencies: pip install kagglehub pandas numpy lightgbm scikit-learn
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import required libraries
try:
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
    import lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error, mean_squared_error
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("\nInstall with:")
    print("   pip install kagglehub pandas numpy lightgbm scikit-learn")
    sys.exit(1)

# Configuration
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

DATASET_NAME = "eoinamoore/historical-nba-data-and-player-box-scores"
STATS_TO_MODEL = ["points", "assists", "rebounds", "threepoint_goals"]
TEST_SEASONS = ['2023-24', '2023-2024']  # Hold out for testing

print("=" * 80)
print("🤖 AUTOMATED NBA PROP MODEL TRAINING")
print("=" * 80)
print()

# ========== STEP 1: DOWNLOAD DATASET ==========
print("📥 STEP 1: Downloading dataset from Kaggle...")
print("-" * 80)

try:
    # Hardcoded Kaggle credentials (for venv compatibility - can't import from other files)
    KAGGLE_KEY = "YOUR_KEY_HERE"  # Your Kaggle key - run ./setup_once.sh to inject
    KAGGLE_USERNAME = ""  # Optional - leave empty if not needed

    if KAGGLE_KEY and KAGGLE_KEY != "YOUR_KEY_HERE":
        os.environ['KAGGLE_KEY'] = KAGGLE_KEY
        if KAGGLE_USERNAME:
            os.environ['KAGGLE_USERNAME'] = KAGGLE_USERNAME
        print("✅ Kaggle credentials loaded (hardcoded)")
    else:
        # Fall back to kaggle.json
        kaggle_dir = Path.home() / ".kaggle" / "kaggle.json"
        if not kaggle_dir.exists():
            print("❌ Kaggle credentials not found!")
            print("\nEdit this file (train_auto.py) around line 62 and add your Kaggle key")
            print("Or use: python setup_kaggle.py")
            sys.exit(1)
        print("✅ Kaggle credentials found in ~/.kaggle/kaggle.json")

    # Download dataset
    print(f"📦 Downloading: {DATASET_NAME}")
    print("   (This may take a few minutes on first run...)")

    dataset_path = kagglehub.dataset_download(DATASET_NAME)
    print(f"✅ Dataset downloaded to: {dataset_path}")

    # List available files
    csv_files = list(Path(dataset_path).glob("*.csv"))
    print(f"\n📁 Found {len(csv_files)} CSV file(s):")
    for f in csv_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"   - {f.name:<50s} ({size_mb:.1f} MB)")

except Exception as e:
    print(f"❌ Error downloading dataset: {e}")
    print("\nTroubleshooting:")
    print("   1. Check your internet connection")
    print("   2. Verify Kaggle credentials: python setup_kaggle.py")
    print("   3. Make sure you have Kaggle account and accepted dataset terms")
    sys.exit(1)

print()

# ========== STEP 2: LOAD AND PROCESS DATA ==========
print("📊 STEP 2: Loading and processing data...")
print("-" * 80)

try:
    # Find the main player box scores file
    # Common names: player_box_scores.csv, game_stats.csv, etc.
    box_score_files = [
        f for f in csv_files
        if any(term in f.name.lower() for term in ['player', 'box', 'score', 'game', 'stats'])
    ]

    if not box_score_files:
        print("⚠️  Could not auto-detect box score file. Available files:")
        for f in csv_files:
            print(f"   - {f.name}")
        print("\nPlease manually specify the file containing player game stats.")
        sys.exit(1)

    # Use the first matching file (or largest if multiple)
    data_file = max(box_score_files, key=lambda f: f.stat().st_size)
    print(f"📖 Loading: {data_file.name}")

    # Load data
    df = pd.read_csv(data_file)
    print(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns")

    # Show column names (for debugging)
    print(f"\n📋 Columns ({len(df.columns)}):")
    for i, col in enumerate(df.columns[:20], 1):  # Show first 20
        print(f"   {i:2d}. {col}")
    if len(df.columns) > 20:
        print(f"   ... and {len(df.columns) - 20} more")

    # Auto-detect column mappings
    print("\n🔍 Auto-detecting column mappings...")

    col_map = {}

    # Try to find key columns
    def find_column(patterns, df_cols):
        for pattern in patterns:
            matches = [col for col in df_cols if pattern.lower() in col.lower()]
            if matches:
                return matches[0]
        return None

    # Map critical columns
    col_map['player_id'] = find_column(['player_id', 'playerId', 'player'], df.columns)
    col_map['player_name'] = find_column(['player_name', 'playerName', 'name'], df.columns)
    col_map['date'] = find_column(['date', 'game_date', 'gameDate'], df.columns)
    col_map['season'] = find_column(['season'], df.columns)
    col_map['team_id'] = find_column(['team_id', 'teamId', 'team'], df.columns)
    col_map['opponent_id'] = find_column(['opponent', 'opp', 'vs'], df.columns)
    col_map['home_away'] = find_column(['home', 'location', 'venue'], df.columns)

    # Map stat columns
    col_map['points'] = find_column(['points', 'pts', 'PTS'], df.columns)
    col_map['assists'] = find_column(['assists', 'ast', 'AST'], df.columns)
    col_map['rebounds'] = find_column(['rebounds', 'reb', 'REB', 'total_rebounds'], df.columns)
    col_map['threepoint_goals'] = find_column(['three', '3pt', '3pm', 'threepoint'], df.columns)

    print("\n📌 Detected mappings:")
    for key, val in col_map.items():
        status = "✅" if val else "❌"
        print(f"   {status} {key:<20s} → {val if val else 'NOT FOUND'}")

    # Check if we have minimum required columns
    required = ['date', 'points', 'assists', 'rebounds']
    missing = [k for k in required if not col_map.get(k)]

    if missing:
        print(f"\n❌ Missing required columns: {missing}")
        print("\n💡 The dataset structure is different than expected.")
        print("   Please check the data and manually adjust column mappings.")
        print("\n   You can:")
        print("   1. Inspect the data: python explore_dataset.py")
        print("   2. Manually edit train_auto.py to fix column names")
        sys.exit(1)

    # Rename columns to standard names
    rename_map = {v: k for k, v in col_map.items() if v}
    df = df.rename(columns=rename_map)

    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Sort by player and date
    if 'player_id' in df.columns:
        df = df.sort_values(['player_id', 'date'])
    elif 'player_name' in df.columns:
        df = df.sort_values(['player_name', 'date'])

    print(f"\n✅ Data processed: {len(df):,} games")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")

    if 'season' in df.columns:
        seasons = df['season'].unique()
        print(f"   Seasons: {', '.join(map(str, sorted(seasons)))}")

except Exception as e:
    print(f"❌ Error processing data: {e}")
    import traceback
    traceback.print_exc()
    print("\n💡 The dataset structure may differ from expected format.")
    print("   Run: python explore_dataset.py")
    print("   To inspect the actual data structure.")
    sys.exit(1)

print()

# ========== STEP 3: FEATURE ENGINEERING ==========
print("🔧 STEP 3: Engineering features...")
print("-" * 80)

try:
    def create_rolling_features(df, stat, windows=[3, 5, 10]):
        """Create rolling average features"""
        group_col = 'player_id' if 'player_id' in df.columns else 'player_name'

        for window in windows:
            # Shift by 1 to avoid leakage (don't use current game)
            df[f'{stat}_avg_{window}g'] = df.groupby(group_col)[stat].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
            df[f'{stat}_std_{window}g'] = df.groupby(group_col)[stat].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).std()
            )
            df[f'{stat}_max_{window}g'] = df.groupby(group_col)[stat].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).max()
            )

        # Trend (3g vs 10g)
        if 3 in windows and 10 in windows:
            df[f'{stat}_trend'] = (df[f'{stat}_avg_3g'] - df[f'{stat}_avg_10g']) / (df[f'{stat}_avg_10g'] + 1e-6)

        return df

    # Create features for each stat
    for stat in ['points', 'assists', 'rebounds']:
        if stat in df.columns:
            print(f"   Creating features for {stat}...")
            df = create_rolling_features(df, stat, windows=[3, 5, 10])

    # Handle threepoint_goals if available
    if 'threepoint_goals' in df.columns:
        print(f"   Creating features for threepoint_goals...")
        df = create_rolling_features(df, 'threepoint_goals', windows=[3, 5, 10])

    # Situational features
    if 'home_away' in df.columns:
        df['is_home'] = df['home_away'].str.lower().str.contains('home').astype(int)

    # Game number in season
    group_cols = []
    if 'player_id' in df.columns:
        group_cols.append('player_id')
    elif 'player_name' in df.columns:
        group_cols.append('player_name')
    if 'season' in df.columns:
        group_cols.append('season')

    if group_cols:
        df['game_num'] = df.groupby(group_cols).cumcount() + 1

    print(f"✅ Features created: {len([c for c in df.columns if '_avg_' in c or '_std_' in c])} rolling features")

except Exception as e:
    print(f"❌ Error creating features: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ========== STEP 4: TRAIN MODELS ==========
print("🎯 STEP 4: Training models...")
print("-" * 80)

models = {}

for stat in STATS_TO_MODEL:
    if stat not in df.columns:
        print(f"⏭️  Skipping {stat} (not in dataset)")
        continue

    print(f"\n{'='*80}")
    print(f"Training: {stat.upper()}")
    print(f"{'='*80}")

    try:
        # Feature columns
        feature_cols = [col for col in df.columns if (
            col.endswith('_avg_3g') or
            col.endswith('_avg_5g') or
            col.endswith('_avg_10g') or
            col.endswith('_std_3g') or
            col.endswith('_std_5g') or
            col.endswith('_max_3g') or
            col.endswith('_trend') or
            col in ['is_home', 'game_num']
        )]

        # Filter to rows with valid data
        train_df = df.copy()
        train_df = train_df[train_df[stat].notna()]
        train_df = train_df.dropna(subset=feature_cols)
        train_df = train_df[train_df['game_num'] >= 5]  # Need at least 5 games

        # Split by season if available
        if 'season' in train_df.columns:
            test_mask = train_df['season'].astype(str).isin(TEST_SEASONS)
            train_data = train_df[~test_mask]
            test_data = train_df[test_mask]
        else:
            # Use last 20% as test
            split_idx = int(len(train_df) * 0.8)
            train_data = train_df.iloc[:split_idx]
            test_data = train_df.iloc[split_idx:]

        X_train = train_data[feature_cols].astype('float32')
        y_train = train_data[stat]
        X_test = test_data[feature_cols].astype('float32')
        y_test = test_data[stat]

        print(f"Train: {len(X_train):,} games | Test: {len(X_test):,} games")
        print(f"Features: {len(feature_cols)}")

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

        lgb_train = lgb.Dataset(X_train, label=y_train, free_raw_data=True)
        lgb_test = lgb.Dataset(X_test, label=y_test, reference=lgb_train, free_raw_data=True)

        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=500,
            valid_sets=[lgb_test],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(50)]
        )

        # Evaluate
        y_pred_test = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

        print(f"\n📊 Test Metrics:")
        print(f"   MAE:  {mae:.2f}")
        print(f"   RMSE: {rmse:.2f}")

        # Feature importance
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)

        print(f"\n🔑 Top 5 Features:")
        for idx, row in importance.head(5).iterrows():
            print(f"   {row['feature']:<30s} {row['importance']:>10.0f}")

        # Save model
        model_info = {
            'model': model,
            'features': feature_cols,
            'stat': stat,
            'metrics': {'test_mae': mae, 'test_rmse': rmse},
            'importance': importance,
            'trained_date': datetime.now().isoformat(),
            'train_size': len(X_train),
            'test_size': len(X_test)
        }

        model_path = MODEL_DIR / f"{stat}_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model_info, f)

        print(f"✅ Saved: {model_path}")

        models[stat] = model_info

    except Exception as e:
        print(f"❌ Error training {stat}: {e}")
        import traceback
        traceback.print_exc()
        continue

print()

# ========== STEP 5: SAVE MODEL REGISTRY ==========
print("📝 STEP 5: Saving model registry...")
print("-" * 80)

registry = {}
for stat, info in models.items():
    registry[stat] = {
        'version': 'v1.0',
        'trained_date': info['trained_date'],
        'test_mae': info['metrics']['test_mae'],
        'test_rmse': info['metrics']['test_rmse'],
        'train_size': info['train_size'],
        'test_size': info['test_size'],
        'features': info['features'],
        'model_file': f"{stat}_model.pkl"
    }

registry_path = MODEL_DIR / "model_registry.json"
with open(registry_path, 'w') as f:
    json.dump(registry, f, indent=2)

print(f"✅ Saved registry: {registry_path}")

print()
print("=" * 80)
print("🎉 TRAINING COMPLETE!")
print("=" * 80)
print(f"\n📦 Models saved to: {MODEL_DIR.absolute()}")
print(f"\n📊 Summary:")
for stat, info in models.items():
    print(f"   {stat:<20s} MAE: {info['metrics']['test_mae']:.2f}  RMSE: {info['metrics']['test_rmse']:.2f}")

print(f"\n🚀 Next steps:")
print(f"   1. Review models in {MODEL_DIR}/")
print(f"   2. Check model_registry.json for details")
print(f"   3. Integrate with RIQ analyzer (see MODEL_INTEGRATION.md)")
print()