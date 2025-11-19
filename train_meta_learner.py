#!/usr/bin/env python
"""
Train Context-Aware Meta-Learner for NBA Props

Trains meta-learner on 2024-2025 season data to use for 2025-2026 predictions.

Usage:
    python train_meta_learner.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from typing import Dict, List
from ensemble_predictor import load_all_window_models, predict_with_window
from meta_learner_ensemble import ContextAwareMetaLearner, extract_player_context

# Configuration
TRAINING_SEASON = '2024-2025'  # Last complete season
DATA_FILE = 'PlayerStatistics.csv'  # Kaggle CSV
OUTPUT_FILE = 'model_cache/meta_learner_2025_2026.pkl'
MIN_SAMPLES_PER_PROP = 100  # Minimum games needed to train


def generate_features_for_prediction(df_games):
    """
    Generate features from PlayerStatistics.csv that match training features.
    This produces up to 182 features; early windows will align to their subset.

    Args:
        df_games: DataFrame with player game stats from Kaggle CSV

    Returns:
        DataFrame with ~80-180 features per game
    """

    # Sort by player and date
    df = df_games.sort_values(['playerId', 'gameDate']).copy()

    # Basic stats (already in CSV)
    features = pd.DataFrame(index=df.index)

    # Direct stats
    for col in ['points', 'assists', 'reboundsTotal', 'threePointersMade',
                'numMinutes', 'fieldGoalsAttempted', 'fieldGoalsMade',
                'freeThrowsAttempted', 'freeThrowsMade', 'turnovers',
                'steals', 'blocks', 'reboundsDefensive', 'reboundsOffensive']:
        if col in df.columns:
            features[col] = df[col].fillna(0)

    # Rolling averages (L3, L5, L7, L10)
    for window in [3, 5, 7, 10]:
        for stat in ['points', 'assists', 'reboundsTotal', 'threePointersMade', 'numMinutes']:
            if stat in df.columns:
                features[f'{stat}_L{window}_avg'] = df.groupby('playerId')[stat].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                ).fillna(0)

    # Shooting percentages
    if 'fieldGoalsMade' in df.columns and 'fieldGoalsAttempted' in df.columns:
        features['fg_pct'] = (df['fieldGoalsMade'] / df['fieldGoalsAttempted'].replace(0, 1)).fillna(0)
        features['fg_pct_L5'] = df.groupby('playerId')['fg_pct'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        ).fillna(0)

    if 'freeThrowsMade' in df.columns and 'freeThrowsAttempted' in df.columns:
        features['ft_pct'] = (df['freeThrowsMade'] / df['freeThrowsAttempted'].replace(0, 1)).fillna(0)

    # Usage proxy
    if 'fieldGoalsAttempted' in df.columns and 'freeThrowsAttempted' in df.columns:
        features['usage'] = (df['fieldGoalsAttempted'].fillna(0) +
                           df['freeThrowsAttempted'].fillna(0) * 0.44)
        features['usage_L5'] = df.groupby('playerId')['usage'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        ).fillna(0)

    # Per-minute stats
    if 'numMinutes' in df.columns:
        minutes_safe = df['numMinutes'].replace(0, 1)
        for stat in ['points', 'assists', 'reboundsTotal']:
            if stat in df.columns:
                features[f'{stat}_per_min'] = (df[stat] / minutes_safe).fillna(0)
                features[f'{stat}_per_min_L5'] = df.groupby('playerId')[f'{stat}_per_min'].transform(
                    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
                ).fillna(0)

    # Home/away
    if 'home' in df.columns:
        features['home'] = df['home'].fillna(0).astype(int)

    # Days rest (if we have gameDate)
    if 'gameDate' in df.columns:
        df['gameDate'] = pd.to_datetime(df['gameDate'])
        features['days_rest'] = df.groupby('playerId')['gameDate'].diff().dt.days.fillna(2).clip(0, 7)

    # Fill any remaining NaN
    features = features.fillna(0)

    return features


def load_player_statistics_csv(csv_path: str = 'PlayerStatistics.csv') -> pd.DataFrame:
    """
    Load PlayerStatistics.csv with robust column handling.

    Args:
        csv_path: Path to the Kaggle CSV file

    Returns:
        DataFrame with player game stats
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"PlayerStatistics.csv not found at {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows from {csv_path}")
    print(f"Columns: {list(df.columns)[:20]}...")
    return df


def filter_to_season(df: pd.DataFrame, season: str = '2024-2025') -> pd.DataFrame:
    """
    Filter DataFrame to a specific NBA season.

    Args:
        df: DataFrame with gameDate column
        season: Season string like '2024-2025'

    Returns:
        Filtered DataFrame
    """
    if 'gameDate' not in df.columns:
        raise ValueError("gameDate column required for season filtering")

    df['gameDate'] = pd.to_datetime(df['gameDate'], errors='coerce')

    # NBA season logic: Oct 2024 - Jun 2025 for '2024-2025'
    if season == '2024-2025':
        start_date = pd.to_datetime('2024-10-01')
        end_date = pd.to_datetime('2025-06-30')
    elif season == '2023-2024':
        start_date = pd.to_datetime('2023-10-01')
        end_date = pd.to_datetime('2024-06-30')
    else:
        # Generic logic for YYYY-YYYY format
        start_year = int(season.split('-')[0])
        start_date = pd.to_datetime(f'{start_year}-10-01')
        end_date = pd.to_datetime(f'{start_year+1}-06-30')

    season_df = df[
        (df['gameDate'] >= start_date) & (df['gameDate'] <= end_date)
    ].copy().reset_index(drop=True)

    print(f"Filtered to {season}: {len(season_df):,} rows")
    print(f"Date range: {season_df['gameDate'].min()} to {season_df['gameDate'].max()}")

    return season_df


def load_training_data(season: str = TRAINING_SEASON) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load PlayerStatistics.csv, filter to the target season, and generate features.

    Returns:
        Tuple of (df_raw, df_features)
            - df_raw: Original CSV with actuals for targets/context
            - df_features: Engineered features for window models
    """
    print(f"\n{'='*70}")
    print(f"LOADING TRAINING DATA: {season}")
    print(f"{'='*70}")

    # Load the Kaggle CSV
    df_raw = load_player_statistics_csv(DATA_FILE)

    # Filter to the target season
    df_raw = filter_to_season(df_raw, season)

    # Generate features for window models
    print("\n[*] Generating features for window models...")
    df_features = generate_features_for_prediction(df_raw)
    print(f"Generated {len(df_features.columns)} features")
    print(f"Feature sample: {list(df_features.columns)[:20]}")

    return df_raw, df_features


def create_features_for_game(game_row: pd.Series, window_models: Dict) -> pd.DataFrame:
    """
    Create feature row for a single game to feed into window models.

    This is a simplified version - you'll need to match your actual feature engineering.
    """
    features = pd.DataFrame([{
        'fieldGoalsAttempted': game_row.get('FGA', 0),
        'freeThrowsAttempted': game_row.get('FTA', 0),
        'assists': game_row.get('AST', 0),
        'rebounds': game_row.get('REB', 0),
        'threes': game_row.get('FG3M', 0),
        'points': game_row.get('PTS', 0),
        'numMinutes': game_row.get('MIN', 0),
        # Add other features your models expect...
    }])

    return features


def extract_context_from_game(game_row: pd.Series) -> Dict:
    """Extract player context features from game row"""
    return {
        'position_encoded': 2,  # Default to SF if unknown
        'usage_rate': game_row.get('USG_PCT', 0.20),
        'minutes_avg': game_row.get('MIN', 30),
        'is_home': 1 if game_row.get('LOCATION', '') == 'HOME' else 0,
        'games_played': game_row.get('GP', 50),
    }


def collect_window_predictions(games_df: pd.DataFrame, window_models: Dict, prop: str) -> Dict:
    """
    Collect predictions from all 27 windows for each game.

    Returns:
        Dict with 'predictions' (n_samples, 27) and 'actuals' (n_samples,)
    """
    print(f"\n  Collecting predictions for: {prop.upper()}")

    window_predictions = []
    player_contexts = []
    actuals = []

    prop_col_map = {
        'points': 'PTS',
        'rebounds': 'REB',
        'assists': 'AST',
        'threes': 'FG3M'
    }
    actual_col = prop_col_map.get(prop)

    if actual_col not in games_df.columns:
        print(f"    [!] Column {actual_col} not found in data, skipping {prop}")
        return None

    for idx, game in games_df.iterrows():
        # Get actual outcome
        actual = game.get(actual_col)
        if pd.isna(actual) or actual < 0:
            continue

        # Get predictions from each window
        preds_for_game = []
        for window_name, models in window_models.items():
            try:
                # Create features
                X_game = create_features_for_game(game, models)

                # Predict
                pred = predict_with_window(models, X_game, prop)
                if pred is not None:
                    preds_for_game.append(pred)
                else:
                    preds_for_game.append(0.0)  # Fallback
            except Exception as e:
                preds_for_game.append(0.0)  # Fallback on error

        # Need predictions from most windows
        if len(preds_for_game) < 20:
            continue

        # Pad to 27 if some failed
        while len(preds_for_game) < 27:
            preds_for_game.append(np.mean(preds_for_game))

        window_predictions.append(preds_for_game[:27])  # Ensure exactly 27

        # Get player context
        context = extract_context_from_game(game)
        player_contexts.append(context)

        actuals.append(actual)

    if len(actuals) < MIN_SAMPLES_PER_PROP:
        print(f"    [!] Not enough samples: {len(actuals)} < {MIN_SAMPLES_PER_PROP}")
        return None

    print(f"    ✓ Collected {len(actuals):,} valid samples")

    return {
        'window_predictions': np.array(window_predictions),  # (n, 27)
        'player_context': pd.DataFrame(player_contexts),     # (n, context_features)
        'actuals': np.array(actuals)                         # (n,)
    }


def train_meta_learner():
    """Main training function"""
    print(f"\n{'='*70}")
    print(f"META-LEARNER TRAINING")
    print(f"{'='*70}")
    print(f"  Training Season: {TRAINING_SEASON}")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"{'='*70}\n")

    # 1. Load window models
    print("Loading 27 window models...")
    window_models = load_all_window_models('model_cache')
    print(f"  ✓ Loaded {len(window_models)} windows")

    # 2. Load training data
    games_df = load_training_data(TRAINING_SEASON)

    # 3. Initialize meta-learner
    meta_learner = ContextAwareMetaLearner(n_windows=27)

    # 4. Train for each prop type
    props_to_train = ['points', 'rebounds', 'assists', 'threes']

    for prop in props_to_train:
        print(f"\n{'='*70}")
        print(f"PROP: {prop.upper()}")
        print(f"{'='*70}")

        # Collect window predictions
        data = collect_window_predictions(games_df, window_models, prop)

        if data is None:
            print(f"  ⚠ Skipping {prop} - insufficient data")
            continue

        # Train meta-learner with OOF
        metrics = meta_learner.fit_oof(
            window_predictions=data['window_predictions'],
            y_true=data['actuals'],
            player_context=data['player_context'],
            prop_name=prop
        )

        print(f"\n  ✅ Meta-learner trained for {prop}")
        print(f"     Improvement: {metrics['improvement_rmse_pct']:+.1f}% RMSE")

    # 5. Save meta-learner
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(exist_ok=True)

    meta_learner.save(str(output_path))

    print(f"\n{'='*70}")
    print(f"✅ TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Saved to: {output_path}")
    print(f"  Props trained: {len(meta_learner.meta_models)}")
    print(f"\nNext steps:")
    print(f"  1. Upload to Modal: modal volume put nba-models {output_path}")
    print(f"  2. Run analyzer with ensemble: modal run modal_analyzer.py")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    try:
        train_meta_learner()
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
