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
DATA_FILE = 'aggregated_nba_data.parquet'
OUTPUT_FILE = 'model_cache/meta_learner_2025_2026.pkl'
MIN_SAMPLES_PER_PROP = 100  # Minimum games needed to train


def load_training_data(season: str = TRAINING_SEASON) -> pd.DataFrame:
    """
    Load historical games from aggregated dataset.

    Returns:
        DataFrame with columns: player, game_date, points, rebounds, assists, threes, etc.
    """
    print(f"\n{'='*70}")
    print(f"LOADING TRAINING DATA: {season}")
    print(f"{'='*70}")

    data_path = Path(DATA_FILE)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")

    print(f"  Reading: {data_path}")
    df = pd.read_parquet(data_path)

    # Filter to training season
    if 'SEASON' in df.columns:
        df = df[df['SEASON'] == season]
    elif 'season' in df.columns:
        df = df[df['season'] == season]

    print(f"  Games loaded: {len(df):,}")
    print(f"  Date range: {df['GAME_DATE'].min()} to {df['GAME_DATE'].max()}" if 'GAME_DATE' in df.columns else "")

    return df


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
