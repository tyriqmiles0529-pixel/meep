# Context-Aware Meta-Learner Implementation Plan

## Overview
Replace simple averaging with a stacked ensemble meta-learner that learns optimal weights for the 27 window models based on player context.

## Architecture

### Input Features (Meta-Learner Training Data)
For each prediction, concatenate:

1. **Base Model Predictions** (27 features)
   - Prediction from each window model (1947-1949, 1950-1952, ..., 2025-2026)
   - These are the primary signals

2. **Prediction Statistics** (4 features)
   - Mean of all window predictions
   - Std deviation across windows
   - Min prediction across windows
   - Max prediction across windows

3. **Player Context** (8-12 features)
   - Position encoded (PG=0, SG=1, SF=2, PF=3, C=4)
   - Career games played (log-scaled)
   - Recent form: Last 5 games average
   - Usage rate (FGA + 0.44*FTA + AST) / team possessions
   - Age (if available)
   - Experience (seasons played)
   - Minutes per game (current season avg)
   - Team pace factor

4. **Opponent Context** (3-5 features)
   - Opponent defensive rating
   - Opponent pace factor
   - Home/Away indicator
   - Rest days since last game

**Total Features**: ~42-48 dimensions

### Target
- Actual player stat (points/rebounds/assists/threes)

## Implementation Steps

### Phase 1: Data Collection
```python
# File: train_meta_learner.py

1. Load all 27 window models
2. For each historical game in validation set (2024-2025 season):
   - Get 27 base predictions
   - Extract player context
   - Extract opponent context
   - Record actual outcome
3. Create meta-training dataset: (N_games, 42-48 features)
```

### Phase 2: Out-of-Fold (OOF) Predictions
**Critical for preventing leakage:**

```python
# Split 2024-2025 season into 5 folds
for fold in range(5):
    # Train base models on folds 0-3 (excluding current fold)
    # Predict on current fold to get OOF predictions
    # Store OOF predictions for meta-learner training
```

**Alternative** (if OOF too expensive):
- Train meta-learner on 2023-2024 season
- Base models predict on 2023-2024
- Validate meta-learner on 2024-2025

### Phase 3: Meta-Learner Training
```python
from lightgbm import LGBMRegressor

# Separate meta-learner per prop type
meta_models = {}

for prop in ['points', 'rebounds', 'assists', 'threes']:
    # Filter to prop-specific data
    X_meta = meta_features[prop]  # (N_samples, 42-48)
    y_meta = actual_stats[prop]   # (N_samples,)

    # Train LightGBM meta-learner
    meta_model = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,    # L1 regularization
        reg_lambda=1.0,   # L2 regularization
        random_state=42,
        objective='rmse'
    )

    meta_model.fit(
        X_meta, y_meta,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50)]
    )

    meta_models[prop] = meta_model
```

### Phase 4: Save & Integration
```python
class ContextAwareMetaLearner:
    def __init__(self):
        self.meta_models = {}  # prop -> LGBMRegressor

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def predict(self, window_predictions, player_context, prop_name):
        """
        Args:
            window_predictions: (n_samples, 27) - predictions from 27 windows
            player_context: (n_samples, 8-12) - player metadata
            prop_name: 'points', 'rebounds', 'assists', or 'threes'

        Returns:
            predictions: (n_samples,) - meta-learner predictions
        """
        # Compute prediction statistics
        pred_mean = window_predictions.mean(axis=1, keepdims=True)
        pred_std = window_predictions.std(axis=1, keepdims=True)
        pred_min = window_predictions.min(axis=1, keepdims=True)
        pred_max = window_predictions.max(axis=1, keepdims=True)

        # Concatenate all features
        X_meta = np.hstack([
            window_predictions,  # 27 features
            pred_mean, pred_std, pred_min, pred_max,  # 4 features
            player_context  # 8-12 features
        ])

        # Predict with meta-learner
        return self.meta_models[prop_name].predict(X_meta)
```

## File Structure

```
nba_predictor/
├── train_meta_learner.py          # Main training script
├── meta_learner_ensemble.py        # Class definition (already exists)
├── model_cache/
│   ├── player_models_*.pkl         # 27 window models
│   └── meta_learner_2024_2025.pkl  # Output: trained meta-learner
└── validation_data/
    └── 2024_2025_games.parquet     # Historical games for training
```

## Training Script Template

```python
# train_meta_learner.py

import pandas as pd
import numpy as np
from pathlib import Path
from ensemble_predictor import load_all_window_models, predict_with_window
from meta_learner_ensemble import ContextAwareMetaLearner
from lightgbm import LGBMRegressor
import lightgbm as lgb

def load_historical_games(season='2024-2025'):
    """Load historical games with actual outcomes"""
    # Load from your aggregated dataset or API
    df = pd.read_parquet('aggregated_nba_data.parquet')
    df = df[df['SEASON'] == season]
    return df

def extract_player_context(player_row):
    """Extract contextual features for a player"""
    return pd.Series({
        'position_encoded': position_map.get(player_row['POS'], 2),
        'career_games': np.log1p(player_row.get('GAMES_PLAYED', 50)),
        'recent_avg': player_row.get('L5_AVG', player_row.get('SEASON_AVG', 0)),
        'usage_rate': player_row.get('USAGE_RATE', 0.20),
        'age': player_row.get('AGE', 25),
        'experience': player_row.get('EXPERIENCE', 3),
        'mpg': player_row.get('MIN_PER_GAME', 30),
        'team_pace': player_row.get('TEAM_PACE', 100),
    })

def create_meta_training_data(games_df, window_models):
    """Generate meta-training dataset from historical games"""
    meta_data = []

    for idx, game in games_df.iterrows():
        for prop in ['points', 'rebounds', 'assists', 'threes']:
            # Get 27 base predictions
            window_preds = []
            for window_name, models in window_models.items():
                try:
                    # Create feature row for this player/game
                    X_game = create_features_for_game(game)
                    pred = predict_with_window(models, X_game, prop)
                    window_preds.append(pred[0])
                except:
                    continue

            if len(window_preds) < 20:  # Need at least 20 windows
                continue

            # Pad to 27 if some failed
            window_preds += [np.mean(window_preds)] * (27 - len(window_preds))

            # Get player context
            player_ctx = extract_player_context(game)

            # Get actual stat
            actual = game[prop.upper()]  # e.g., 'POINTS'

            meta_data.append({
                'prop': prop,
                'window_preds': window_preds,
                'player_context': player_ctx.values,
                'actual': actual
            })

    return pd.DataFrame(meta_data)

def train_meta_learner():
    """Main training function"""
    print("Loading window models...")
    window_models = load_all_window_models('model_cache')

    print("Loading historical games...")
    games = load_historical_games('2024-2025')

    # Split train/val (80/20)
    train_games = games.iloc[:int(len(games) * 0.8)]
    val_games = games.iloc[int(len(games) * 0.8):]

    print("Creating meta-training data...")
    train_meta = create_meta_training_data(train_games, window_models)
    val_meta = create_meta_training_data(val_games, window_models)

    # Train separate model per prop
    meta_learner = ContextAwareMetaLearner()

    for prop in ['points', 'rebounds', 'assists', 'threes']:
        print(f"\nTraining meta-learner for {prop}...")

        # Filter to prop
        train_prop = train_meta[train_meta['prop'] == prop]
        val_prop = val_meta[val_meta['prop'] == prop]

        # Build feature matrix
        X_train = np.column_stack([
            np.vstack(train_prop['window_preds'].values),
            np.vstack(train_prop['player_context'].values)
        ])
        y_train = train_prop['actual'].values

        X_val = np.column_stack([
            np.vstack(val_prop['window_preds'].values),
            np.vstack(val_prop['player_context'].values)
        ])
        y_val = val_prop['actual'].values

        # Add prediction statistics
        pred_stats_train = np.column_stack([
            np.mean(np.vstack(train_prop['window_preds'].values), axis=1),
            np.std(np.vstack(train_prop['window_preds'].values), axis=1),
            np.min(np.vstack(train_prop['window_preds'].values), axis=1),
            np.max(np.vstack(train_prop['window_preds'].values), axis=1),
        ])

        pred_stats_val = np.column_stack([
            np.mean(np.vstack(val_prop['window_preds'].values), axis=1),
            np.std(np.vstack(val_prop['window_preds'].values), axis=1),
            np.min(np.vstack(val_prop['window_preds'].values), axis=1),
            np.max(np.vstack(val_prop['window_preds'].values), axis=1),
        ])

        X_train = np.hstack([X_train, pred_stats_train])
        X_val = np.hstack([X_val, pred_stats_val])

        # Train LightGBM
        model = LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )

        # Evaluate
        train_rmse = np.sqrt(np.mean((model.predict(X_train) - y_train) ** 2))
        val_rmse = np.sqrt(np.mean((model.predict(X_val) - y_val) ** 2))

        print(f"  Train RMSE: {train_rmse:.3f}")
        print(f"  Val RMSE: {val_rmse:.3f}")

        meta_learner.meta_models[prop] = model

    # Save
    output_path = Path('model_cache/meta_learner_2024_2025.pkl')
    meta_learner.save(str(output_path))
    print(f"\n✅ Meta-learner saved to {output_path}")

if __name__ == '__main__':
    train_meta_learner()
```

## Expected Improvements

**Current**: Simple averaging RMSE
- Points: ~4.5-5.0
- Rebounds: ~2.5-3.0
- Assists: ~2.0-2.5
- Threes: ~1.5-2.0

**With Meta-Learner**: Expected 10-15% RMSE reduction
- Points: ~4.0-4.5
- Rebounds: ~2.2-2.7
- Assists: ~1.8-2.2
- Threes: ~1.3-1.8

## Next Steps

1. **Implement training script** (`train_meta_learner.py`)
2. **Load historical data** (use `aggregated_nba_data.parquet`)
3. **Train on 2023-2024** season (validate on early 2024-2025)
4. **Upload to Modal** volume with `modal volume put`
5. **Test** predictions improve vs simple averaging

## Timeline
- Day 1: Implement training script
- Day 2: Generate meta-training data
- Day 3: Train & validate meta-learner
- Day 4: Deploy to Modal & A/B test
