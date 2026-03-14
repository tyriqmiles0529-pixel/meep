"""
Multi-Task Learning for Game Predictions
=========================================

Trains ONE model to predict BOTH moneyline and spread simultaneously:
- Moneyline: Win probability (classification)
- Spread: Point margin (regression)

Benefits:
1. Shared embeddings learn team strength indicators
2. Faster training (one TabNet pass vs two)
3. Better accuracy from joint learning
4. Natural correlation: strong teams win AND cover spreads

Architecture:
    Input Features (team stats, pace, ratings, etc.)
            ↓
    Shared TabNet Encoder (32-dim embeddings)
            ↓
    ┌───────────────┴───────────────┐
    ↓                               ↓
Moneyline Head                 Spread Head
(Classification)               (Regression)
    ↓                               ↓
LightGBM                       LightGBM
(Win Probability)              (Point Margin)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss, accuracy_score, mean_absolute_error
import pickle
from pathlib import Path

try:
    from pytorch_tabnet.tab_model import TabNetRegressor
    import torch
    import torch.nn as nn
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
    print("⚠️  Install: pip install pytorch-tabnet torch")


class GameMultiTaskModel:
    """
    Multi-task model for game predictions (moneyline + spread).

    Uses shared TabNet encoder with two task-specific LightGBM heads.
    """

    def __init__(self, use_gpu=False):
        """
        Args:
            use_gpu: Use GPU for TabNet training
        """
        self.use_gpu = use_gpu and TABNET_AVAILABLE

        # Shared TabNet parameters (learns features useful for BOTH tasks)
        self.tabnet_params = {
            'n_d': 32,                    # Embedding dimension
            'n_a': 32,
            'n_steps': 5,                 # Attention steps
            'gamma': 1.5,
            'n_independent': 2,
            'n_shared': 3,                # More shared = better multi-task
            'lambda_sparse': 1e-3,        # Stronger sparsity for small dataset
            'momentum': 0.3,
            'clip_value': 1.0,
            'mask_type': 'entmax',
            'verbose': 1,
            'device_name': 'cuda' if self.use_gpu else 'cpu'
        }

        if TABNET_AVAILABLE:
            self.tabnet_params['optimizer_fn'] = torch.optim.AdamW
            self.tabnet_params['optimizer_params'] = {'lr': 1e-2, 'weight_decay': 1e-4}
            self.tabnet_params['scheduler_fn'] = torch.optim.lr_scheduler.ReduceLROnPlateau
            self.tabnet_params['scheduler_params'] = {'mode': 'min', 'patience': 5, 'factor': 0.5}

        # Task-specific LightGBM parameters
        self.lgbm_classifier_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'verbose': -1
        }

        self.lgbm_regressor_params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'verbose': -1
        }

        self.tabnet = None                # Shared encoder
        self.moneyline_model = None       # Classification head
        self.spread_model = None          # Regression head
        self.feature_names = None

    def fit(self, X, y_moneyline, y_spread,
            X_val=None, y_moneyline_val=None, y_spread_val=None,
            epochs=30, batch_size=2048):
        """
        Train multi-task game model.

        Args:
            X: Training features (DataFrame)
            y_moneyline: Binary labels (1 = home team wins, 0 = away wins)
            y_spread: Point margin (positive = home team wins by X points)
            X_val: Validation features
            y_moneyline_val: Validation moneyline labels
            y_spread_val: Validation spread values
            epochs: TabNet epochs
            batch_size: Batch size
        """
        self.feature_names = list(X.columns)
        X_np = X.values.astype(np.float32)

        if X_val is not None:
            X_val_np = X_val.values.astype(np.float32)

        print("\n" + "="*70)
        print("GAME MULTI-TASK TRAINING (Moneyline + Spread)")
        print("="*70)

        if not TABNET_AVAILABLE:
            print("⚠️  TabNet not available - using LightGBM only")
            return self._fit_lgbm_only(X, y_moneyline, y_spread, X_val, y_moneyline_val, y_spread_val)

        # ===================================================================
        # PHASE 1: Train Shared TabNet Encoder
        # ===================================================================
        print("\nPHASE 1: Training Shared TabNet Encoder")
        print("-" * 70)

        # Create composite target for multi-task learning
        # Normalize both tasks to 0-1 range, then average
        # This lets TabNet learn features useful for BOTH tasks

        # Moneyline: already 0-1
        y_ml_norm = y_moneyline

        # Spread: normalize to 0-1 range
        spread_min, spread_max = y_spread.min(), y_spread.max()
        y_spread_norm = (y_spread - spread_min) / (spread_max - spread_min + 1e-8)

        # Composite target (50% moneyline, 50% spread)
        y_composite = 0.5 * y_ml_norm + 0.5 * y_spread_norm

        if X_val is not None:
            y_ml_val_norm = y_moneyline_val
            y_spread_val_norm = (y_spread_val - spread_min) / (spread_max - spread_min + 1e-8)
            y_val_composite = 0.5 * y_ml_val_norm + 0.5 * y_spread_val_norm
            eval_set = [(X_val_np, y_val_composite)]
        else:
            eval_set = None

        # Train shared TabNet
        print("Training TabNet on composite target (learns patterns for both tasks)...")
        self.tabnet = TabNetRegressor(**self.tabnet_params)
        self.tabnet.fit(
            X_np, y_composite,
            eval_set=eval_set,
            max_epochs=epochs,
            patience=10,
            batch_size=batch_size,
            virtual_batch_size=256,
            eval_metric=['mae']
        )

        print(f"✓ Shared TabNet trained - 32-dim embeddings")

        # ===================================================================
        # PHASE 2: Extract Shared Embeddings
        # ===================================================================
        print("\nPHASE 2: Extracting Shared Embeddings")
        print("-" * 70)

        _, train_embeddings = self.tabnet.predict(X_np, return_embeddings=True)
        X_train_combined = np.hstack([X_np, train_embeddings])
        print(f"✓ Train: {X_np.shape} → {X_train_combined.shape} (+{train_embeddings.shape[1]} embeddings)")

        if X_val is not None:
            _, val_embeddings = self.tabnet.predict(X_val_np, return_embeddings=True)
            X_val_combined = np.hstack([X_val_np, val_embeddings])
            print(f"✓ Val: {X_val_np.shape} → {X_val_combined.shape}")

        # ===================================================================
        # PHASE 3: Train Task-Specific LightGBM Models
        # ===================================================================
        print("\nPHASE 3: Training Task-Specific Models")
        print("-" * 70)

        # Moneyline (Classification)
        print("\nMONEYLINE (Win Probability):")
        self.moneyline_model = lgb.LGBMClassifier(**self.lgbm_classifier_params, n_estimators=500)

        if X_val is not None:
            self.moneyline_model.fit(
                X_train_combined, y_moneyline,
                eval_set=[(X_val_combined, y_moneyline_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )

            # Metrics
            y_pred_proba = self.moneyline_model.predict_proba(X_val_combined)[:, 1]
            y_pred_binary = (y_pred_proba >= 0.5).astype(int)

            logloss = log_loss(y_moneyline_val, y_pred_proba)
            accuracy = accuracy_score(y_moneyline_val, y_pred_binary)

            print(f"  Validation Logloss: {logloss:.4f}")
            print(f"  Validation Accuracy: {accuracy:.2%}")
        else:
            self.moneyline_model.fit(X_train_combined, y_moneyline)

        # Spread (Regression)
        print("\nSPREAD (Point Margin):")
        self.spread_model = lgb.LGBMRegressor(**self.lgbm_regressor_params, n_estimators=500)

        if X_val is not None:
            self.spread_model.fit(
                X_train_combined, y_spread,
                eval_set=[(X_val_combined, y_spread_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )

            # Metrics
            y_pred = self.spread_model.predict(X_val_combined)
            mae = mean_absolute_error(y_spread_val, y_pred)
            rmse = np.sqrt(np.mean((y_spread_val - y_pred) ** 2))

            print(f"  Validation MAE: {mae:.3f}")
            print(f"  Validation RMSE: {rmse:.3f}")
        else:
            self.spread_model.fit(X_train_combined, y_spread)

        print("\n" + "="*70)
        print("GAME MULTI-TASK TRAINING COMPLETE")
        print("="*70)

        metrics = {}
        if X_val is not None:
            metrics = {
                'moneyline_logloss': logloss,
                'moneyline_accuracy': accuracy,
                'spread_mae': mae,
                'spread_rmse': rmse
            }

        return metrics

    def _fit_lgbm_only(self, X, y_moneyline, y_spread, X_val, y_moneyline_val, y_spread_val):
        """Fallback if TabNet unavailable"""
        X_np = X.values
        X_val_np = X_val.values if X_val is not None else None

        # Moneyline
        self.moneyline_model = lgb.LGBMClassifier(**self.lgbm_classifier_params, n_estimators=500)
        if X_val is not None:
            self.moneyline_model.fit(
                X_np, y_moneyline,
                eval_set=[(X_val_np, y_moneyline_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
        else:
            self.moneyline_model.fit(X_np, y_moneyline)

        # Spread
        self.spread_model = lgb.LGBMRegressor(**self.lgbm_regressor_params, n_estimators=500)
        if X_val is not None:
            self.spread_model.fit(
                X_np, y_spread,
                eval_set=[(X_val_np, y_spread_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
        else:
            self.spread_model.fit(X_np, y_spread)

        return {}

    def predict_moneyline(self, X):
        """Predict win probability (0-1)"""
        X_np = X.values.astype(np.float32) if hasattr(X, 'values') else X.astype(np.float32)

        if self.tabnet is not None:
            _, embeddings = self.tabnet.predict(X_np, return_embeddings=True)
            X_combined = np.hstack([X_np, embeddings])
        else:
            X_combined = X_np

        return self.moneyline_model.predict_proba(X_combined)[:, 1]

    def predict_spread(self, X):
        """Predict point margin (positive = home team wins)"""
        X_np = X.values.astype(np.float32) if hasattr(X, 'values') else X.astype(np.float32)

        if self.tabnet is not None:
            _, embeddings = self.tabnet.predict(X_np, return_embeddings=True)
            X_combined = np.hstack([X_np, embeddings])
        else:
            X_combined = X_np

        return self.spread_model.predict(X_combined)

    def predict(self, X):
        """Predict both moneyline and spread"""
        return {
            'moneyline_prob': self.predict_moneyline(X),
            'spread': self.predict_spread(X)
        }

    def save(self, filepath):
        """Save multi-task game model"""
        filepath = Path(filepath)

        # Save TabNet
        tabnet_path = None
        if self.tabnet is not None:
            tabnet_path = str(filepath.parent / f"{filepath.stem}_tabnet.zip")
            self.tabnet.save_model(tabnet_path)

        # Save everything else
        state = {
            'moneyline_model': self.moneyline_model,
            'spread_model': self.spread_model,
            'feature_names': self.feature_names,
            'tabnet_path': tabnet_path,
            'tabnet_params': self.tabnet_params
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        print(f"✓ Saved game multi-task model: {filepath}")

    @staticmethod
    def load(filepath):
        """Load multi-task game model"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        model = GameMultiTaskModel()
        model.moneyline_model = state['moneyline_model']
        model.spread_model = state['spread_model']
        model.feature_names = state['feature_names']
        model.tabnet_params = state.get('tabnet_params', {})

        # Load TabNet
        if state['tabnet_path'] and Path(state['tabnet_path']).exists():
            model.tabnet = TabNetRegressor(**model.tabnet_params)
            model.tabnet.load_model(state['tabnet_path'])

        return model


# ===================================================================
# Example Usage
# ===================================================================
"""
# Training
model = GameMultiTaskModel(use_gpu=True)

# Prepare data
y_moneyline = (games_df['home_score'] > games_df['away_score']).astype(int)
y_spread = games_df['home_score'] - games_df['away_score']

# Train
metrics = model.fit(
    X_train, y_moneyline_train, y_spread_train,
    X_val, y_moneyline_val, y_spread_val,
    epochs=30
)

model.save('models/game_multi_task.pkl')

# Prediction
model = GameMultiTaskModel.load('models/game_multi_task.pkl')

# Both predictions
preds = model.predict(X_test)
print(f"Win prob: {preds['moneyline_prob'][0]:.1%}")
print(f"Spread: {preds['spread'][0]:.1f} points")

# Individual predictions
win_prob = model.predict_moneyline(X_test)
spread = model.predict_spread(X_test)
"""
