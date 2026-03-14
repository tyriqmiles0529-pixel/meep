"""
Multi-Task Learning for Player Props
=====================================

Trains ONE model to predict ALL player stats simultaneously:
- Minutes, Points, Rebounds, Assists, Threes

Benefits:
1. Shared embeddings learn correlations between stats
2. 5x faster training (one pass vs five)
3. Better accuracy from joint learning
4. Natural support for combo props (PRA = Points + Rebounds + Assists)

Architecture:
    Input Features (150+ features)
            ↓
    Shared TabNet Encoder (24-dim embeddings)
            ↓
    ┌───────┼───────┬─────────┬─────────┐
    ↓       ↓       ↓         ↓         ↓
  Minutes Points Rebounds Assists  Threes
   Head    Head    Head     Head    Head
    ↓       ↓       ↓         ↓         ↓
  LightGBM LightGBM LightGBM LightGBM LightGBM
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import pickle
from pathlib import Path

try:
    from pytorch_tabnet.tab_model import TabNetRegressor
    import torch
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
    print("⚠️  Install: pip install pytorch-tabnet torch")


class MultiTaskPlayerModel:
    """
    Single model for all 5 player props using shared TabNet embeddings.
    """

    def __init__(self, use_gpu=False):
        """
        Args:
            use_gpu: Use GPU for TabNet training
        """
        self.use_gpu = use_gpu and TABNET_AVAILABLE
        self.prop_names = ['minutes', 'points', 'rebounds', 'assists', 'threes']

        # Shared TabNet for ALL props (learns common patterns)
        self.tabnet_params = {
            'n_d': 32,                    # Embedding dimension
            'n_a': 32,
            'n_steps': 5,                 # Attention steps
            'gamma': 1.5,
            'n_independent': 2,
            'n_shared': 3,                # More shared layers = better multi-task
            'lambda_sparse': 1e-4,
            'momentum': 0.3,
            'clip_value': 2.0,
            'mask_type': 'entmax',
            'verbose': 1,
            'device_name': 'cuda' if self.use_gpu else 'cpu'
        }

        if TABNET_AVAILABLE:
            self.tabnet_params['optimizer_fn'] = torch.optim.AdamW
            self.tabnet_params['optimizer_params'] = {'lr': 2e-2, 'weight_decay': 1e-5}
            self.tabnet_params['scheduler_fn'] = torch.optim.lr_scheduler.ReduceLROnPlateau
            self.tabnet_params['scheduler_params'] = {'mode': 'min', 'patience': 5, 'factor': 0.5}

        # Separate LightGBM for each prop (uses shared embeddings + raw features)
        self.lgbm_params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.85,
            'bagging_fraction': 0.85,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'verbose': -1
        }

        self.tabnet = None                      # Shared encoder
        self.lgbm_models = {}                   # One per prop
        self.sigma_models = {}                  # Uncertainty models
        self.feature_names = None

    def fit(self, X, y_dict, X_val=None, y_val_dict=None, epochs=50, batch_size=4096):
        """
        Train multi-task model.

        Args:
            X: Training features (DataFrame)
            y_dict: {prop_name: y_array} for all 5 props
            X_val: Validation features
            y_val_dict: Validation targets
            epochs: TabNet epochs
            batch_size: Batch size

        Example:
            y_dict = {
                'minutes': df['minutes'].values,
                'points': df['points'].values,
                'rebounds': df['rebounds'].values,
                'assists': df['assists'].values,
                'threes': df['threes'].values
            }
        """
        self.feature_names = list(X.columns)
        X_np = X.values.astype(np.float32)

        if X_val is not None:
            X_val_np = X_val.values.astype(np.float32)

        # === PHASE 1: Train Shared TabNet ===
        # We'll use a weighted combination of all props as the target
        print("\n" + "="*70)
        print("PHASE 1: Training Shared TabNet Encoder")
        print("="*70)

        if not TABNET_AVAILABLE:
            print("⚠️  TabNet not available - using LightGBM only")
            return self._fit_lgbm_only(X, y_dict, X_val, y_val_dict)

        # Create composite target (weighted average for multi-task learning)
        # Normalize each prop to 0-1 range, then average
        y_composite = np.zeros(len(X))
        weights = {'minutes': 0.15, 'points': 0.30, 'rebounds': 0.20,
                   'assists': 0.20, 'threes': 0.15}

        for prop, weight in weights.items():
            y_normalized = (y_dict[prop] - y_dict[prop].min()) / (y_dict[prop].max() - y_dict[prop].min() + 1e-8)
            y_composite += weight * y_normalized

        if X_val is not None:
            y_val_composite = np.zeros(len(X_val))
            for prop, weight in weights.items():
                y_val_norm = (y_val_dict[prop] - y_dict[prop].min()) / (y_dict[prop].max() - y_dict[prop].min() + 1e-8)
                y_val_composite += weight * y_val_norm
            eval_set = [(X_val_np, y_val_composite)]
        else:
            eval_set = None

        # Train TabNet
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

        print(f"✓ TabNet trained - learned 32-dim embeddings")

        # === PHASE 2: Extract Shared Embeddings ===
        print("\n" + "="*70)
        print("PHASE 2: Extracting Shared Embeddings")
        print("="*70)

        _, train_embeddings = self.tabnet.predict(X_np, return_embeddings=True)
        print(f"✓ Train embeddings: {train_embeddings.shape}")

        if X_val is not None:
            _, val_embeddings = self.tabnet.predict(X_val_np, return_embeddings=True)
            print(f"✓ Val embeddings: {val_embeddings.shape}")

        # Combine raw features + embeddings
        X_combined = np.hstack([X_np, train_embeddings])
        if X_val is not None:
            X_val_combined = np.hstack([X_val_np, val_embeddings])

        # === PHASE 3: Train Task-Specific LightGBM Models ===
        print("\n" + "="*70)
        print("PHASE 3: Training Task-Specific LightGBM Models")
        print("="*70)

        metrics = {}
        for prop in self.prop_names:
            print(f"\nTraining {prop.upper()} model...")

            # Main model
            lgbm = lgb.LGBMRegressor(**self.lgbm_params, n_estimators=500)

            if X_val is not None:
                lgbm.fit(
                    X_combined, y_dict[prop],
                    eval_set=[(X_val_combined, y_val_dict[prop])],
                    callbacks=[lgb.early_stopping(50, verbose=False)]
                )

                # Metrics
                y_pred = lgbm.predict(X_val_combined)
                mae = mean_absolute_error(y_val_dict[prop], y_pred)
                print(f"  ✓ Validation MAE: {mae:.3f}")
                metrics[prop] = mae
            else:
                lgbm.fit(X_combined, y_dict[prop])

            self.lgbm_models[prop] = lgbm

            # Train sigma model (uncertainty)
            residuals = np.abs(y_dict[prop] - lgbm.predict(X_combined))
            sigma_lgbm = lgb.LGBMRegressor(**self.lgbm_params, n_estimators=200)
            sigma_lgbm.fit(X_combined, residuals)
            self.sigma_models[prop] = sigma_lgbm

        print("\n" + "="*70)
        print("MULTI-TASK TRAINING COMPLETE")
        print("="*70)
        for prop, mae in metrics.items():
            print(f"  {prop:10s}: MAE = {mae:.3f}")

        return metrics

    def _fit_lgbm_only(self, X, y_dict, X_val, y_val_dict):
        """Fallback if TabNet unavailable"""
        X_np = X.values
        X_val_np = X_val.values if X_val is not None else None

        for prop in self.prop_names:
            lgbm = lgb.LGBMRegressor(**self.lgbm_params, n_estimators=500)

            if X_val is not None:
                lgbm.fit(
                    X_np, y_dict[prop],
                    eval_set=[(X_val_np, y_val_dict[prop])],
                    callbacks=[lgb.early_stopping(50, verbose=False)]
                )
            else:
                lgbm.fit(X_np, y_dict[prop])

            self.lgbm_models[prop] = lgbm

        return {}

    def predict(self, X, prop_name=None, return_uncertainty=False):
        """
        Predict player stats.

        Args:
            X: Features
            prop_name: Specific prop to predict (or None for all)
            return_uncertainty: Return prediction intervals

        Returns:
            If prop_name specified: predictions array
            If prop_name=None: dict of predictions for all props
        """
        X_np = X.values.astype(np.float32) if hasattr(X, 'values') else X.astype(np.float32)

        # Get embeddings
        if self.tabnet is not None:
            _, embeddings = self.tabnet.predict(X_np, return_embeddings=True)
            X_combined = np.hstack([X_np, embeddings])
        else:
            X_combined = X_np

        # Predict
        if prop_name:
            preds = self.lgbm_models[prop_name].predict(X_combined)
            if return_uncertainty and prop_name in self.sigma_models:
                sigma = self.sigma_models[prop_name].predict(X_combined)
                return preds, sigma
            return preds
        else:
            results = {}
            for prop in self.prop_names:
                results[prop] = self.lgbm_models[prop].predict(X_combined)
            return results

    def predict_combo(self, X, combo_type='PRA'):
        """
        Predict combo props.

        Args:
            combo_type: 'PRA' (Points+Rebounds+Assists), 'PR', 'PA', 'RA'
        """
        preds = self.predict(X)

        if combo_type == 'PRA':
            return preds['points'] + preds['rebounds'] + preds['assists']
        elif combo_type == 'PR':
            return preds['points'] + preds['rebounds']
        elif combo_type == 'PA':
            return preds['points'] + preds['assists']
        elif combo_type == 'RA':
            return preds['rebounds'] + preds['assists']
        else:
            raise ValueError(f"Unknown combo: {combo_type}")

    def save(self, filepath):
        """Save multi-task model"""
        filepath = Path(filepath)

        # Save TabNet
        tabnet_path = None
        if self.tabnet is not None:
            tabnet_path = str(filepath.parent / f"{filepath.stem}_tabnet.zip")
            self.tabnet.save_model(tabnet_path)

        # Save everything else
        state = {
            'lgbm_models': self.lgbm_models,
            'sigma_models': self.sigma_models,
            'feature_names': self.feature_names,
            'tabnet_path': tabnet_path,
            'tabnet_params': self.tabnet_params,
            'prop_names': self.prop_names
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        print(f"✓ Saved multi-task model: {filepath}")

    @staticmethod
    def load(filepath):
        """Load multi-task model"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        model = MultiTaskPlayerModel()
        model.lgbm_models = state['lgbm_models']
        model.sigma_models = state['sigma_models']
        model.feature_names = state['feature_names']
        model.prop_names = state['prop_names']
        model.tabnet_params = state.get('tabnet_params', {})

        # Load TabNet
        if state['tabnet_path'] and Path(state['tabnet_path']).exists():
            model.tabnet = TabNetRegressor(**model.tabnet_params)
            model.tabnet.load_model(state['tabnet_path'])

        return model


# Example usage:
"""
# Training
model = MultiTaskPlayerModel(use_gpu=True)

y_dict = {
    'minutes': train_df['minutes'].values,
    'points': train_df['points'].values,
    'rebounds': train_df['rebounds'].values,
    'assists': train_df['assists'].values,
    'threes': train_df['threes'].values
}

model.fit(X_train, y_dict, X_val, y_val_dict, epochs=50)
model.save('models/multi_task_player.pkl')

# Prediction
model = MultiTaskPlayerModel.load('models/multi_task_player.pkl')

# Individual props
points_pred = model.predict(X_test, 'points')

# All props at once
all_preds = model.predict(X_test)
print(all_preds['points'], all_preds['rebounds'], all_preds['assists'])

# Combo props
pra_pred = model.predict_combo(X_test, 'PRA')  # Points + Rebounds + Assists
"""
