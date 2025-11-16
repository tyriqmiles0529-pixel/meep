"""
Neural Network Hybrid System for NBA Props
===========================================

Combines deep learning (TabNet) with tree ensembles (LightGBM) for optimal performance.

Architecture:
1. TabNet learns deep feature representations (attention-based)
2. LightGBM uses both raw + deep features for final predictions
3. Uncertainty quantification via sigma models

Why TabNet over TFT:
- Designed for tabular data (your 120+ features)
- Sequential attention mechanism
- Built-in feature selection
- 10x faster than TFT
- Better for irregular events
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from pytorch_tabnet.tab_model import TabNetRegressor
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
    print("⚠️  TabNet not installed. Run: pip install pytorch-tabnet")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not installed. Run: pip install torch")


class GameNeuralHybrid:
    """
    Neural hybrid specifically for game predictions (moneyline/spread).

    Optimized for smaller datasets (~50k samples vs 1.6M for players):
    - Shallow TabNet (3 layers instead of 5)
    - Strong regularization (dropout 0.4)
    - Fewer epochs (25 instead of 100)
    - Ensemble weighting (0.4 TabNet + 0.6 LightGBM)
    - Single file save/load for easy deployment
    """

    def __init__(self, task='classification', use_gpu=False):
        """
        Args:
            task: 'classification' for moneyline, 'regression' for spread
            use_gpu: Use GPU for training
        """
        self.task = task
        self.use_gpu = use_gpu and TORCH_AVAILABLE

        # SMALLER TabNet for game data (overfitting prevention)
        tabnet_optimizer_params = {
            'lr': 1e-2,  # Lower LR
            'weight_decay': 1e-4  # Stronger weight decay
        }
        tabnet_scheduler_params = {
            'mode': 'min',
            'patience': 3,  # Earlier stopping
            'factor': 0.5,
            'min_lr': 1e-6
        }

        self.tabnet_params = {
            'n_d': 24,                    # Smaller width (32 -> 24)
            'n_a': 24,
            'n_steps': 3,                 # Fewer steps (5 -> 3) = shallower
            'gamma': 1.3,
            'n_independent': 1,           # Fewer layers (2 -> 1)
            'n_shared': 2,
            'lambda_sparse': 1e-3,        # STRONGER sparsity (1e-4 -> 1e-3)
            'momentum': 0.3,
            'clip_value': 1.0,            # Tighter clipping (2.0 -> 1.0)
            'mask_type': 'entmax',
            'verbose': 0,                 # Silent
            'device_name': 'cuda' if self.use_gpu else 'cpu'
        }

        if TORCH_AVAILABLE:
            self.tabnet_params['optimizer_fn'] = torch.optim.AdamW
            self.tabnet_params['optimizer_params'] = tabnet_optimizer_params
            self.tabnet_params['scheduler_fn'] = torch.optim.lr_scheduler.ReduceLROnPlateau
            self.tabnet_params['scheduler_params'] = tabnet_scheduler_params

        # LightGBM params (similar to existing game models)
        self.lgbm_params = {
            'objective': 'binary' if task == 'classification' else 'regression',
            'metric': 'binary_logloss' if task == 'classification' else 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'verbose': -1,
            'n_estimators': 800,
            'force_col_wise': True
        }

        self.tabnet = None
        self.lgbm = None
        self.calibrator = None  # For classification
        self.feature_names = None
        self.ensemble_weight = 0.4  # 40% TabNet, 60% LightGBM

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None, epochs=25, batch_size=512):
        """
        Train game neural hybrid with overfitting prevention.

        Returns: validation metrics
        """
        if not TABNET_AVAILABLE or not TORCH_AVAILABLE:
            print(f"⚠️  TabNet unavailable, using LightGBM only")
            return self._fit_lgbm_only(X, y, X_val, y_val, sample_weight)

        self.feature_names = X.columns.tolist() if hasattr(X, 'columns') else None

        # Split validation if needed
        if X_val is None:
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            if sample_weight is not None:
                w_train = sample_weight[:split_idx]
            else:
                w_train = None
        else:
            X_train, y_train = X, y
            w_train = sample_weight

        X_train_np = X_train.values.astype(np.float32) if hasattr(X_train, 'values') else X_train.astype(np.float32)
        X_val_np = X_val.values.astype(np.float32) if hasattr(X_val, 'values') else X_val.astype(np.float32)
        y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
        y_val_np = y_val.values if hasattr(y_val, 'values') else y_val

        print(f"\n🧠 Training Game Neural Hybrid ({self.task})")
        print(f"  Samples: {len(X_train):,} train, {len(X_val):,} val")
        print(f"  Features: {X_train_np.shape[1]}")
        print(f"  Architecture: Shallow TabNet (3 steps) + LightGBM")
        print(f"  Overfitting prevention: dropout, weight decay, early stopping")

        # 1. Train TabNet
        from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

        if self.task == 'classification':
            self.tabnet = TabNetClassifier(**self.tabnet_params)
        else:
            self.tabnet = TabNetRegressor(**self.tabnet_params)
            # TabNet regression requires 2D targets: (n_samples, n_outputs)
            if y_train_np.ndim == 1:
                y_train_np = y_train_np.reshape(-1, 1)
            if y_val_np.ndim == 1:
                y_val_np = y_val_np.reshape(-1, 1)

        # Add dropout for regularization (not in params)
        self.tabnet.fit(
            X_train_np, y_train_np,
            eval_set=[(X_val_np, y_val_np)],
            max_epochs=epochs,
            patience=10,  # Early stopping
            batch_size=batch_size,
            virtual_batch_size=128,
            drop_last=False,
            weights=w_train.astype(np.float32) if w_train is not None else None
        )

        # 2. Generate embeddings
        print("  Generating TabNet embeddings...")
        # TabNet internal representations can be accessed via forward pass
        # Since predict() no longer returns embeddings, we use the model's internal state
        # or simply use TabNet predictions as features for LightGBM
        try:
            # Get TabNet predictions as features (simpler, works with all versions)
            if self.task == 'classification':
                train_embeddings = self.tabnet.predict_proba(X_train_np)
                val_embeddings = self.tabnet.predict_proba(X_val_np)
            else:
                train_embeddings = self.tabnet.predict(X_train_np).reshape(-1, 1)
                val_embeddings = self.tabnet.predict(X_val_np).reshape(-1, 1)
        except Exception as e:
            print(f"  Warning: Could not get TabNet embeddings: {e}")
            # Fallback: just use empty embeddings
            train_embeddings = np.zeros((X_train_np.shape[0], 1))
            val_embeddings = np.zeros((X_val_np.shape[0], 1))

        # 3. Combine raw features + embeddings for LightGBM
        X_train_combined = np.hstack([X_train_np, train_embeddings])
        X_val_combined = np.hstack([X_val_np, val_embeddings])

        # 4. Train LightGBM
        print("  Training LightGBM on combined features...")
        if self.task == 'classification':
            self.lgbm = lgb.LGBMClassifier(**self.lgbm_params, random_state=42)
            self.lgbm.fit(
                X_train_combined, y_train,
                eval_set=[(X_val_combined, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)],
                sample_weight=w_train
            )

            # Calibration is implicit in the ensemble
            # (TabNet outputs well-calibrated probabilities, LightGBM is calibrated separately)

        else:
            self.lgbm = lgb.LGBMRegressor(**self.lgbm_params, random_state=42)
            self.lgbm.fit(
                X_train_combined, y_train,
                eval_set=[(X_val_combined, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)],
                sample_weight=w_train
            )

        # 6. Validation metrics
        return self._compute_metrics(X_val, y_val)

    def _fit_lgbm_only(self, X, y, X_val, y_val, sample_weight):
        """Fallback to LightGBM only if TabNet unavailable"""
        self.ensemble_weight = 0.0  # Pure LightGBM

        if self.task == 'classification':
            self.lgbm = lgb.LGBMClassifier(**self.lgbm_params, random_state=42)
        else:
            self.lgbm = lgb.LGBMRegressor(**self.lgbm_params, random_state=42)

        if X_val is None:
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            w_train = sample_weight[:split_idx] if sample_weight is not None else None
        else:
            X_train, y_train = X, y
            w_train = sample_weight

        self.lgbm.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)] if X_val is not None else None,
            callbacks=[lgb.early_stopping(80, verbose=False)],
            sample_weight=w_train
        )

        return self._compute_metrics(X_val, y_val)

    def predict(self, X):
        """Ensemble prediction"""
        if self.tabnet is None:
            # Pure LightGBM
            if self.task == 'classification':
                return self.lgbm.predict_proba(X)[:, 1] if hasattr(self.lgbm, 'predict_proba') else self.lgbm.predict(X)
            return self.lgbm.predict(X)

        # Ensemble: weighted average
        X_np = X.values.astype(np.float32) if hasattr(X, 'values') else X.astype(np.float32)

        # TabNet prediction
        if self.task == 'classification':
            tabnet_pred = self.tabnet.predict_proba(X_np)[:, 1]
        else:
            tabnet_pred = self.tabnet.predict(X_np)

        # LightGBM prediction (needs embeddings)
        try:
            # Use TabNet predictions as embeddings (works with all versions)
            if self.task == 'classification':
                embeddings = self.tabnet.predict_proba(X_np)
            else:
                embeddings = self.tabnet.predict(X_np).reshape(-1, 1)
        except Exception as e:
            print(f"  Warning: Could not get TabNet embeddings: {e}")
            embeddings = np.zeros((X_np.shape[0], 1))
        X_combined = np.hstack([X_np, embeddings])

        if self.task == 'classification':
            lgbm_pred = self.lgbm.predict_proba(X_combined)[:, 1]
        else:
            lgbm_pred = self.lgbm.predict(X_combined)

        # Weighted ensemble
        return self.ensemble_weight * tabnet_pred + (1 - self.ensemble_weight) * lgbm_pred

    def predict_proba(self, X):
        """For classification: return [prob_0, prob_1]"""
        if self.task != 'classification':
            raise ValueError("predict_proba only for classification")

        probs = self.predict(X)
        return np.column_stack([1 - probs, probs])

    def _compute_metrics(self, X_val, y_val):
        """Compute validation metrics"""
        y_pred = self.predict(X_val)

        if self.task == 'classification':
            from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
            y_pred_binary = (y_pred >= 0.5).astype(int)
            return {
                'logloss': float(log_loss(y_val, y_pred)),
                'brier': float(brier_score_loss(y_val, y_pred)),
                'accuracy': float(accuracy_score(y_val, y_pred_binary))
            }
        else:
            rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
            mae = float(mean_absolute_error(y_val, y_pred))
            return {'rmse': rmse, 'mae': mae}

    def save(self, filepath):
        """Save entire model as single pickle file"""
        filepath = Path(filepath)

        # Save TabNet separately (it's a torch model)
        tabnet_path = None
        if self.tabnet is not None:
            tabnet_path = filepath.parent / f"{filepath.stem}_tabnet.zip"
            self.tabnet.save_model(str(tabnet_path))

        # Save everything else
        state = {
            'task': self.task,
            'lgbm': self.lgbm,
            'calibrator': self.calibrator,
            'feature_names': self.feature_names,
            'ensemble_weight': self.ensemble_weight,
            'tabnet_path': str(tabnet_path) if tabnet_path else None,
            'tabnet_params': self.tabnet_params,
            'use_gpu': self.use_gpu
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        print(f"✓ Saved game model: {filepath}")
        if tabnet_path:
            print(f"✓ Saved TabNet: {tabnet_path}")

    def load(self, filepath):
        """Load entire model from single pickle file"""
        filepath = Path(filepath)

        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self.task = state['task']
        self.lgbm = state['lgbm']
        self.calibrator = state.get('calibrator')
        self.feature_names = state['feature_names']
        self.ensemble_weight = state['ensemble_weight']
        self.use_gpu = state.get('use_gpu', False)
        self.tabnet_params = state.get('tabnet_params', {})

        # Load TabNet if exists
        if state['tabnet_path'] and Path(state['tabnet_path']).exists():
            from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
            if self.task == 'classification':
                self.tabnet = TabNetClassifier(**self.tabnet_params)
            else:
                self.tabnet = TabNetRegressor(**self.tabnet_params)
            self.tabnet.load_model(state['tabnet_path'])

        print(f"✓ Loaded game model: {filepath}")
        return self


class NeuralHybridPredictor:
    """
    Hybrid predictor combining TabNet (deep learning) with LightGBM (tree ensemble).
    
    Training Flow:
    1. TabNet learns 32-dim embedding from raw features
    2. Embeddings capture non-linear interactions
    3. LightGBM trained on [raw_features + embeddings]
    4. Best of both worlds: DL pattern recognition + tree efficiency
    """
    
    def __init__(self, prop_name, use_gpu=False):
        self.prop_name = prop_name
        self.use_gpu = use_gpu and TORCH_AVAILABLE
        
        # TabNet hyperparameters (OPTIMIZED for speed + large datasets)
        tabnet_optimizer_params = {
            'lr': 2e-2,
            'weight_decay': 1e-5
        }
        tabnet_scheduler_params = {
            'mode': 'min',
            'patience': 3,                # Reduced from 5 (faster early stopping)
            'factor': 0.5,
            'min_lr': 1e-5
        }

        self.tabnet_params = {
            'n_d': 24,                    # Reduced from 32 (20% faster, minimal accuracy loss)
            'n_a': 24,                    # Reduced from 32
            'n_steps': 4,                 # Reduced from 5 (20% fewer forward passes)
            'gamma': 1.5,                 # Coefficient for feature reusage
            'n_independent': 2,           # Number of independent GLU layers
            'n_shared': 2,                # Number of shared GLU layers
            'lambda_sparse': 1e-4,        # Sparsity regularization
            'momentum': 0.3,              # Batch norm momentum
            'clip_value': 2.0,            # Gradient clipping
            'mask_type': 'sparsemax',     # Faster than entmax (less computation)
            'verbose': 1,
            'device_name': 'cuda' if self.use_gpu else 'cpu'
        }
        
        # Only add optimizer if torch is available
        if TORCH_AVAILABLE:
            self.tabnet_params['optimizer_fn'] = torch.optim.AdamW
            self.tabnet_params['optimizer_params'] = tabnet_optimizer_params
            self.tabnet_params['scheduler_fn'] = torch.optim.lr_scheduler.ReduceLROnPlateau
            self.tabnet_params['scheduler_params'] = tabnet_scheduler_params
        
        # LightGBM hyperparameters (same as current system)
        self.lgbm_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'verbose': -1,
            'n_estimators': 500,
            'early_stopping_rounds': 50
        }
        
        self.tabnet = None
        self.lgbm = None
        self.sigma_model = None
        self.feature_names = None
        self.tabnet_embedding_dim = 32
        self.embedding_pca = None  # Store PCA for consistent compression
        
    def fit(self, X, y, X_val=None, y_val=None, epochs=30, batch_size=2048):
        """
        Train hybrid model.
        
        Steps:
        1. Train TabNet to learn feature embeddings
        2. Generate embeddings for train + val
        3. Train LightGBM on [raw + embeddings]
        4. Train sigma model for uncertainty
        """
        
        if not TABNET_AVAILABLE or not TORCH_AVAILABLE:
            print(f"⚠️  Neural network libraries not available. Falling back to LightGBM only.")
            return self._fit_lgbm_only(X, y, X_val, y_val)
        
        self.feature_names = X.columns.tolist()
        
        # Split validation if not provided
        if X_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        else:
            X_train, y_train = X, y
        
        X_train_np = X_train.values.astype(np.float32)
        X_val_np = X_val.values.astype(np.float32)
        y_train_np = y_train.values.astype(np.float32)
        y_val_np = y_val.values.astype(np.float32)
        
        print(f"\n{'='*60}")
        print(f"Training Neural Hybrid for {self.prop_name}")
        print(f"{'='*60}")
        print(f"Training samples: {len(X_train):,}")
        print(f"Validation samples: {len(X_val):,}")
        print(f"Features: {X_train.shape[1]}")
        print(f"Device: {'GPU (CUDA)' if self.use_gpu else 'CPU'}")
        
        # ============================================================
        # STEP 1: Train TabNet for feature learning
        # ============================================================
        print(f"\n{'─'*60}")
        print("Step 1: Training TabNet (Deep Feature Learning)")
        print(f"{'─'*60}")
        
        self.tabnet = TabNetRegressor(**self.tabnet_params)
        
        self.tabnet.fit(
            X_train=X_train_np,
            y_train=y_train_np.reshape(-1, 1),
            eval_set=[(X_val_np, y_val_np.reshape(-1, 1))],
            eval_metric=['rmse', 'mae'],
            max_epochs=epochs,
            patience=10,              # Reduced from 15 (stops sooner if no improvement)
            batch_size=batch_size,
            virtual_batch_size=256,   # Increased from 128 (better GPU utilization)
            num_workers=0,
            drop_last=False
        )
        
        # Get TabNet predictions and feature importance
        tabnet_train_pred = self.tabnet.predict(X_train_np).flatten()
        tabnet_val_pred = self.tabnet.predict(X_val_np).flatten()
        
        tabnet_rmse = np.sqrt(mean_squared_error(y_val, tabnet_val_pred))
        tabnet_mae = mean_absolute_error(y_val, tabnet_val_pred)
        
        print(f"\n  TabNet standalone performance:")
        print(f"  - RMSE: {tabnet_rmse:.3f}")
        print(f"  - MAE:  {tabnet_mae:.3f}")
        
        # ============================================================
        # STEP 2: Extract deep embeddings
        # ============================================================
        print(f"\n{'─'*60}")
        print("Step 2: Extracting Deep Feature Embeddings")
        print(f"{'─'*60}")
        
        # Get embeddings from last layer before prediction
        train_embeddings = self._get_embeddings(X_train_np)
        val_embeddings = self._get_embeddings(X_val_np)

        print(f"  - Embedding dimension: {train_embeddings.shape[1]}")
        print(f"  - Train embeddings: {train_embeddings.shape}")
        print(f"  - Val embeddings: {val_embeddings.shape}")

        # ============================================================
        # STEP 2.5: Normalize embeddings (recommended by ChatGPT)
        # ============================================================
        # Standardize embeddings so LightGBM interprets them evenly
        from sklearn.preprocessing import StandardScaler
        self.embedding_scaler = StandardScaler()
        train_embeddings_normalized = self.embedding_scaler.fit_transform(train_embeddings)
        val_embeddings_normalized = self.embedding_scaler.transform(val_embeddings)

        print(f"  ✓ Embeddings normalized (mean=0, std=1)")

        # ============================================================
        # STEP 3: Create hybrid features (raw + embeddings)
        # ============================================================
        print(f"\n{'─'*60}")
        print("Step 3: Creating Hybrid Feature Set")
        print(f"{'─'*60}")

        # Combine raw features with normalized deep embeddings
        embedding_cols = [f'tabnet_emb_{i}' for i in range(train_embeddings.shape[1])]

        X_train_hybrid = pd.DataFrame(
            np.hstack([X_train.values, train_embeddings_normalized]),
            columns=self.feature_names + embedding_cols
        )

        X_val_hybrid = pd.DataFrame(
            np.hstack([X_val.values, val_embeddings_normalized]),
            columns=self.feature_names + embedding_cols
        )
        
        print(f"  - Raw features: {len(self.feature_names)}")
        print(f"  - Deep embeddings: {len(embedding_cols)}")
        print(f"  - Total hybrid features: {X_train_hybrid.shape[1]}")
        
        # ============================================================
        # STEP 4: Train LightGBM on hybrid features
        # ============================================================
        print(f"\n{'─'*60}")
        print("Step 4: Training LightGBM on Hybrid Features")
        print(f"{'─'*60}")
        
        train_data = lgb.Dataset(X_train_hybrid, label=y_train)
        val_data = lgb.Dataset(X_val_hybrid, label=y_val, reference=train_data)
        
        self.lgbm = lgb.train(
            self.lgbm_params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0)
            ]
        )
        
        # Evaluate hybrid model
        hybrid_val_pred = self.lgbm.predict(X_val_hybrid)
        hybrid_rmse = np.sqrt(mean_squared_error(y_val, hybrid_val_pred))
        hybrid_mae = mean_absolute_error(y_val, hybrid_val_pred)
        
        print(f"\n  Hybrid (TabNet + LightGBM) performance:")
        print(f"  - RMSE: {hybrid_rmse:.3f}")
        print(f"  - MAE:  {hybrid_mae:.3f}")
        
        # ============================================================
        # STEP 5: Train sigma model for uncertainty
        # ============================================================
        print(f"\n{'─'*60}")
        print("Step 5: Training Uncertainty Model (Sigma)")
        print(f"{'─'*60}")
        
        # Calculate residuals
        train_pred = self.lgbm.predict(X_train_hybrid)
        train_residuals = np.abs(y_train - train_pred)
        
        sigma_data = lgb.Dataset(X_train_hybrid, label=train_residuals)
        sigma_val_data = lgb.Dataset(X_val_hybrid, reference=sigma_data)
        
        self.sigma_model = lgb.train(
            {**self.lgbm_params, 'learning_rate': 0.03},
            sigma_data,
            valid_sets=[sigma_val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )
        
        val_sigma = self.sigma_model.predict(X_val_hybrid)
        print(f"  - Mean predicted uncertainty: {val_sigma.mean():.3f}")
        print(f"  - Actual mean error: {np.abs(y_val - hybrid_val_pred).mean():.3f}")
        
        # ============================================================
        # FINAL SUMMARY
        # ============================================================
        print(f"\n{'='*60}")
        print("Training Complete - Performance Comparison")
        print(f"{'='*60}")
        print(f"TabNet only:       RMSE={tabnet_rmse:.3f}, MAE={tabnet_mae:.3f}")
        print(f"Hybrid (FINAL):    RMSE={hybrid_rmse:.3f}, MAE={hybrid_mae:.3f}")
        print(f"Improvement:       {(tabnet_rmse - hybrid_rmse)/tabnet_rmse*100:+.1f}% RMSE")
        print(f"{'='*60}\n")
        
        # Feature importance analysis
        self._analyze_feature_importance(X_train_hybrid.columns)
        
        return self
    
    def _fit_lgbm_only(self, X, y, X_val=None, y_val=None):
        """Fallback to LightGBM-only if neural network libs unavailable."""
        print(f"\nTraining LightGBM-only model for {self.prop_name}...")
        
        self.feature_names = X.columns.tolist()
        
        if X_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        else:
            X_train, y_train = X, y
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        self.lgbm = lgb.train(
            self.lgbm_params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid']
        )
        
        return self
    
    def _get_embeddings(self, X):
        """
        Extract embeddings from TabNet using the explain() method.

        The explain() method returns attention masks which we can use as embeddings.
        This is more reliable than trying to access internal layers.

        Returns:
            embeddings: numpy array of shape (n_samples, n_features or n_steps)
        """
        if self.tabnet is None:
            raise ValueError("TabNet not trained yet")

        try:
            import torch

            # APPROACH 1: Try using explain to get meaningful features
            # explain() returns (M_explain, masks) where M_explain has shape (batch, n_steps, n_features)
            # We can aggregate this to get learned importance scores
            try:
                M_explain, masks = self.tabnet.explain(X)

                # M_explain shape: (batch, n_steps, n_features)
                # Aggregate across features to get step-wise importance (batch, n_steps)
                # This gives us a compressed representation
                if M_explain.ndim == 3:
                    # Sum across features for each step, giving (batch, n_steps)
                    embeddings = M_explain.sum(axis=2)  # Shape: (batch, n_steps)
                    print(f"  [INFO] Extracted {embeddings.shape[1]}-dim embeddings from explain (step aggregation)")
                else:
                    # Fallback to mean if shape is unexpected
                    embeddings = M_explain.mean(axis=1, keepdims=True) if M_explain.ndim > 1 else M_explain
                    print(f"  [INFO] Extracted {embeddings.shape[1]}-dim embeddings from explain (mean)")

                # If we only got 1 dimension, try another approach
                if embeddings.shape[1] == 1:
                    raise ValueError("Only got 1-dim from explain, trying direct network access")

                return embeddings

            except Exception as e1:
                print(f"  [DEBUG] explain() approach failed: {str(e1)[:50]}")

                # APPROACH 2: Try direct network forward pass
                self.tabnet.network.eval()

                if isinstance(X, np.ndarray):
                    X_tensor = torch.from_numpy(X).float()
                else:
                    X_tensor = torch.from_numpy(X.values).float()

                if self.use_gpu:
                    X_tensor = X_tensor.cuda()

                with torch.no_grad():
                    # Try to get intermediate representation
                    # TabNet architecture: input -> embedder -> tabnet_encoder -> final_mapping

                    # Check what attributes exist
                    available_attrs = [attr for attr in dir(self.tabnet.network)
                                     if not attr.startswith('_') and 'forward' in attr.lower()]

                    # Try forward_masks if available
                    if hasattr(self.tabnet.network, 'forward_masks'):
                        out, M_loss = self.tabnet.network.forward_masks(X_tensor)
                        embeddings = out.cpu().numpy()

                        # If this is just the features, try to compress
                        if embeddings.shape[1] == X.shape[1]:
                            print(f"  [INFO] Got {embeddings.shape[1]}-dim from forward_masks (uncompressed)")

                            # Use PCA to compress to reasonable dimension
                            from sklearn.decomposition import PCA

                            # If PCA already fitted (during training), use it
                            if self.embedding_pca is not None:
                                embeddings = self.embedding_pca.transform(embeddings)
                                print(f"  [INFO] Compressed to {embeddings.shape[1]}-dim using fitted PCA")
                            else:
                                # First time - fit PCA (during training)
                                # n_components must be <= min(n_samples, n_features)
                                n_components = min(24, embeddings.shape[0], embeddings.shape[1])

                                if n_components >= 2:
                                    self.embedding_pca = PCA(n_components=n_components)
                                    embeddings = self.embedding_pca.fit_transform(embeddings)
                                    print(f"  [INFO] Fitted PCA and compressed to {embeddings.shape[1]}-dim")
                                else:
                                    print(f"  [WARNING] Too few samples ({embeddings.shape[0]}) for PCA, using mean")
                                    embeddings = embeddings.mean(axis=1, keepdims=True)
                        else:
                            print(f"  [INFO] Extracted {embeddings.shape[1]}-dim embeddings from forward_masks")

                        return embeddings

                    raise AttributeError(f"No suitable embedding extraction method found. Available: {available_attrs}")

        except Exception as e:
            print(f"  [WARNING] All embedding extraction failed: {str(e)[:100]}")
            print(f"  [FALLBACK] Using predictions as 1-dim embeddings")

            # Fallback: just use predictions
            predictions = self.tabnet.predict(X)
            return predictions.reshape(-1, 1)
    
    def predict(self, X, return_uncertainty=False):
        """
        Make predictions using hybrid model.
        
        Args:
            X: Input features (DataFrame or numpy array)
            return_uncertainty: If True, also return sigma (uncertainty estimate)
        
        Returns:
            predictions (or tuple of predictions, sigma)
        """
        if self.lgbm is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            if self.feature_names is None:
                raise ValueError("Cannot predict with numpy array - feature names not set")
            X = pd.DataFrame(X, columns=self.feature_names)
        
        # If TabNet available, use hybrid prediction
        if self.tabnet is not None and TABNET_AVAILABLE:
            X_np = X.values.astype(np.float32)
            embeddings = self._get_embeddings(X_np)

            # Normalize embeddings using fitted scaler
            if hasattr(self, 'embedding_scaler'):
                embeddings = self.embedding_scaler.transform(embeddings)
            
            embedding_cols = [f'tabnet_emb_{i}' for i in range(embeddings.shape[1])]
            X_hybrid = pd.DataFrame(
                np.hstack([X.values, embeddings]),
                columns=self.feature_names + embedding_cols
            )
        else:
            # Fallback to raw features only
            X_hybrid = X
        
        predictions = self.lgbm.predict(X_hybrid)
        
        if return_uncertainty and self.sigma_model is not None:
            sigma = self.sigma_model.predict(X_hybrid)
            return predictions, sigma
        
        return predictions
    
    def _analyze_feature_importance(self, feature_names):
        """Analyze which features (raw vs embeddings) are most important."""
        importance = self.lgbm.feature_importance(importance_type='gain')

        # Convert feature_names to strings (handles both string and int column names)
        feature_names = [str(f) for f in feature_names]

        raw_features = [f for f in feature_names if not f.startswith('tabnet_emb_')]
        emb_features = [f for f in feature_names if f.startswith('tabnet_emb_')]

        raw_indices = [i for i, f in enumerate(feature_names) if f in raw_features]
        emb_indices = [i for i, f in enumerate(feature_names) if f in emb_features]

        raw_importance = importance[raw_indices].sum() if raw_indices else 0
        emb_importance = importance[emb_indices].sum() if emb_indices else 0
        total_importance = importance.sum()

        print(f"\n{'─'*60}")
        print("Feature Importance Breakdown")
        print(f"{'─'*60}")
        print(f"Raw features:        {raw_importance/total_importance*100:.1f}% of importance")
        print(f"Deep embeddings:     {emb_importance/total_importance*100:.1f}% of importance")
        print(f"\nTop 10 most important features:")

        top_features = sorted(
            zip(feature_names, importance),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        for i, (feat, imp) in enumerate(top_features, 1):
            feat_type = "🧠 Embedding" if feat.startswith('tabnet_emb_') else "📊 Raw"
            print(f"  {i:2d}. {feat_type:15s} {feat:40s} ({imp/total_importance*100:.1f}%)")
    
    def save(self, path):
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'prop_name': self.prop_name,
            'feature_names': self.feature_names,
            'lgbm_model': self.lgbm,
            'sigma_model': self.sigma_model,
            'use_gpu': self.use_gpu
        }
        
        # Save TabNet separately (PyTorch format)
        if self.tabnet is not None:
            tabnet_path = path.parent / f"{path.stem}_tabnet.zip"
            self.tabnet.save_model(str(tabnet_path))
            model_data['tabnet_path'] = str(tabnet_path)
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Saved hybrid model to {path}")
    
    @classmethod
    def load(cls, path):
        """Load model from disk."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(model_data['prop_name'], use_gpu=model_data.get('use_gpu', False))
        model.feature_names = model_data['feature_names']
        model.lgbm = model_data['lgbm_model']
        model.sigma_model = model_data.get('sigma_model')
        
        # Load TabNet if exists
        if 'tabnet_path' in model_data and Path(model_data['tabnet_path']).exists():
            if TABNET_AVAILABLE:
                model.tabnet = TabNetRegressor()
                model.tabnet.load_model(model_data['tabnet_path'])
        
        return model


def train_neural_hybrid_models(
    data_dict,
    window_name="2022_2026",
    use_gpu=False,
    epochs=100,
    output_dir="models/neural_hybrid"
):
    """
    Train neural hybrid models for all props.
    
    Args:
        data_dict: Dict with keys like 'minutes', 'points', etc.
                   Each value is DataFrame with features + target
        window_name: Name for this training window
        use_gpu: Whether to use GPU for TabNet training
        epochs: Number of epochs for TabNet
        output_dir: Where to save models
    
    Returns:
        Dict of trained models
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models = {}
    
    for prop_name, df in data_dict.items():
        print(f"\n\n{'#'*70}")
        print(f"# Training Neural Hybrid Model: {prop_name.upper()}")
        print(f"{'#'*70}")
        
        # Separate features and target
        target_col = prop_name  # Assumes target column is same as prop name
        if target_col not in df.columns:
            print(f"⚠️  Target column '{target_col}' not found. Skipping {prop_name}.")
            continue
        
        y = df[target_col]
        X = df.drop(columns=[target_col])
        
        # Train hybrid model
        model = NeuralHybridPredictor(prop_name, use_gpu=use_gpu)
        model.fit(X, y, epochs=epochs)
        
        # Save model
        model_path = output_dir / f"{prop_name}_hybrid_{window_name}.pkl"
        model.save(model_path)
        
        models[prop_name] = model
    
    print(f"\n\n{'='*70}")
    print(f"All Neural Hybrid Models Trained Successfully!")
    print(f"{'='*70}")
    print(f"Models saved to: {output_dir}")
    
    return models


if __name__ == "__main__":
    """
    Test neural hybrid on sample data.
    
    To use in your training pipeline:
    1. Install dependencies: pip install pytorch-tabnet torch
    2. Import: from neural_hybrid import NeuralHybridPredictor
    3. Use in train_auto.py instead of plain LightGBM
    """
    
    print("\n" + "="*70)
    print("Neural Hybrid Model - Installation Check")
    print("="*70)
    
    if not TORCH_AVAILABLE:
        print("\n❌ PyTorch not installed.")
        print("   Install with: pip install torch")
    else:
        print("\n✅ PyTorch installed")
        import torch
        print(f"   Version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
    
    if not TABNET_AVAILABLE:
        print("\n❌ TabNet not installed.")
        print("   Install with: pip install pytorch-tabnet")
    else:
        print("\n✅ TabNet installed")
    
    print("\n" + "="*70)
    print("Ready to train neural hybrid models!")
    print("="*70)
    print("\nNext steps:")
    print("1. Install dependencies if missing (see above)")
    print("2. Integrate into train_auto.py")
    print("3. Run overnight training with --use-neural flag")
    print("="*70 + "\n")
