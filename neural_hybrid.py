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
        
        # TabNet hyperparameters (tuned for sports data)
        self.tabnet_params = {
            'n_d': 32,                    # Width of decision prediction layer
            'n_a': 32,                    # Width of attention embedding
            'n_steps': 5,                 # Number of sequential attention steps
            'gamma': 1.5,                 # Coefficient for feature reusage
            'n_independent': 2,           # Number of independent GLU layers
            'n_shared': 2,                # Number of shared GLU layers
            'lambda_sparse': 1e-4,        # Sparsity regularization
            'momentum': 0.3,              # Batch norm momentum
            'clip_value': 2.0,            # Gradient clipping
            'optimizer_fn': None,         # Will use AdamW
            'optimizer_params': {
                'lr': 2e-2,
                'weight_decay': 1e-5
            },
            'scheduler_fn': None,         # Will use ReduceLROnPlateau
            'scheduler_params': {
                'mode': 'min',
                'patience': 5,
                'factor': 0.5,
                'min_lr': 1e-5
            },
            'mask_type': 'entmax',        # Sparse attention (better than sparsemax)
            'verbose': 1,
            'device_name': 'cuda' if self.use_gpu else 'cpu'
        }
        
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
        
    def fit(self, X, y, X_val=None, y_val=None, epochs=100, batch_size=1024):
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
            patience=15,
            batch_size=batch_size,
            virtual_batch_size=128,
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
        # STEP 3: Create hybrid features (raw + embeddings)
        # ============================================================
        print(f"\n{'─'*60}")
        print("Step 3: Creating Hybrid Feature Set")
        print(f"{'─'*60}")
        
        # Combine raw features with deep embeddings
        embedding_cols = [f'tabnet_emb_{i}' for i in range(train_embeddings.shape[1])]
        
        X_train_hybrid = pd.DataFrame(
            np.hstack([X_train.values, train_embeddings]),
            columns=self.feature_names + embedding_cols
        )
        
        X_val_hybrid = pd.DataFrame(
            np.hstack([X_val.values, val_embeddings]),
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
        """Extract embeddings from TabNet's last hidden layer."""
        # Use TabNet's internal embedding extraction
        # This gets the representation before final prediction layer
        self.tabnet.network.eval()
        
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float()
            if self.use_gpu:
                X_tensor = X_tensor.cuda()
            
            # Forward pass through TabNet encoder
            # Get embeddings from final attention step
            steps_output, _ = self.tabnet.network.encoder(X_tensor)
            embeddings = steps_output[:, -1, :]  # Last step embeddings
            
            if self.use_gpu:
                embeddings = embeddings.cpu()
            
            return embeddings.numpy()
    
    def predict(self, X, return_uncertainty=False):
        """
        Make predictions using hybrid model.
        
        Args:
            X: Input features (DataFrame)
            return_uncertainty: If True, also return sigma (uncertainty estimate)
        
        Returns:
            predictions (or tuple of predictions, sigma)
        """
        if self.lgbm is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        # If TabNet available, use hybrid prediction
        if self.tabnet is not None and TABNET_AVAILABLE:
            X_np = X.values.astype(np.float32)
            embeddings = self._get_embeddings(X_np)
            
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
        
        raw_features = [f for f in feature_names if not f.startswith('tabnet_emb_')]
        emb_features = [f for f in feature_names if f.startswith('tabnet_emb_')]
        
        raw_indices = [i for i, f in enumerate(feature_names) if f in raw_features]
        emb_indices = [i for i, f in enumerate(feature_names) if f in emb_features]
        
        raw_importance = importance[raw_indices].sum()
        emb_importance = importance[emb_indices].sum()
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
