#!/usr/bin/env python3
"""
Quick Neural Hybrid Test - Train 1 epoch just to test embedding extraction

This trains a minimal model just to verify:
1. TabNet can train
2. Embeddings can be extracted
3. Hybrid architecture works
4. Models can be saved

Run time: ~2-3 minutes instead of 1.5 hours
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

# Check imports
try:
    from pytorch_tabnet.tab_model import TabNetRegressor
    import torch
    import lightgbm as lgb
    print("OK All packages available")
except ImportError as e:
    print(f"ERROR Missing package: {e}")
    print("   Run: pip install pytorch-tabnet lightgbm scikit-learn")
    exit(1)

# Create dummy data (mimics your real data structure)
print("\n📊 Creating dummy training data...")
n_samples = 10000
n_features = 56  # Same as your real models

np.random.seed(42)

# Feature names (same as your real model)
feature_names = [
    'is_home', 'season_end_year', 'season_decade',
    'team_recent_pace', 'team_off_strength', 'team_def_strength', 'team_recent_winrate',
    'opp_recent_pace', 'opp_off_strength', 'opp_def_strength', 'opp_recent_winrate',
    'match_off_edge', 'match_def_edge', 'match_pace_sum', 'winrate_diff',
    'starter_flag', 'minutes',
    'points_L3', 'points_L5', 'points_L10',
    'rebounds_L3', 'rebounds_L5', 'rebounds_L10',
    'assists_L3', 'assists_L5', 'assists_L10',
    'threepoint_goals_L3', 'threepoint_goals_L5', 'threepoint_goals_L10',
    'fieldGoalsAttempted_L3', 'fieldGoalsAttempted_L5', 'fieldGoalsAttempted_L10',
    'threePointersAttempted_L3', 'threePointersAttempted_L5', 'threePointersAttempted_L10',
    'freeThrowsAttempted_L3', 'freeThrowsAttempted_L5', 'freeThrowsAttempted_L10',
    'rate_fga', 'rate_3pa', 'rate_fta',
    'ts_pct_L5', 'ts_pct_L10', 'ts_pct_season',
    'three_pct_L5', 'ft_pct_L5',
    'matchup_pace', 'pace_factor', 'def_matchup_difficulty', 'offensive_environment',
    'usage_rate_L5', 'rebound_rate_L5', 'assist_rate_L5',
    'points_home_avg', 'points_away_avg', 'opp_def_strength'
]

X = pd.DataFrame(
    np.random.randn(n_samples, n_features),
    columns=feature_names
)
y = pd.Series(np.random.randn(n_samples) * 5 + 20)  # Target (e.g., points)

# Split
split = int(n_samples * 0.8)
X_train, X_val = X.iloc[:split], X.iloc[split:]
y_train, y_val = y.iloc[:split], y.iloc[split:]

print(f"   Training: {len(X_train):,} samples")
print(f"   Validation: {len(X_val):,} samples")
print(f"   Features: {n_features}")

# Test 1: Train TabNet (1 epoch only)
print("\n" + "="*70)
print("TEST 1: TabNet Training (1 epoch)")
print("="*70)

tabnet_params = {
    'n_d': 24,
    'n_a': 24,
    'n_steps': 4,
    'gamma': 1.5,
    'n_independent': 2,
    'n_shared': 2,
    'lambda_sparse': 1e-4,
    'momentum': 0.3,
    'clip_value': 2.0,
    'mask_type': 'sparsemax',
    'device_name': 'cuda' if torch.cuda.is_available() else 'cpu',
    'verbose': 1
}

if torch.cuda.is_available():
    tabnet_params['optimizer_fn'] = torch.optim.AdamW
    tabnet_params['optimizer_params'] = {'lr': 2e-2, 'weight_decay': 1e-5}

print(f"Device: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")

tabnet = TabNetRegressor(**tabnet_params)

print("\nTraining TabNet (1 epoch only)...")
tabnet.fit(
    X_train=X_train.values.astype(np.float32),
    y_train=y_train.values.astype(np.float32).reshape(-1, 1),
    eval_set=[(X_val.values.astype(np.float32), y_val.values.astype(np.float32).reshape(-1, 1))],
    eval_metric=['rmse', 'mae'],
    max_epochs=1,  # JUST 1 EPOCH!
    batch_size=2048,
    virtual_batch_size=256,
    num_workers=0
)

print("OK TabNet training complete")

# Test 2: Extract embeddings
print("\n" + "="*70)
print("TEST 2: Embedding Extraction")
print("="*70)

try:
    tabnet.network.eval()

    with torch.no_grad():
        X_tensor = torch.from_numpy(X_val.values.astype(np.float32))
        if torch.cuda.is_available():
            X_tensor = X_tensor.cuda()

        # Get embeddings from TabNet internals
        if hasattr(tabnet.network, 'embedder'):
            x = tabnet.network.embedder(X_tensor)
        else:
            x = X_tensor

        if hasattr(tabnet.network, 'tabnet'):
            steps_output, _ = tabnet.network.tabnet(x)

            print(f"   Steps output shape: {steps_output.shape}")
            print(f"   Steps output ndim: {steps_output.ndim}")

            # Handle different shapes
            if steps_output.ndim == 3:
                embeddings = steps_output[:, -1, :].cpu().numpy()
                print("   OK Extracted 3D tensor → 2D embeddings")
            elif steps_output.ndim == 2:
                embeddings = steps_output.cpu().numpy()
                print("   OK Extracted 2D embeddings directly")
            else:
                raise ValueError(f"Unexpected shape: {steps_output.shape}")

            print(f"   Final embeddings shape: {embeddings.shape}")
            print(f"   Expected: ({len(X_val)}, 24)")

            if embeddings.shape[1] == 24:
                print("   OK SUCCESS: Got 24-dimensional embeddings!")
            elif embeddings.shape[1] == 1:
                print("   WARNING  WARNING: Only 1-dimensional (predictions, not embeddings)")
            else:
                print(f"   WARNING  Unexpected: {embeddings.shape[1]} dimensions")

        else:
            print("   ERROR Cannot find tabnet encoder in network")
            embeddings = None

except Exception as e:
    print(f"   ERROR Embedding extraction failed: {e}")
    import traceback
    traceback.print_exc()
    embeddings = None

# Test 3: Create hybrid features
if embeddings is not None:
    print("\n" + "="*70)
    print("TEST 3: Hybrid Feature Creation")
    print("="*70)

    embedding_cols = [f'tabnet_emb_{i}' for i in range(embeddings.shape[1])]

    X_hybrid = pd.DataFrame(
        np.hstack([X_val.values, embeddings]),
        columns=feature_names + embedding_cols
    )

    print(f"   Raw features: {len(feature_names)}")
    print(f"   Embeddings: {len(embedding_cols)}")
    print(f"   Total hybrid: {len(X_hybrid.columns)}")
    print("   OK Hybrid features created")

    # Test 4: Train LightGBM on hybrid
    print("\n" + "="*70)
    print("TEST 4: LightGBM on Hybrid Features")
    print("="*70)

    lgbm = lgb.LGBMRegressor(
        objective='regression',
        learning_rate=0.05,
        num_leaves=31,
        n_estimators=50,  # Quick training
        random_state=42,
        n_jobs=-1,
        force_col_wise=True,
        verbosity=-1
    )

    # Need train embeddings too
    with torch.no_grad():
        X_train_tensor = torch.from_numpy(X_train.values.astype(np.float32))
        if torch.cuda.is_available():
            X_train_tensor = X_train_tensor.cuda()

        if hasattr(tabnet.network, 'embedder'):
            x_train = tabnet.network.embedder(X_train_tensor)
        else:
            x_train = X_train_tensor

        steps_train, _ = tabnet.network.tabnet(x_train)
        if steps_train.ndim == 3:
            embeddings_train = steps_train[:, -1, :].cpu().numpy()
        else:
            embeddings_train = steps_train.cpu().numpy()

    X_train_hybrid = pd.DataFrame(
        np.hstack([X_train.values, embeddings_train]),
        columns=feature_names + embedding_cols
    )

    print("   Training LightGBM...")
    lgbm.fit(X_train_hybrid, y_train)

    pred_hybrid = lgbm.predict(X_hybrid)

    from sklearn.metrics import mean_squared_error, mean_absolute_error
    rmse = np.sqrt(mean_squared_error(y_val, pred_hybrid))
    mae = mean_absolute_error(y_val, pred_hybrid)

    print(f"   OK LightGBM trained")
    print(f"   RMSE: {rmse:.3f}")
    print(f"   MAE: {mae:.3f}")

    # Test 5: Save model
    print("\n" + "="*70)
    print("TEST 5: Model Saving")
    print("="*70)

    # Create a simple model object
    class TestHybridModel:
        def __init__(self, tabnet, lgbm, feature_names):
            self.tabnet = tabnet
            self.lgbm = lgbm
            self.feature_names = feature_names
            self.sigma_model = None

        def predict(self, X):
            # Extract embeddings
            with torch.no_grad():
                X_tensor = torch.from_numpy(X.values.astype(np.float32))
                if torch.cuda.is_available():
                    X_tensor = X_tensor.cuda()

                if hasattr(self.tabnet.network, 'embedder'):
                    x = self.tabnet.network.embedder(X_tensor)
                else:
                    x = X_tensor

                steps, _ = self.tabnet.network.tabnet(x)
                if steps.ndim == 3:
                    emb = steps[:, -1, :].cpu().numpy()
                else:
                    emb = steps.cpu().numpy()

            # Create hybrid features
            embedding_cols = [f'tabnet_emb_{i}' for i in range(emb.shape[1])]
            X_hybrid = pd.DataFrame(
                np.hstack([X.values, emb]),
                columns=self.feature_names + embedding_cols
            )

            return self.lgbm.predict(X_hybrid)

    test_model = TestHybridModel(tabnet, lgbm, feature_names)

    # Save it
    Path('./test_models').mkdir(exist_ok=True)
    with open('./test_models/test_hybrid.pkl', 'wb') as f:
        pickle.dump(test_model, f)

    print("   OK Model saved to ./test_models/test_hybrid.pkl")

    # Test loading
    with open('./test_models/test_hybrid.pkl', 'rb') as f:
        loaded_model = pickle.load(f)

    test_pred = loaded_model.predict(X_val.head(5))
    print(f"   OK Model loaded and prediction works")
    print(f"   Sample predictions: {test_pred}")

# Final summary
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

if embeddings is not None and embeddings.shape[1] == 24:
    print("OK SUCCESS! Everything works:")
    print("   1. TabNet training: OK")
    print("   2. Embedding extraction: OK (24 dimensions)")
    print("   3. Hybrid features: OK")
    print("   4. LightGBM training: OK")
    print("   5. Model save/load: OK")
    print("\n🎉 Your embedding extraction code IS WORKING!")
    print("   You can proceed with full training")
elif embeddings is not None and embeddings.shape[1] == 1:
    print("WARNING  PARTIAL SUCCESS:")
    print("   1. TabNet training: OK")
    print("   2. Embedding extraction: WARNING  (only 1D, not 24D)")
    print("   3. Hybrid features: OK")
    print("   4. LightGBM training: OK")
    print("   5. Model save/load: OK")
    print("\n💡 Embedding extraction needs more debugging")
    print("   But 1D embeddings still work (as you saw in training)")
else:
    print("ERROR Embedding extraction failed")
    print("   Need to debug the TabNet internals access")

print("\n" + "="*70 + "\n")
