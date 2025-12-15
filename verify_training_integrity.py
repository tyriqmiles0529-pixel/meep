import pandas as pd
import numpy as np
import joblib
import os
import torch
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import json

MODEL_DIR = 'models'
DATA_FILE = 'final_feature_matrix_with_per_min_1997_onward.csv'

def verify_models():
    print("=== 1. Verifying Models ===")
    
    # XGBoost
    try:
        xgb_model = joblib.load(os.path.join(MODEL_DIR, 'xgb_model_points.pkl'))
        print(f"[XGBoost] Loaded. Best Iteration: {xgb_model.best_iteration}/{xgb_model.n_estimators}")
        if xgb_model.best_iteration < 10:
            print("WARNING: XGBoost stopped very early!")
        else:
            print("OK: XGBoost trained significantly.")
    except Exception as e:
        print(f"FAIL: XGBoost load error: {e}")

    # LightGBM
    try:
        lgb_model = joblib.load(os.path.join(MODEL_DIR, 'lgb_model_points.pkl'))
        # LGBM stores best iteration in best_iteration_
        print(f"[LightGBM] Loaded. Best Iteration: {lgb_model.best_iteration_}/{lgb_model.n_estimators}")
        if lgb_model.best_iteration_ < 10:
             print("WARNING: LightGBM stopped very early!")
        else:
             print("OK: LightGBM trained significantly.")
    except Exception as e:
        print(f"FAIL: LightGBM load error: {e}")

    # CatBoost
    try:
        cat_model = cb.CatBoostRegressor()
        cat_model.load_model(os.path.join(MODEL_DIR, 'cat_model_points.cbm'))
        print(f"[CatBoost] Loaded. Tree Count: {cat_model.tree_count_}")
        if cat_model.tree_count_ < 10:
            print("WARNING: CatBoost stopped very early!")
        else:
            print("OK: CatBoost trained significantly.")
    except Exception as e:
        print(f"FAIL: CatBoost load error: {e}")

    # Ridge
    try:
        ridge_pipeline = joblib.load(os.path.join(MODEL_DIR, 'ridge_model_points.pkl'))
        # It's a pipeline, access 'model' step
        if hasattr(ridge_pipeline, 'named_steps'):
            ridge_model = ridge_pipeline.named_steps['model']
        else:
            ridge_model = ridge_pipeline
            
        print(f"[Ridge] Loaded. Coefficients shape: {ridge_model.coef_.shape}")
        if np.all(ridge_model.coef_ == 0):
            print("FAIL: Ridge coefficients are all zero!")
        else:
            print(f"OK: Ridge has non-zero coefficients. Max coef: {np.max(np.abs(ridge_model.coef_)):.4f}")
    except Exception as e:
        print(f"FAIL: Ridge load error: {e}")

    # FT-Transformer
    try:
        # Check for .pth extension if .pt fails, or check file listing
        ft_files = [f for f in os.listdir(MODEL_DIR) if 'ft_transformer' in f]
        if not ft_files:
             print("FAIL: No FT-Transformer model found in models/ directory.")
        else:
             ft_path = os.path.join(MODEL_DIR, ft_files[0])
             print(f"[FT-Transformer] Found file: {ft_path}")
             state_dict = torch.load(ft_path, map_location='cpu')
             print(f"[FT-Transformer] Loaded state_dict. Keys: {len(state_dict)}")
             print("OK: FT-Transformer loaded on CPU.")
    except Exception as e:
        print(f"FAIL: FT-Transformer load error: {e}")

def verify_data():
    print("\n=== 2. Verifying Data Integrity ===")
    if not os.path.exists(DATA_FILE):
        print(f"FAIL: Data file {DATA_FILE} not found.")
        return

    df = pd.read_csv(DATA_FILE)
    print(f"Dataset Shape: {df.shape}")
    
    # Check IDs
    if 'player_id' not in df.columns or 'gameId' not in df.columns:
        print("FAIL: Missing ID columns!")
    else:
        print("OK: ID columns present.")
        
    # Check Nulls in Target
    null_targets = df['points'].isnull().sum()
    print(f"Null Targets: {null_targets}")
    
    # Check Features
    expected_lags = ['points_last_game', 'points_last_10_avg']
    for f in expected_lags:
        if f not in df.columns:
            print(f"FAIL: Missing feature {f}")
        else:
            print(f"OK: Feature {f} present.")

    # Check for all-zero rows (suspicious)
    # We check numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    zero_rows = (numeric_df == 0).all(axis=1).sum()
    print(f"All-Zero Rows: {zero_rows}")

def main():
    verify_models()
    verify_data()

if __name__ == "__main__":
    main()
