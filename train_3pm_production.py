"""
Phase J.6: Train 3PM (FG3M) Regression Model for Production
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import joblib
import os

FEATURES_PATH = "data/strict_features_v4.csv"
RAW_LOGS_PATH = "data/nba_game_logs_1997_2024.csv"
MODEL_DIR = "models/production_v4"

def train_3pm():
    print("=== Phase J.6: Training 3PM (FG3M) Regression Models ===")
    
    # 1. Load Raw Logs for FG3M Target
    print("Loading raw game logs for FG3M target...")
    raw_cols = ['PLAYER_NAME', 'GAME_DATE', 'FG3M']
    raw_logs = pd.read_csv(RAW_LOGS_PATH, usecols=raw_cols, skipinitialspace=True)
    raw_logs = raw_logs.dropna(subset=['FG3M'])
    print(f"Raw logs with FG3M: {len(raw_logs)} rows")
    
    # 2. Load Features
    print("Loading features...")
    features_list = joblib.load(os.path.join(MODEL_DIR, "features.joblib"))
    
    # Load data in chunks (large file)
    chunk_size = 200000
    matched_rows = []
    
    # Create lookup set
    lookup = set(zip(raw_logs['PLAYER_NAME'].str.upper(), raw_logs['GAME_DATE'].astype(str)))
    
    print("Matching features with FG3M targets (chunked)...")
    for chunk in pd.read_csv(FEATURES_PATH, chunksize=chunk_size, skipinitialspace=True):
        mask = chunk.apply(lambda row: (str(row['PLAYER_NAME']).upper(), str(row['GAME_DATE'])) in lookup, axis=1)
        if mask.any():
            matched_rows.append(chunk[mask])
            
    if not matched_rows:
        print("[FATAL] No feature-target matches found. Aborting.")
        return
        
    df_features = pd.concat(matched_rows).drop_duplicates(subset=['PLAYER_NAME', 'GAME_DATE'])
    print(f"Matched {len(df_features)} feature rows.")
    
    # 3. Join FG3M
    df_features['player_key'] = df_features['PLAYER_NAME'].str.upper()
    raw_logs['player_key'] = raw_logs['PLAYER_NAME'].str.upper()
    raw_logs['GAME_DATE'] = raw_logs['GAME_DATE'].astype(str)
    df_features['GAME_DATE'] = df_features['GAME_DATE'].astype(str)
    
    df = pd.merge(df_features, raw_logs[['player_key', 'GAME_DATE', 'FG3M']], 
                  on=['player_key', 'GAME_DATE'], how='inner')
    print(f"Final training set with FG3M: {len(df)} rows")
    
    # 4. Prepare X and y
    X = df[features_list].fillna(0)
    y = df['FG3M']
    
    # 5. Train Models
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # XGBoost
    print("\nTraining XGBRegressor for FG3M...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=200, 
        learning_rate=0.05, 
        max_depth=5, 
        objective='reg:squarederror'
    )
    xgb_model.fit(X, y)
    joblib.dump(xgb_model, os.path.join(MODEL_DIR, "xgb_FG3M.joblib"))
    print(f"XGB Sample Predictions: {xgb_model.predict(X.head(5))}")
    print(f"XGB Sample Actuals: {y.head(5).values}")
    
    # LightGBM
    print("\nTraining LGBMRegressor for FG3M...")
    lgb_model = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        objective='regression'
    )
    lgb_model.fit(X, y)
    joblib.dump(lgb_model, os.path.join(MODEL_DIR, "lgb_FG3M.joblib"))
    
    # CatBoost
    print("\nTraining CatBoostRegressor for FG3M...")
    cat_model = CatBoostRegressor(
        iterations=200,
        learning_rate=0.05,
        depth=5,
        loss_function='RMSE',
        verbose=0
    )
    cat_model.fit(X, y)
    cat_model.save_model(os.path.join(MODEL_DIR, "cat_FG3M.cbm"))
    
    # 6. Compute RMSE (on training data as sanity check)
    preds = xgb_model.predict(X)
    rmse = np.sqrt(((preds - y) ** 2).mean())
    print(f"\n[SUCCESS] FG3M Models Trained.")
    print(f"Training RMSE (XGB): {rmse:.4f}")
    print(f"Locked sigma_3PM: {rmse:.2f}")
    
    # Save RMSE for later use
    with open(os.path.join(MODEL_DIR, "sigma_3PM.txt"), 'w') as f:
        f.write(str(rmse))
    
    return rmse

if __name__ == "__main__":
    train_3pm()
