
import pandas as pd
import numpy as np
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
import joblib
import os
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Config
TRAIN_DATA = "data/pro_training_set.csv"
MODEL_DIR = "models/base_v1"
OOF_DIR = "data/oof"
TARGETS = ['PTS', 'AST', 'REB']

def train_base_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OOF_DIR, exist_ok=True)
    
    print(f"Loading {TRAIN_DATA}...")
    df = pd.read_csv(TRAIN_DATA, low_memory=False)
    
    # Drop IDs and Leaks
    drop_cols = [
        'SEASON_ID', 'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 
        'TEAM_NAME', 'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL', 
        'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 
        'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'PLUS_MINUS', 
        'FANTASY_PTS', 'VIDEO_AVAILABLE', 'season_type', 'season_start_year',
        'prev_date', 'TEAM_ID_OPP'
    ]
    
    feature_cols = [c for c in df.columns if c not in drop_cols]
    print(f"Features: {feature_cols}")
    
    # Split
    # Train <= 2023, Val = 2024
    train_mask = df['season_start_year'] <= 2023
    val_mask = df['season_start_year'] == 2024
    
    X_train = df[train_mask][feature_cols]
    X_val = df[val_mask][feature_cols]
    
    # Store OOF Preds (actually Val preds here)
    # We will build a dataframe for 2024 with all preds
    oof_df = df[val_mask][['GAME_ID', 'GAME_DATE', 'PLAYER_ID', 'PLAYER_NAME', 'season_start_year'] + TARGETS].copy()
    
    models = {
        'xgb': lambda: xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42, early_stopping_rounds=50),
        'lgb': lambda: lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42),
        'cat': lambda: cb.CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, subsample=0.8, verbose=100, random_state=42, allow_writing_files=False)
    }
    
    results = []
    
    for target in TARGETS:
        print(f"\n=== TARGET: {target} ===")
        y_train = df[train_mask][target]
        y_val = df[val_mask][target]
        
        for name, model_factory in models.items():
            print(f"Training {name}...")
            model = model_factory()
            
            # Fitting
            if name == 'xgb':
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            elif name == 'lgb':
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='rmse', callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            elif name == 'cat':
                model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
                
            # Predict
            preds = model.predict(X_val)
            oof_df[f'pred_{target}_{name}'] = preds
            
            # Evaluate
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            r2 = r2_score(y_val, preds)
            print(f"  {name} RMSE: {rmse:.4f}, R2: {r2:.4f}")
            results.append({'target': target, 'model': name, 'rmse': rmse, 'r2': r2})
            
            # Save
            joblib.dump(model, os.path.join(MODEL_DIR, f"{name}_{target}.joblib"))
            
    # Save OOF
    oof_path = os.path.join(OOF_DIR, "oof_validation_2024.csv")
    oof_df.to_csv(oof_path, index=False)
    print(f"\nOOF Predictions saved to {oof_path}")
    
    # Print Summary
    res_df = pd.DataFrame(results)
    print("\nSummary:")
    print(res_df)
    res_df.to_csv(os.path.join(MODEL_DIR, "training_metrics.csv"), index=False)

if __name__ == "__main__":
    train_base_models()
