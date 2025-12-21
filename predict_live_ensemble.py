
import pandas as pd
import joblib
import os
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
import numpy as np

LIVE_DATA = "data/live_inference_set.csv"
BASE_DIR = "models/base_v1"
META_DIR = "models/meta_v1"
OUTPUT_FILE = "predictions/live_ensemble_2025.csv"
TARGETS = ['PTS', 'AST', 'REB']

def predict_live_ensemble():
    print(f"Loading Live Data: {LIVE_DATA}...")
    df = pd.read_csv(LIVE_DATA, low_memory=False)
    
    # Feature Selection (must match training)
    # Re-defining here for safety (same as training script)
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
    
    X = df[feature_cols]
    
    results = df[['GAME_DATE', 'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'MATCHUP', 'PTS', 'AST', 'REB']].copy()
    results.rename(columns={'PTS': 'actual_PTS', 'AST': 'actual_AST', 'REB': 'actual_REB'}, inplace=True)
    
    for target in TARGETS:
        print(f"\n=== TARGET: {target} ===")
        
        # 1. Base Model Predictions
        base_preds = pd.DataFrame(index=X.index)
        
        for name in ['xgb', 'lgb', 'cat']:
            path = os.path.join(BASE_DIR, f"{name}_{target}.joblib")
            if not os.path.exists(path):
                print(f"Warning: Model {path} not found.")
                continue
                
            print(f"Predicting with {name}...")
            model = joblib.load(path)
            base_preds[f'pred_{target}_{name}'] = model.predict(X)
            
            # Save Base Preds to results
            results[f'base_{target}_{name}'] = base_preds[f'pred_{target}_{name}']
            
        # 2. Meta Learner Prediction
        stacker_path = os.path.join(META_DIR, f"stacker_{target}.joblib")
        if os.path.exists(stacker_path):
            print("Predicting with Meta-Learner...")
            stacker = joblib.load(stacker_path)
            
            # Predict
            final_pred = stacker.predict(base_preds)
            results[f'pred_{target}'] = final_pred
        else:
            print("Meta-Learner not found, using simple average.")
            results[f'pred_{target}'] = base_preds.mean(axis=1)
            
    # Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved Ensemble Predictions to {OUTPUT_FILE}")
    print(results.head())

if __name__ == "__main__":
    predict_live_ensemble()
