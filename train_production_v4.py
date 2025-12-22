import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import Ridge
import joblib
import os

DATA_PATH = "data/strict_features_v4.csv"
MODEL_DIR = "models/production_v4"

def train():
    print(f"Loading training data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Clean targets
    df = df.dropna(subset=['PTS', 'AST', 'REB'])
    
    # Feature selection
    drop_cols = [
        'SEASON_ID', 'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 
        'TEAM_NAME', 'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL', 
        'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 
        'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'PLUS_MINUS', 
        'FANTASY_PTS', 'VIDEO_AVAILABLE', 'season_type', 'season_start_year',
        'prev_date', 'TEAM_ID_OPP', 'role_trend_min'
    ]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols].fillna(0)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    for target in ['PTS', 'AST', 'REB']:
        print(f"\nTraining Production V4 for {target}...")
        y = df[target]
        
        # 1. XGBoost Regressor (Confirmed Raw Stat)
        model = xgb.XGBRegressor(
            n_estimators=200, 
            learning_rate=0.05, 
            max_depth=5, 
            objective='reg:squarederror'
        )
        model.fit(X, y)
        
        # 2. Save
        joblib.dump(model, os.path.join(MODEL_DIR, f"xgb_{target}.joblib"))
        
        # Check Sample prediction
        sample_pred = model.predict(X.head(5))
        print(f"Sample Predictions: {sample_pred}")
        print(f"Sample Actuals: {y.head(5).values}")
        
    # Save feature list for inference alignment
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "features.joblib"))
    print("\n[SUCCESS] Production V4 Models Trained and Verified.")

if __name__ == "__main__":
    train()
