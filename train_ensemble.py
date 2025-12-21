
import pandas as pd
import numpy as np
import xgboost as xgb
import argparse
import os
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configuration
TRAIN_DATA = "data/pro_training_set.csv"
MODEL_DIR = "models/clean_v1"
OUTPUT_DIR = "predictions"
TARGETS = ['PTS', 'AST', 'REB']

def train_base_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading Training Data: {TRAIN_DATA}...")
    df = pd.read_csv(TRAIN_DATA, low_memory=False)
    
    # Define Features (Exclude identifiers and future info)
    # We use the 'strict' features we built
    drop_cols = [
        'SEASON_ID', 'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 
        'TEAM_NAME', 'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL', 
        'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 
        'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'PLUS_MINUS', 
        'FANTASY_PTS', 'VIDEO_AVAILABLE', 'season_type', 'season_start_year',
        'prev_date', 'TEAM_ID_OPP'
    ]
    
    # Also drop any other leak candidates if they exist
    # (The strict pipeline should have only generated safe features + metadata)
    
    feature_cols = [c for c in df.columns if c not in drop_cols]
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    
    # Split Train (<=2023) / Validation (2024)
    # Note: 'season_start_year' 2023 means 2023-24 season. 2024 is 2024-25.
    # The user request said: "Validation: Strict time-series split (train <= 2023, validate on 2024)"
    # Our 'pro_training_set' contains data up to season 2024.
    
    train_mask = df['season_start_year'] <= 2023
    val_mask = df['season_start_year'] == 2024
    
    X_train = df[train_mask][feature_cols]
    X_val = df[val_mask][feature_cols]
    
    results = {}
    
    for target in TARGETS:
        print(f"\n--- Training Target: {target} ---")
        y_train = df[train_mask][target]
        y_val = df[val_mask][target]
        
        print(f"Train Rows: {len(X_train)}, Val Rows: {len(X_val)}")
        
        # XGBoost Regressor
        model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=50,
            n_jobs=-1,
            random_state=42
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=100
        )
        
        # Evaluate
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        mae = mean_absolute_error(y_val, preds)
        r2 = r2_score(y_val, preds)
        
        print(f"Validation Metrics ({target}):")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R2:   {r2:.4f}")
        
        results[target] = {'rmse': rmse, 'mae': mae, 'r2': r2}
        
        # Save Model
        model_path = os.path.join(MODEL_DIR, f"xgb_{target}.joblib")
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
        
    print("\n--- Summary Results ---")
    for t, m in results.items():
        print(f"{t}: RMSE={m['rmse']:.4f}, R2={m['r2']:.4f}")

if __name__ == "__main__":
    train_base_models()
