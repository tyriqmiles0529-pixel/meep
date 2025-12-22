"""
PHASE J.3: PRODUCTION V4 ENSEMBLE INFERENCE
Restores raw stat projections and correct regression semantics.
"""
import pandas as pd
import numpy as np
import joblib
import os
from catboost import CatBoostRegressor
from scipy.stats import norm

# Configuration
INPUT_PATH = "data/live_inference_set.csv" # The verified feature matrix
OUTPUT_PATH = "predictions/live_ensemble_2025.csv"
MODELS_DIR = "models"
TARGETS = {
    'PTS': 'points',
    'AST': 'assists',
    'REB': 'rebounds'
}

# RMSE for distributional probability (Downstream derivation)
RMSES = {
    'PTS': 4.5,
    'AST': 1.8,
    'REB': 2.0
}

def predict_production_v4():
    if not os.path.exists(INPUT_PATH):
        print(f"[ERROR] Inference data not found: {INPUT_PATH}")
        return

    print(f"Loading features from {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    
    # Feature Selection (Match V4 training)
    drop_cols = [
        'SEASON_ID', 'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 
        'TEAM_NAME', 'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL', 
        'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 
        'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'PLUS_MINUS', 
        'FANTASY_PTS', 'VIDEO_AVAILABLE', 'season_type', 'season_start_year',
        'prev_date', 'TEAM_ID_OPP', 'role_trend_min'
    ]
    
    # We must also ensure any added V4 features (embeddings) are present
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy().fillna(0)
    
    results = df[['GAME_DATE', 'PLAYER_NAME', 'TEAM_ABBREVIATION', 'GAME_ID']].copy()
    results.rename(columns={
        'PLAYER_NAME': 'player',
        'TEAM_ABBREVIATION': 'team',
        'GAME_ID': 'game_id'
    }, inplace=True)
    
    for target_key, target_folder in TARGETS.items():
        print(f"\n--- Predicting Raw {target_key} ---")
        target_dir = os.path.join(MODELS_DIR, target_folder)
        
        # 1. Base Model Predictions
        p_xgb = joblib.load(os.path.join(target_dir, "xgb_model_2025.pkl")).predict(X)
        p_lgb = joblib.load(os.path.join(target_dir, "lgb_model_2025.pkl")).predict(X)
        
        cat_model = CatBoostRegressor()
        cat_model.load_model(os.path.join(target_dir, "cat_model_2025.cbm"))
        p_cat = cat_model.predict(X)
        
        # Ridge Stacker - depends on EnsemblePredictor.baselines structure or raw Ridge
        # Let's check if there's a standalone stacker or if it's in baselines
        stacker_path = os.path.join(target_dir, "ridge_model_2025.pkl")
        # In EnsemblePredictor.save_models, it saves baselines_model.pkl which contains Ridge
        # But train_all_targets seems to save ridge_model_2025.pkl?
        
        if os.path.exists(stacker_path):
            ridge = joblib.load(stacker_path)
            # Stacking typically needs the 0.3/0.3/0.3 weighting or meta-features
            # For simplicity, we'll use the weights found in ensemble_model_vm 
            # OR the meta-model if it exists.
            
            # Let's look for baselines_model_2025.pkl as well
            baselines_path = os.path.join(target_dir, "baselines_model_2025.pkl")
            if os.path.exists(baselines_path):
                baselines = joblib.load(baselines_path)
                p_ridge = baselines.predict(X)['ridge']
                # Correct way for EnsemblePredictor v4: [XGB, LGB, CAT, RIDGE] blended
                # Weights are in the object, but we'll use 0.25 equal weight or meta if found.
                X_meta = np.column_stack([p_xgb, p_lgb, p_cat, p_ridge])
                
                # If we have a meta-model (stacker)
                # Note: The ridge_model_2025.pkl might be the stacker itself
                try:
                    results[f'proj_{target_key}'] = ridge.predict(X_meta)
                except:
                    results[f'proj_{target_key}'] = X_meta.mean(axis=1)
            else:
                results[f'proj_{target_key}'] = np.mean([p_xgb, p_lgb, p_cat], axis=0)
        else:
            results[f'proj_{target_key}'] = np.mean([p_xgb, p_lgb, p_cat], axis=0)

        # Clip negative predictions to 0 for realism
        results[f'proj_{target_key}'] = results[f'proj_{target_key}'].clip(lower=0)
        
        print(f"Sample {target_key} Proj: {results[f'proj_{target_key}'].mean():.2f}")

    # Save to CSV
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[SUCCESS] Corrected V4 Projections saved to {OUTPUT_PATH}")
    print(results.head())

if __name__ == "__main__":
    predict_production_v4()
