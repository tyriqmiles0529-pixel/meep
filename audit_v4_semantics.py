import pandas as pd
import joblib
import os
import numpy as np
from catboost import CatBoostRegressor

def audit_model():
    print("=== Model Semantic Audit (V4 2025 Models) ===")
    
    # 1. Load data
    DATA_PATH = "data/strict_features_v4.csv"
    df = pd.read_csv(DATA_PATH, nrows=10)
    
    # Identify feature columns (match predict_today_v4.py logic)
    drop_cols = [
        'SEASON_ID', 'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 
        'TEAM_NAME', 'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL', 
        'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 
        'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'PLUS_MINUS', 
        'FANTASY_PTS', 'VIDEO_AVAILABLE', 'season_type', 'season_start_year',
        'prev_date', 'TEAM_ID_OPP'
    ]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X_test = df[feature_cols].head(1)
    
    # 2. Audit Points Models
    target = "points"
    target_dir = os.path.join("models", target)
    
    models_to_check = [
        ("XGB", f"xgb_model_2025.pkl", "joblib"),
        ("LGB", f"lgb_model_2025.pkl", "joblib"),
        ("CAT", f"cat_model_2025.cbm", "catboost"),
        ("Ridge (Stacker)", f"ridge_model_2025.pkl", "joblib")
    ]
    
    base_preds = {}
    
    for name, filename, mtype in models_to_check:
        path = os.path.join(target_dir, filename)
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
            
        try:
            if mtype == "joblib":
                model = joblib.load(path)
                if name == "Ridge (Stacker)":
                    # Stacker expects columns of base preds, we'll simulate if we have them
                    pass
                else:
                    p = model.predict(X_test)[0]
                    base_preds[name] = p
                    print(f"{name} Prediction: {p:.4f}")
            elif mtype == "catboost":
                model = CatBoostRegressor()
                model.load_model(path)
                p = model.predict(X_test)[0]
                base_preds[name] = p
                print(f"{name} Prediction: {p:.4f}")
        except Exception as e:
            print(f"Error checking {name}: {e}")

    actual_pts = df['PTS'].iloc[0]
    expected_mins = df['MIN'].iloc[0] if 'MIN' in df.columns else 30
    print(f"\nActual PTS in training row: {actual_pts}")
    print(f"Minutes in training row: {expected_mins}")
    
    if len(base_preds) > 0:
        avg_pred = np.mean(list(base_preds.values()))
        print(f"Average Base Prediction: {avg_pred:.4f}")
        
        # Check Per-Minute hypothesis
        per_min_scaled = avg_pred * expected_mins
        print(f"Hypothesis: If this is Per-Minute, Raw PTS = {per_min_scaled:.2f}")
        
        if abs(per_min_scaled - actual_pts) < abs(avg_pred - actual_pts):
            print(">>> CONCLUSION: Models are likely predicting PER-MINUTE stats.")
        elif 0 <= avg_pred <= 1:
            print(">>> CONCLUSION: Models are likely predicting PROBABILITIES.")
        else:
            print(">>> CONCLUSION: Models are predicting RAW stats but may be poorly calibrated.")

if __name__ == "__main__":
    audit_model()
