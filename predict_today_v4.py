
import pandas as pd
import joblib
import os

# Configuration
# Configuration
INPUT_PATH = "data/today_inference.csv"
OUTPUT_PATH = "predictions/live_ensemble_2025.csv" # Updated to match Betting Strategy requirement
BASE_DIR = "models/base_v4"
META_DIR = "models/meta_v4"

TARGETS = ['PTS', 'AST', 'REB']

def predict_today():
    if not os.path.exists(INPUT_PATH):
        print(f"Input {INPUT_PATH} not found.")
        return

    print(f"Loading features from {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    
    # Feature Selection
    # Must match training
    drop_cols = [
        'SEASON_ID', 'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 
        'TEAM_NAME', 'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL', 
        'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 
        'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'PLUS_MINUS', 
        'FANTASY_PTS', 'VIDEO_AVAILABLE', 'season_type', 'season_start_year',
        'prev_date', 'TEAM_ID_OPP', 'role_trend_min'
    ]
    
    feature_cols = [c for c in df.columns if c not in drop_cols]
    print(f"Features: {len(feature_cols)} columns")
    
    X = df[feature_cols].copy()
    
    # Fill NaN with 0 for safety (inference time)
    X = X.fillna(0)
    
    results = df[['GAME_DATE', 'PLAYER_NAME', 'TEAM_ABBREVIATION', 'MATCHUP']].copy()
    
    # Need catboost generic info
    from catboost import CatBoostRegressor
    
    for target in TARGETS:
        print(f"Predicting {target}...")
        
        # Base Models
        base_preds = pd.DataFrame(index=X.index)
        for name in ['xgb', 'lgb', 'cat']:
            try:
                if name == 'cat':
                    path = os.path.join(BASE_DIR, f"{name}_{target}.cbm")
                    model = CatBoostRegressor()
                    model.load_model(path)
                    base_preds[f'pred_{target}_{name}'] = model.predict(X)
                else:
                    path = os.path.join(BASE_DIR, f"{name}_{target}.joblib")
                    model = joblib.load(path)
                    base_preds[f'pred_{target}_{name}'] = model.predict(X)
            except Exception as e:
                print(f"  Error loading {name}: {e}")
                base_preds[f'pred_{target}_{name}'] = 0
        
        # Meta
        stacker_path = os.path.join(META_DIR, f"stacker_{target}.joblib")
        if os.path.exists(stacker_path):
            stacker = joblib.load(stacker_path)
            results[f'pred_{target}'] = stacker.predict(base_preds)
        else:
            results[f'pred_{target}'] = base_preds.mean(axis=1)
            
    print(f"Saving predictions to {OUTPUT_PATH}...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)
    print("Done.")

if __name__ == "__main__":
    predict_today()
