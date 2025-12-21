
import pandas as pd
import joblib
import os

LIVE_DATA = "data/live_inference_set.csv"
MODEL_DIR = "models/clean_v1"
OUTPUT_FILE = "predictions/live_2025_predictions.csv"
TARGETS = ['PTS', 'AST', 'REB']

def predict_live():
    print(f"Loading Live Data: {LIVE_DATA}...")
    df = pd.read_csv(LIVE_DATA, low_memory=False)
    
    # Feature Selection (Must match training exactly)
    # We can infer features from the training script logic or model metadata if saved
    # Re-defining here for safety
    drop_cols = [
        'SEASON_ID', 'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 
        'TEAM_NAME', 'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL', 
        'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 
        'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'PLUS_MINUS', 
        'FANTASY_PTS', 'VIDEO_AVAILABLE', 'season_type', 'season_start_year',
        'prev_date', 'TEAM_ID_OPP'
    ]
    
    # We need to ensure we only use columns that exist in df
    # And match column order? XGBoost usually handles by name if dataframe is passed, 
    # but strictly it's better to match.
    
    # Let's filter df to feature cols
    # We can get features from the loaded model if it's sklearn/xgb wrapper
    # But let's assume the drop logic is consistent.
    
    # Identify feature columns
    feature_cols = [c for c in df.columns if c not in drop_cols]
    print(f"Using {len(feature_cols)} features.")
    
    X_live = df[feature_cols]
    
    # Predictions
    results = df[['GAME_DATE', 'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'MATCHUP', 'PTS', 'AST', 'REB']].copy()
    # Note: PTS/AST/REB in results are the ACTUALS (if available, for checking accuracy so far)
    results.rename(columns={'PTS': 'actual_PTS', 'AST': 'actual_AST', 'REB': 'actual_REB'}, inplace=True)
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    for target in TARGETS:
        model_path = os.path.join(MODEL_DIR, f"xgb_{target}.joblib")
        print(f"Loading model {target} from {model_path}...")
        
        if not os.path.exists(model_path):
            print(f"Model for {target} not found!")
            continue
            
        model = joblib.load(model_path)
        
        # Predict
        print(f"Predicting {target}...")
        preds = model.predict(X_live)
        
        results[f'pred_{target}'] = preds
        
    # Save
    results.to_csv(OUTPUT_FILE, index=False)
    print(f"Predictions saved to {OUTPUT_FILE}")
    print(results.head())

if __name__ == "__main__":
    predict_live()
