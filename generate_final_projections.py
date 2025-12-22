import pandas as pd
import numpy as np
import joblib
import os

MODELS_DIR = "models/production_v4"
DATA_SOURCE = "data/strict_features_v4.csv"
OUTPUT_FILE = "predictions/live_ensemble_2025.csv"

def generate():
    print("=== Generating Production V4 Projections (Correct Columns) ===")
    
    # 1. Get player list
    print("Fetching player list...")
    # TEAM_ABBREVIATION is missing, use TEAM_ID
    df_players = pd.read_csv(DATA_SOURCE, usecols=['PLAYER_NAME', 'TEAM_ID', 'GAME_DATE', 'GAME_ID'], skipinitialspace=True)
    df_players = df_players.sort_values(['PLAYER_NAME', 'GAME_DATE'], ascending=[True, False])
    latest_per_player = df_players.groupby('PLAYER_NAME').head(1)
    
    print(f"Found {len(latest_per_player)} unique players.")
    
    # 2. Load Models
    features = joblib.load(os.path.join(MODELS_DIR, "features.joblib"))
    models = {}
    for target in ['PTS', 'AST', 'REB']:
        models[target] = joblib.load(os.path.join(MODELS_DIR, f"xgb_{target}.joblib"))
        
    # 3. Process data in chunks to find features
    print("Processing features in chunks...")
    latest_features = []
    lookup = latest_per_player[['PLAYER_NAME', 'GAME_DATE']].drop_duplicates()
    
    chunk_size = 100000
    for chunk in pd.read_csv(DATA_SOURCE, chunksize=chunk_size, skipinitialspace=True):
        match = pd.merge(chunk, lookup, on=['PLAYER_NAME', 'GAME_DATE'])
        if not match.empty:
            latest_features.append(match)
            
    if not latest_features:
        print("Error: No matching features found.")
        return
        
    df_final_features = pd.concat(latest_features).drop_duplicates(subset=['PLAYER_NAME'])
    print(f"Extracted features for {len(df_final_features)} players.")
    
    # 4. Predict
    results = df_final_features[['PLAYER_NAME', 'TEAM_ID', 'GAME_ID']].copy()
    # Placeholder for Team Abbreviation (since it's missing in this file)
    results['team'] = "UNK" 
    results.rename(columns={'PLAYER_NAME': 'player', 'GAME_ID': 'game_id'}, inplace=True)
    
    X = df_final_features[features].fillna(0)
    for target, model in models.items():
        print(f"Predicting {target}...")
        results[f'proj_{target}'] = model.predict(X).clip(0)
        
    # 5. Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[SUCCESS] Saved to {OUTPUT_FILE}")
    print(results[['player', 'proj_PTS', 'proj_AST', 'proj_REB']].head())

if __name__ == "__main__":
    generate()
