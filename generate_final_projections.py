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
    # J.6: Added FG3M (3PM)
    for target in ['PTS', 'AST', 'REB', 'FG3M']:
        model_path = os.path.join(MODELS_DIR, f"xgb_{target}.joblib")
        if os.path.exists(model_path):
            models[target] = joblib.load(model_path)
        
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
    
    # Team Mapping for Display
    team_map = {
        1610612737: 'ATL', 1610612738: 'BOS', 1610612739: 'CLE', 1610612740: 'NOH', 1610612741: 'CHI',
        1610612742: 'DAL', 1610612743: 'DEN', 1610612744: 'GSW', 1610612745: 'HOU', 1610612746: 'LAC',
        1610612747: 'LAL', 1610612748: 'MIA', 1610612749: 'MIL', 1610612750: 'MIN', 1610612751: 'BKN',
        1610612752: 'NYK', 1610612753: 'ORL', 1610612754: 'IND', 1610612755: 'PHI', 1610612756: 'PHX',
        1610612757: 'POR', 1610612758: 'SAC', 1610612759: 'SAS', 1610612760: 'OKC', 1610612761: 'TOR',
        1610612762: 'UTA', 1610612763: 'MEM', 1610612764: 'WAS', 1610612765: 'DET', 1610612766: 'CHA'
    }
    results['team'] = results['TEAM_ID'].map(team_map).fillna("UNK")
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
    if 'proj_FG3M' in results.columns:
        print("3PM (FG3M) projections included.")

if __name__ == "__main__":
    generate()
