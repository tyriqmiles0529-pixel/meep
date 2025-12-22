import pandas as pd
import numpy as np
import joblib
import os

MODELS_DIR = "models/production_v4"
ODDS_API_FILE = "historical_data/the_odds_api_historical.csv"
STRICT_FEATURES = "data/strict_features_v4.csv"
OUTPUT_FILE = "data/validation_projections_v4.csv"

def generate_validation_data():
    print("=== Phase J.4: Validation Projection Generation (Fixed) ===")
    
    # 1. Load Historical Odds
    odds = pd.read_csv(ODDS_API_FILE)
    market_map = {
        'player_points': 'PTS',
        'player_assists': 'AST',
        'player_rebounds': 'REB'
    }
    odds = odds[odds['market'].isin(market_map.keys())]
    print(f"Loaded {len(odds)} historical prop lines.")
    
    validation_pairs = odds[['player_name', 'game_date']].drop_duplicates()
    players_to_find = set(validation_pairs['player_name'].str.upper())
    print(f"Unique players to match: {len(players_to_find)}")
    
    # Needs (player, date) set for final filtering
    lookup = set(zip(validation_pairs['player_name'].str.upper(), validation_pairs['game_date']))
    
    # 2. Extract features
    features_list = joblib.load(os.path.join(MODELS_DIR, "features.joblib"))
    
    matched_features = []
    chunk_size = 200000
    print("Scanning strict matrix...")
    for chunk in pd.read_csv(STRICT_FEATURES, chunksize=chunk_size):
        # Vectorized player name filter first
        mask = chunk['PLAYER_NAME'].str.upper().isin(players_to_find)
        if mask.any():
            sub_chunk = chunk[mask]
            # Precise date match
            sub_chunk = sub_chunk[sub_chunk.apply(lambda row: (str(row['PLAYER_NAME']).upper(), str(row['GAME_DATE'])) in lookup, axis=1)]
            if not sub_chunk.empty:
                matched_features.append(sub_chunk)
            
    if not matched_features:
        print("Error: No feature matches found.")
        return
        
    df_features = pd.concat(matched_features).drop_duplicates(subset=['PLAYER_NAME', 'GAME_DATE'])
    print(f"Matched {len(df_features)} feature records.")
    
    # 3. Predict
    results = df_features[['PLAYER_NAME', 'GAME_DATE', 'TEAM_ID']].copy()
    X = df_features[features_list].fillna(0)
    
    for target_key, target_col in market_map.items():
        print(f"Projecting {target_col}...")
        model = joblib.load(os.path.join(MODELS_DIR, f"xgb_{target_col}.joblib"))
        results[f'proj_{target_col}'] = model.predict(X).clip(0)
        
    results.to_csv(OUTPUT_FILE, index=False)
    print(f"[SUCCESS] Validation Projections saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_validation_data()
