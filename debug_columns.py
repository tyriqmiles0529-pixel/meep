
import pandas as pd
import os
from datetime import datetime

PREDICTIONS_PATH = "predictions/live_ensemble_2025.csv"

def debug_data():
    if not os.path.exists(PREDICTIONS_PATH):
        print(f"[FAIL] Predictions missing: {PREDICTIONS_PATH}")
        return
        
    print(f"Loading predictions from {PREDICTIONS_PATH}...")
    preds = pd.read_csv(PREDICTIONS_PATH)
    
    col_map = {
        'player': 'player_name', 
        'proj_PTS': 'pred_points', 
        'proj_AST': 'pred_assists', 
        'proj_REB': 'pred_rebounds',
        'proj_FG3M': 'pred_three_pointers'
    }
    preds = preds.rename(columns=col_map)
    preds['player_name_key'] = preds['player_name'].str.lower().str.strip()
    
    print("Predictions columns:", preds.columns.tolist())
    print("First 5 rows of preds['player_name']:")
    print(preds['player_name'].head())

    # Mock odds_df
    odds_list = [
        {'player_name': 'Grant Long', 'event_id': '123', 'market': 'points', 'line': 10.5, 'odds_over': -110, 'odds_under': -110, 'bookmaker': 'fanduel'}
    ]
    for p in odds_list:
        p['player_name_key'] = p['player_name'].lower().strip()
    
    odds_df = pd.DataFrame(odds_list)
    print("Odds columns:", odds_df.columns.tolist())
    
    merged = pd.merge(preds, odds_df, on='player_name_key', how='inner', suffixes=('', '_odds'))
    print("Merged columns:", merged.columns.tolist())
    
    if 'player_name' in merged.columns:
        print("SUCCESS: 'player_name' found in merged.")
    else:
        print("FAIL: 'player_name' MISSING from merged.")

if __name__ == "__main__":
    debug_data()
