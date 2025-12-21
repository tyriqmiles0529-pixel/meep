import pandas as pd
import numpy as np

def check_leakage():
    print("Loading snippet of dataset...")
    # Load enough rows to get a full season for a player
    df = pd.read_csv('final_feature_matrix_with_per_min_1997_onward.csv', nrows=50000)
    
    # Standardize columns to lower case for easy check
    df.columns = [c.lower() for c in df.columns]

    print("All Columns:", df.columns.tolist())
    
    # Check for name variations
    player_col = next((c for c in df.columns if 'player' in c and 'name' in c), None)
    if not player_col: player_col = next((c for c in df.columns if 'player' in c), None)
    
    season_col = next((c for c in df.columns if 'season' in c), None)
    
    if not player_col or not season_col:
        print(f"Required columns missing. Found Player: {player_col}, Season: {season_col}")
        return

    # Renaming for consistency
    df = df.rename(columns={player_col: 'player_name', season_col: 'season'})


    # Pick a player with > 50 games in a season
    player = "Luka Doncic"
    season = 2021
    
    subset = df[(df['player_name'] == player) & (df['season'] == season)].sort_values('date')
    
    if len(subset) == 0:
        # Fallback to mostly frequent player
        top_player = df['player_name'].mode()[0]
        season = df[df['player_name'] == top_player]['season'].mode()[0]
        subset = df[(df['player_name'] == top_player) & (df['season'] == season)].sort_values('date')
        player = top_player
        
    print(f"\nAnalyzing {player} Season {season} ({len(subset)} games)")
    
    # Check potential leakage columns
    leak_candidates = [c for c in df.columns if 'season_avg' in c]
    
    if not leak_candidates:
        print("No 'season_avg' columns found. Checking for other patterns...")
        leak_candidates = [c for c in df.columns if 'avg' in c and 'season' in c]

    print(f"Checking {len(leak_candidates)} potential leakage features.")
    
    for col in leak_candidates[:5]: # Check first 5
        print(f"\nChecking Feature: {col}")
        
        # Get the value from the FIRST game of the season
        first_game_val = subset.iloc[0][col]
        last_game_val = subset.iloc[-1][col]
        
        # Calculate actual season average
        # Heuristic: assume 'points' or equivalent is the raw metric
        # We need to guess the raw metric. 'pts_season_avg' -> 'pts' or 'points'?
        raw_col = None
        if 'points' in col: raw_col = 'points'
        elif 'assists' in col: raw_col = 'assists'
        elif 'rebounds' in col or 'trb' in col: raw_col = 'rebounds'
        
        if raw_col and raw_col in df.columns:
            actual_mean = subset[raw_col].mean()
            print(f"  Game 1 Value: {first_game_val}")
            print(f"  Game N Value: {last_game_val}")
            print(f"  Actual Season Mean: {actual_mean}")
            
            if abs(first_game_val - actual_mean) < 0.01:
                print("  [CRITICAL LEAK] Game 1 feature equals final season mean!")
            else:
                print("  [PASS] Feature varies or does not match final mean.")
        else:
            print("  (Could not identify raw column to verify)")

    # Also check the Opponent Leak spotted in code
    # We saw: opp_def_points
    if 'opp_def_points' in df.columns:
        print("\nChecking Computed Opponent Leak (opp_def_points from code)...")
        # In the CSV, this might not exist yet if it's computed in-memory.
        # But if it IS in the CSV, let's check.
        pass
    else:
        print("\n'opp_def_points' not in CSV (likely computed in-memory). Leak is in data_processor.py code.")

if __name__ == "__main__":
    check_leakage()
