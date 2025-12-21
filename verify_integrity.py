
import pandas as pd
import numpy as np
import datetime

DATA_PATH = "data/strict_features_1997_2024.csv"

def verify_dataset():
    print(f"Loading {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Sort for sequential checking
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(['PLAYER_ID', 'GAME_DATE'])
    
    print(f"Loaded {len(df)} rows. Columns: {df.columns.tolist()}")

    # 1. Leakage Check: Does Feature(t) depend on Stats(t)?
    # We expect Feature(t) to be derived from Stats(t-1).
    # Specifically, `roll_PTS_3` at index T should contain the average of PTS at T-1, T-2, T-3.
    # It should NOT include PTS at T.
    
    print("\n--- TEST 1: CAUSALITY AUDIT ---")
    player_id = df['PLAYER_ID'].iloc[0]
    subset = df[df['PLAYER_ID'] == player_id].head(10).copy()
    
    print(f"Checking Player {player_id} (first 10 games)...")
    
    # Check Roll lag
    # Manual calculation
    pts = subset['PTS'].values
    roll_pts_3_feature = subset['roll_PTS_3'].values
    
    errors = 0
    for i in range(1, 10):
        # Correct: avg(pts[i-3:i]) approximately
        # Our shift(1) logic means `roll_PTS_3` at `i` is rolling mean of shifted column.
        # Shifted column at `i` is `pts[i-1]`.
        # So `roll_PTS_3` at `i` should use `pts[i-1], pts[i-2], pts[i-3]`.
        
        # Taking i=3 (4th game)
        # It relies on index 2, 1, 0.
        
        if i < 3: continue
            
        manual_calc = np.mean(pts[i-3:i])
        feature_val = roll_pts_3_feature[i]
        
        print(f"Game {i}: PTS history={pts[i-3:i]}, Mean={manual_calc:.2f}, Feature={feature_val:.2f}")
        
        if abs(manual_calc - feature_val) > 0.01:
            print("  >> MISMATCH!")
            errors += 1
        else:
            print("  >> OK")
            
    if errors == 0:
        print("SUCCESS: Rolling stats obey strict causality (shift-1 confirmed).")
    else:
        print(f"FAILURE: {errors} mismatches found.")

    # 2. Season Leakage Check
    print("\n--- TEST 2: SEASON LEAKAGE ---")
    # `season_PTS_avg` at game T should NOT equal the final average of the season.
    # It should range and converge.
    
    # Pick a player with >50 games
    counts = df['PLAYER_ID'].value_counts()
    player_id = counts[counts > 50].index[0]
    
    subset = df[df['PLAYER_ID'] == player_id]
    season_id = subset['SEASON_ID'].iloc[10] # Pick a season (skipping first few mixed seasons if any)
    season_subset = subset[subset['SEASON_ID'] == season_id]
    
    if len(season_subset) < 20: 
        print("Skipping season test (too few games)")
    else:
        first_val = season_subset['season_PTS_avg'].iloc[5]
        last_val = season_subset['season_PTS_avg'].iloc[-1]
        
        print(f"Player {player_id}, Season {season_id}")
        print(f"  Game 5 Expanding Avg: {first_val}")
        print(f"  Game N Expanding Avg: {last_val}")
        
        if first_val != last_val:
            print("SUCCESS: Season average evolves over time (Expanding Window).")
        else:
            print("FAILURE: Season average is constant (Future Leakage).")

    # 3. Lag Features
    print("\n--- TEST 3: LAG FEATURES ---")
    # lag_PTS_1 at T should equal PTS at T-1
    pts = season_subset['PTS'].values
    lags = season_subset['lag_PTS_1'].values
    
    mismatches = 0
    for i in range(1, len(pts)):
        if abs(pts[i-1] - lags[i]) > 0.01:
            mismatches += 1
            
    if mismatches == 0:
        print("SUCCESS: lag_PTS_1 matches previous game's PTS exactly.")
    else:
        print(f"FAILURE: {mismatches} lag errors.")
        
    print("\n--- DATA PREVIEW ---")
    print(subset[['GAME_DATE', 'PTS', 'lag_PTS_1', 'roll_PTS_3', 'season_PTS_avg', 'rest_days']].head(10))

if __name__ == "__main__":
    verify_dataset()
