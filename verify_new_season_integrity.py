
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

    # 4. Opponent Strength Check
    print("\n--- TEST 4: OPPONENT STRENGTH CAUSALITY ---")
    
    # We need to pick a specific Game and Opponent
    # Let's pick a game in the middle of a season
    # Row 1000
    row = df.iloc[1000]
    game_date = row['GAME_DATE']
    game_id = row['GAME_ID']
    team_id = row['TEAM_ID']
    # We need to find the opponent team ID.
    # The dataset might not have TEAM_ID_OPP explicitly unless we preserved it?
    # build_features merged it but then dropped TEAM_ID_trash.
    # But usually we can infer it or we might have kept 'MATCHUP'.
    
    print(f"Checking Game {game_id} ({game_date}) for Player {row['PLAYER_ID']}...")
    print(f"Matchup: {row['MATCHUP']}")
    
    # We need to find who the opponent is.
    # In this dataset, we don't have TEAM_ID_OPP column?
    # Let's check columns again.
    # df.columns from previous step: ... 'matchup_map' was merged but columns might not be there if not explicitly selected?
    # Wait, build_features did:
    # self.df = self.df.merge(matchup_map, on=['GAME_ID', 'TEAM_ID'], how='left')
    # matchup_map had TEAM_ID_OPP. So it SHOULD be there.
    
    if 'TEAM_ID_OPP' not in df.columns:
        print("WARNING: TEAM_ID_OPP not found in columns. Attempting to deduce...")
        # We can't easily deduce without fetching all rows for the game.
        # Let's assume the previous merge worked and check if we can see it in columns.
        pass
        
    # Let's verify 'opp_allow_pts_roll_5'
    target_feat = row['opp_allow_pts_roll_5']
    print(f"Feature Value (Avg Pts Allowed by Opponent Last 5): {target_feat}")
    
    if pd.isna(target_feat) or target_feat == 0:
        print("Feature is NaN or 0 (early season?). Skipping precise check.")
    else:
        # We need to find the Opponent's previous games.
        # 1. Identify Opponent Team ID.
        # If we can't find it in the row, we look at the raw logs for this game ID.
        subset_game = df[df['GAME_ID'] == game_id]
        teams = subset_game['TEAM_ID'].unique()
        opp_team_id = [t for t in teams if t != team_id]
        if len(opp_team_id) > 0:
            opp_team_id = opp_team_id[0]
            print(f" identified Opponent Team ID: {opp_team_id}")
            
            # 2. Get all games for Opponent BEFORE this date
            opp_games = df[(df['TEAM_ID'] == opp_team_id) & (df['GAME_DATE'] < game_date)]
            # We need unique games (one row per game)
            opp_games_unique = opp_games.drop_duplicates(subset=['GAME_ID'])
            opp_games_unique = opp_games_unique.sort_values('GAME_DATE', ascending=False).head(5)
            
            print(f"Opponent's Last 5 Games Dates: {opp_games_unique['GAME_DATE'].astype(str).tolist()}")
            
            # 3. Calculate Points Allowed in those games
            # Points Allowed = Points scored by THEIR opponent.
            # We need to find the points scored by the team playing AGAINST opp_team_id in those games.
            # This is hard to get from just 'df' which is player-centric? 
            # Actually, we have 'PTS' in df.
            # In a game where Team X plays Team Y:
            # We have rows for Team X players (sum PTS = Team X Score)
            # We have rows for Team Y players (sum PTS = Team Y Score)
            
            manual_allows = []
            for og_id in opp_games_unique['GAME_ID']:
                # Who did they play?
                og_rows = df[df['GAME_ID'] == og_id]
                # We want sum(PTS) of the OTHER team
                other_team_rows = og_rows[og_rows['TEAM_ID'] != opp_team_id]
                allowed = other_team_rows['PTS'].sum()
                manual_allows.append(allowed)
                
            print(f"Manual Pts Allowed: {manual_allows}")
            manual_avg = np.mean(manual_allows)
            print(f"Manual Avg: {manual_avg}")
            
            if abs(manual_avg - target_feat) < 0.1:
                print("SUCCESS: Opponent Strength matches manual calculation.")
            else:
                print("FAILURE: Mismatch.")
        else:
            print("Could not identify opponent from local subset.")
    # 5. Season 2025-26 Check
    print("\n--- TEST 5: SEASON 2025-26 INTEGRITY ---")
    
    # 2025 season start ~ Oct 2025
    season_start = pd.Timestamp('2025-10-01')
    recent_games = df[df['GAME_DATE'] > season_start].copy()
    
    print(f"2025-26 Games Found: {len(recent_games)}")
    
    if len(recent_games) == 0:
        print("FAILURE: No 2025-26 data found!")
        return
        
    # Check max date
    max_date = recent_games['GAME_DATE'].max()
    print(f"Latest Game Date: {max_date}")
    
    # Check features for a recent game
    sample_row = recent_games.iloc[-1]
    print(f"Sample Row ({sample_row['GAME_DATE']} - {sample_row['PLAYER_NAME']}):")
    cols = ['roll_PTS_5', 'season_PTS_avg', 'lag_PTS_1', 'opp_allow_pts_roll_10']
    for c in cols:
        print(f"  {c}: {sample_row[c]}")
        if pd.isna(sample_row[c]) or sample_row[c] == 0:
            # First few games might be 0/NaN, but mid-season (Dec) should not be.
            if sample_row['game_number_in_season'] > 10: # We don't have game_number explicit?
                 pass 
                 # We can check simple logic
    
    # Verify Lag for a recent game
    # Lag PTS 1 should be PTS of previous game
    player_id = sample_row['PLAYER_ID']
    p_subset = df[df['PLAYER_ID'] == player_id].sort_values('GAME_DATE')
    
    last_pts = p_subset.iloc[-1]['PTS']
    prev_pts = p_subset.iloc[-2]['PTS']
    lag_val = p_subset.iloc[-1]['lag_PTS_1']
    
    print(f"Recent Game PTS: {last_pts}")
    print(f"Previous Game PTS: {prev_pts}")
    print(f"Lag Feature Value: {lag_val}")
    
    if abs(prev_pts - lag_val) < 0.01:
        print("SUCCESS: 2025-26 Lag features are strictly causal.")
    else:
        print("FAILURE: Lag mismatch in 2025-26.")

if __name__ == "__main__":
    verify_dataset()
