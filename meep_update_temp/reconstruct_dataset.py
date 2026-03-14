import pandas as pd
import numpy as np
import os
import re

# Paths
RAW_EOIN = 'raw_data/eoinamoore'
RAW_SUMIT = 'raw_data/sumitrodatta'
OUTPUT_FILE = 'final_feature_matrix_with_per_min_1997_onward.csv'

def clean_name(name):
    """Standardizes player names for merging."""
    if pd.isna(name):
        return ""
    # Lowercase, remove periods, strip whitespace
    name = str(name).lower().replace('.', '').strip()
    # Handle common suffixes if needed (Jr, III) - keeping simple for now
    return name

def get_season(date_obj):
    """Derives NBA season from date. Oct-Dec is start of season (Year+1)."""
    if date_obj.month >= 10:
        return date_obj.year + 1
    return date_obj.year

def main():
    print("Loading Games data...")
    games = pd.read_csv(os.path.join(RAW_EOIN, 'Games.csv'))
    # Handle timezone offsets by converting to UTC with mixed format support
    try:
        games['gameDateTimeEst'] = pd.to_datetime(games['gameDateTimeEst'], utc=True, format='mixed')
    except ValueError:
        # Fallback for older pandas versions or stubborn formats
        games['gameDateTimeEst'] = pd.to_datetime(games['gameDateTimeEst'], utc=True, errors='coerce')
        
    games['season'] = games['gameDateTimeEst'].apply(get_season)
    
    # Filter 1997+
    games = games[games['season'] >= 1997]
    print(f"Filtered to {len(games)} games from 1997-present.")
    
    print("Loading Player Game Logs...")
    p_stats = pd.read_csv(os.path.join(RAW_EOIN, 'PlayerStatistics.csv'))
    
    # Merge Games info (Date, Season)
    print("Merging Games info...")
    # Drop gameDateTimeEst from p_stats to avoid collision/suffixes, as we want the parsed one from games
    p_stats = p_stats.drop(columns=['gameDateTimeEst'], errors='ignore')
    df = p_stats.merge(games[['gameId', 'gameDateTimeEst', 'season', 'winner']], on='gameId', how='inner')
    print(f"Merged dataset size: {len(df)} rows.")
    
    # Create clean name for joining
    df['clean_name'] = (df['firstName'] + ' ' + df['lastName']).apply(clean_name)
    
    # Load Season Stats (Target for some features, or just useful context)
    print("Loading Season Stats...")
    season_stats = pd.read_csv(os.path.join(RAW_SUMIT, 'Player Per Game.csv'))
    season_stats['clean_name'] = season_stats['player'].apply(clean_name)
    
    # Load Fuzzy Mapping
    if os.path.exists('player_name_mapping.csv'):
        print("Loading fuzzy name mapping...")
        mapping_df = pd.read_csv('player_name_mapping.csv')
        # Create dictionary: eoin_name -> sumit_name
        name_map = dict(zip(mapping_df['eoin_name'], mapping_df['sumit_name']))
        # Apply mapping to df['clean_name'] where applicable
        df['clean_name'] = df['clean_name'].replace(name_map)
        
    # Rename columns to avoid collision
    season_stats = season_stats.rename(columns={'age': 'season_age', 'team': 'season_team'})
    
    # Deduplicate season stats (keep row with most games played)
    season_stats = season_stats.sort_values('g', ascending=False).drop_duplicates(subset=['clean_name', 'season'])
    
    # Merge Season Stats (Advanced)
    # CRITICAL: We merge on (clean_name, season - 1) to use PREVIOUS season's stats as priors.
    # This is safe and powerful. Using current season stats is leakage.
    print("Merging Previous Season Stats (Priors)...")
    season_stats['next_season'] = season_stats['season'] + 1
    
    # Select useful advanced columns (add more if needed from Advanced.csv)
    # For now, Player Per Game has: pts_per_game, trb_per_game, ast_per_game, etc.
    # We can use these as "prior" expectations.
    cols_to_use = ['clean_name', 'next_season', 'pts_per_game', 'trb_per_game', 'ast_per_game', 'mp_per_game']
    
    season_stats_prior = season_stats[cols_to_use].rename(columns={
        'pts_per_game': 'prior_pts',
        'trb_per_game': 'prior_reb',
        'ast_per_game': 'prior_ast',
        'mp_per_game': 'prior_mp'
    })
    
    df = df.merge(season_stats_prior, left_on=['clean_name', 'season'], right_on=['clean_name', 'next_season'], how='left')
    df = df.drop(columns=['next_season'])
    
    # Fill missing priors with 0 or global average? 
    # For now 0, or maybe we can fill with current season rolling avg later?
    for c in ['prior_pts', 'prior_reb', 'prior_ast', 'prior_mp']:
        df[c] = df[c].fillna(0)
    
    # --- Feature Engineering: Lags ---
    print("Generating Lag Features...")
    # Sort by Player ID and Date
    df = df.sort_values(['personId', 'gameDateTimeEst'])
    
    # Explicit Lags
    lag_cols = ['points', 'assists', 'reboundsTotal', 'numMinutes', 'threePointersMade']
    windows = [3, 5, 10, 20]
    
    for col in lag_cols:
        if col in df.columns:
            print(f"  Lagging {col}...")
            # Last game (lag 1)
            df[f'{col}_last_game'] = df.groupby('personId')[col].shift(1)
            
            # Rolling averages
            for w in windows:
                df[f'{col}_last_{w}_avg'] = df.groupby('personId')[col].transform(lambda x: x.shift(1).rolling(w).mean())
            
    # --- Feature Engineering: Rolling Opponent Strength ---
    print("Generating Opponent Defensive Metrics...")
    # We can use the 'games' df to calculate points allowed by each team
    # Create a long-form team stats df
    home_games = games[['gameDateTimeEst', 'hometeamId', 'awayScore']].rename(columns={'hometeamId': 'teamId', 'awayScore': 'points_allowed'})
    away_games = games[['gameDateTimeEst', 'awayteamId', 'homeScore']].rename(columns={'awayteamId': 'teamId', 'homeScore': 'points_allowed'})
    team_defense = pd.concat([home_games, away_games]).sort_values(['teamId', 'gameDateTimeEst'])
    
    # Calculate rolling points allowed
    for w in [10, 20]:
        team_defense[f'opp_pts_allowed_last_{w}'] = team_defense.groupby('teamId')['points_allowed'].transform(lambda x: x.shift(1).rolling(w).mean())
    
    # Merge back into main df based on opponentteamId
    # Note: In p_stats, we have 'opponentteamId' (we might need to map it if names are used, but eoin uses IDs usually)
    # Let's check columns. Eoin's PlayerStatistics.csv usually has 'opponentTeamId' or similar.
    # The head command showed 'opponentteamCity', 'opponentteamName'. It might not have ID directly?
    # Wait, the head output showed 'player_id', 'gameId', ... but I didn't see 'opponentteamId' explicitly in the CSV head output I saw earlier.
    # I saw 'opponentteamCity', 'opponentteamName'.
    # I can reconstruct opponentTeamId from the games merge.
    
    # In the initial merge: df = p_stats.merge(games[['gameId', ...]], ...)
    # Games has hometeamId and awayteamId.
    # We need to know which one is the opponent.
    # p_stats has 'teamId' usually? Or 'playerteamCity'?
    # Let's assume we can match on gameId and figure out opponent.
    
    # Actually, let's look at the columns again.
    # The head output showed: ..., playerteamCity, playerteamName, opponentteamCity, opponentteamName, ...
    # It didn't show team IDs.
    # But 'games' has team IDs.
    # Let's merge team IDs from games into df first.
    
    # Re-merge games to get team IDs if needed, or just use the team names if unique.
    # IDs are safer.
    # Let's update the initial merge to include team IDs from games.
    
    # ... Wait, I can't easily change the initial merge up top without a big replace.
    # But I can do a separate merge here.
    
    games_teams = games[['gameId', 'hometeamId', 'awayteamId']]
    df = df.merge(games_teams, on='gameId', how='left')
    
    # Determine opponentTeamId
    # We don't have player's teamId explicitly in the head output I saw (it had playerteamCity).
    # But we can infer it. If playerteamName == games.homeTeamName? No, we have IDs in games.
    # Let's assume we can map names to IDs or just use the fact that we have 'home' column (boolean) in df?
    # The head output showed 'home' column! (1 or 0)
    
    if 'home' in df.columns:
        df['opponentTeamId'] = np.where(df['home'] == 1, df['awayteamId'], df['hometeamId'])
    else:
        # Fallback if 'home' is missing (it was in the merge list earlier)
        # We dropped 'home' from games merge in step 1029?
        # "df = p_stats.merge(games[['gameId', 'gameDateTimeEst', 'season', 'winner']], ...)"
        # Yes, I removed 'home' from the merge list!
        # I need to add it back or re-derive it.
        # p_stats usually has 'home' or 'isHome'?
        # The head output showed 'home' is NOT in the columns list I saw in step 1457.
        # Wait, step 1457 output: ...,win,home,minutes,...
        # It IS there! So p_stats must have it.
        df['opponentTeamId'] = np.where(df['home'] == 1, df['awayteamId'], df['hometeamId'])

    # Now merge defense stats
    df = df.merge(team_defense[['teamId', 'gameDateTimeEst', 'opp_pts_allowed_last_10', 'opp_pts_allowed_last_20']], 
                  left_on=['opponentTeamId', 'gameDateTimeEst'], 
                  right_on=['teamId', 'gameDateTimeEst'], 
                  how='left')
                  
    # Fill NaNs with global average
    for w in [10, 20]:
        col = f'opp_pts_allowed_last_{w}'
        df[col] = df[col].fillna(df[col].mean())
        
    df = df.drop(columns=['teamId', 'gameDateTimeEst_y', 'hometeamId', 'awayteamId', 'opponentTeamId'], errors='ignore')
    
    # --- Final Cleanup ---
    # Rename columns to match pipeline expectations if needed
    # Pipeline expects: 'points', 'minutes', 'player_name'
    df = df.rename(columns={
        'personId': 'player_id',
        'firstName': 'first_name',
        'lastName': 'last_name',
        'reboundsTotal': 'rebounds',
        'numMinutes': 'minutes',
        'threePointersMade': 'three_pointers',
        'gameDateTimeEst': 'date'
    })
    
    df['player_name'] = df['first_name'] + ' ' + df['last_name']
    
    # Fill NaNs in lags with 0 (start of career/season)
    lag_features = [c for c in df.columns if 'last_game' in c or '_avg' in c]
    df[lag_features] = df[lag_features].fillna(0)
    
    print(f"Saving final dataset with {len(df)} rows and {len(df.columns)} columns...")
    df.to_csv(OUTPUT_FILE, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
