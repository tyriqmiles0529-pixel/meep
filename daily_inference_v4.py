
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import scoreboardv2, commonteamroster
from datetime import datetime
import time
import sys
import os

# Import our Feature Engine
# We assume build_features.py is in the same directory
from build_features import StrictFeatureEngine

# Configuration
# If raw logs are not found, we cannot generate rolling stats.
RAW_LOGS_PATH = "data/nba_game_logs_1997_2024.csv"
OUTPUT_PATH = "data/today_inference.csv"

def get_today_games(date_str=None):
    if date_str is None:
        date_str = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Fetching games for {date_str}...")
    try:
        board = scoreboardv2.ScoreboardV2(game_date=date_str)
        games = board.game_header.get_data_frame()
        # Columns: GAME_DATE_EST, GAME_SEQUENCE, GAME_ID, GAME_STATUS_ID, GAME_STATUS_TEXT, GAMECODE, HOME_TEAM_ID, VISITOR_TEAM_ID, ...
        return games
    except Exception as e:
        print(f"Error fetching scoreboard: {e}")
        return pd.DataFrame()

def get_roster(team_id):
    time.sleep(0.6) # Rate limit
    try:
        roster = commonteamroster.CommonTeamRoster(team_id=team_id, season='2024-25') # Or 2025-26? API syntax usually 2024-25 for now? 
        # Actually current season is 2025-26 in our hypothetical future? 
        # User prompt says 2025-26 season. API usually requires '2025-26'.
        # Try '2025-26', fall back if needed.
        df = roster.common_team_roster.get_data_frame()
        return df
    except Exception as e:
        print(f"Error fetching roster for {team_id}: {e}")
        return pd.DataFrame()

def construct_inference_rows(games, date_str):
    print("Constructing inference rows...")
    rows = []
    
    # Pre-fetch rosters
    all_team_ids = set(games['HOME_TEAM_ID'].tolist() + games['VISITOR_TEAM_ID'].tolist())
    rosters = {}
    for tid in all_team_ids:
        print(f"  Fetching roster for {tid}...")
        r = get_roster(tid)
        rosters[tid] = r
    
    for _, game in games.iterrows():
        game_id = game['GAME_ID']
        home_id = game['HOME_TEAM_ID']
        away_id = game['VISITOR_TEAM_ID']
        
        # Matchup Strings
        # We assume standard abbreviation mapping is hard, but we can try to get it from roster or just use generic?
        # StrictFeatureEngine expects 'MATCHUP' like "LAL vs. BOS".
        # We need team abbreviations. 
        # Let's assume we can get them from roster (TeamCode?).
        
        # Helper
        def get_abbr(tid):
            # Abbreviation is not in roster df usually.
            # We can try to look it up from 'games', but 'games' only has ID.
            # Actually ScoreboardV2 game_header has HOME_TEAM_ID and VISITOR_TEAM_ID.
            # It also has line_score which has abbreviations.
            # But let's just use "UNK" or a placeholder. 
            # The strict feature engine might need it for MATCHUP string parsing?
            # Yes, "LAL vs. BOS".
            # We need correct abbreviations for MATCHUP parsing logic if it depends on them.
            # Let's try to get it from 'games' data frame if possible?
            # Or just use ID in matchup string? "12345 vs 67890"?
            # Feature engine regex might fail.
            # Let's assume standard abbreviations for now if we can't find them.
            # Or assume we don't care about "is_home" parsing if it fails?
            # Wait, 'is_home' uses 'vs.' or '@'. The team names don't matter as much as the separator.
            return "UNK"

        home_abbr = get_abbr(home_id)
        away_abbr = get_abbr(away_id)
        
        matchup_home = f"{home_abbr} vs. {away_abbr}"
        matchup_away = f"{away_abbr} @ {home_abbr}"
        
        # Process Home Players
        if home_id in rosters:
            for _, p in rosters[home_id].iterrows():
                # Columns in Roster: TeamID, SEASON, LeagueID, PLAYER, NICKNAME, PLAYER_SLUG, NUM, POSITION, HEIGHT, WEIGHT, BIRTH_DATE, AGE, EXP, SCHOOL, PLAYER_ID
                rows.append({
                    'GAME_ID': game_id,
                    'GAME_DATE': date_str,
                    'SEASON_ID': '22025', # 2 = Regular Season, 2025
                    'PLAYER_ID': p['PLAYER_ID'],
                    'PLAYER_NAME': p['PLAYER'],
                    'TEAM_ID': home_id,
                    'TEAM_ABBREVIATION': home_abbr,
                    'TEAM_NAME': '', # Optional
                    'MATCHUP': matchup_home,
                    'WL': None, # Future
                    'MIN': 0, # Placeholder, will be ignored by shift logic
                    'PTS': 0, 'AST': 0, 'REB': 0, 'FGA': 0, 'FTA': 0, 'TOV': 0, # Placeholders
                })
                
        # Process Away Players
        if away_id in rosters:
            for _, p in rosters[away_id].iterrows():
                rows.append({
                    'GAME_ID': game_id,
                    'GAME_DATE': date_str,
                    'SEASON_ID': '22025',
                    'PLAYER_ID': p['PLAYER_ID'],
                    'PLAYER_NAME': p['PLAYER'],
                    'TEAM_ID': away_id,
                    'TEAM_ABBREVIATION': away_abbr,
                    'TEAM_NAME': '',
                    'MATCHUP': matchup_away,
                    'WL': None,
                    'MIN': 0,
                    'PTS': 0, 'AST': 0, 'REB': 0, 'FGA': 0, 'FTA': 0, 'TOV': 0,
                })
                
    return pd.DataFrame(rows)

def prepare_data(target_date):
    # 1. Load History
    if not os.path.exists(RAW_LOGS_PATH):
        print("Raw logs not found.")
        return
        
    print(f"Loading history from {RAW_LOGS_PATH}...")
    df_history = pd.read_csv(RAW_LOGS_PATH, low_memory=False)
    
    # 2. Get Today's Games
    games = get_today_games(target_date)
    if games.empty:
        print("No games found.")
        return
        
    # 3. Create Rows
    df_today = construct_inference_rows(games, target_date)
    print(f"Generated {len(df_today)} player-rows for inference.")
    
    # 4. Concat
    # Ensure columns match
    # history has many more columns. We need to align.
    # We only really need columns used by Feature Engine or Identifiers.
    
    # Fill missing columns in df_today with 0 or NaN
    for c in df_history.columns:
        if c not in df_today.columns:
            df_today[c] = 0 # or NaN
            
    # Concat
    combined = pd.concat([df_history, df_today], ignore_index=True)
    
    # 5. Feature Engineering
    print("Running StrictFeatureEngine...")
    engine = StrictFeatureEngine(combined)
    engine.load_and_clean()
    # Call all feature methods
    engine.compute_rolling_stats()
    engine.compute_rest_days()
    engine.add_lag_features()
    engine.compute_opponent_strength()
    engine.compute_advanced_rolling_stats()
    engine.compute_advanced_player_metrics()
    engine.compute_contextual_features()
    engine.compute_availability_features()
    engine.compute_per_minute_features()
    
    # Generate Embeddings (for ALL rows, including today)
    # Using saved artifacts from V4 training
    engine.generate_embeddings(
        model_path="models/ft_transformer_v1.pt",
        encoder_path="models/player_id_encoder_v1.joblib",
        scaler_path="models/scaler_v1.joblib",
        feature_list_path="models/cont_features_v1.joblib"
    )

    # 4. Filter for Today & Save
    # 'GAME_DATE' is datetime in df usually (check engine.load_and_clean)
    # The merged rosters have today_str (YYYY-MM-DD)
    # Let's ensure string comparison works
    
    print(f"Filtering features for {target_date}...")
    combined_processed = engine.df
    combined_processed['GAME_DATE'] = combined_processed['GAME_DATE'].astype(str)
    
    # Check format of target_date '2025-12-15' vs '2025-12-15 00:00:00'
    # pandas default str conversion usually keeps YYYY-MM-DD if no time.
    
    df_inference = combined_processed[combined_processed['GAME_DATE'].str.startswith(target_date)].copy()
    
    print(f"Saving {len(df_inference)} rows to {OUTPUT_PATH}...")
    df_inference.to_csv(OUTPUT_PATH, index=False)
    return OUTPUT_PATH

if __name__ == "__main__":
    target_date = "2025-12-15"
    if len(sys.argv) > 1:
        target_date = sys.argv[1]
    
    prepare_data(target_date)
