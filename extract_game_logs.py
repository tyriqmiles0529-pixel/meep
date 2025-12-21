import pandas as pd
import time
import os
import json
from nba_api.stats.endpoints import leaguegamelog
from nba_api.stats.static import players, teams

# Configuration
START_SEASON = 1997
# END_SEASON = 2024 # Current ongoing season
END_SEASON = 2025
DATA_DIR = "data/raw_logs"

def get_season_string(year):
    """
    Converts 2023 -> '2023-24'
    """
    next_year = str(year + 1)[-2:]
    return f"{year}-{next_year}"

def fetch_all_logs():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    all_logs = []
    
    for year in range(START_SEASON, END_SEASON + 1):
        season_str = get_season_string(year)
        cache_file = os.path.join(DATA_DIR, f"gamelog_{season_str}.csv")
        
        if os.path.exists(cache_file):
            print(f"[CACHE] Loading {season_str}...")
            df = pd.read_csv(cache_file)
        else:
            print(f"[FETCH] Downloading {season_str}...")
            try:
                # Regular Season
                log_reg = leaguegamelog.LeagueGameLog(
                    season=season_str, 
                    season_type_all_star='Regular Season',
                    player_or_team_abbreviation='P' # Player stats
                ).get_data_frames()[0]
                
                # Playoffs
                log_playoff = leaguegamelog.LeagueGameLog(
                    season=season_str, 
                    season_type_all_star='Playoffs',
                    player_or_team_abbreviation='P'
                ).get_data_frames()[0]
                
                log_reg['season_type'] = 'Regular'
                log_playoff['season_type'] = 'Playoffs'
                
                df = pd.concat([log_reg, log_playoff], ignore_index=True)
                
                # Save Raw
                df.to_csv(cache_file, index=False)
                
                # Sleep to respect API limits
                time.sleep(2.0)
                
            except Exception as e:
                print(f"[ERROR] Failed {season_str}: {e}")
                continue
        
        # Add numeric season column for convenience
        df['season_start_year'] = year
        all_logs.append(df)
        
    # Combine
    print("Merging all seasons...")
    full_df = pd.concat(all_logs, ignore_index=True)
    
    # Sort by Date
    full_df['GAME_DATE'] = pd.to_datetime(full_df['GAME_DATE'])
    full_df = full_df.sort_values(['GAME_DATE', 'GAME_ID'])
    
    output_file = "data/nba_game_logs_1997_2024.csv"
    full_df.to_csv(output_file, index=False)
    print(f"saved full dataset to {output_file} ({len(full_df)} rows)")
    
    return full_df

if __name__ == "__main__":
    fetch_all_logs()
