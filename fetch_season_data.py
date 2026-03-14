from nba_api.stats.endpoints import playergamelogs
import pandas as pd
from datetime import datetime

def fetch_season(season='2025-26'):
    print(f"Fetching full player logs for season {season}...")
    import time
    headers = {
        'Host': 'stats.nba.com',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/117.0',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.nba.com/',
        'Origin': 'https://www.nba.com',
        'Connection': 'keep-alive',
    }
    # Try PlayerGameLogs first (it has better metadata)
    df = None
    for attempt in range(2):
        try:
            logs = playergamelogs.PlayerGameLogs(
                season_nullable=season,
                measure_type_player_game_logs_nullable='Base',
                timeout=10,
                headers=headers
            )
            df = logs.player_game_logs.get_data_frame()
            print(f"Successfully fetched {len(df)} player-game logs via PlayerGameLogs.")
            break
        except Exception as e:
            print(f"PlayerGameLogs attempt {attempt+1} failed (Timeout 10s).")
            if attempt < 1: time.sleep(2)

    # Fallback to LeagueGameLog (more robust/faster)
    if df is None:
        from nba_api.stats.endpoints import leaguegamelog
        print(">>> [FALLBACK] Attempting LeagueGameLog retrieval...")
        for attempt in range(1):
            try:
                logs = leaguegamelog.LeagueGameLog(
                    season=season,
                    player_or_team_abbreviation='P', # P for Player
                    timeout=10,
                    headers=headers
                )
                df = logs.league_game_log.get_data_frame()
                print(f"Successfully fetched {len(df)} logs via LeagueGameLog.")
                break
            except Exception as e:
                print(f"LeagueGameLog attempt {attempt+1} failed (Timeout 10s).")
    
    if df is None:
        print(">>> [FATAL] All NBA API endpoints timed out. Check network or stats.nba.com status.")
        return None
        
    # Standardize columns to match update_feature_matrix expectations
    rename_map = {
        'PLAYER_ID': 'player_id',
        'PLAYER_NAME': 'player_name',
        'GAME_ID': 'gameId',
        'GAME_DATE': 'date',
        'TEAM_ID': 'team_id',
        'TEAM_ABBREVIATION': 'team',
        'PTS': 'points',
        'AST': 'assists',
        'REB': 'reboundsTotal',
        'MIN': 'minutes',
        'FG3M': 'three_pointers'
    }
    # Ensure season metadata is present for the engine
    if 'SEASON_ID' not in df.columns:
        # 22025 for Regular Season 2025-26
        year_part = season.split('-')[0]
        df['SEASON_ID'] = f"2{year_part}"
    
    # Extract numeric start year for filtering
    try:
        df['season_start_year'] = int(season.split('-')[0])
    except:
        df['season_start_year'] = 2025

    # Save for update
    output_file = f"data/season_logs_{season}.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved to {output_file} with Season ID {df['SEASON_ID'].iloc[0] if not df.empty else 'N/A'}")
    return output_file

if __name__ == "__main__":
    fetch_season()