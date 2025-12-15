import argparse
import pandas as pd
from datetime import datetime, timedelta
from nba_api.stats.endpoints import playergamelogs, scoreboardv2
import time

def get_games_for_date(date_str):
    """
    Fetches games and player stats for a specific date.
    date_str: YYYY-MM-DD
    """
    print(f"Fetching data for {date_str}...")
    
    # 1. Get Scoreboard to find game IDs and team info
    try:
        board = scoreboardv2.ScoreboardV2(game_date=date_str)
        games_df = board.game_header.get_data_frame()
        line_score = board.line_score.get_data_frame()
    except Exception as e:
        print(f"Error fetching scoreboard: {e}")
        return None, None

    if games_df.empty:
        print("No games found.")
        return None, None
        
    # Filter for completed games if needed, or just all games
    # For daily updates, we usually run this the NEXT day for previous day's games
    # Or live. Let's assume we want FINAL scores for training/updates.
    
    # 2. Get Player Stats for these games
    # We can fetch daily logs for the whole league for this season, then filter by date.
    # OR iterate dates. playergamelogs gets a batch.
    
    # Re-formatting date for Season (e.g., '2023-24')
    # Actually, PlayerGameLogs requires 'Season' (e.g. '2023-24')
    # Let's determine season from date.
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    year = dt.year
    if dt.month >= 10:
        season_str = f"{year}-{str(year+1)[-2:]}"
    else:
        season_str = f"{year-1}-{str(year)[-2:]}"
        
    print(f"Derived Season: {season_str}")
    
    try:
        # Fetching all logs for the season is heavy if we just want one day.
        # But 'DateFrom' and 'DateTo' params exist!
        # Format: MM/DD/YYYY
        formatted_date = dt.strftime("%m/%d/%Y")
        
        logs = playergamelogs.PlayerGameLogs(
            season_nullable=season_str,
            date_from_nullable=formatted_date,
            date_to_nullable=formatted_date
        )
        player_stats = logs.player_game_logs.get_data_frame()
        
    except Exception as e:
        print(f"Error fetching player logs: {e}")
        return None, None
        
    if player_stats.empty:
        print("No player stats found.")
        return None, None

    # 3. Merge Scoreboard Info (Home/Away, Win/Loss consistency)
    # The player logs usually have WL, but maybe not opponent score directly in a clean way?
    # It has 'MATCHUP' like 'LAL vs. BOS'.
    # Let's standardize columns to match our training schema.
    
    return player_stats, games_df

def standardize_data(player_stats, games_df):
    """
    Maps nba_api columns to our project schema.
    """
    # Mapping
    # API -> Our Schema
    # PLAYER_ID -> player_id
    # PLAYER_NAME -> player_name
    # GAME_ID -> gameId
    # TEAM_ABBREVIATION -> team (we used IDs or Names? let's check dataset)
    # MIN -> minutes (needs conversion from "MM:SS" or float)
    # PTS -> points
    # REB -> reboundsTotal
    # AST -> assists
    # FG3M -> three_pointers
    # ...
    
    df = player_stats.copy()
    
    # Rename columns
    rename_map = {
        'PLAYER_ID': 'player_id',
        'PLAYER_NAME': 'player_name',
        'GAME_ID': 'gameId',
        'TEAM_ID': 'teamId',
        'TEAM_NAME': 'playerteamName', # Verify this exists
        'MIN': 'minutes',
        'PTS': 'points',
        'AST': 'assists',
        'REB': 'reboundsTotal',
        'OREB': 'reboundsOffensive',
        'DREB': 'reboundsDefensive',
        'FG3M': 'three_pointers',
        'FGA': 'fieldGoalsAttempted',
        'FGM': 'fieldGoalsMade',
        'FG_PCT': 'fieldGoalsPercentage',
        'FG3A': 'threePointersAttempted',
        'FG3_PCT': 'threePointersPercentage',
        'FTA': 'freeThrowsAttempted',
        'FTM': 'freeThrowsMade',
        'FT_PCT': 'freeThrowsPercentage',
        'PF': 'foulsPersonal',
        'TOV': 'turnovers',
        'STL': 'steals',
        'BLK': 'blocks',
        'PLUS_MINUS': 'plusMinusPoints',
        'GAME_DATE': 'date', # usually formatted ISO
        'MATCHUP': 'matchup'
    }
    
    df = df.rename(columns=rename_map)
    
    # Fix Minutes (API returns float usually for this endpoint, check type)
    # If str "MM:SS"? Data inspection needed. PlayerGameLogs usually returns float.
    
    # Fix Date (API: '2023-10-24T00:00:00', Project: '2023-10-24')
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    
    # Add Season
    # We can infer or pass it.
    
    # Add Home/Away and Opponent Info from Matchup
    # Matchup: "PHX vs. GSW" (Home) or "PHX @ GSW" (Away)
    def parse_matchup(row):
        m = row['matchup']
        if ' vs. ' in m:
            row['home'] = 1
            row['opponent'] = m.split(' vs. ')[1]
        elif ' @ ' in m:
            row['home'] = 0
            row['opponent'] = m.split(' @ ')[1]
        else:
             row['home'] = 0 # Fallback
             row['opponent'] = 'UNK'
        return row
        
    df = df.apply(parse_matchup, axis=1)
    
    # Add Win
    df['win'] = df['WL'].apply(lambda x: 1 if x == 'W' else 0)
    
    # Select Columns
    final_cols = [
        'player_id', 'player_name', 'gameId', 'date', 'minutes', 'points', 'assists', 
        'reboundsTotal', 'three_pointers', 'fieldGoalsAttempted', 'fieldGoalsMade', 
        'threePointersAttempted', 'freeThrowsAttempted', 'freeThrowsMade',
        'reboundsOffensive', 'reboundsDefensive', 'foulsPersonal', 'turnovers', 
        'steals', 'blocks', 'plusMinusPoints', 'home', 'win'
        # Add others if needed
    ]
    
    # Keep only what exists
    existing = [c for c in final_cols if c in df.columns]
    
    return df[existing]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default=None, help='Date YYYY-MM-DD. Defaults to Yesterday.')
    args = parser.parse_args()
    
    target_date = args.date
    if not target_date:
        # Default to yesterday
        target_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
    print(f"Target Date: {target_date}")
    
    stats, games = get_games_for_date(target_date)
    
    if stats is not None:
        clean_df = standardize_data(stats, games)
        print("Scraped Data Sample:")
        print(clean_df.head())
        
        # Save raw daily file
        filename = f"daily_games_{target_date}.csv"
        clean_df.to_csv(filename, index=False)
        print(f"Saved to {filename}")
    else:
        print("No data retrieved.")
