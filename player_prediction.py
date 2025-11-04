from nba_api.stats.endpoints import playergamelog
import numpy as np

def predict_player_stats(player_id, season='2025-26', min_minutes=15, opponent_adjust=1.0, recent_games=10):
    """
    Predicts a player's points, rebounds, and assists for the next game.
    """
    try:
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
        df = gamelog.get_data_frames()[0]
    except Exception as e:
        print(f"Error fetching player log: {e}")
        return {"PTS": 0, "REB": 0, "AST": 0}

    df = df[['GAME_DATE', 'MATCHUP', 'MIN', 'PTS', 'REB', 'AST']]
    df = df[df['MIN'] >= min_minutes].head(recent_games)
    if df.empty:
        return {"PTS": 0, "REB": 0, "AST": 0}

    weights = np.arange(1, len(df)+1) / np.sum(np.arange(1, len(df)+1))
    preds = {stat: np.dot(df[stat], weights) * opponent_adjust for stat in ['PTS','REB','AST']}
    return preds
