from nba_api.stats.static import players, teams
from player_prediction import predict_player_stats
import pandas as pd

def get_players_today(teams_filter=None):
    all_players = players.get_active_players()
    if teams_filter:
        all_teams = teams.get_teams()
        team_name_to_id = {team['full_name']: team['id'] for team in all_teams}
        filtered_team_ids = [team_name_to_id[name] for name in teams_filter if name in team_name_to_id]
        return [p for p in all_players if p['team_id'] in filtered_team_ids]
    return all_players

def generate_daily_predictions(season='2025-26', opponent_adjust=1.0, min_minutes=15, recent_games=10, teams_filter=None):
    player_list = get_players_today(teams_filter)
    predictions = []

    for player in player_list:
        stats = predict_player_stats(player['id'], season, min_minutes, opponent_adjust, recent_games)
        predictions.append({
            "player_id": player['id'],
            "player_name": player['full_name'],
            "PTS_pred": stats['PTS'],
            "REB_pred": stats['REB'],
            "AST_pred": stats['AST']
        })

    df = pd.DataFrame(predictions)
    return df
