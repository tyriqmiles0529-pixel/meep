from fetch_schedule import get_today_games
from daily_player_predictions import generate_daily_predictions

games_today = get_today_games()
if not games_today:
    print("No games found today.")
    predictions_df = pd.DataFrame()  # avoid exit
else:
    teams_today = {team for game in games_today for team in [game['home_team'], game['away_team']]}
    print(f"Teams playing today: {teams_today}")

    predictions_df = generate_daily_predictions(
        teams_filter=list(teams_today),
        opponent_adjust=1.05
    )
    predictions_df.to_csv("today_player_predictions.csv", index=False)
    print("Predictions saved to today_player_predictions.csv")
    print(predictions_df.head())
