import pandas as pd

predictions_df = pd.read_csv("today_player_predictions.csv")
games_df = pd.read_csv("today_game_lines.csv")

disparities = []

for _, game in games_df.iterrows():
    home_players = predictions_df[predictions_df['player_name'].str.contains(game['home_team'])].head(7)
    away_players = predictions_df[predictions_df['player_name'].str.contains(game['away_team'])].head(7)

    predicted_diff = home_players['PTS_pred'].sum() - away_players['PTS_pred'].sum()
    spread_disp = abs(predicted_diff - game['spread_home'])

    disparities.append({
        "home_team": game['home_team'],
        "away_team": game['away_team'],
        "predicted_point_diff": predicted_diff,
        "spread_disparity": spread_disp
    })

pd.DataFrame(disparities).to_csv("spread_disparity.csv", index=False)
print("Spread disparities saved.")
