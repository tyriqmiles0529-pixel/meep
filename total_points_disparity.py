import pandas as pd

# Load player predictions & game lines
predictions_df = pd.read_csv("today_player_predictions.csv")
games_df = pd.read_csv("today_game_lines.csv")

disparities = []

for _, game in games_df.iterrows():
    home_players = predictions_df[predictions_df['player_name'].isin(
        # Replace with function to get top 7 players for the home team
        predictions_df[predictions_df['player_name'].str.contains(game['home_team'])]
    )].head(7)
    away_players = predictions_df[predictions_df['player_name'].isin(
        predictions_df[predictions_df['player_name'].str.contains(game['away_team'])]
    )].head(7)

    predicted_total = home_players['PTS_pred'].sum() + away_players['PTS_pred'].sum()
    total_disp = abs(predicted_total - game['total_points'])

    disparities.append({
        "home_team": game['home_team'],
        "away_team": game['away_team'],
        "predicted_total_points": predicted_total,
        "total_points_disparity": total_disp
    })

pd.DataFrame(disparities).to_csv("total_points_disparity.csv", index=False)
print("Total points disparities saved.")
