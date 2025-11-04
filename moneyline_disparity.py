import pandas as pd
import numpy as np

predictions_df = pd.read_csv("today_player_predictions.csv")
games_df = pd.read_csv("today_game_lines.csv")

def predicted_win_prob(home_points, away_points):
    # Simple probability model: logistic based on predicted point diff
    diff = home_points - away_points
    return 1 / (1 + np.exp(-0.1 * diff))  # scale factor 0.1

disparities = []

for _, game in games_df.iterrows():
    home_players = predictions_df[predictions_df['player_name'].str.contains(game['home_team'])].head(7)
    away_players = predictions_df[predictions_df['player_name'].str.contains(game['away_team'])].head(7)

    home_points = home_players['PTS_pred'].sum()
    away_points = away_players['PTS_pred'].sum()

    predicted_prob = predicted_win_prob(home_points, away_points)
    implied_prob = 1 / (1 + 10**((game['moneyline_away'] - game['moneyline_home'])/400))  # simple approx

    moneyline_disp = abs(predicted_prob - implied_prob)

    disparities.append({
        "home_team": game['home_team'],
        "away_team": game['away_team'],
        "predicted_home_win_prob": predicted_prob,
        "implied_home_win_prob": implied_prob,
        "moneyline_disparity": moneyline_disp
    })

pd.DataFrame(disparities).to_csv("moneyline_disparity.csv", index=False)
print("Moneyline disparities saved.")
