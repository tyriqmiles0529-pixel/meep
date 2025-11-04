import pandas as pd
import requests

# Load predictions
try:
    predictions_df = pd.read_csv("today_player_predictions.csv")
except FileNotFoundError:
    print("Predictions CSV not found.")
    predictions_df = pd.DataFrame()

if not predictions_df.empty:
    API_KEY = "3ee00eb314b80853c6c77920c5bf74f7"
    LEAGUE_ID = "NBA"
    PLAYER_STATS = ["points_line", "rebounds_line", "assists_line"]

    # Fetch player props
    player_props_url = f"https://api.sportsgameodds.com/v2/sports/player_props?league={LEAGUE_ID}"
    headers = {'X-Api-Key': API_KEY}
    resp = requests.get(player_props_url, headers=headers)
    player_data = resp.json().get("players", [])

    player_odds_df = pd.DataFrame([
        {**{"player_name": p["name"]}, **{stat: p.get(stat, 0) for stat in PLAYER_STATS}}
        for p in player_data
    ])

    # Merge and calculate disparities
    merged = pd.merge(predictions_df, player_odds_df, on="player_name", how="inner")
    for stat in PLAYER_STATS:
        pred_col = stat.replace("_line", "_pred")
        disp_col = stat.replace("_line", "_disparity")
        merged[disp_col] = (merged[pred_col] - merged[stat]).abs()
    merged["total_disparity"] = merged[[s.replace("_line","_disparity") for s in PLAYER_STATS]].sum(axis=1)
    merged.sort_values(by="total_disparity", ascending=False, inplace=True)
    merged.to_csv("top_player_disparities.csv", index=False)
    print(merged.head(10))
