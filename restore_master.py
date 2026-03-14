import pandas as pd
import os

master_file = 'final_feature_matrix_with_per_min_1997_onward.csv'
history_file = 'data/strict_features_v4.csv'
season_logs = 'data/season_logs_2025-26.csv'

print("Restoring master matrix from history and season logs...")

# 1. Load History (1997-2025)
df_hist = pd.read_csv(history_file, low_memory=False)
df_hist.rename(columns={'GAME_DATE': 'gameDate', 'PLAYER_ID': 'player_id', 'PLAYER_NAME': 'player_name', 'TEAM_ABBREVIATION': 'team'}, inplace=True, errors='ignore')

# 2. Load Season Logs (2025-26)
df_season = pd.read_csv(season_logs, low_memory=False)
# fetch_season_data.py renames them to 'date', 'player_id', 'player_name'
df_season.rename(columns={'date': 'gameDate', 'PLAYER_ID': 'player_id', 'PLAYER_NAME': 'player_name'}, inplace=True, errors='ignore')

# We need features for the season logs, so we'll let update_feature_matrix handle it
# But for now, let's just combine the RAW logs and re-run the engine.

print(f"History rows: {len(df_hist)}")
print(f"Season logs rows: {len(df_season)}")

# Instead of manual combine, let's just use update_dataset on the history file with the season logs
from update_feature_matrix import update_dataset

# Temporarily save a combined base if needed, or just use history as master
# Note: update_dataset takes (daily_file, master_file)
# We want history to be the master.
df_hist.to_csv(master_file, index=False)
update_dataset(season_logs, master_file)

print("Restoration complete. Now run sync for Feb 17-24.")
