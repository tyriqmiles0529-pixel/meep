import pandas as pd
import os

master_file = 'final_feature_matrix_with_per_min_1997_onward.csv'
daily_file = 'data/season_logs_2025-26.csv'

print(f"--- Checking {master_file} ---")
df_master = pd.read_csv(master_file, nrows=5)
print("Columns:", df_master.columns.tolist()[:15])
print("Player ID sample:", df_master.iloc[0].get('player_id', df_master.iloc[0].get('PLAYER_ID')))
print("Date sample:", df_master.iloc[0].get('date', df_master.iloc[0].get('GAME_DATE')))

print(f"\n--- Checking {daily_file} ---")
df_daily = pd.read_csv(daily_file, nrows=5)
print("Columns:", df_daily.columns.tolist()[:15])
print("Player ID sample:", df_daily.iloc[0].get('player_id', df_daily.iloc[0].get('PLAYER_ID')))
print("Date sample:", df_daily.iloc[0].get('date', df_daily.iloc[0].get('GAME_DATE')))

# Check if mandatory columns for update_feature_matrix exist
mandatory = ['player_id', 'gameId', 'date']
for name, df in [("Master", df_master), ("Daily", df_daily)]:
    # Standardize for check
    cols = [c.lower() for c in df.columns]
    missing = [m for m in mandatory if m.lower() not in cols]
    if missing:
        print(f"!!! {name} is MISSING: {missing}")
    else:
        print(f"OK: {name} has all mandatory columns (case-insensitive check)")
