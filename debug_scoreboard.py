"""
Debug ScoreboardV2 to see what columns are available
"""
from nba_api.stats.endpoints import scoreboardv2
import pandas as pd

# Test scoreboard for Oct 28, 2025 (6 days ago - should be completed)
scoreboard = scoreboardv2.ScoreboardV2(game_date='10/28/2025')

# Get all dataframes
dfs = scoreboard.get_data_frames()

print(f"Number of DataFrames: {len(dfs)}")

for i, df in enumerate(dfs):
    print(f"\n{'='*70}")
    print(f"DataFrame {i}: {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    if len(df) > 0:
        print(f"\nFirst row:")
        print(df.iloc[0])
