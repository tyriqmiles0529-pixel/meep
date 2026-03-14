import pandas as pd
import os

filesize = os.path.getsize('final_feature_matrix_with_per_min_1997_onward.csv')
# Read a chunk from the end
try:
    df = pd.read_csv('final_feature_matrix_with_per_min_1997_onward.csv', skiprows=range(1, 1000000), nrows=100)
    print(df[['date', 'season_start_year', 'player_name']])
except:
    # If file smaller than 1M, just read normally
    df = pd.read_csv('final_feature_matrix_with_per_min_1997_onward.csv').tail(20)
    print(df[['date', 'season_start_year', 'player_name']])
