import pandas as pd
df = pd.read_csv('final_feature_matrix_with_per_min_1997_onward.csv', usecols=['date', 'season_start_year', 'player_name']).tail(20)
print(df)
