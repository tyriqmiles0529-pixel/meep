import pandas as pd
try:
    df = pd.read_csv("final_feature_matrix_with_per_min_1997_onward.csv", nrows=100)
    print("Columns:", df.columns.tolist())
    df = pd.read_csv("final_feature_matrix_with_per_min_1997_onward.csv", usecols=['season_start_year'])
    print("Unique season_start_year:", df['season_start_year'].unique())
except Exception as e:
    print(e)
