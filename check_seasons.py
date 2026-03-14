import pandas as pd
df = pd.read_csv("final_feature_matrix_with_per_min_1997_onward.csv", usecols=['season'])
print(df['season'].unique())
