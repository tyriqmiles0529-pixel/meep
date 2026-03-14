import pandas as pd
import numpy as np

try:
    df = pd.read_csv('meep/nba_predictor/final_feature_matrix_with_per_min_1997_onward.csv')
    print(f"NaNs in points: {df['points'].isna().sum()}")
    print(f"Infs in points: {np.isinf(df['points']).sum()}")
    
    # Also check other potential targets or features
    print(f"NaNs in minutes: {df['minutes'].isna().sum()}")
except Exception as e:
    print(f"Error: {e}")
