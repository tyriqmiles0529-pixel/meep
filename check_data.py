
import pandas as pd
import numpy as np

file = "final_feature_matrix_with_per_min_1997_onward.csv"
try:
    # Read just a bit to check columns and NaNs
    df = pd.read_csv(file, nrows=10000)
    print("Columns:", df.columns.tolist()[:20])
    print("Total columns:", len(df.columns))
    
    # Check for NaNs in first 10k rows
    nan_counts = df.isna().sum()
    cols_with_nans = nan_counts[nan_counts > 0]
    print("\nColumns with NaNs (first 10k):")
    print(cols_with_nans.head(20))
    
    # Check specifically for 3PM columns
    three_cols = [c for c in df.columns if '3P' in c or 'three' in c.lower()]
    print("\n3PM related columns:", three_cols[:10])
    
except Exception as e:
    print(f"Error: {e}")
