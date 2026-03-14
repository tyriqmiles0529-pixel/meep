import pandas as pd
import sys

try:
    df = pd.read_csv('final_feature_matrix_with_per_min_1997_onward.csv', nrows=1)
    
    # Dump all columns to file for full inspection
    all_cols = df.columns.tolist()
    with open('all_columns.txt', 'w') as f:
        for i, col in enumerate(all_cols):
             f.write(f"{i}: {col}\n")
             
    print(f"Dumped {len(all_cols)} columns to all_columns.txt")
    
    # Also print any column containing 'player' or 'id'
    print("\nPotentially relevant columns found:")
    for c in all_cols:
        cl = c.lower()
        if 'player' in cl or 'id' in cl or 'name' in cl or 'date' in cl:
            print(f" - {c}")
            
except Exception as e:
    print(f"Error: {e}")
