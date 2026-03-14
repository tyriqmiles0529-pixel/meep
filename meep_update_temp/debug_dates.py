from data_processor import BasketballDataProcessor
import pandas as pd
import numpy as np

try:
    processor = BasketballDataProcessor('final_feature_matrix_with_per_min_1997_onward.csv')
    processor.load_data(nrows=1000) # Load small chunk
    
    print("Dtypes:")
    print(processor.df.dtypes)
    
    if 'date' in processor.df.columns:
        print("\nDate column head:")
        print(processor.df['date'].head())
        
        unique_dates = processor.df['date'].unique()
        print(f"\nUnique dates type: {type(unique_dates)}")
        if len(unique_dates) > 0:
            first_date = unique_dates[0]
            print(f"First date: {first_date} (type: {type(first_date)})")
            
            print("\nFiltering check:")
            filtered = processor.df[processor.df['date'] == first_date]
            print(f"Filtered shape: {filtered.shape}")
            
    else:
        print("Date column missing!")

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
