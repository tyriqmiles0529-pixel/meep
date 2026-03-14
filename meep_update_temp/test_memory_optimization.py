"""Test memory optimization for aggregated data loading"""
import pandas as pd
import gc
from pathlib import Path

# Simulate the memory optimization code
def load_with_optimization(file_path):
    print("Loading with optimizations...")

    # Load with optimized settings
    df = pd.read_csv(
        file_path,
        low_memory=False,
        dtype_backend='numpy_nullable',
    )

    print(f"Loaded {len(df):,} rows")
    print(f"Initial memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    # Convert object columns to category if low cardinality
    for col in df.select_dtypes(include=['object']).columns:
        num_unique = df[col].nunique()
        num_total = len(df)
        if num_unique / num_total < 0.5:
            df[col] = df[col].astype('category')
            print(f"  Converted {col} to category ({num_unique} unique values)")

    # Downcast numeric types
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')

    for col in df.select_dtypes(include=['integer']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')

    gc.collect()

    print(f"Optimized memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    return df

# Test if aggregated file exists
agg_path = Path("aggregated_nba_data.csv.gzip")
if agg_path.exists():
    df = load_with_optimization(agg_path)
    print("\n✅ Memory optimization test PASSED")
else:
    print(f"❌ File not found: {agg_path}")
    print("This test should be run in Kaggle where the dataset is available")
