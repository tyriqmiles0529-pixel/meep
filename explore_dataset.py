#!/usr/bin/env python3
"""
Explore the Kaggle NBA historical dataset
"""
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd

print("=" * 70)
print("EXPLORING KAGGLE NBA DATASET")
print("=" * 70)

# First, let's see what files are available in the dataset
print("\n📦 Downloading dataset metadata...")
try:
    # Download the dataset to see what files are available
    dataset_path = kagglehub.dataset_download("eoinamoore/historical-nba-data-and-player-box-scores")
    print(f"✅ Dataset downloaded to: {dataset_path}")

    # List all files in the dataset
    import os
    files = []
    for root, dirs, filenames in os.walk(dataset_path):
        for filename in filenames:
            filepath = os.path.join(root, filename)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            files.append((filename, f"{size_mb:.2f} MB"))

    print(f"\n📁 Available files in dataset ({len(files)} total):")
    for fname, size in sorted(files):
        print(f"   - {fname:<50s} {size:>10s}")

    # Try to load and preview each CSV file
    print("\n" + "=" * 70)
    print("PREVIEWING DATA FILES")
    print("=" * 70)

    for fname, _ in sorted(files):
        if fname.endswith('.csv'):
            print(f"\n📊 {fname}")
            print("-" * 70)
            try:
                filepath = os.path.join(dataset_path, fname)
                df = pd.read_csv(filepath, nrows=5)
                print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns (first 5 rows shown)")
                print(f"Columns: {', '.join(df.columns.tolist())}")
                print("\nFirst 5 rows:")
                print(df.to_string(index=False))

                # Show data types and missing values
                print(f"\nData types:")
                for col in df.columns[:10]:  # Show first 10 columns
                    dtype = df[col].dtype
                    print(f"   {col}: {dtype}")
                if len(df.columns) > 10:
                    print(f"   ... and {len(df.columns) - 10} more columns")

            except Exception as e:
                print(f"   ⚠️ Error loading {fname}: {e}")

    print("\n" + "=" * 70)
    print("✅ Exploration complete!")
    print("=" * 70)

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
