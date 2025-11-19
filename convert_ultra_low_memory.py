#!/usr/bin/env python
"""
ULTRA LOW MEMORY: Convert to Parquet first, then add features.
Two-pass approach to minimize memory usage.
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import gc
from tqdm import tqdm
import os

csv_path = 'aggregated_nba_data.csv.gzip'
temp_parquet = 'temp_filtered_2000.parquet'
final_parquet = 'aggregated_nba_data_2000_features.parquet'
MIN_YEAR = 2000

print("="*60)
print("ULTRA LOW MEMORY: Two-Pass Feature Engineering")
print("="*60)
print("Pass 1: Filter to 2000+ and save as Parquet")
print("Pass 2: Add features in small batches")
print("Peak RAM: ~2-3 GB\n")

# ========== PASS 1: Filter and save to Parquet ==========
print("PASS 1: Converting CSV to filtered Parquet...")
chunk_size = 200000  # Smaller chunks
writer = None
total_rows = 0

for i, chunk in enumerate(pd.read_csv(csv_path, compression='gzip', chunksize=chunk_size, low_memory=False)):
    # Filter immediately
    if 'season' in chunk.columns:
        chunk = chunk[chunk['season'] >= MIN_YEAR]

    if len(chunk) == 0:
        continue

    total_rows += len(chunk)

    # Convert strings
    for col in chunk.select_dtypes(include=['object']).columns:
        chunk[col] = chunk[col].astype(str)

    # Convert to PyArrow
    table = pa.Table.from_pandas(chunk, preserve_index=False)

    if writer is None:
        writer = pq.ParquetWriter(temp_parquet, table.schema, compression='snappy')
    else:
        table = table.cast(writer.schema)

    writer.write_table(table)

    print(f"  Chunk {i+1}: total {total_rows:,} rows (2000+)")

    del chunk, table
    gc.collect()

writer.close()
print(f"\nPass 1 complete: {total_rows:,} rows saved to temp Parquet")
print(f"File size: {os.path.getsize(temp_parquet)/1024**2:.1f} MB\n")

# ========== PASS 2: Add features by reading from Parquet ==========
print("PASS 2: Adding temporal features...")

# Read the filtered Parquet (much more memory efficient than CSV)
print("Loading filtered data from Parquet...")
df = pd.read_parquet(temp_parquet)
print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

mem_mb = df.memory_usage(deep=True).sum() / 1024**2
print(f"  Memory: {mem_mb:.1f} MB")

# Sort by player and date
print("\nSorting by player and game date...")
if 'gameDate' in df.columns:
    df['gameDate'] = pd.to_datetime(df['gameDate'], format='mixed', utc=True, errors='coerce')
    df = df.sort_values(['personId', 'gameDate']).reset_index(drop=True)
gc.collect()

# Add rolling features (minimal set to reduce memory)
print("\nComputing rolling features...")
stats_to_roll = ['points', 'assists', 'reboundsTotal', 'numMinutes', 'threePointersMade']
stats_found = [s for s in stats_to_roll if s in df.columns]

for stat in tqdm(stats_found, desc="Rolling L5/L10"):
    df[f'{stat}_L5'] = df.groupby('personId')[stat].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    ).astype('float32')

    df[f'{stat}_L10'] = df.groupby('personId')[stat].transform(
        lambda x: x.shift(1).rolling(window=10, min_periods=1).mean()
    ).astype('float32')

    df[f'{stat}_trend'] = (df[f'{stat}_L5'] - df[f'{stat}_L10']).astype('float32')
    gc.collect()

# Per-minute rates
print("\nComputing per-minute rates...")
if 'numMinutes' in df.columns:
    for stat in ['points', 'assists', 'reboundsTotal']:
        if stat in df.columns:
            df[f'{stat}_per_min'] = (df[stat] / df['numMinutes'].replace(0, np.nan)).fillna(0).astype('float32')

# Rest features
print("Computing rest features...")
if 'gameDate' in df.columns:
    df['days_rest'] = df.groupby('personId')['gameDate'].diff().dt.days.fillna(3).clip(0, 10).astype('float32')
    df['is_back_to_back'] = (df['days_rest'] <= 1).astype('int8')

gc.collect()

# Write final Parquet
print("\nWriting final Parquet with features...")
table = pa.Table.from_pandas(df, preserve_index=False)
pq.write_table(table, final_parquet, compression='snappy')
del table
gc.collect()

# Cleanup temp file
os.remove(temp_parquet)
print(f"Removed temp file: {temp_parquet}")

# Verify
pf = pq.ParquetFile(final_parquet)
size_mb = os.path.getsize(final_parquet) / 1024**2

print(f"\n" + "="*60)
print(f"DONE!")
print(f"  Total rows: {len(df):,}")
print(f"  Total columns: {len(df.columns)}")
print(f"  Parquet file: {size_mb:.1f} MB")
print(f"  Season range: {df['season'].min()} - {df['season'].max()}")

# List new features
new_features = [c for c in df.columns if any(x in c for x in ['_L5', '_L10', '_trend', '_per_min', 'days_rest', 'is_back'])]
print(f"\nNew features: {len(new_features)}")
for f in sorted(new_features):
    print(f"  - {f}")

print(f"\nUpload {final_parquet} to Kaggle!")
print("="*60)
