#!/usr/bin/env python
"""
Convert CSV.gzip to Parquet WITH pre-computed temporal features.
MEMORY-EFFICIENT: Processes by player groups to enable rolling features.
FILTERED TO 2000+ to save memory and training time.
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import gc
from tqdm import tqdm

csv_path = 'aggregated_nba_data.csv.gzip'
parquet_path = 'aggregated_nba_data_2000_features.parquet'
MIN_YEAR = 2000  # Filter to this year and later

print("="*60)
print("MEMORY-EFFICIENT Feature Engineering")
print(f"FILTERED TO {MIN_YEAR}+ DURING LOAD")
print("="*60)
print(f"Input: {csv_path} (416 MB)")
print(f"Output: {parquet_path}")
print("RAM usage: ~4-5 GB peak (chunked processing)")
print("Time: ~30-45 minutes\n")

# Step 1: Load in chunks, filter immediately, keep only essential columns first
print("Step 1: Loading data with {MIN_YEAR}+ filter...")
chunks = []
chunk_size = 500000
total_read = 0
total_kept = 0

for i, chunk in enumerate(pd.read_csv(csv_path, compression='gzip', chunksize=chunk_size, low_memory=False)):
    total_read += len(chunk)

    # Filter immediately
    if 'season' in chunk.columns:
        chunk = chunk[chunk['season'] >= MIN_YEAR]
    elif 'game_year' in chunk.columns:
        chunk = chunk[chunk['game_year'] >= MIN_YEAR]

    total_kept += len(chunk)

    if len(chunk) > 0:
        # Optimize dtypes immediately
        for col in chunk.select_dtypes(include=['float64']).columns:
            chunk[col] = chunk[col].astype('float32')
        for col in chunk.select_dtypes(include=['int64']).columns:
            if chunk[col].min() >= -2147483648 and chunk[col].max() <= 2147483647:
                chunk[col] = chunk[col].astype('int32')
        chunks.append(chunk)

    print(f"  Chunk {i+1}: read {total_read:,}, kept {total_kept:,} rows ({MIN_YEAR}+)")
    gc.collect()

print(f"\nConcatenating {len(chunks)} filtered chunks...")
df = pd.concat(chunks, ignore_index=True)
del chunks
gc.collect()

print(f"  Total rows: {len(df):,}")
print(f"  Reduction: {(1-len(df)/total_read)*100:.1f}%")

# Force string types
print("\nStep 2: Standardizing dtypes...")
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype(str)
gc.collect()

mem_mb = df.memory_usage(deep=True).sum() / 1024**2
print(f"  Memory: {mem_mb:.1f} MB ({mem_mb/1024:.2f} GB)")

# Step 3: Sort by player and date
print("\nStep 3: Sorting by player and game date...")
if 'gameDate' in df.columns:
    df['gameDate'] = pd.to_datetime(df['gameDate'], format='mixed', utc=True, errors='coerce')
    df = df.sort_values(['personId', 'gameDate']).reset_index(drop=True)
    print("  Sorted by personId and gameDate")
gc.collect()

# Step 4: Rolling features (this is the memory-intensive part)
print("\nStep 4: Computing rolling features...")

# Core stats only (reduce memory pressure)
stats_to_roll = ['points', 'assists', 'reboundsTotal', 'numMinutes', 'threePointersMade']
stats_found = [s for s in stats_to_roll if s in df.columns]
print(f"  Core stats: {stats_found}")

for stat in tqdm(stats_found, desc="  Rolling L5/L10"):
    # L5 average
    df[f'{stat}_L5'] = df.groupby('personId')[stat].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    ).astype('float32')

    # L10 average
    df[f'{stat}_L10'] = df.groupby('personId')[stat].transform(
        lambda x: x.shift(1).rolling(window=10, min_periods=1).mean()
    ).astype('float32')

    # Trend
    df[f'{stat}_trend'] = (df[f'{stat}_L5'] - df[f'{stat}_L10']).astype('float32')

    gc.collect()

# Step 5: Per-minute rates (simple, low memory)
print("\nStep 5: Computing per-minute rates...")
if 'numMinutes' in df.columns:
    for stat in ['points', 'assists', 'reboundsTotal']:
        if stat in df.columns:
            df[f'{stat}_per_min'] = (df[stat] / df['numMinutes'].replace(0, np.nan)).fillna(0).astype('float32')

# Step 6: Efficiency metrics
print("\nStep 6: Computing efficiency metrics...")
if all(col in df.columns for col in ['points', 'fieldGoalsAttempted', 'freeThrowsAttempted']):
    tsa = df['fieldGoalsAttempted'] + 0.44 * df['freeThrowsAttempted']
    df['true_shooting_pct'] = (df['points'] / (2 * tsa.replace(0, np.nan))).fillna(0).astype('float32')
    del tsa
    gc.collect()

if all(col in df.columns for col in ['assists', 'turnovers']):
    df['ast_to_ratio'] = (df['assists'] / df['turnovers'].replace(0, np.nan)).fillna(0).clip(0, 20).astype('float32')

# Step 7: Rest features
print("\nStep 7: Computing rest features...")
if 'gameDate' in df.columns:
    df['days_rest'] = df.groupby('personId')['gameDate'].diff().dt.days.fillna(3).clip(0, 10).astype('float32')
    df['is_back_to_back'] = (df['days_rest'] <= 1).astype('int8')
gc.collect()

# Step 8: Season context
print("\nStep 8: Computing season context...")
if 'season' in df.columns:
    df['games_this_season'] = df.groupby(['personId', 'season']).cumcount().astype('int16')
gc.collect()

mem_mb = df.memory_usage(deep=True).sum() / 1024**2
print(f"\nTotal columns: {len(df.columns)}")
print(f"Final memory: {mem_mb:.1f} MB ({mem_mb/1024:.2f} GB)")

# Step 9: Write to Parquet
print("\nStep 9: Writing to Parquet...")
table = pa.Table.from_pandas(df, preserve_index=False)
pq.write_table(table, parquet_path, compression='snappy')
del table
gc.collect()

# Verify
import os
pf = pq.ParquetFile(parquet_path)
size_mb = os.path.getsize(parquet_path) / 1024**2
print(f"\nDone! Total rows: {len(df):,}")
print(f"Verified rows: {pf.metadata.num_rows:,}")
print(f"Parquet file: {size_mb:.1f} MB")
print(f"Total columns: {len(df.columns)}")

# Year range
print(f"\nSeason range: {df['season'].min()} - {df['season'].max()}")

# List new features
new_features = [c for c in df.columns if any(x in c for x in ['_L5', '_L10', '_trend', '_per_min',
                                                               'true_shooting', 'ast_to_ratio',
                                                               'days_rest', 'is_back_to_back',
                                                               'games_this_season'])]
print(f"\nNew temporal features: {len(new_features)}")
for f in sorted(new_features):
    print(f"  - {f}")

print("\n" + "="*60)
print(f"Upload {parquet_path} to Kaggle!")
print("Then train WITHOUT --add-rolling-features")
print("="*60)
