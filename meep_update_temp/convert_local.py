#!/usr/bin/env python
"""Convert CSV.gzip to Parquet locally - memory efficient"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import gc

csv_path = 'aggregated_nba_data.csv.gzip'
parquet_path = 'aggregated_nba_data.parquet'

print("="*60)
print("Converting CSV.gzip to Parquet")
print("="*60)
print(f"Input: {csv_path} (416 MB)")
print(f"Output: {parquet_path}")
print("RAM usage: ~2-3 GB peak")
print("Time: ~10-15 minutes\n")

chunk_size = 500000
total_rows = 0
writer = None

for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False, compression='gzip')):
    total_rows += len(chunk)
    print(f"Chunk {i+1}: {total_rows:,} rows processed...")

    # Force all object columns to string for consistent schema
    for col in chunk.select_dtypes(include=['object']).columns:
        chunk[col] = chunk[col].astype(str)

    table = pa.Table.from_pandas(chunk, preserve_index=False)

    if writer is None:
        writer = pq.ParquetWriter(parquet_path, table.schema, compression='snappy')
        print(f"  Schema: {len(table.schema)} columns")
    else:
        table = table.cast(writer.schema)

    writer.write_table(table)
    del chunk, table
    gc.collect()

writer.close()

# Verify
pf = pq.ParquetFile(parquet_path)
print(f"\nDone! Total rows: {total_rows:,}")
print(f"Verified rows: {pf.metadata.num_rows:,}")

import os
size_mb = os.path.getsize(parquet_path) / 1024**2
print(f"Parquet file: {size_mb:.1f} MB")

# Check year range
sample = pd.read_parquet(parquet_path, columns=['season', 'game_year'])
print(f"Season range: {sample['season'].min()} - {sample['season'].max()}")
print(f"Game year range: {sample['game_year'].min()} - {sample['game_year'].max()}")
print("\nUpload this file to Kaggle!")
