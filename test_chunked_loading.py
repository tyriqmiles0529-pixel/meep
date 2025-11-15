"""Test chunked CSV loading with memory limit"""
import pandas as pd
import gc
from pathlib import Path

def test_chunked_loading(file_path, year_filter=2002):
    """Simulate the chunked loading approach"""
    print(f"Testing chunked loading from: {file_path}")
    print(f"Filter: season_end_year >= {year_filter}\n")

    chunks = []
    chunk_size = 100000  # 100K rows at a time
    total_rows_read = 0
    total_rows_kept = 0

    try:
        for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size, low_memory=False)):
            total_rows_read += len(chunk)

            # Filter BEFORE appending (key memory savings)
            if 'season_end_year' in chunk.columns:
                chunk = chunk[chunk['season_end_year'] >= year_filter]

            if len(chunk) > 0:
                total_rows_kept += len(chunk)
                chunks.append(chunk)

            if (i + 1) % 5 == 0:  # Progress every 500K rows
                print(f"  Processed {total_rows_read:,} rows, kept {total_rows_kept:,} ({year_filter}+)...")
                gc.collect()

        print(f"\n✅ Chunked read complete:")
        print(f"   Read: {total_rows_read:,} total rows")
        print(f"   Kept: {total_rows_kept:,} rows ({year_filter}+)")
        print(f"   Reduction: {(1 - total_rows_kept/total_rows_read)*100:.1f}%")

        print(f"\n   Concatenating {len(chunks)} chunks...")
        final_df = pd.concat(chunks, ignore_index=True)
        del chunks
        gc.collect()

        print(f"✅ Final dataframe: {len(final_df):,} rows")
        print(f"   Memory usage: {final_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

        return final_df

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

# Test with aggregated file if it exists
agg_path = Path("aggregated_nba_data.csv.gzip")
if agg_path.exists():
    print("="*70)
    print("TESTING CHUNKED LOADING WITH MEMORY LIMIT")
    print("="*70 + "\n")
    df = test_chunked_loading(agg_path, year_filter=2002)
    if df is not None:
        print("\n" + "="*70)
        print("TEST PASSED - Chunked loading works!")
        print("="*70)
else:
    print(f"❌ File not found: {agg_path}")
    print("This test should be run where the aggregated dataset exists")
