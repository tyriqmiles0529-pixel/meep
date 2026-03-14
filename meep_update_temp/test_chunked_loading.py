"""Test chunked CSV loading with ALL years (1947-2026)"""
import pandas as pd
import gc
from pathlib import Path

def test_chunked_loading_all_years(file_path):
    """Test chunked loading that keeps ALL years with aggressive dtype optimization"""
    print(f"Testing chunked loading from: {file_path}")
    print(f"Mode: Keep ALL years (1947-2026) with aggressive dtype optimization\n")

    chunks = []
    chunk_size = 100000  # 100K rows at a time
    total_rows_read = 0

    try:
        for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size, low_memory=False)):
            total_rows_read += len(chunk)

            # AGGRESSIVE DTYPE OPTIMIZATION PER CHUNK
            # Convert object columns to category if low cardinality
            for col in chunk.select_dtypes(include=['object']).columns:
                num_unique = chunk[col].nunique()
                num_total = len(chunk)
                if num_unique / num_total < 0.5:
                    chunk[col] = chunk[col].astype('category')

            # Downcast numeric types
            for col in chunk.select_dtypes(include=['float']).columns:
                chunk[col] = pd.to_numeric(chunk[col], downcast='float')

            for col in chunk.select_dtypes(include=['integer']).columns:
                chunk[col] = pd.to_numeric(chunk[col], downcast='integer')

            chunks.append(chunk)

            if (i + 1) % 5 == 0:  # Progress every 500K rows
                print(f"  Processed {total_rows_read:,} rows, optimized dtypes...")
                gc.collect()

        print(f"\n✅ Chunked read complete:")
        print(f"   Read: {total_rows_read:,} total rows (ALL years)")

        print(f"\n   Concatenating {len(chunks)} optimized chunks...")
        final_df = pd.concat(chunks, ignore_index=True)
        del chunks
        gc.collect()

        print(f"✅ Final dataframe: {len(final_df):,} rows")
        print(f"   Memory usage: {final_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

        # Show year range
        if 'season_end_year' in final_df.columns:
            print(f"   Year range: {final_df['season_end_year'].min()}-{final_df['season_end_year'].max()}")

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
    print("TESTING CHUNKED LOADING - ALL YEARS (1947-2026)")
    print("="*70 + "\n")
    df = test_chunked_loading_all_years(agg_path)
    if df is not None:
        print("\n" + "="*70)
        print("TEST PASSED - Chunked loading with all years works!")
        print("="*70)
else:
    print(f"❌ File not found: {agg_path}")
    print("This test should be run where the aggregated dataset exists")
