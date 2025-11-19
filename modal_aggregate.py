#!/usr/bin/env python
"""
Aggregate NBA Data on Modal with 64GB RAM

Runs the fuzzy-match aggregation script on Modal to create a properly merged
Parquet file with all advanced stats matched to players.
"""

import modal

app = modal.App("nba-aggregate")

# Volumes
data_volume = modal.Volume.from_name("nba-data")

# Image with all dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        "pandas",
        "numpy",
        "pyarrow",
        "rapidfuzz",
        "tqdm"
    )
    .add_local_dir("shared", remote_path="/root/shared")
)


@app.function(
    image=image,
    memory=65536,  # 64GB RAM for aggregation
    timeout=7200,  # 2 hours
    volumes={"/data": data_volume}
)
def aggregate_data():
    """
    Aggregate all CSVs using fuzzy name matching.
    Creates aggregated_nba_data.parquet with all advanced stats.
    """
    import sys
    sys.path.insert(0, "/root")

    from shared.csv_aggregation import aggregate_player_data

    print("="*70)
    print("AGGREGATING NBA DATA ON MODAL")
    print("="*70)
    print("Memory: 64GB RAM")
    print("Processing: All 9 CSVs with fuzzy name matching")
    print("="*70)

    # Run aggregation
    df = aggregate_player_data(
        data_dir="/data/csv_dir",
        min_year=None,  # Include all years (1947-2026)
        max_year=None,
        verbose=True
    )

    # Save to Parquet
    output_path = "/data/aggregated_nba_data.parquet"
    print(f"\nSaving aggregated data to {output_path}...")
    df.to_parquet(output_path, index=False, compression='snappy')

    # Commit volume
    data_volume.commit()

    file_size_mb = import os; os.path.getsize(output_path) / (1024**2)

    print("="*70)
    print("AGGREGATION COMPLETE!")
    print("="*70)
    print(f"Output: {output_path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"File size: {file_size_mb:.1f} MB")
    print(f"Year range: {df['season'].min()}-{df['season'].max()}")
    print("="*70)

    return {
        'rows': len(df),
        'columns': len(df.columns),
        'file_size_mb': file_size_mb
    }


@app.local_entrypoint()
def main():
    print("="*70)
    print("STARTING DATA AGGREGATION ON MODAL")
    print("="*70)
    print("This will:")
    print("  1. Load all 9 CSVs from Modal volume")
    print("  2. Use fuzzy matching to merge Basketball Reference stats")
    print("  3. Create aggregated_nba_data.parquet with ALL advanced stats")
    print("  4. Save to Modal volume for training")
    print("="*70)

    result = aggregate_data.remote()

    print("\n" + "="*70)
    print("SUCCESS!")
    print("="*70)
    print(f"Aggregated {result['rows']:,} rows with {result['columns']} columns")
    print(f"File size: {result['file_size_mb']:.1f} MB")
    print("\nNext step: Train models with properly merged data")
    print("  py -3.12 -m modal run modal_train.py")
    print("="*70)


if __name__ == "__main__":
    pass
