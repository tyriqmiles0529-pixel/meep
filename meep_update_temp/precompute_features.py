"""
Precompute Rolling Features for NBA Player Data

This script:
1. Loads raw aggregated data
2. Adds rolling features (L5, L10 averages, trends, variance)
3. Saves enhanced data to new Parquet file

Run ONCE to create enhanced dataset, then use for all training.

Usage:
    python precompute_features.py
"""

import pandas as pd
from pathlib import Path
from shared.data_loading import load_player_data
from rolling_features import add_rolling_features

def main():
    print("="*70)
    print("PRECOMPUTING ROLLING FEATURES")
    print("="*70)

    # Input/output paths
    input_path = "aggregated_nba_data.parquet"
    output_path = "aggregated_nba_data_with_features.parquet"

    # Load raw data
    print(f"\nLoading raw data from {input_path}...")
    df = load_player_data(input_path, verbose=True)

    print(f"  • Loaded: {len(df):,} rows, {len(df.columns)} columns")

    # Add rolling features
    print(f"\nAdding rolling features...")
    df_enhanced = add_rolling_features(
        df,
        windows=[5, 10, 20],  # L5, L10, L20
        add_variance=True,    # Add std deviation
        add_trend=True,       # Add momentum indicators
        low_memory=False,     # Use full feature set
        verbose=True
    )

    print(f"\n  • Enhanced: {len(df_enhanced):,} rows, {len(df_enhanced.columns)} columns")
    print(f"  • Added {len(df_enhanced.columns) - len(df.columns)} new features")

    # Save enhanced data
    print(f"\nSaving enhanced data to {output_path}...")
    df_enhanced.to_parquet(output_path, index=False, compression='gzip')

    # Verify file size
    file_size_mb = Path(output_path).stat().st_size / (1024**2)
    print(f"  • Saved: {file_size_mb:.1f} MB")

    print("\n" + "="*70)
    print("✓ FEATURE PRECOMPUTATION COMPLETE")
    print("="*70)
    print(f"\nNext steps:")
    print(f"  1. Upload {output_path} to Modal volume:")
    print(f"     modal volume put nba-data {output_path} /data/{output_path}")
    print(f"  2. Update modal_train.py to use {output_path}")
    print(f"  3. Retrain models with enhanced features")

if __name__ == "__main__":
    main()
