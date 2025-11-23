#!/usr/bin/env python
"""
Data Loading Module for NBA Prediction Models

Handles loading aggregated player data with memory optimization.
"""

import pandas as pd
import pyarrow.parquet as pq
import gc
from pathlib import Path
from typing import Optional, Tuple, Union


def load_aggregated_player_data(
    parquet_path: str,
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load aggregated player data from Parquet file with memory optimization.

    Args:
        parquet_path: Path to aggregated_nba_data.parquet
        min_year: Optional minimum season year (e.g., 1997, 2002)
        max_year: Optional maximum season year (e.g., 2024)
        verbose: Print loading progress

    Returns:
        DataFrame with player-game rows
    """
    if verbose:
        print("="*70)
        print("LOADING AGGREGATED PLAYER DATA")
        print("="*70)
        print(f"- Loading from: {parquet_path}")
        print("- PARQUET FORMAT: Fast loading with chunked memory optimization")
        if min_year:
            print(f"- FILTER: Keeping only {min_year}+ data to reduce memory")
        if max_year:
            print(f"- FILTER: Keeping only data up to {max_year}")

    # Open Parquet file
    parquet_file = pq.ParquetFile(parquet_path)
    total_rows = parquet_file.metadata.num_rows
    num_row_groups = parquet_file.num_row_groups

    if verbose:
        print(f"- Total rows in file: {total_rows:,}")
        print(f"- Parquet file has {num_row_groups} row groups")

    # Load ALL columns (no filtering - keep everything!)
    all_columns = [field.name for field in parquet_file.schema_arrow]
    columns_to_load = all_columns  # Keep everything

    if verbose:
        print(f"- Loading ALL {len(columns_to_load)} columns (no filtering)")

    # Load entire dataset at once (works on Modal with 64GB+ RAM)
    if verbose:
        print("- Loading entire Parquet file at once...")

    agg_df = parquet_file.read(columns=columns_to_load).to_pandas()

    # Apply year filter if specified
    if min_year or max_year:
        year_col = None
        for col_name in ['season', 'game_year', 'season_end_year', 'year']:
            if col_name in agg_df.columns:
                year_col = col_name
                break

        if year_col:
            rows_before = len(agg_df)
            if min_year:
                agg_df = agg_df[agg_df[year_col] >= min_year]
            if max_year:
                agg_df = agg_df[agg_df[year_col] <= max_year]

            if verbose and len(agg_df) < rows_before:
                print(f"  Filtered by year: {rows_before:,} → {len(agg_df):,} rows")

    # Optimize dtypes for memory efficiency
    if verbose:
        print("- Optimizing dtypes...")

    for col in agg_df.select_dtypes(include=['object']).columns:
        agg_df[col] = agg_df[col].astype('category')

    for col in agg_df.select_dtypes(include=['float64']).columns:
        agg_df[col] = pd.to_numeric(agg_df[col], downcast='float', errors='ignore')

    for col in agg_df.select_dtypes(include=['int64']).columns:
        agg_df[col] = pd.to_numeric(agg_df[col], downcast='integer', errors='ignore')

    gc.collect()

    if verbose:
        mem_mb = agg_df.memory_usage(deep=True).sum() / 1024**2
        print(f"[OK] Loaded {len(agg_df):,} player-game rows")
        print(f"[OK] Memory usage: {mem_mb:.1f} MB ({mem_mb/1024:.1f} GB)")
        print(f"[OK] Columns: {len(agg_df.columns)}")

        # Show year range
        year_col = None
        for col_name in ['season', 'game_year', 'season_end_year', 'year']:
            if col_name in agg_df.columns:
                year_col = col_name
                break
        if year_col:
            min_yr = int(agg_df[year_col].min())
            max_yr = int(agg_df[year_col].max())
            print(f"[OK] Year range: {min_yr}-{max_yr}")
        print("="*70)

    return agg_df


def load_player_data(
    data_path: str,
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load player data from either Parquet file or CSV directory.

    Auto-detects the input type:
    - If data_path ends with .parquet: loads from Parquet file
    - If data_path is a directory: loads from CSV directory using csv_aggregation
    - If data_path ends with .csv: loads single CSV file

    Args:
        data_path: Path to Parquet file, CSV file, or CSV directory
        min_year: Optional minimum season year
        max_year: Optional maximum season year
        verbose: Print loading progress

    Returns:
        DataFrame with player-game rows
    """
    path = Path(data_path)

    # Check if it's a Parquet file
    if path.is_file() and path.suffix == '.parquet':
        return load_aggregated_player_data(str(path), min_year, max_year, verbose)

    # Check if it's a single CSV file
    if path.is_file() and path.suffix == '.csv':
        return load_from_kaggle_csvs(str(path), min_year, max_year, verbose)

    # Check if it's a directory with CSVs
    if path.is_dir():
        # Use CSV aggregation to load all 9 CSVs
        from shared.csv_aggregation import aggregate_player_data
        return aggregate_player_data(str(path), min_year, max_year, verbose)

    raise ValueError(f"Invalid data_path: {data_path}. Must be .parquet file, .csv file, or directory containing CSVs")


def get_year_column(df: pd.DataFrame) -> str:
    """Find the year column in the dataframe."""
    for col_name in ['season', 'season_end_year', 'game_year', 'year']:
        if col_name in df.columns:
            return col_name
    raise KeyError(f"No year column found. Available columns: {list(df.columns)[:20]}")


def get_season_range(df: pd.DataFrame) -> Tuple[int, int]:
    """Get the min and max season years from the dataframe."""
    year_col = get_year_column(df)
    seasons = df[year_col].dropna().unique()
    return int(min(seasons)), int(max(seasons))


def load_from_kaggle_csvs(
    player_csv_path: str,
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load player data directly from Kaggle CSV files.

    Loads PlayerStatistics.csv from the Kaggle dataset:
    /kaggle/input/historical-nba-data-and-player-box-scores/PlayerStatistics.csv

    Args:
        player_csv_path: Path to PlayerStatistics.csv
        min_year: Optional minimum season year
        max_year: Optional maximum season year
        verbose: Print loading progress

    Returns:
        DataFrame with player-game rows
    """
    if verbose:
        print("="*70)
        print("LOADING PLAYER DATA FROM CSV")
        print("="*70)
        print(f"- Loading from: {player_csv_path}")
        if min_year:
            print(f"- FILTER: Keeping only {min_year}+ data")
        if max_year:
            print(f"- FILTER: Keeping only data up to {max_year}")

    # Load PlayerStatistics.csv
    # This has all game-level player stats from 1947-2026
    if verbose:
        print("- Reading CSV file...")

    df = pd.read_csv(player_csv_path, low_memory=False)

    if verbose:
        print(f"- Loaded {len(df):,} rows")
        print(f"- Columns: {len(df.columns)}")

    # Apply year filter if specified
    if min_year or max_year:
        # Find year column
        year_col = None
        for col_name in ['season', 'season_end_year', 'game_year', 'year']:
            if col_name in df.columns:
                year_col = col_name
                break

        if year_col:
            rows_before = len(df)
            if min_year:
                df = df[df[year_col] >= min_year]
            if max_year:
                df = df[df[year_col] <= max_year]

            if verbose:
                print(f"- Filtered by year: {rows_before:,} → {len(df):,} rows")

    # Optimize dtypes to save memory
    if verbose:
        print("- Optimizing dtypes...")

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')

    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float', errors='ignore')

    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer', errors='ignore')

    if verbose:
        mem_mb = df.memory_usage(deep=True).sum() / 1024**2
        print(f"[OK] Final data: {len(df):,} rows, {len(df.columns)} columns")
        print(f"[OK] Memory usage: {mem_mb:.1f} MB ({mem_mb/1024:.1f} GB)")

        # Show year range
        try:
            year_col = get_year_column(df)
            min_yr = int(df[year_col].min())
            max_yr = int(df[year_col].max())
            print(f"[OK] Year range: {min_yr}-{max_yr}")
        except:
            pass

        print("="*70)

    return df


def load_player_data(
    data_source: Union[str, Path],
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load player data from Parquet, CSV file, or CSV directory (auto-detects format).

    Args:
        data_source: Path to:
            - .parquet file (aggregated data)
            - .csv file (PlayerStatistics.csv only)
            - directory containing all 7 Basketball Reference CSVs
        min_year: Optional minimum season year
        max_year: Optional maximum season year
        verbose: Print loading progress

    Returns:
        DataFrame with player-game rows (with advanced stats if CSVs available)
    """
    data_path = Path(data_source)

    if not data_path.exists():
        raise FileNotFoundError(f"Data source not found: {data_source}")

    # Auto-detect format
    if data_path.is_dir():
        # Directory with multiple CSVs - aggregate them all
        if verbose:
            print("📁 Detected CSV directory (will merge all Basketball Reference tables)")
        from .csv_aggregation import load_and_merge_csvs
        return load_and_merge_csvs(str(data_path), min_year, max_year, verbose)

    elif data_path.suffix.lower() == '.parquet':
        if verbose:
            print("📦 Detected Parquet format")
        return load_aggregated_player_data(str(data_path), min_year, max_year, verbose)

    elif data_path.suffix.lower() == '.csv':
        # Single CSV file - check if it's in a directory with other BR tables
        parent_dir = data_path.parent
        has_advanced = (parent_dir / "Player Advanced.csv").exists()
        has_per100 = (parent_dir / "Player Per 100 Poss.csv").exists()

        if has_advanced or has_per100:
            if verbose:
                print("📄 Detected CSV file with Basketball Reference tables in same directory")
                print("   Will merge all available tables for advanced stats")
            from .csv_aggregation import load_and_merge_csvs
            return load_and_merge_csvs(str(parent_dir), min_year, max_year, verbose)
        else:
            if verbose:
                print("📄 Detected CSV format (PlayerStatistics.csv only, no advanced stats)")
            return load_from_kaggle_csvs(str(data_path), min_year, max_year, verbose)

    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}. Use .parquet, .csv, or directory")
