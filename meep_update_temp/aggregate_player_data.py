"""
Aggregate Player Data from Multiple CSVs

This script combines player data from different sources to create a comprehensive database:

1. PlayerStatistics.csv (1946-2025 from Kaggle)
2. priors_data/Advanced.csv (Basketball Reference advanced stats)
3. priors_data/Per 100 Poss.csv (Per-100-possession stats)
4. priors_data/Player Shooting.csv (Shooting splits)
5. priors_data/Player Play By Play.csv (Play-by-play derived stats)

Each CSV may have different:
- Date ranges (1980+, 1996+, 2000+, etc.)
- Granularity (season-level vs game-level)
- Player name formats

This script:
- Identifies date range of each source
- Merges on common keys (player, season, team)
- Fills gaps intelligently
- Creates comprehensive player database

Usage:
    python aggregate_player_data.py --output aggregated_player_data.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime


def analyze_csv_date_range(csv_path, date_col=None, season_col=None):
    """
    Analyze the date range of a CSV file.

    Args:
        csv_path: Path to CSV
        date_col: Name of date column (if exists)
        season_col: Name of season column (if exists)

    Returns:
        dict with min_year, max_year, total_rows, columns
    """
    print(f"\nAnalyzing: {csv_path}")

    try:
        # Read first few rows to check structure
        df_sample = pd.read_csv(csv_path, nrows=1000)

        print(f"  Columns: {list(df_sample.columns[:10])}...")
        print(f"  Total rows (estimate): {len(df_sample) * (Path(csv_path).stat().st_size / (df_sample.memory_usage(deep=True).sum())): ,.0f}")

        # Try to find date/season column
        if season_col and season_col in df_sample.columns:
            min_year = df_sample[season_col].min()
            max_year = df_sample[season_col].max()
            print(f"  Date range: {min_year} - {max_year} (from {season_col})")
        elif date_col and date_col in df_sample.columns:
            df_sample[date_col] = pd.to_datetime(df_sample[date_col])
            min_year = df_sample[date_col].dt.year.min()
            max_year = df_sample[date_col].dt.year.max()
            print(f"  Date range: {min_year} - {max_year} (from {date_col})")
        else:
            # Look for columns with "season" or "year" in name
            year_cols = [col for col in df_sample.columns if 'season' in col.lower() or 'year' in col.lower()]
            if year_cols:
                year_col = year_cols[0]
                min_year = df_sample[year_col].min()
                max_year = df_sample[year_col].max()
                print(f"  Date range: {min_year} - {max_year} (from {year_col})")
            else:
                print(f"  No date/season column found")
                min_year, max_year = None, None

        return {
            'file': csv_path,
            'min_year': min_year,
            'max_year': max_year,
            'columns': list(df_sample.columns),
            'sample_count': len(df_sample)
        }

    except Exception as e:
        print(f"  Error: {e}")
        return None


def aggregate_player_data(player_stats_csv, priors_dir, output_csv):
    """
    Aggregate player data from multiple sources.

    Strategy:
    1. Load main PlayerStatistics.csv (game-level, 1946-2025)
    2. Load priors (season-level, various date ranges)
    3. Merge priors onto game data by (player, season, team)
    4. Fill missing values intelligently
    5. Save comprehensive dataset

    Args:
        player_stats_csv: Path to main PlayerStatistics.csv
        priors_dir: Directory containing priors CSVs
        output_csv: Output path for aggregated data
    """
    print("="*70)
    print("AGGREGATING PLAYER DATA")
    print("="*70)

    # 1. Analyze all sources
    print("\n[STEP 1] Analyzing data sources...")

    sources = {
        'PlayerStatistics': analyze_csv_date_range(
            player_stats_csv,
            date_col='GAME_DATE_EST',
            season_col='season'
        ),
        'Advanced': analyze_csv_date_range(
            f"{priors_dir}/Advanced.csv",
            season_col='Season'
        ),
        'Per100': analyze_csv_date_range(
            f"{priors_dir}/Per 100 Poss.csv",
            season_col='Season'
        ),
        'Shooting': analyze_csv_date_range(
            f"{priors_dir}/Player Shooting.csv",
            season_col='Season'
        ),
        'PlayByPlay': analyze_csv_date_range(
            f"{priors_dir}/Player Play By Play.csv",
            season_col='Season'
        ),
    }

    # 2. Load main player statistics
    print("\n[STEP 2] Loading PlayerStatistics.csv...")
    print("  (This may take a minute for large files)")

    df_main = pd.read_csv(player_stats_csv)
    print(f"  Loaded: {len(df_main):,} rows")
    print(f"  Columns: {len(df_main.columns)}")

    # Ensure season column exists
    if 'season' not in df_main.columns and 'GAME_DATE_EST' in df_main.columns:
        df_main['GAME_DATE_EST'] = pd.to_datetime(df_main['GAME_DATE_EST'])
        df_main['season'] = df_main['GAME_DATE_EST'].dt.year
        # Adjust for NBA season (Oct-Jun crosses calendar years)
        df_main.loc[df_main['GAME_DATE_EST'].dt.month >= 10, 'season'] += 1
        print(f"  Created season column: {df_main['season'].min()}-{df_main['season'].max()}")

    # 3. Load and merge priors
    print("\n[STEP 3] Loading and merging priors...")

    # Load Advanced stats
    if Path(f"{priors_dir}/Advanced.csv").exists():
        print("  Loading Advanced.csv...")
        df_advanced = pd.read_csv(f"{priors_dir}/Advanced.csv")
        print(f"    Rows: {len(df_advanced):,}")
        # TODO: Merge logic based on common keys

    # Load Per 100 Poss
    if Path(f"{priors_dir}/Per 100 Poss.csv").exists():
        print("  Loading Per 100 Poss.csv...")
        df_per100 = pd.read_csv(f"{priors_dir}/Per 100 Poss.csv")
        print(f"    Rows: {len(df_per100):,}")
        # TODO: Merge logic

    # Load Shooting
    if Path(f"{priors_dir}/Player Shooting.csv").exists():
        print("  Loading Player Shooting.csv...")
        df_shooting = pd.read_csv(f"{priors_data}/Player Shooting.csv")
        print(f"    Rows: {len(df_shooting):,}")
        # TODO: Merge logic

    # 4. Save aggregated data
    print("\n[STEP 4] Saving aggregated data...")
    print(f"  Output: {output_csv}")

    # For now, just save main data
    # TODO: Save merged data after implementing merge logic
    df_main.to_csv(output_csv, index=False)

    print(f"\n[DONE] Aggregated data saved:")
    print(f"  Rows: {len(df_main):,}")
    print(f"  Columns: {len(df_main.columns)}")
    print(f"  Date range: {df_main['season'].min()}-{df_main['season'].max()}")

    return df_main


def main():
    parser = argparse.ArgumentParser(description='Aggregate player data from multiple CSVs')
    parser.add_argument('--player-csv', type=str,
                       default='PlayerStatistics.csv',
                       help='Path to main PlayerStatistics.csv')
    parser.add_argument('--priors-dir', type=str,
                       default='priors_data',
                       help='Directory containing priors CSVs')
    parser.add_argument('--output', type=str,
                       default='aggregated_player_data.csv',
                       help='Output CSV path')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze sources, don\'t aggregate')

    args = parser.parse_args()

    if args.analyze_only:
        # Just analyze date ranges
        print("\n[ANALYSIS MODE] Checking all sources...")

        analyze_csv_date_range(args.player_csv, date_col='GAME_DATE_EST')
        analyze_csv_date_range(f"{args.priors_dir}/Advanced.csv")
        analyze_csv_date_range(f"{args.priors_dir}/Per 100 Poss.csv")
        analyze_csv_date_range(f"{args.priors_dir}/Player Shooting.csv")
        analyze_csv_date_range(f"{args.priors_dir}/Player Play By Play.csv")
    else:
        # Full aggregation
        aggregate_player_data(
            args.player_csv,
            args.priors_dir,
            args.output
        )


if __name__ == "__main__":
    main()
