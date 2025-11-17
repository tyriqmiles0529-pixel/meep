#!/usr/bin/env python
"""
CSV Aggregation Module

Aggregates all 7 Basketball Reference CSV tables to recreate the full
aggregated dataset with advanced stats, per-100, PBP, and shooting data.

Maintains the same high merge rate achieved in original aggregation.
"""

import pandas as pd
import gc
from pathlib import Path
from typing import Optional, Dict


def load_and_merge_csvs(
    data_dir: str,
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load all 7 Basketball Reference CSVs and merge them.

    Expected files in data_dir:
    1. PlayerStatistics.csv - Base box scores (game_id, player_id, points, assists, etc.)
    2. Player Advanced.csv - Advanced stats (PER, TS%, BPM, VORP, etc.)
    3. Player Per 100 Poss.csv - Per-100 possession stats
    4. Player Play-By-Play.csv - PBP stats (plus/minus, turnovers, fouls)
    5. Player Shooting.csv - Shooting zones and percentages
    6. (Optional) Player Totals.csv
    7. (Optional) Team stats

    Args:
        data_dir: Directory containing all CSV files
        min_year: Optional minimum season year
        max_year: Optional maximum season year
        verbose: Print progress

    Returns:
        Merged DataFrame with all advanced stats
    """
    data_path = Path(data_dir)

    if verbose:
        print("="*70)
        print("AGGREGATING CSV DATA FROM BASKETBALL REFERENCE")
        print("="*70)
        print(f"Data directory: {data_dir}")

    # =====================================================================
    # Step 1: Load base player statistics (game-level box scores)
    # =====================================================================
    player_stats_path = data_path / "PlayerStatistics.csv"
    if not player_stats_path.exists():
        raise FileNotFoundError(f"PlayerStatistics.csv not found in {data_dir}")

    if verbose:
        print("\n[1/5] Loading PlayerStatistics.csv (base box scores)...")

    df = pd.read_csv(player_stats_path, low_memory=False)

    if verbose:
        print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Apply year filter early
    if min_year or max_year:
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
                print(f"  Filtered by year: {rows_before:,} → {len(df):,} rows")

    # Optimize dtypes
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() < len(df) * 0.5:  # Only categorize if < 50% unique
            df[col] = df[col].astype('category')

    # =====================================================================
    # Step 2: Load and merge Basketball Reference advanced stats
    # =====================================================================
    merge_keys = ['player_id', 'season']  # Keys for merging BR tables
    merge_successful = {}

    # Advanced stats (PER, BPM, VORP, etc.)
    advanced_path = data_path / "Player Advanced.csv"
    if advanced_path.exists():
        if verbose:
            print("\n[2/5] Loading Player Advanced.csv...")

        adv_df = pd.read_csv(advanced_path, low_memory=False)

        # Rename columns with adv_ prefix
        adv_cols = {col: f'adv_{col.lower().replace(" ", "_")}'
                   for col in adv_df.columns if col not in ['player_id', 'season']}
        adv_df = adv_df.rename(columns=adv_cols)

        if verbose:
            print(f"  Loaded {len(adv_df):,} rows, {len(adv_df.columns)} columns")

        # Merge
        before_merge = len(df)
        df = df.merge(adv_df, on=merge_keys, how='left', suffixes=('', '_adv_dup'))

        # Calculate merge rate
        non_null_rate = df[[c for c in df.columns if c.startswith('adv_')]].notna().any(axis=1).mean()
        merge_successful['advanced'] = non_null_rate * 100

        if verbose:
            print(f"  Merged: {before_merge:,} rows → {len(df):,} rows")
            print(f"  Match rate: {non_null_rate*100:.1f}%")

        del adv_df
        gc.collect()
    else:
        if verbose:
            print("\n[2/5] Player Advanced.csv not found - skipping")

    # Per-100 possession stats
    per100_path = data_path / "Player Per 100 Poss.csv"
    if per100_path.exists():
        if verbose:
            print("\n[3/5] Loading Player Per 100 Poss.csv...")

        per100_df = pd.read_csv(per100_path, low_memory=False)

        # Rename columns with per100_ prefix
        per100_cols = {col: f'per100_{col.lower().replace(" ", "_").replace("/", "_")}'
                      for col in per100_df.columns if col not in ['player_id', 'season']}
        per100_df = per100_df.rename(columns=per100_cols)

        if verbose:
            print(f"  Loaded {len(per100_df):,} rows, {len(per100_df.columns)} columns")

        # Merge
        before_merge = len(df)
        df = df.merge(per100_df, on=merge_keys, how='left', suffixes=('', '_per100_dup'))

        non_null_rate = df[[c for c in df.columns if c.startswith('per100_')]].notna().any(axis=1).mean()
        merge_successful['per100'] = non_null_rate * 100

        if verbose:
            print(f"  Merged: {before_merge:,} rows → {len(df):,} rows")
            print(f"  Match rate: {non_null_rate*100:.1f}%")

        del per100_df
        gc.collect()
    else:
        if verbose:
            print("\n[3/5] Player Per 100 Poss.csv not found - skipping")

    # Play-by-Play stats (plus/minus, turnovers, fouls)
    pbp_path = data_path / "Player Play-By-Play.csv"
    if pbp_path.exists():
        if verbose:
            print("\n[4/5] Loading Player Play-By-Play.csv...")

        pbp_df = pd.read_csv(pbp_path, low_memory=False)

        # Rename columns with pbp_ prefix
        pbp_cols = {col: f'pbp_{col.lower().replace(" ", "_").replace("/", "_").replace("+", "plus").replace("-", "_")}'
                   for col in pbp_df.columns if col not in ['player_id', 'season']}
        pbp_df = pbp_df.rename(columns=pbp_cols)

        if verbose:
            print(f"  Loaded {len(pbp_df):,} rows, {len(pbp_df.columns)} columns")

        # Merge
        before_merge = len(df)
        df = df.merge(pbp_df, on=merge_keys, how='left', suffixes=('', '_pbp_dup'))

        non_null_rate = df[[c for c in df.columns if c.startswith('pbp_')]].notna().any(axis=1).mean()
        merge_successful['pbp'] = non_null_rate * 100

        if verbose:
            print(f"  Merged: {before_merge:,} rows → {len(df):,} rows")
            print(f"  Match rate: {non_null_rate*100:.1f}%")

        del pbp_df
        gc.collect()
    else:
        if verbose:
            print("\n[4/5] Player Play-By-Play.csv not found - skipping")

    # Shooting stats (zones, percentages)
    shoot_path = data_path / "Player Shooting.csv"
    if shoot_path.exists():
        if verbose:
            print("\n[5/5] Loading Player Shooting.csv...")

        shoot_df = pd.read_csv(shoot_path, low_memory=False)

        # Rename columns with shoot_ prefix
        shoot_cols = {col: f'shoot_{col.lower().replace(" ", "_").replace("%", "percent").replace("/", "_")}'
                     for col in shoot_df.columns if col not in ['player_id', 'season']}
        shoot_df = shoot_df.rename(columns=shoot_cols)

        if verbose:
            print(f"  Loaded {len(shoot_df):,} rows, {len(shoot_df.columns)} columns")

        # Merge
        before_merge = len(df)
        df = df.merge(shoot_df, on=merge_keys, how='left', suffixes=('', '_shoot_dup'))

        non_null_rate = df[[c for c in df.columns if c.startswith('shoot_')]].notna().any(axis=1).mean()
        merge_successful['shooting'] = non_null_rate * 100

        if verbose:
            print(f"  Merged: {before_merge:,} rows → {len(df):,} rows")
            print(f"  Match rate: {non_null_rate*100:.1f}%")

        del shoot_df
        gc.collect()
    else:
        if verbose:
            print("\n[5/5] Player Shooting.csv not found - skipping")

    # =====================================================================
    # Final optimization
    # =====================================================================
    if verbose:
        print("\n" + "="*70)
        print("AGGREGATION COMPLETE")
        print("="*70)
        print(f"Final dataset: {len(df):,} rows, {len(df.columns)} columns")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

        if merge_successful:
            print("\nMerge Success Rates:")
            for table, rate in merge_successful.items():
                print(f"  {table:12s}: {rate:.1f}%")
            avg_rate = sum(merge_successful.values()) / len(merge_successful)
            print(f"  {'Average':12s}: {avg_rate:.1f}%")

        # Show year range
        try:
            year_col = None
            for col_name in ['season', 'season_end_year', 'game_year', 'year']:
                if col_name in df.columns:
                    year_col = col_name
                    break
            if year_col:
                min_yr = int(df[year_col].min())
                max_yr = int(df[year_col].max())
                print(f"\nYear range: {min_yr}-{max_yr}")
        except:
            pass

        print("="*70)

    return df
