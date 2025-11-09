"""
Create Pre-Aggregated Dataset - EFFICIENT MERGE VERSION

This script is optimized for both speed and memory by using efficient
vectorized operations for merging instead of slower, row-by-row methods.
It is configured to read from the two separate Kaggle datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
from tqdm import tqdm
from rapidfuzz import fuzz, process
import gc


def load_player_statistics(csv_path):
    """Load main player game logs."""
    print(f"\n[1/7] Loading PlayerStatistics.csv...")
    print(f"  Path: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")

    # Parse dates
    if 'gameDate' in df.columns:
        df['gameDate'] = pd.to_datetime(df['gameDate'], format='mixed', utc=True, errors='coerce')
        df['game_year'] = df['gameDate'].dt.year
        # NBA season (Oct-Jun crosses calendar years)
        df['season'] = df['game_year'].copy()
        df.loc[df['gameDate'].dt.month >= 10, 'season'] += 1
        print(f"  Date range: {df['gameDate'].min()} to {df['gameDate'].max()}")
        print(f"  Season range: {df['season'].min()} - {df['season'].max()}")

    # Standardize player identifier
    if 'personId' in df.columns:
        df['player_game_id'] = df['personId'].astype(str)

    # Create full player name for matching
    if 'firstName' in df.columns and 'lastName' in df.columns:
        df['player_name'] = (df['firstName'].fillna('') + ' ' + df['lastName'].fillna('')).str.strip()

    return df


def load_advanced_stats(csv_path):
    """Load Basketball Reference advanced stats."""
    print(f"\n[2/7] Loading Advanced.csv...")

    df = pd.read_csv(csv_path)
    print(f"  Rows: {len(df):,}")
    print(f"  Season range: {df['season'].min()} - {df['season'].max()}")

    # Prefix columns to avoid conflicts
    rename_cols = {col: f'adv_{col}' for col in df.columns
                   if col not in ['season', 'player', 'player_id', 'team']}
    df = df.rename(columns=rename_cols)

    return df


def load_per_100_poss(csv_path):
    """Load per-100-possession stats."""
    print(f"\n[3/7] Loading Per 100 Poss.csv...")

    df = pd.read_csv(csv_path)
    print(f"  Rows: {len(df):,}")
    print(f"  Season range: {df['season'].min()} - {df['season'].max()}")

    # Prefix columns
    rename_cols = {col: f'per100_{col}' for col in df.columns
                   if col not in ['season', 'player', 'player_id', 'team']}
    df = df.rename(columns=rename_cols)

    return df


def load_shooting_stats(csv_path):
    """Load shooting splits."""
    print(f"\n[4/7] Loading Player Shooting.csv...")

    df = pd.read_csv(csv_path)
    print(f"  Rows: {len(df):,}")
    print(f"  Season range: {df['season'].min()} - {df['season'].max()}")

    # Prefix columns
    rename_cols = {col: f'shoot_{col}' for col in df.columns
                   if col not in ['season', 'player', 'player_id', 'team']}
    df = df.rename(columns=rename_cols)

    return df


def load_playbyplay_stats(csv_path):
    """Load play-by-play derived stats."""
    print(f"\n[5/7] Loading Player Play By Play.csv...")

    df = pd.read_csv(csv_path)
    print(f"  Rows: {len(df):,}")
    print(f"  Season range: {df['season'].min()} - {df['season'].max()}")

    # Prefix columns
    rename_cols = {col: f'pbp_{col}' for col in df.columns
                   if col not in ['season', 'player', 'player_id', 'team']}
    df = df.rename(columns=rename_cols)

    return df


def load_team_summaries(csv_path):
    """Load team season summaries."""
    print(f"\n[6/7] Loading Team Summaries.csv...")

    df = pd.read_csv(csv_path)
    print(f"  Rows: {len(df):,}")
    print(f"  Season range: {df['season'].min()} - {df['season'].max()}")

    # Prefix columns
    rename_cols = {col: f'team_{col}' for col in df.columns
                   if col not in ['season', 'team', 'abbreviation']}
    df = df.rename(columns=rename_cols)

    return df


def load_team_abbrev(csv_path):
    """Load team abbreviation mappings."""
    print(f"\n[7/7] Loading Team Abbrev.csv...")

    df = pd.read_csv(csv_path)
    print(f"  Rows: {len(df):,}")

    return df


def create_fuzzy_match_mapping(df_main, df_prior, season_col='season'):
    """
    Create player name mapping using fuzzy matching.
    Memory efficient - only stores unique player-season combinations.
    """
    print(f"  Building fuzzy match mapping...")

    # Get unique player-season combinations
    main_combos = df_main[['player_name', season_col]].drop_duplicates()
    prior_combos = df_prior[['player', season_col]].drop_duplicates()

    print(f"    Main players: {len(main_combos):,}")
    print(f"    Prior players: {len(prior_combos):,}")

    # Create mapping dict
    mapping = {}

    # For each season, do fuzzy matching within that season only (faster)
    seasons = sorted(df_main[season_col].unique())
    for season in tqdm(seasons, desc="    Fuzzy matching by season"):
        main_season = main_combos[main_combos[season_col] == season]['player_name'].tolist()
        prior_season = prior_combos[prior_combos[season_col] == season]['player'].tolist()

        if not prior_season:  # No priors for this season
            continue

        # Fuzzy match each player in main to best match in prior
        for main_player in main_season:
            if pd.isna(main_player) or main_player == '':
                continue

            # Find best match
            result = process.extractOne(
                main_player,
                prior_season,
                scorer=fuzz.ratio
            )
            
            if result is None:
                continue
                
            match, score, _ = result

            # Only accept if score > 85 (pretty confident match)
            if score > 85:
                mapping[(main_player, season)] = match

    print(f"    Matched: {len(mapping):,} player-season combinations")

    return mapping


def merge_one_prior(df_main, df_prior, mapping, prior_name):
    """
    Memory-efficient merge using vectorized operations instead of apply().
    """
    print(f"\n  Merging {prior_name}...")
    
    # Create a MultiIndex for efficient mapping. This is much more memory-friendly
    # and faster than creating an intermediate series of tuples with .apply().
    multi_index = pd.MultiIndex.from_frame(df_main[['player_name', 'season']])
    merge_key = multi_index.map(mapping)
    
    df_main[f'_temp_match'] = merge_key
    
    # Merge on the temporary key
    df_merged = df_main.merge(
        df_prior,
        left_on=['_temp_match', 'season'],
        right_on=['player', 'season'],
        how='left',
        suffixes=('', f'_{prior_name}_dup')
    )
    
    # Clean up
    df_merged = df_merged.drop(columns=['_temp_match'], errors='ignore')
    
    # Drop original df_main to free memory
    del df_main
    gc.collect()
    
    cols_added = len([c for c in df_merged.columns if c.startswith(f'{prior_name}_')])
    print(f"    Added {cols_added} columns")
    
    # Check match rate (first numeric column from prior)
    numeric_cols = [c for c in df_merged.columns if c.startswith(f'{prior_name}_')]
    if numeric_cols:
        match_rate = (~df_merged[numeric_cols[0]].isna()).sum() / len(df_merged) * 100
        print(f"    Match rate: {match_rate:.1f}%")
    
    return df_merged


def merge_player_priors(df_main, df_adv, df_per100, df_shoot, df_pbp):
    """
    Merge all player season-level priors onto game-level data.
    """
    print(f"\n{'='*70}")
    print("MERGING PLAYER PRIORS (WITH FUZZY MATCHING)")
    print(f"{'='*70}")

    initial_rows = len(df_main)

    # 1. Merge Advanced stats
    print(f"\n[1/4] Advanced stats...")
    player_mapping = create_fuzzy_match_mapping(df_main, df_adv, 'season')
    df_merged = merge_one_prior(df_main, df_adv, player_mapping, 'adv')
    del player_mapping
    gc.collect()

    # 2. Merge Per 100 Poss
    print(f"\n[2/4] Per 100 Poss...")
    player_mapping_per100 = create_fuzzy_match_mapping(df_merged, df_per100, 'season')
    df_merged = merge_one_prior(df_merged, df_per100, player_mapping_per100, 'per100')
    del player_mapping_per100
    gc.collect()

    # 3. Merge Shooting
    print(f"\n[3/4] Shooting splits...")
    player_mapping_shoot = create_fuzzy_match_mapping(df_merged, df_shoot, 'season')
    df_merged = merge_one_prior(df_merged, df_shoot, player_mapping_shoot, 'shoot')
    del player_mapping_shoot
    gc.collect()

    # 4. Merge Play-by-Play
    print(f"\n[4/4] Play-by-Play...")
    player_mapping_pbp = create_fuzzy_match_mapping(df_merged, df_pbp, 'season')
    df_merged = merge_one_prior(df_merged, df_pbp, player_mapping_pbp, 'pbp')
    del player_mapping_pbp
    gc.collect()

    # Verify no row multiplication
    final_rows = len(df_merged)
    if final_rows != initial_rows:
        print(f"\n  ⚠️ WARNING: Row count changed from {initial_rows:,} to {final_rows:,}")
    else:
        print(f"\n  ✓ Row count preserved: {final_rows:,}")

    return df_merged


def merge_team_priors(df_main, df_team_sum, df_team_abbrev):
    """Merge team season-level context."""
    print(f"\n{'='*70}")
    print("MERGING TEAM PRIORS")
    print(f"{'='*70}")

    # Merge team summaries (player's team)
    print(f"\n  Merging team summaries...")
    merge_keys = ['season', 'team']

    df_merged = df_main.merge(
        df_team_sum,
        on=merge_keys,
        how='left',
        suffixes=('', '_team_dup')
    )

    team_cols_added = len([c for c in df_merged.columns if c.startswith('team_')])
    print(f"    Added {team_cols_added} team stat columns")
    
    # Free memory
    del df_main
    gc.collect()

    return df_merged


def fill_missing_values(df):
    """Intelligently fill missing values based on era/context."""
    print(f"\n{'='*70}")
    print("FILLING MISSING VALUES")
    print(f"{'='*70}")

    # Count missing before
    missing_before = df.isna().sum().sum()
    print(f"\n  Missing values before: {missing_before:,}")

    # Fill priors with 0 (player didn't have that stat recorded)
    prior_cols = [c for c in df.columns if c.startswith(('adv_', 'per100_', 'shoot_', 'pbp_', 'team_'))]

    print(f"\n  Filling {len(prior_cols)} prior columns with 0...")
    df[prior_cols] = df[prior_cols].fillna(0)

    # Count missing after
    missing_after = df.isna().sum().sum()
    print(f"  Missing values after: {missing_after:,}")
    print(f"  Filled: {missing_before - missing_after:,}")

    return df


def main():
    parser = argparse.ArgumentParser(description='Create pre-aggregated NBA dataset (efficient merge version)')
    parser.add_argument('--player-csv', type=str,
                       default='/kaggle/input/historical-nba-data-and-player-box-scores/PlayerStatistics.csv',
                       help='Path to PlayerStatistics.csv')
    parser.add_argument('--priors-dir', type=str,
                       default='/kaggle/input/nba-aba-baa-stats',
                       help='Directory with priors CSVs')
    parser.add_argument('--output', type=str,
                       default='aggregated_nba_data.csv',
                       help='Output CSV path')
    parser.add_argument('--compression', type=str,
                       default='gzip',
                       choices=['gzip', 'bz2', 'zip', 'none'],
                       help='Compression for output')

    args, _ = parser.parse_known_args()

    print("="*70)
    print("PRE-AGGREGATING NBA DATASET (EFFICIENT MERGE VERSION)")
    print("="*70)
    print(f"Expected time: 5-10 minutes")
    print(f"Expected size: ~500-800 MB compressed")

    # Load all data sources
    df_player = load_player_statistics(args.player_csv)
    df_adv = load_advanced_stats(f"{args.priors_dir}/Advanced.csv")
    df_per100 = load_per_100_poss(f"{args.priors_dir}/Per 100 Poss.csv")
    df_shoot = load_shooting_stats(f"{args.priors_dir}/Player Shooting.csv")
    df_pbp = load_playbyplay_stats(f"{args.priors_dir}/Player Play By Play.csv")
    df_team_sum = load_team_summaries(f"{args.priors_dir}/Team Summaries.csv")
    df_team_abbrev = load_team_abbrev(f"{args.priors_dir}/Team Abbrev.csv")

    # Merge player priors
    df_merged = merge_player_priors(df_player, df_adv, df_per100, df_shoot, df_pbp)

    # Merge team priors
    df_merged = merge_team_priors(df_merged, df_team_sum, df_team_abbrev)

    # Fill missing values
    df_merged = fill_missing_values(df_merged)

    # Save
    print(f"\n{'='*70}")
    print("SAVING AGGREGATED DATASET")
    print(f"{'='*70}")

    output_path = args.output
    if args.compression != 'none':
        output_path = f"{args.output}.{args.compression}"

    print(f"\n  Output: {output_path}")
    print(f"  Compression: {args.compression}")
    print(f"  Writing...")

    compression_arg = None if args.compression == 'none' else args.compression
    df_merged.to_csv(output_path, index=False, compression=compression_arg)

    # Stats
    file_size_mb = Path(output_path).stat().st_size / 1024 / 1024

    print(f"\n{'='*70}")
    print("COMPLETE!")
    print(f"{'='*70}")
    print(f"\n  File: {output_path}")
    print(f"  Size: {file_size_mb:.1f} MB")
    print(f"  Rows: {len(df_merged):,}")
    print(f"  Columns: {len(df_merged.columns)}")
    print(f"  Date range: {df_merged['season'].min()} - {df_merged['season'].max()}")

    print(f"\n  Time saved per training run: ~10-20 minutes")
    print(f"  (No more expensive fuzzy matching during training!)")

    print(f"\n  Next: Use in training with:")
    print(f"    python train_auto.py --aggregated-data {output_path}")


if __name__ == "__main__":
    main()
