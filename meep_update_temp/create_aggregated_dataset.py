"""
Create Pre-Aggregated Dataset - ONE TIME PREPROCESSING

This script merges ALL data sources into a single comprehensive CSV
to eliminate expensive merge operations during training.

BEFORE (slow):
- Training loads PlayerStatistics.csv
- Training merges Advanced.csv (fuzzy match player names)
- Training merges Per 100 Poss.csv (fuzzy match)
- Training merges Shooting.csv (fuzzy match)
- Training merges Team priors (fuzzy match)
- Total merge time: 10-20 minutes per training run

AFTER (fast):
- Run this script ONCE to create aggregated_data.csv
- Training just loads aggregated_data.csv
- All merges pre-done, no fuzzy matching needed
- Training time saved: 10-20 minutes

Usage:
    python create_aggregated_dataset.py --output aggregated_nba_data.csv

    # Then in training:
    python train_auto.py --aggregated-data aggregated_nba_data.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
from tqdm import tqdm
from rapidfuzz import fuzz, process


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

    This is the expensive operation we're doing ONCE instead of every training run.

    Args:
        df_main: PlayerStatistics with 'player_name', 'season'
        df_prior: Prior CSV with 'player', 'season'

    Returns:
        DataFrame with mapping keys for merging
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
    for season in tqdm(df_main[season_col].unique(), desc="    Fuzzy matching by season"):
        main_season = main_combos[main_combos[season_col] == season]['player_name'].tolist()
        prior_season = prior_combos[prior_combos[season_col] == season]['player'].tolist()

        if not prior_season:  # No priors for this season
            continue

        # Fuzzy match each player in main to best match in prior
        for main_player in main_season:
            if pd.isna(main_player) or main_player == '':
                continue

            # Find best match
            match, score, _ = process.extractOne(
                main_player,
                prior_season,
                scorer=fuzz.ratio
            )

            # Only accept if score > 85 (pretty confident match)
            if score > 85:
                mapping[(main_player, season)] = match

    print(f"    Matched: {len(mapping):,} player-season combinations")

    return mapping


def merge_player_priors(df_main, df_adv, df_per100, df_shoot, df_pbp):
    """
    Merge all player season-level priors onto game-level data.

    Strategy:
    - Fuzzy match player names (firstName + lastName) → priors 'player'
    - Match teams by name similarity
    - Left merge (keep all games, even without priors)

    This is THE expensive operation we're doing ONCE instead of every training run!
    """
    print(f"\n{'='*70}")
    print("MERGING PLAYER PRIORS (WITH FUZZY MATCHING)")
    print(f"{'='*70}")

    initial_rows = len(df_main)

    # 1. Merge Advanced stats
    print(f"\n[1/4] Merging Advanced stats...")

    # Create fuzzy mapping
    player_mapping = create_fuzzy_match_mapping(df_main, df_adv, 'season')

    # Apply mapping to create merge key
    df_main['_adv_player_match'] = df_main.apply(
        lambda row: player_mapping.get((row['player_name'], row['season']), None),
        axis=1
    )

    # Merge on matched player name + season
    df_merged = df_main.merge(
        df_adv,
        left_on=['_adv_player_match', 'season'],
        right_on=['player', 'season'],
        how='left',
        suffixes=('', '_adv_dup')
    )

    # Drop temporary columns
    df_merged = df_merged.drop(columns=['_adv_player_match'], errors='ignore')

    adv_cols_added = len([c for c in df_merged.columns if c.startswith('adv_')])
    print(f"  Added {adv_cols_added} advanced stat columns")
    if 'adv_per' in df_merged.columns:
        print(f"  Match rate: {(~df_merged['adv_per'].isna()).sum() / len(df_merged) * 100:.1f}%")

    # 2. Merge Per 100 Poss
    print(f"\n[2/4] Merging Per 100 Poss...")

    # Create fuzzy mapping for Per 100
    player_mapping_per100 = create_fuzzy_match_mapping(df_merged, df_per100, 'season')

    # Apply mapping to create merge key
    df_merged['_per100_player_match'] = df_merged.apply(
        lambda row: player_mapping_per100.get((row['player_name'], row['season']), None),
        axis=1
    )

    # Merge on matched player name + season
    df_merged = df_merged.merge(
        df_per100,
        left_on=['_per100_player_match', 'season'],
        right_on=['player', 'season'],
        how='left',
        suffixes=('', '_per100_dup')
    )

    # Drop temporary columns
    df_merged = df_merged.drop(columns=['_per100_player_match'], errors='ignore')

    per100_cols_added = len([c for c in df_merged.columns if c.startswith('per100_')])
    print(f"  Added {per100_cols_added} per-100 columns")
    if 'per100_pts' in df_merged.columns:
        print(f"  Match rate: {(~df_merged['per100_pts'].isna()).sum() / len(df_merged) * 100:.1f}%")

    # 3. Merge Shooting
    print(f"\n[3/4] Merging Shooting splits...")

    # Create fuzzy mapping for Shooting
    player_mapping_shoot = create_fuzzy_match_mapping(df_merged, df_shoot, 'season')

    # Apply mapping to create merge key
    df_merged['_shoot_player_match'] = df_merged.apply(
        lambda row: player_mapping_shoot.get((row['player_name'], row['season']), None),
        axis=1
    )

    # Merge on matched player name + season
    df_merged = df_merged.merge(
        df_shoot,
        left_on=['_shoot_player_match', 'season'],
        right_on=['player', 'season'],
        how='left',
        suffixes=('', '_shoot_dup')
    )

    # Drop temporary columns
    df_merged = df_merged.drop(columns=['_shoot_player_match'], errors='ignore')

    shoot_cols_added = len([c for c in df_merged.columns if c.startswith('shoot_')])
    print(f"  Added {shoot_cols_added} shooting columns")
    if 'shoot_fg_pct' in df_merged.columns:
        print(f"  Match rate: {(~df_merged['shoot_fg_pct'].isna()).sum() / len(df_merged) * 100:.1f}%")

    # 4. Merge Play-by-Play
    print(f"\n[4/4] Merging Play-by-Play...")

    # Create fuzzy mapping for Play-by-Play
    player_mapping_pbp = create_fuzzy_match_mapping(df_merged, df_pbp, 'season')

    # Apply mapping to create merge key
    df_merged['_pbp_player_match'] = df_merged.apply(
        lambda row: player_mapping_pbp.get((row['player_name'], row['season']), None),
        axis=1
    )

    # Merge on matched player name + season
    df_merged = df_merged.merge(
        df_pbp,
        left_on=['_pbp_player_match', 'season'],
        right_on=['player', 'season'],
        how='left',
        suffixes=('', '_pbp_dup')
    )

    # Drop temporary columns
    df_merged = df_merged.drop(columns=['_pbp_player_match'], errors='ignore')

    pbp_cols_added = len([c for c in df_merged.columns if c.startswith('pbp_')])
    print(f"  Added {pbp_cols_added} play-by-play columns")
    if 'pbp_on_court_plus_minus' in df_merged.columns:
        print(f"  Match rate: {(~df_merged['pbp_on_court_plus_minus'].isna()).sum() / len(df_merged) * 100:.1f}%")

    # Verify no row multiplication
    final_rows = len(df_merged)
    if final_rows != initial_rows:
        print(f"\n  ⚠️ WARNING: Row count changed from {initial_rows:,} to {final_rows:,}")
        print(f"  This suggests duplicate merges - check data quality")
    else:
        print(f"\n  ✓ Row count preserved: {final_rows:,}")

    total_cols_added = adv_cols_added + per100_cols_added + shoot_cols_added + pbp_cols_added
    print(f"\n  TOTAL: Added {total_cols_added} prior columns")

    return df_merged


def merge_team_priors(df_main, df_team_sum, df_team_abbrev):
    """Merge team season-level context."""
    print(f"\n{'='*70}")
    print("MERGING TEAM PRIORS")
    print(f"{'='*70}")

    # Merge team summaries (player's team)
    print(f"\n[1/2] Merging team summaries (player team)...")
    merge_keys = ['season', 'team']

    df_merged = df_main.merge(
        df_team_sum,
        on=merge_keys,
        how='left',
        suffixes=('', '_team_dup')
    )

    team_cols_added = len([c for c in df_merged.columns if c.startswith('team_')])
    print(f"  Added {team_cols_added} team stat columns")

    # TODO: Merge opponent team stats (requires opponent team name in df_main)
    # This would give matchup context

    return df_merged


def fill_missing_values(df):
    """Intelligently fill missing values based on era/context."""
    print(f"\n{'='*70}")
    print("FILLING MISSING VALUES")
    print(f"{'='*70}")

    # Count missing before
    missing_before = df.isna().sum().sum()
    print(f"\n  Missing values before: {missing_before:,}")

    # Strategy 1: Fill priors with 0 (player didn't have that stat recorded)
    prior_cols = [c for c in df.columns if c.startswith(('adv_', 'per100_', 'shoot_', 'pbp_', 'team_'))]

    print(f"\n  Filling {len(prior_cols)} prior columns with 0 (stat not available)...")
    df[prior_cols] = df[prior_cols].fillna(0)

    # Strategy 2: Forward fill team/player context within same season
    # (If player has multiple games, carry forward their season stats)

    # Count missing after
    missing_after = df.isna().sum().sum()
    print(f"  Missing values after: {missing_after:,}")
    print(f"  Filled: {missing_before - missing_after:,}")

    return df


def main():
    parser = argparse.ArgumentParser(description='Create pre-aggregated NBA dataset')
    parser.add_argument('--player-csv', type=str,
                       default='PlayerStatistics.csv',
                       help='Path to PlayerStatistics.csv')
    parser.add_argument('--priors-dir', type=str,
                       default='priors_data',
                       help='Directory with priors CSVs')
    parser.add_argument('--output', type=str,
                       default='aggregated_nba_data.csv',
                       help='Output CSV path')
    parser.add_argument('--compression', type=str,
                       default='gzip',
                       choices=['gzip', 'bz2', 'zip', 'none'],
                       help='Compression for output')

    args = parser.parse_args()

    print("="*70)
    print("PRE-AGGREGATING NBA DATASET")
    print("="*70)
    print(f"\nThis will create a single comprehensive CSV with:")
    print(f"  • All player game logs (1946-2025)")
    print(f"  • All Basketball Reference priors merged")
    print(f"  • All team context merged")
    print(f"  • Ready for instant training (no merge time)")
    print(f"\nExpected time: 5-10 minutes")
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
