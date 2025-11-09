"""
Create Pre-Aggregated Dataset - EFFICIENT MERGE & DTYPE VERSION

This script is optimized for speed and memory. It loads all data into
memory but uses efficient vectorized operations and memory-optimized
data types to stay within high-RAM environment limits.

FIXES:
- Replaced inefficient apply() with MultiIndex mapping for merging.
- Aggressively optimizes dtypes on load, converting strings to 'category'
  and downcasting numeric types to reduce memory footprint.
- Added garbage collection between merge steps.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
from tqdm import tqdm
from rapidfuzz import fuzz, process
import gc

def optimize_df(df):
    """Aggressively downcast numeric types and convert object columns to category."""
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif df[col].dtype == 'object':
            # Only convert to category if cardinality is reasonably low
            if df[col].nunique() / len(df[col]) < 0.5:
                df[col] = df[col].astype('category')
    return df

def load_player_statistics(csv_path):
    """Load main player game logs."""
    print(f"\n[1/7] Loading PlayerStatistics.csv...")
    print(f"  Path: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    
    # Perform fillna on firstName/lastName BEFORE optimizing dtypes
    if 'firstName' in df.columns:
        df['firstName'] = df['firstName'].fillna('')
    if 'lastName' in df.columns:
        df['lastName'] = df['lastName'].fillna('')

    df = optimize_df(df)
    print(f"  Rows: {len(df):,}, Columns: {len(df.columns)}")

    if 'gameDate' in df.columns:
        df['gameDate'] = pd.to_datetime(df['gameDate'], format='mixed', utc=True, errors='coerce')
        df['game_year'] = df['gameDate'].dt.year
        df['season'] = df['game_year'].copy()
        df.loc[df['gameDate'].dt.month >= 10, 'season'] += 1
    
    if 'personId' in df.columns:
        df['player_game_id'] = df['personId'].astype(str)

    if 'firstName' in df.columns and 'lastName' in df.columns:
        # fillna already done, just concatenate
        df['player_name'] = (df['firstName'].astype(str) + ' ' + df['lastName'].astype(str)).str.strip()

    return df

def load_prior_df(csv_path, prefix):
    """Loads a prior CSV, optimizes dtypes, and prefixes its columns."""
    print(f"\nLoading {Path(csv_path).name}...")
    df = pd.read_csv(csv_path)
    df = optimize_df(df)
    print(f"  Rows: {len(df):,}")
    
    rename_cols = {col: f'{prefix}_{col}' for col in df.columns if col not in ['season', 'player', 'player_id', 'team']}
    df = df.rename(columns=rename_cols)
    return df

def create_fuzzy_match_mapping(df_main, df_prior, season_col='season'):
    """Create player name mapping using fuzzy matching."""
    print(f"  Building fuzzy match mapping...")
    main_combos = df_main[['player_name', season_col]].drop_duplicates()
    prior_combos = df_prior[['player', season_col]].drop_duplicates()
    print(f"    Main players: {len(main_combos):,}, Prior players: {len(prior_combos):,}")
    
    mapping = {}
    seasons = sorted(df_main[season_col].unique())
    for season in tqdm(seasons, desc="    Fuzzy matching by season"):
        main_season_players = main_combos[main_combos[season_col] == season]['player_name'].tolist()
        prior_season_players = prior_combos[prior_combos[season_col] == season]['player'].tolist()

        if not prior_season_players: continue

        for main_player in main_season_players:
            if pd.isna(main_player) or main_player == '': continue
            
            result = process.extractOne(main_player, prior_season_players, scorer=fuzz.ratio)
            if result and result[1] > 85:
                mapping[(main_player, season)] = result[0]

    print(f"    Matched: {len(mapping):,} player-season combinations")
    return mapping

def merge_one_prior(df_main, df_prior, mapping, prior_name):
    """Memory-efficient merge using vectorized operations."""
    print(f"\n  Merging {prior_name}...")
    
    multi_index = pd.MultiIndex.from_frame(df_main[['player_name', 'season']])
    merge_key = multi_index.map(mapping)
    df_main['_temp_match'] = merge_key
    
    df_merged = pd.merge(df_main, df_prior, left_on=['_temp_match', 'season'], right_on=['player', 'season'], how='left', suffixes=('', f'_{prior_name}_dup'))
    df_merged = df_merged.drop(columns=['_temp_match'])
    
    del df_main, merge_key, multi_index
    gc.collect()
    
    cols_added = len([c for c in df_merged.columns if c.startswith(f'{prior_name}_')])
    print(f"    Added {cols_added} columns")
    return df_merged

def merge_player_priors(df_main, priors):
    """Merge all player season-level priors onto game-level data."""
    print(f"\n{'='*70}\nMERGING PLAYER PRIORS (WITH FUZZY MATCHING)\n{'='*70}")
    
    df_merged = df_main
    for prior_name, df_prior in priors.items():
        if prior_name.startswith('team_'): continue
        player_mapping = create_fuzzy_match_mapping(df_merged, df_prior)
        df_merged = merge_one_prior(df_merged, df_prior, player_mapping, prior_name)
        del player_mapping
        gc.collect()
        
    return df_merged

def merge_team_priors(df_main, df_team_sum):
    """Merge team season-level context."""
    print(f"\n{'='*70}\nMERGING TEAM PRIORS\n{'='*70}")
    df_merged = pd.merge(df_main, df_team_sum, on=['season', 'team'], how='left', suffixes=('', '_team_dup'))
    del df_main
    gc.collect()
    return df_merged

def fill_missing_values(df):
    """Intelligently fill missing values."""
    print(f"\n{'='*70}\nFILLING MISSING VALUES\n{'='*70}")
    prior_cols = [c for c in df.columns if c.startswith(('adv_', 'per100_', 'shoot_', 'pbp_', 'team_'))]
    print(f"\n  Filling {len(prior_cols)} prior columns with 0...")
    df[prior_cols] = df[prior_cols].fillna(0)
    return df

def main():
    parser = argparse.ArgumentParser(description='Create pre-aggregated NBA dataset (efficient merge version)')
    parser.add_argument('--player-csv', type=str, default='/kaggle/input/historical-nba-data-and-player-box-scores/PlayerStatistics.csv')
    parser.add_argument('--priors-dir', type=str, default='/kaggle/input/nba-aba-baa-stats')
    parser.add_argument('--output', type=str, default='aggregated_nba_data.csv')
    parser.add_argument('--compression', type=str, default='gzip', choices=['gzip', 'bz2', 'zip', 'none'])
    args, _ = parser.parse_known_args()

    print("="*70 + "\nPRE-AGGREGATING NBA DATASET (EFFICIENT MERGE & DTYPE VERSION)\n" + "="*70)

    df_player = load_player_statistics(args.player_csv)
    
    priors = {
        'adv': load_prior_df(f"{args.priors_dir}/Advanced.csv", 'adv'),
        'per100': load_prior_df(f"{args.priors_dir}/Per 100 Poss.csv", 'per100'),
        'shoot': load_prior_df(f"{args.priors_dir}/Player Shooting.csv", 'shoot'),
        'pbp': load_prior_df(f"{args.priors_dir}/Player Play By Play.csv", 'pbp'),
        'team_sum': load_prior_df(f"{args.priors_dir}/Team Summaries.csv", 'team_sum')
    }

    df_merged = merge_player_priors(df_player, priors)
    df_merged = merge_team_priors(df_merged, priors['team_sum'])
    df_merged = fill_missing_values(df_merged)

    print(f"\n{'='*70}\nSAVING AGGREGATED DATASET\n{'='*70}")
    output_path = f"{args.output}.{args.compression}" if args.compression != 'none' else args.output
    print(f"\n  Output: {output_path}")
    df_merged.to_csv(output_path, index=False, compression=args.compression if args.compression != 'none' else None)

    print(f"\n{'='*70}\nCOMPLETE!\n{'='*70}")

if __name__ == "__main__":
    main()
