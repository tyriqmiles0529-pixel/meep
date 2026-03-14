"""
Create Pre-Aggregated Dataset - ROBUST CHUNKING VERSION

This script is the most robust version, designed to run in any environment
by processing the main data file in chunks and writing directly to disk.
This guarantees completion without memory crashes, even on large datasets.

It is configured to read from the two separate Kaggle datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
from rapidfuzz import fuzz, process
import gc

# --- Configuration ---
CHUNK_SIZE = 750000  # Process 750,000 rows at a time

def get_memory_usage():
    """Returns current memory usage of the script in MB."""
    import os, psutil
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except (ImportError, AttributeError):
        return -1 # psutil not available

def optimize_df(df):
    """Aggressively downcast numeric types and convert object columns to category."""
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif df[col].dtype == 'object':
            if df[col].nunique() / len(df[col]) < 0.5:
                df[col] = df[col].astype('category')
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

def create_fuzzy_match_mapping(main_player_seasons, df_prior, season_col='season'):
    """Creates a memory-efficient fuzzy match mapping."""
    print(f"  Building fuzzy match mapping for {len(df_prior):,} prior rows...")

    prior_combos = df_prior[['player', season_col]].drop_duplicates()
    prior_player_seasons = {
        season: prior_combos[prior_combos[season_col] == season]['player'].tolist()
        for season in prior_combos[season_col].unique()
    }

    mapping = {}
    main_player_seasons_list = sorted(list(main_player_seasons))

    for main_player, season in tqdm(main_player_seasons_list, desc="    Fuzzy matching players"):
        if pd.isna(main_player) or main_player == '' or season not in prior_player_seasons:
            continue

        prior_season_players = prior_player_seasons[season]
        if not prior_season_players:
            continue

        result = process.extractOne(main_player, prior_season_players, scorer=fuzz.ratio)
        if result and result[1] > 85:
            mapping[(main_player, season)] = result[0]

    print(f"    Matched: {len(mapping):,} player-season combinations")
    return mapping

def process_chunk(chunk, mappings, priors):
    """Processes a single chunk of the main player data."""
    # 1. Basic preprocessing for the chunk
    if 'gameDate' in chunk.columns:
        chunk['gameDate'] = pd.to_datetime(chunk['gameDate'], format='mixed', utc=True, errors='coerce')
        chunk['game_year'] = chunk['gameDate'].dt.year
        chunk['season'] = chunk['game_year'].copy()
        chunk.loc[chunk['gameDate'].dt.month >= 10, 'season'] += 1
    
    if 'firstName' in chunk.columns:
        chunk['firstName'] = chunk['firstName'].fillna('')
    if 'lastName' in chunk.columns:
        chunk['lastName'] = chunk['lastName'].fillna('')
    
    if 'personId' in chunk.columns:
        chunk['player_game_id'] = chunk['personId'].astype(str)
    if 'firstName' in chunk.columns and 'lastName' in chunk.columns:
        chunk['player_name'] = (chunk['firstName'].astype(str) + ' ' + chunk['lastName'].astype(str)).str.strip()

    chunk = optimize_df(chunk)
    
    # 2. Merge player priors
    for prior_name, mapping in mappings.items():
        df_prior = priors[prior_name]
        
        multi_index = pd.MultiIndex.from_frame(chunk[['player_name', 'season']])
        merge_key_series = multi_index.map(mapping)
        chunk[f'_temp_match'] = merge_key_series.values

        chunk = pd.merge(chunk, df_prior, left_on=['_temp_match', 'season'], right_on=['player', 'season'], how='left', suffixes=('', f'_{prior_name}_dup'))
        chunk = chunk.drop(columns=['_temp_match'])

    # 3. Merge team priors
    if 'team_sum' in priors:
        chunk = pd.merge(chunk, priors['team_sum'], on=['season', 'team'], how='left', suffixes=('', '_team_dup'))

    # 4. Fill missing values
    prior_cols = [c for c in chunk.columns if c.startswith(('adv_', 'per100_', 'shoot_', 'pbp_', 'team_'))]
    numeric_prior_cols = chunk[prior_cols].select_dtypes(include=np.number).columns.tolist()
    chunk[numeric_prior_cols] = chunk[numeric_prior_cols].fillna(0)

    return chunk

def main():
    parser = argparse.ArgumentParser(description='Create pre-aggregated NBA dataset (robust chunking version)')
    parser.add_argument('--player-csv', type=str, default='/kaggle/input/historical-nba-data-and-player-box-scores/PlayerStatistics.csv')
    parser.add_argument('--priors-dir', type=str, default='/kaggle/input/nba-aba-baa-stats')
    parser.add_argument('--output', type=str, default='aggregated_nba_data.csv')
    parser.add_argument('--compression', type=str, default='gzip', choices=['gzip', 'bz2', 'zip', 'none'])
    args, _ = parser.parse_known_args()

    print(f"--- Starting Aggregation (Robust Chunking Version) ---")
    mem_usage = get_memory_usage()
    if mem_usage != -1: print(f"Initial memory usage: {mem_usage:.2f} MB")

    # --- 1. Load all smaller "prior" files into memory ---
    print("\n[Step 1/5] Loading all prior datasets into memory...")
    priors = {
        'adv': load_prior_df(f"{args.priors_dir}/Advanced.csv", 'adv'),
        'per100': load_prior_df(f"{args.priors_dir}/Per 100 Poss.csv", 'per100'),
        'shoot': load_prior_df(f"{args.priors_dir}/Player Shooting.csv", 'shoot'),
        'pbp': load_prior_df(f"{args.priors_dir}/Player Play By Play.csv", 'pbp'),
        'team_sum': load_prior_df(f"{args.priors_dir}/Team Summaries.csv", 'team_sum')
    }
    if mem_usage != -1: print(f"Memory after loading priors: {get_memory_usage():.2f} MB")

    # --- 2. Get all unique player-season combos from the large file ---
    print("\n[Step 2/5] Scanning main CSV for all unique player-season combinations...")
    reader = pd.read_csv(args.player_csv, usecols=['firstName', 'lastName', 'gameDate'], chunksize=CHUNK_SIZE)
    main_player_seasons = set()
    for chunk in tqdm(reader, desc="Scanning for players"):
        chunk['gameDate'] = pd.to_datetime(chunk['gameDate'], format='mixed', utc=True, errors='coerce')
        chunk['game_year'] = chunk['gameDate'].dt.year
        chunk['season'] = chunk['game_year']
        chunk.loc[chunk['gameDate'].dt.month >= 10, 'season'] += 1
        chunk['player_name'] = (chunk['firstName'].fillna('') + ' ' + chunk['lastName'].fillna('')).str.strip()
        
        for record in chunk[['player_name', 'season']].drop_duplicates().to_records(index=False):
            main_player_seasons.add(tuple(record))
    del reader, chunk
    gc.collect()
    print(f"Found {len(main_player_seasons):,} unique player-season combinations.")
    if mem_usage != -1: print(f"Memory after scanning: {get_memory_usage():.2f} MB")

    # --- 3. Create all fuzzy match mappings once ---
    print("\n[Step 3/5] Pre-building all fuzzy match mappings...")
    mappings = {
        'adv': create_fuzzy_match_mapping(main_player_seasons, priors['adv']),
        'per100': create_fuzzy_match_mapping(main_player_seasons, priors['per100']),
        'shoot': create_fuzzy_match_mapping(main_player_seasons, priors['shoot']),
        'pbp': create_fuzzy_match_mapping(main_player_seasons, priors['pbp']),
    }
    if mem_usage != -1: print(f"Memory after creating mappings: {get_memory_usage():.2f} MB")

    # --- 4. Process and Write Chunks Directly to File ---
    print(f"\n[Step 4/5] Processing {args.player_csv} and writing to CSV in chunks...")
    
    output_path = f"{args.output}.{args.compression}" if args.compression != 'none' else args.output
    compression_arg = args.compression if args.compression != 'none' else None

    if Path(output_path).exists():
        Path(output_path).unlink()
        print(f"Removed existing file: {output_path}")

    reader = pd.read_csv(args.player_csv, chunksize=CHUNK_SIZE, low_memory=False)
    
    is_first_chunk = True
    for i, chunk in enumerate(reader, start=1):
        print(f"\n--- Processing chunk {i} ---")
        processed_chunk = process_chunk(chunk, mappings, priors)
        
        processed_chunk.to_csv(output_path, index=False, compression=compression_arg, mode='a', header=is_first_chunk)
        
        print(f"Wrote chunk {i} to {output_path}")
        is_first_chunk = False
        
        if mem_usage != -1: print(f"Memory after chunk {i}: {get_memory_usage():.2f} MB")
        
        del chunk, processed_chunk
        gc.collect()

    # --- 5. Finalize ---
    print("\n[Step 5/5] Finalizing...")
    file_size_mb = Path(output_path).stat().st_size / 1024 / 1024
    
    print("\n--- COMPLETE! ---")
    print(f"  File: {output_path} ({file_size_mb:.1f} MB)")
    print(f"\nNext: Use in training with `python train_auto.py --aggregated-data {output_path}`")

if __name__ == "__main__":
    main()