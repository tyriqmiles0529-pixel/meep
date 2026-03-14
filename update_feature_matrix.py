import pandas as pd
import numpy as np
import os
import argparse
from datetime import datetime
from build_features import StrictFeatureEngine

def update_dataset(daily_file, master_file):
    print(f"\n>>> [UPGRADE] Updating Master Matrix via StrictFeatureEngine...")
    
    # 1. Load Master (Optimized)
    if not os.path.exists(master_file):
        print(f"Master file {master_file} not found.")
        return
        
    # Load Master
    # We only need the last few months of data to compute features for today's new logs, 
    # but we eventually want to save a complete historical file.
    # For now, let's load all data from 2024 onwards to be safe and avoid memory issues.
    print(f"   Loading master database...")
    peek = pd.read_csv(master_file, nrows=5)
    season_col = 'season_start_year' if 'season_start_year' in peek.columns else 'season' if 'season' in peek.columns else None
    
    chunks = []
    for chunk in pd.read_csv(master_file, chunksize=100000, low_memory=False):
        if season_col and season_col in chunk.columns:
            # Keep 2023+ to ensure enough rolling lookback for 2025-26 season
            filtered = chunk[chunk[season_col] >= 2023]
            chunks.append(filtered)
        else:
            chunks.append(chunk)
    df_master = pd.concat(chunks, ignore_index=True)
    
    # Standardize Master to Case-Sensitive Keys expected by StrictFeatureEngine
    mapping = {
        'player_id': 'PLAYER_ID',
        'gameId': 'GAME_ID',
        'points': 'PTS',
        'assists': 'AST',
        'reboundsTotal': 'REB',
        'minutes': 'MIN',
        'date': 'GAME_DATE',
        'gameDate': 'GAME_DATE',
        'matchup': 'MATCHUP',
        'WL': 'WL',
        'TEAM_ID': 'TEAM_ID',
        'player_name': 'PLAYER_NAME',
        'three_pointers': 'FG3M',
        'FGA': 'FGA',
        'FG3A': 'FG3A',
        'TOV': 'TOV',
        'FTA': 'FTA'
    }
    # Reverse mapping for later
    rev_mapping = {v: k for k, v in mapping.items()}
    
    # Clean Master Headers
    df_master = df_master.rename(columns={k: v for k, v in mapping.items() if k in df_master.columns})
    
    # 2. Load Daily
    if not os.path.exists(daily_file):
        print(f"Daily file {daily_file} not found.")
        return
    df_daily = pd.read_csv(daily_file)
    print(f"   Loaded {len(df_daily)} new events.")
    
    # Clean Daily Headers
    df_daily = df_daily.rename(columns={k: v for k, v in mapping.items() if k in df_daily.columns})
    # Handle specific common names in daily logs
    if 'PLAYER_ID' not in df_daily.columns and 'player_id' in df_daily.columns: df_daily.rename(columns={'player_id': 'PLAYER_ID'}, inplace=True)
    if 'GAME_ID' not in df_daily.columns and 'gameId' in df_daily.columns: df_daily.rename(columns={'gameId': 'GAME_ID'}, inplace=True)
    
    # 3. Handle Duplicates
    df_master['unique_key'] = df_master['PLAYER_ID'].astype(str) + "_" + df_master['GAME_ID'].astype(str)
    df_daily['unique_key'] = df_daily['PLAYER_ID'].astype(str) + "_" + df_daily['GAME_ID'].astype(str)
    
    existing_keys = set(df_master['unique_key'])
    new_rows = df_daily[~df_daily['unique_key'].isin(existing_keys)].copy()
    
    if new_rows.empty:
        print("   >>> [OK] No new unique rows to add.")
        return
        
    print(f"   Adding {len(new_rows)} new unique games to master...")
    
    # 4. Integrate & Run Engine
    df_combined = pd.concat([df_master, new_rows], ignore_index=True)
    df_combined.drop(columns=['unique_key'], inplace=True, errors='ignore')
    
    # RUN THE ROOT ENGINE (StrictFeatureEngine)
    engine = StrictFeatureEngine(df_combined)
    engine.load_and_clean()
    engine.compute_rolling_stats()
    engine.compute_rest_days()
    engine.add_lag_features()
    engine.compute_opponent_strength()
    engine.compute_advanced_rolling_stats()
    engine.compute_advanced_player_metrics()
    engine.compute_contextual_features()
    engine.compute_availability_features()
    engine.compute_per_minute_features()
    
    # 5. Handle Embeddings Sync
    # If the combined df already had embeddings from history, ffill them for the new rows
    emb_cols = [c for c in engine.df.columns if c.startswith('emb_')]
    if emb_cols:
        print(f"   Syncing {len(emb_cols)} Latent DNA Embeddings...")
        engine.df = engine.df.sort_values(['PLAYER_ID', 'GAME_DATE'])
        engine.df[emb_cols] = engine.df.groupby('PLAYER_ID')[emb_cols].ffill()
        engine.df[emb_cols] = engine.df.groupby('PLAYER_ID')[emb_cols].bfill() # Fallback
    
    # 5.5 Final Scrub (Zero NaNs)
    engine.finalize_features()
    
    # 6. Final Save
    # Convert back to lowercase-leaning names if preferred, but build_features save_features 
    # uses a specific set. Let's just use its output logic but ensure filenames match.
    print(f"   Saving enriched dataset to {master_file}...")
    
    # Standardize back some columns to keep terminal happy
    engine.df = engine.df.rename(columns={'FG3M': 'three_pointers'})
    
    # We want to overwrite the master file with the complete processed data
    engine.df.to_csv(master_file, index=False)
    print(f">>> [SUCCESS] Master matrix updated with {len(new_rows)} new high-integrity rows.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--daily', type=str, required=True, help='Path to daily CSV')
    parser.add_argument('--master', type=str, default='final_feature_matrix_with_per_min_1997_onward.csv')
    args = parser.parse_args()
    
    update_dataset(args.daily, args.master)
