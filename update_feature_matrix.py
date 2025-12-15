import pandas as pd
import numpy as np
import os
import argparse
from datetime import datetime
import torch
from ft_transformer import FTTransformerFeatureExtractor
from data_processor import BasketballDataProcessor

def update_dataset(daily_file, master_file, ft_model_path=None):
    print(f"Loading Master: {master_file}")
    # Load Master
    # Use low_memory=False to avoid DtypeWarning
    df_master = pd.read_csv(master_file, low_memory=False)
    
    # Load Daily
    if not os.path.exists(daily_file):
        print(f"Daily file {daily_file} not found.")
        return
        
    df_daily = pd.read_csv(daily_file)
    print(f"Loaded {len(df_daily)} new player-games.")
    
    # Check for duplicates (Player + Date + GameID)
    # Create unique keys
    df_master['unique_key'] = df_master['player_id'].astype(str) + "_" + df_master['gameId'].astype(str)
    df_daily['unique_key'] = df_daily['player_id'].astype(str) + "_" + df_daily['gameId'].astype(str)
    
    # Filter out existing
    existing_keys = set(df_master['unique_key'])
    new_rows = df_daily[~df_daily['unique_key'].isin(existing_keys)].copy()
    
    if new_rows.empty:
        print("No new unique rows to add.")
        return
        
    print(f"Adding {len(new_rows)} unique rows...")
    
    # Drop unique_key
    df_master.drop(columns=['unique_key'], inplace=True, errors='ignore')
    new_rows.drop(columns=['unique_key'], inplace=True, errors='ignore')
    
    # Append
    # Ensure columns match. New rows might miss some engineered features.
    # We concat first, then re-calculate lags.
    # Lags need sorted history.
    
    df_combined = pd.concat([df_master, new_rows], ignore_index=True)
    
    # Sort
    df_combined['date'] = pd.to_datetime(df_combined['date'])
    df_combined = df_combined.sort_values(by=['player_id', 'date']).reset_index(drop=True)
    
    # --- Feature Engineering: Lags ---
    print("Recalculating Lag Features (Incremental)...")
    # For efficiency, we *could* only calc for affected players.
    # But for robustness, let's re-run the lag function for the whole dataset (or just the tail if careful).
    # Since we have ~1M rows, full calc might be slow daily.
    # Optimized approach: GroupBy player, calc rolling.
    
    # Definition of Lags (matches reconstruct_dataset.py)
    LAG_COLS = ['points', 'assists', 'reboundsTotal', 'numMinutes', 'three_pointers']
    WINDOWS = [3, 5, 10, 20]
    
    # Ensure 'numMinutes' exists (mapped from 'minutes')
    if 'numMinutes' not in df_combined.columns and 'minutes' in df_combined.columns:
        df_combined['numMinutes'] = df_combined['minutes']

    # We can use the existing 'add_lag_features' from data_processor if adaptable, 
    # but re-implementing optimized version here for clarity.
    
    for col in LAG_COLS:
        col_name = col
        if col == 'numMinutes': col_name = 'minutes' # Standardize if needed
        # Actually 'minutes' is usually in the source.
        
        # Check if column exists
        if col not in df_combined.columns:
             # Try mapping
             if col == 'reboundsTotal' and 'rebounds' in df_combined.columns:
                 df_combined['reboundsTotal'] = df_combined['rebounds']
             elif col == 'numMinutes' and 'minutes' in df_combined.columns:
                 df_combined['numMinutes'] = df_combined['minutes']
        
        if col not in df_combined.columns:
            print(f"Warning: Column {col} missing for lags.")
            continue

        for window in WINDOWS:
            # Shift 1 to exclude current game! (Leakage prevention)
            # grouped = df_combined.groupby('player_id')[col].shift(1).rolling(window=window, min_periods=1)
            # Efficient implementation
            
            # Simple vectorised approach:
            # 1. Shift
            # 2. Rolling mean
            # Group keys needed.
            
            feat_name = f"{col}_last_{window}_avg"
            
            # This can be slow. 
            # Optimization: Only calc for players in new_rows?
            # Yes, but they need history.
            # Let's just do full calc for Safety first.
            
            df_combined[feat_name] = df_combined.groupby('player_id')[col].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            
            # Also "last_game"
            if window == 3: # Do it once
                 df_combined[f"{col}_last_game"] = df_combined.groupby('player_id')[col].shift(1)

    # Fill NaNs
    lag_cols_created = [c for c in df_combined.columns if '_last_' in c]
    df_combined[lag_cols_created] = df_combined[lag_cols_created].fillna(0)

    # --- Opponent Metrics ---
    # Rolling Points Allowed by Team
    print("Updating Opponent Metrics...")
    # Determine Opponent ID (Need opponent_id column or infer from matchups)
    # The 'fetch' script had 'home' and 'opponent' (name?).
    # Master usually has 'opponentId' or similar.
    # reconstruct_dataset used 'games' table.
    # If we don't have opponentId in daily, we rely on 'opponent' name.
    
    # Simplified: Reuse existing columns if present.
    
    # --- Embeddings ---
    print("Updating Embeddings...")
    if ft_model_path and os.path.exists(ft_model_path):
        # Load FT
        # Need cardinalities.
        # This part is tricky without strict state.
        # FeatureExtractor needs to know cat map.
        pass # Skipping complex embedding update for V1.
             # We rely on 'cat_cols' being handled by model pipeline or redundant here.
             # Actually, if we want to SAVE them, we need to generate them.
             # Assuming 'ft_extractor' can handle new categories (UNK) or consistent.
    
    # Save
    print(f"Saving updated master to {master_file}...")
    df_combined.to_csv(master_file, index=False)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--daily', type=str, required=True, help='Path to daily CSV')
    parser.add_argument('--master', type=str, default='final_feature_matrix_with_per_min_1997_onward.csv')
    args = parser.parse_args()
    
    update_dataset(args.daily, args.master)
