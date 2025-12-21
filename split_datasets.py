import pandas as pd
import os

INPUT_FILE = "data/strict_features_1997_2024.csv"
TRAIN_OUTPUT = "data/pro_training_set.csv"
LIVE_OUTPUT = "data/live_inference_set.csv"

def split_data():
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    
    # Identify Season Column
    # 'season_start_year' was added in extract_game_logs.py
    if 'season_start_year' not in df.columns:
        # Fallback: Parse SEASON_ID
        # SEASON_ID format: 22023 (Regular), 42023 (Playoffs) -> 2023
        df['season_start_year'] = df['SEASON_ID'].astype(str).str[-4:].astype(int)
        
    print(f"Seasons found: {sorted(df['season_start_year'].unique())}")
    
    # Split
    # Training: <= 2024 (Season starting in 2024, i.e., 2024-25)
    # User said: "Training dataset ending at the 2024 season." 
    # Usually "2024 season" implies 2024-25. 
    # And "2025-26" is the live season.
    # So we want <= 2024 in Train. 2025 in Live.
    
    train_mask = df['season_start_year'] <= 2024
    live_mask = df['season_start_year'] >= 2025
    
    train_df = df[train_mask].copy()
    live_df = df[live_mask].copy()
    
    print(f"Training Set: {len(train_df)} rows (Ends {train_df['season_start_year'].max()})")
    print(f"Live Set: {len(live_df)} rows (Starts {live_df['season_start_year'].min()})")
    
    # Verification
    # Ensure no overlap
    print(f"Max Train Season: {train_df['season_start_year'].max()}")
    print(f"Min Live Season: {live_df['season_start_year'].min()}")
    
    # Save
    train_df.to_csv(TRAIN_OUTPUT, index=False)
    live_df.to_csv(LIVE_OUTPUT, index=False)
    print("Datasets saved.")

if __name__ == "__main__":
    split_data()
