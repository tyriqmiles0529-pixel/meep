#!/usr/bin/env python3
"""
Quick test script to debug PlayerIdentityEmbeddings locally
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add local modules
sys.path.insert(0, ".")
from train_meta_learner_v4 import PlayerIdentityEmbeddings

def test_embeddings():
    print("="*60)
    print("TESTING PLAYER EMBEDDINGS LOCALLY")
    print("="*60)
    
    # Load a small sample of the Kaggle data
    csv_path = "PlayerStatistics.csv"
    if not Path(csv_path).exists():
        print(f"[!] {csv_path} not found - download from Kaggle first")
        return
    
    print(f"[*] Loading sample from {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    sample_df = df.head(1000)  # Small sample for quick testing
    print(f"    Sample size: {len(sample_df)} rows")
    
    # Process dates and create playerName
    if 'gameDate' in sample_df.columns:
        sample_df['gameDate'] = pd.to_datetime(sample_df['gameDate'], format='mixed', utc=True)
        sample_df['gameDate'] = sample_df['gameDate'].dt.tz_localize(None)
        sample_df['year'] = sample_df['gameDate'].dt.year
        sample_df['month'] = sample_df['gameDate'].dt.month
        sample_df['season_year'] = sample_df.apply(
            lambda row: row['year'] if row['month'] >= 10 else row['year'] - 1,
            axis=1
        )
    
    # Create playerName if needed
    if 'playerName' not in sample_df.columns:
        if 'firstName' in sample_df.columns and 'lastName' in sample_df.columns:
            sample_df['playerName'] = sample_df['firstName'] + ' ' + sample_df['lastName']
            print("    Created playerName column")
    
    print(f"\n[*] Available columns: {list(sample_df.columns)[:10]}...")
    
    # Test embeddings with debug config
    config = {
        'embedding_dim': 8,
        'min_games_for_embedding': 5,  # Lower threshold for testing
        'player_id_col': 'playerName'
    }
    
    print(f"\n[*] Creating PlayerIdentityEmbeddings with config:")
    for k, v in config.items():
        print(f"    {k}: {v}")
    
    embeddings = PlayerIdentityEmbeddings(config)
    
    print(f"\n[*] Testing fit() with sample data...")
    print(f"    player_id_col: {embeddings.player_id_col}")
    print(f"    min_games: {embeddings.min_games}")
    
    # Debug what the fit method sees
    print(f"\n[*] Data structure before fit():")
    print(f"    Shape: {sample_df.shape}")
    print(f"    Has '{embeddings.player_id_col}': {embeddings.player_id_col in sample_df.columns}")
    
    if embeddings.player_id_col in sample_df.columns:
        player_counts = sample_df[embeddings.player_id_col].value_counts()
        print(f"    Player counts (top 5): {player_counts.head()}")
        eligible_players = player_counts[player_counts >= embeddings.min_games]
        print(f"    Eligible players: {len(eligible_players)}")
    
    try:
        embeddings.fit(sample_df)
        print(f"\n[✓] Embeddings fit completed successfully!")
        print(f"    Fitted: {embeddings.fitted}")
        print(f"    Embeddings learned: {len(embeddings.player_embeddings)}")
        
    except Exception as e:
        print(f"\n[!] Embeddings fit failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_embeddings()
