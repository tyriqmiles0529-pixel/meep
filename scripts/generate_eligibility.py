import pandas as pd
import numpy as np
import os

LOG_PATH = "data/nba_game_logs_1997_2024.csv"
OUTPUT_PATH = "data/eligibility_lookup.csv"

def parse_minutes(min_str):
    if pd.isna(min_str): return 0.0
    min_str = str(min_str)
    if ':' in min_str:
        parts = min_str.split(':')
        return float(parts[0]) + float(parts[1])/60.0
    try:
        return float(min_str)
    except:
        return 0.0

def generate():
    print(f"Loading {LOG_PATH}...")
    df = pd.read_csv(LOG_PATH)
    
    # 1. Standardize
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df['MIN'] = df['MIN'].apply(parse_minutes)
    
    # Sort by player and date
    df = df.sort_values(['PLAYER_NAME', 'GAME_DATE'])
    
    print("Computing rolling averages...")
    # Group by player and calc rolling 10
    # The requirement is "trailing 10 games OR season-to-date if fewer than 10"
    # min_periods=1 in rolling() achieves exactly this (moving average of available history up to 10)
    # CRITICAL: It must be EXCLUDING the current game (trailing). 
    # Use shift(1).
    
    df['avg_min'] = df.groupby('PLAYER_NAME')['MIN']\
                      .shift(1)\
                      .rolling(window=10, min_periods=1)\
                      .mean()
                      
    df['eligible'] = df['avg_min'] >= 20.0
    
    # Filter for dates from 2023-01-01 onwards to save space
    df_filtered = df[df['GAME_DATE'] >= '2023-01-01'].copy()
    
    # Format date as string for easy lookup
    df_filtered['game_date_str'] = df_filtered['GAME_DATE'].dt.strftime('%Y-%m-%d')
    
    # Keep essential columns
    # We use PLAYER_NAME for the API lookup
    lookup = df_filtered[['PLAYER_NAME', 'game_date_str', 'avg_min', 'eligible', 'PLAYER_ID']]
    
    print(f"Saving lookup table with {len(lookup)} rows...")
    lookup.to_csv(OUTPUT_PATH, index=False)
    
    eligible_count = lookup['eligible'].sum()
    print(f"Total entries: {len(lookup)}")
    print(f"Eligible entries (>= 20 MPG): {eligible_count} ({eligible_count/len(lookup):.1%})")

if __name__ == "__main__":
    generate()
