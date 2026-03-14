import pandas as pd
import numpy as np

PROJS_FILE = "data/validation_projections_v4.csv"
ODDS_FILE = "historical_data/the_odds_api_historical.csv"

def analyze():
    print("=== Phase J.4: Validation Analysis (Corrected Outcomes) ===")
    
    # 1. Load Data
    if not os.path.exists(PROJS_FILE):
        print(f"File missing: {PROJS_FILE}")
        return
        
    projs = pd.read_csv(PROJS_FILE)
    odds = pd.read_csv(ODDS_FILE)
    
    # Normalize join keys
    projs['player_key'] = projs['PLAYER_NAME'].str.upper().str.strip()
    odds['player_key'] = odds['player_name'].str.upper().str.strip()
    
    # 2. Join
    df = pd.merge(odds, projs, left_on=['player_key', 'game_date'], right_on=['player_key', 'GAME_DATE'])
    print(f"Matched {len(df)} lines with projections and outcomes.")
    
    # 3. Map Markets to Projection and Outcome
    # market in odds: 'player_points', 'player_assists', 'player_rebounds'
    # proj columns in projs: 'proj_PTS', 'proj_AST', 'proj_REB'
    # outcome columns in projs: 'PTS', 'AST', 'REB'
    
    mapping = {
        'player_points': {'proj': 'proj_PTS', 'actual': 'PTS'},
        'player_assists': {'proj': 'proj_AST', 'actual': 'AST'},
        'player_rebounds': {'proj': 'proj_REB', 'actual': 'REB'}
    }
    
    def extract_metrics(row):
        m = row['market']
        if m in mapping:
            return pd.Series([row[mapping[m]['proj']], row[mapping[m]['actual']]])
        return pd.Series([np.nan, np.nan])
        
    df[['prediction', 'actual_outcome']] = df.apply(extract_metrics, axis=1)
    df = df.dropna(subset=['prediction', 'actual_outcome'])
    
    # 4. Success Criteria
    df['delta'] = df['prediction'] - df['line']
    df['abs_delta'] = df['delta'].abs()
    
    def check_hit(row):
        if row['delta'] >= 0: # Over
            return 1 if row['actual_outcome'] > row['line'] else 0
        else: # Under
            return 1 if row['actual_outcome'] < row['line'] else 0
            
    df['hit'] = df.apply(check_hit, axis=1)
    
    # 5. Bucket Analysis
    bins = [0, 0.5, 1.0, 1.5, 2.0, 5.0, 100]
    labels = ['0.0-0.5', '0.5-1.0', '1.0-1.5', '1.5-2.0', '2.0-5.0', '5.0+']
    df['bucket'] = pd.cut(df['abs_delta'], bins=bins, labels=labels)
    
    bucket_stats = df.groupby('bucket').agg(
        count=('hit', 'count'),
        hit_rate=('hit', 'mean')
    ).reset_index()
    
    print("\n--- Delta Bucket Analysis ---")
    print(bucket_stats.to_string(index=False))
    
    # 6. Book Shading (Is DK more conservative than FD?)
    # Calculate Mean Absolute Error (MAE) by book
    df['error'] = (df['actual_outcome'] - df['line']).abs()
    book_perf = df.groupby('book').agg(
        n=('hit', 'count'),
        avg_line=('line', 'mean'),
        hit_rate=('hit', 'mean'),
        mae=('error', 'mean')
    ).sort_values('hit_rate', ascending=False)
    
    print("\n--- Book Performance Summary ---")
    print(book_perf.to_string())
    
    # Special Filter: Delta >= 1.5
    high_edge = df[df['abs_delta'] >= 1.5]
    print(f"\nSummary for Delta ≥ 1.5 (THRESHOLD)")
    print(f"Count: {len(high_edge)}")
    print(f"Hit Rate: {high_edge['hit'].mean():.1%}")

import os
if __name__ == "__main__":
    analyze()
