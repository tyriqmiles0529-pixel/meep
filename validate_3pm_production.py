"""
Phase J.6.2: Validate 3PM Model using Historical Odds
"""
import pandas as pd
import numpy as np
import joblib
import os

MODEL_DIR = "models/production_v4"
ODDS_FILE = "historical_data/the_odds_api_historical.csv"
FEATURES_PATH = "data/strict_features_v4.csv"
RAW_LOGS_PATH = "data/nba_game_logs_1997_2024.csv"

def validate_3pm():
    print("=== Phase J.6.2: 3PM Bucket Validation ===")
    
    # 1. Load historical 3PM odds
    odds = pd.read_csv(ODDS_FILE)
    odds_3pm = odds[odds['market'] == 'player_threes'].copy()
    print(f"Loaded {len(odds_3pm)} historical 3PM prop lines.")
    
    if odds_3pm.empty:
        print("[FATAL] No player_threes data found. Cannot validate.")
        return
    
    # 2. Get unique (player, date) pairs
    odds_3pm['player_key'] = odds_3pm['player_name'].str.upper()
    validation_pairs = odds_3pm[['player_key', 'game_date']].drop_duplicates()
    print(f"Unique validation pairs: {len(validation_pairs)}")
    
    # 3. Load 3PM model
    xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgb_FG3M.joblib"))
    features_list = joblib.load(os.path.join(MODEL_DIR, "features.joblib"))
    
    # 4. Load Features and FG3M (Actual Outcome)
    print("Loading features and outcomes...")
    raw_logs = pd.read_csv(RAW_LOGS_PATH, usecols=['PLAYER_NAME', 'GAME_DATE', 'FG3M'])
    raw_logs['player_key'] = raw_logs['PLAYER_NAME'].str.upper()
    raw_logs['GAME_DATE'] = raw_logs['GAME_DATE'].astype(str)
    
    lookup = set(zip(validation_pairs['player_key'], validation_pairs['game_date'].astype(str)))
    
    matched_rows = []
    chunk_size = 200000
    for chunk in pd.read_csv(FEATURES_PATH, chunksize=chunk_size):
        chunk['player_key'] = chunk['PLAYER_NAME'].str.upper()
        chunk['GAME_DATE'] = chunk['GAME_DATE'].astype(str)
        mask = chunk.apply(lambda row: (row['player_key'], row['GAME_DATE']) in lookup, axis=1)
        if mask.any():
            matched_rows.append(chunk[mask])
            
    if not matched_rows:
        print("[FATAL] No feature matches.")
        return
        
    df_features = pd.concat(matched_rows).drop_duplicates(subset=['player_key', 'GAME_DATE'])
    
    # Join with FG3M
    df = pd.merge(df_features, raw_logs[['player_key', 'GAME_DATE', 'FG3M']], 
                  on=['player_key', 'GAME_DATE'], how='inner')
    print(f"Matched features with outcomes: {len(df)} rows")
    
    # 5. Predict
    X = df[features_list].fillna(0)
    df['proj_3PM'] = xgb_model.predict(X).clip(0)
    
    # 6. Join back with odds to get line
    odds_3pm['game_date'] = odds_3pm['game_date'].astype(str)
    df_final = pd.merge(df, odds_3pm[['player_key', 'game_date', 'line']], 
                        left_on=['player_key', 'GAME_DATE'], 
                        right_on=['player_key', 'game_date'], how='inner')
    print(f"Final matched with lines: {len(df_final)} rows")
    
    # 7. Calculate Delta and Hit
    df_final['delta'] = df_final['proj_3PM'] - df_final['line']
    df_final['abs_delta'] = df_final['delta'].abs()
    
    def check_hit(row):
        if row['delta'] >= 0:  # Over
            return 1 if row['FG3M'] > row['line'] else 0
        else:  # Under
            return 1 if row['FG3M'] < row['line'] else 0
            
    df_final['hit'] = df_final.apply(check_hit, axis=1)
    
    # 8. Bucket Analysis
    bins = [0, 0.5, 1.0, 1.5, 2.0, 5.0, 100]
    labels = ['0.0-0.5', '0.5-1.0', '1.0-1.5', '1.5-2.0', '2.0-5.0', '5.0+']
    df_final['bucket'] = pd.cut(df_final['abs_delta'], bins=bins, labels=labels)
    
    bucket_stats = df_final.groupby('bucket').agg(
        count=('hit', 'count'),
        hit_rate=('hit', 'mean')
    ).reset_index()
    
    print("\n--- 3PM Delta Bucket Analysis ---")
    print(bucket_stats.to_string(index=False))
    
    # 9. Threshold Decision
    threshold_15 = df_final[df_final['abs_delta'] >= 1.5]
    print(f"\nSummary for Delta >= 1.5:")
    print(f"Count: {len(threshold_15)}")
    if len(threshold_15) > 0:
        print(f"Hit Rate: {threshold_15['hit'].mean():.1%}")
        
        # Decision
        if threshold_15['hit'].mean() >= 0.55:
            print("\n✅ 3PM PRODUCTION APPROVED at ±1.5 threshold")
            return True
        else:
            print("\n⚠️ 3PM hit rate below 55%. Recommend higher threshold or disable.")
            return False
    else:
        print("Not enough data at ±1.5 threshold.")
        return False

if __name__ == "__main__":
    validate_3pm()
