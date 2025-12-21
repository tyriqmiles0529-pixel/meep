import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import time

DATA_PATH = "final_feature_matrix_with_per_min_1997_onward.csv"
TARGET = "points"
TEST_SEASON = 2024

def load_data():
    print(f"Loading {DATA_PATH}...")
    # Load limited columns if needed, but we need to identify season_avg
    df = pd.read_csv(DATA_PATH, low_memory=False)
    
    # Simple cleaning
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Ensure Test Season exists
    if 'season' not in df.columns:
        print("ERROR: 'season' column missing.")
        return None
        
    print(f"Data Loaded: {df.shape}")
    print(f"Seasons: {df['season'].min()} - {df['season'].max()}")
    return df

def train_and_eval(df, drop_leakage=False):
    print(f"\nTraining Model (Drop Leakage: {drop_leakage})...")
    
    # Identify features
    exclude_cols = [TARGET, 'player_name', 'date', 'game_id', 'team', 'opponent', 'opponentteamname', 'win', 'loss', 'season']
    # Also exclude raw stats that are definitely part of the target (points, minutes, fg, etc) if they exist
    # Current game stats are leaks.
    raw_stats = ['points', 'assists', 'rebounds', 'minutes', 'fg', 'fga', 'fg_percent', 'x3p', 'x3pa', 'ft', 'fta']
    exclude_cols += raw_stats
    
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    if drop_leakage:
        # Drop columns containing 'season_avg' or 'avg'
        # Be aggressive
        original_count = len(feature_cols)
        feature_cols = [c for c in feature_cols if 'season_avg' not in c and '_season' not in c]
        print(f"Dropped {original_count - len(feature_cols)} suspected leakage features.")
        
    print(f"Using {len(feature_cols)} features.")
    
    # Split
    train = df[df['season'] < TEST_SEASON]
    test = df[df['season'] == TEST_SEASON]
    
    if len(train) == 0 or len(test) == 0:
        print("Error: Train or Test set empty.")
        return
        
    X_train = train[feature_cols]
    y_train = train[TARGET]
    X_test = test[feature_cols]
    y_test = test[TARGET]
    
    # Train XGBoost
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        n_jobs=-1,
        tree_method='hist' 
    )
    
    start_time = time.time()
    model.fit(X_train, y_train)
    print(f"Training Time: {time.time() - start_time:.2f}s")
    
    # Predict
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    return r2

def main():
    df = load_data()
    if df is None: return
    
    # Run Baseline
    print("\n--- BASELINE (All Features) ---")
    r2_baseline = train_and_eval(df, drop_leakage=False)
    
    # Run Probe
    print("\n--- PROBE (No Season Avg) ---")
    r2_probe = train_and_eval(df, drop_leakage=True)
    
    print("\nData Analysis:")
    if r2_baseline > 0.5 and r2_probe < 0.2:
        print("CONCLUSION: CRITICAL LEAKAGE CONFIRMED.")
        print(f"Model collapses from R2 {r2_baseline:.2f} to {r2_probe:.2f} without season averages.")
    elif r2_baseline - r2_probe > 0.1:
        print("CONCLUSION: SIGNIFICANT RELIANCE on Season Averages.")
    else:
        print("CONCLUSION: Model is robust. Season averages are not the primary driver.")

if __name__ == "__main__":
    main()
