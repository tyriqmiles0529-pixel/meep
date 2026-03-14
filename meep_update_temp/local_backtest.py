#!/usr/bin/env python
"""
Local backtesting using existing data and models
No Modal usage required
"""

import pandas as pd
import numpy as np
import pickle
import dill
from pathlib import Path

def load_local_data():
    """Load your local NBA data"""
    # Try different possible data locations
    data_paths = [
        "PlayerStatistics.csv",
        "data/PlayerStatistics.csv", 
        "shared/PlayerStatistics.csv"
    ]
    
    for path in data_paths:
        if Path(path).exists():
            print(f"✅ Found data: {path}")
            return pd.read_csv(path, low_memory=False)
    
    print("❌ No data found - download from Kaggle manually")
    return None

def load_meta_learner(model_path="meta_learner_v4.pkl"):
    """Load the trained meta-learner"""
    if Path(model_path).exists():
        print(f"✅ Loading model: {model_path}")
        with open(model_path, 'rb') as f:
            return dill.load(f)
    else:
        print(f"❌ Model not found: {model_path}")
        return None

def prepare_backtest_data(df, seasons=[2022, 2023]):
    """Prepare data for backtesting"""
    print(f"[*] Preparing backtest data for seasons: {seasons}")
    
    # Process dates
    df['gameDate'] = pd.to_datetime(df['gameDate'], errors='coerce')
    df['year'] = df['gameDate'].dt.year
    df['month'] = df['gameDate'].dt.month
    df['season_year'] = df.apply(
        lambda row: row['year'] if row['month'] >= 10 else row['year'] - 1,
        axis=1
    )
    
    # Create playerName
    if 'firstName' in df.columns and 'lastName' in df.columns:
        df['playerName'] = df['firstName'] + ' ' + df['lastName']
    
    # Filter to backtest seasons
    backtest_df = df[df['season_year'].isin(seasons)].copy()
    print(f"✅ Backtest data: {len(backtest_df):,} records")
    
    return backtest_df

def run_simple_backtest(meta_learner, backtest_df):
    """Run a simple backtest"""
    print("\n" + "="*50)
    print("RUNNING SIMPLE BACKTEST")
    print("="*50)
    
    # Sample 1000 records for quick test
    sample_df = backtest_df.sample(min(1000, len(backtest_df)), random_state=42)
    
    # Create dummy window predictions (since we can't load all 26 models locally)
    props = ['points', 'rebounds', 'assists', 'threes']
    
    results = {}
    for prop in props:
        if prop == 'points':
            actuals = sample_df['points'].values
        elif prop == 'rebounds':
            actuals = sample_df['reboundsDefensive'] + sample_df['reboundsOffensive']
        elif prop == 'assists':
            actuals = sample_df['assists'].values
        elif prop == 'threes':
            actuals = sample_df['threePointersMade'].values
        
        # Create dummy predictions (26 windows)
        dummy_preds = np.random.normal(actuals.mean(), actuals.std(), (len(actuals), 26))
        
        # Simple MAE calculation
        pred_mean = dummy_preds.mean(axis=1)
        mae = np.mean(np.abs(pred_mean - actuals))
        
        results[prop] = {
            'samples': len(actuals),
            'mae': round(mae, 3),
            'actual_mean': round(actuals.mean(), 3)
        }
    
    return results

def main():
    """Main local backtesting"""
    print("="*60)
    print("LOCAL BACKTESTING - NO MODAL REQUIRED")
    print("="*60)
    
    # Load data
    df = load_local_data()
    if df is None:
        print("❌ Cannot proceed without data")
        return
    
    # Load meta-learner (if available)
    meta_learner = load_meta_learner()
    
    # Prepare backtest data
    backtest_df = prepare_backtest_data(df)
    
    if meta_learner is not None:
        print("✅ Using trained meta-learner")
        # Would run full backtest if we had all window models locally
        print("⚠️  Full backtest requires all 26 window models locally")
    else:
        print("⚠️  Running simple statistical backtest")
        results = run_simple_backtest(None, backtest_df)
        
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        for prop, metrics in results.items():
            print(f"{prop.upper()}:")
            print(f"  Samples: {metrics['samples']:,}")
            print(f"  MAE: {metrics['mae']}")
            print(f"  Actual Mean: {metrics['actual_mean']}")
    
    print("\n🎯 Next Steps:")
    print("1. Download meta-learner from Modal when billing resets")
    print("2. Download all 26 window models for full backtesting")
    print("3. Integrate into your analyzer for production predictions")

if __name__ == "__main__":
    main()
