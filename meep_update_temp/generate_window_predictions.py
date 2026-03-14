#!/usr/bin/env python
"""
Generate Window Predictions for Meta-Learner Training

This script loads all trained window models and generates predictions
on the Kaggle dataset for meta-learner training.

Usage:
    python generate_window_predictions.py --all_windows
    python generate_window_predictions.py --window 2022_2024
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime

# Force CPU mode
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

sys.path.insert(0, ".")
from ensemble_predictor import load_all_window_models, predict_with_window


def setup_kaggle_credentials():
    """Setup Kaggle credentials from environment variables"""
    kaggle_username = os.getenv("KAGGLE_USERNAME")
    kaggle_key = os.getenv("KAGGLE_KEY")
    
    if not kaggle_username or not kaggle_key:
        raise Exception("Set KAGGLE_USERNAME and KAGGLE_KEY environment variables")
    
    os.environ['KAGGLE_USERNAME'] = kaggle_username
    os.environ['KAGGLE_KEY'] = kaggle_key
    return True


def download_kaggle_data():
    """Download PlayerStatistics.csv from Kaggle"""
    print("[*] Downloading PlayerStatistics.csv from Kaggle...")
    
    try:
        import kaggle
        
        # Download dataset
        kaggle.api.dataset_download_files(
            'eoinamoore/historical-nba-data-and-player-box-scores',
            path='./',
            unzip=True
        )
        
        # Find the downloaded CSV
        csv_path = Path("PlayerStatistics.csv")
        if not csv_path.exists():
            # Try alternative names
            for file in Path(".").glob("*.csv"):
                if "player" in file.name.lower() or "statistics" in file.name.lower():
                    csv_path = file
                    break
        
        if not csv_path.exists():
            raise FileNotFoundError("Could not find downloaded CSV file")
            
        print(f"✅ Downloaded: {csv_path}")
        return csv_path
        
    except Exception as e:
        print(f"❌ Failed to download data: {e}")
        return None


def load_and_process_data(csv_path):
    """Load and process the Kaggle data"""
    print(f"[*] Processing data from {csv_path}...")
    
    games_df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Total records: {len(games_df):,}")
    
    # Process dates
    if 'gameDate' in games_df.columns:
        games_df['gameDate'] = pd.to_datetime(games_df['gameDate'], format='mixed', utc=True)
        games_df['gameDate'] = games_df['gameDate'].dt.tz_localize(None)
        games_df['year'] = games_df['gameDate'].dt.year
        games_df['month'] = games_df['gameDate'].dt.month
        games_df['season_year'] = games_df.apply(
            lambda row: row['year'] if row['month'] >= 10 else row['year'] - 1,
            axis=1
        )
    
    return games_df


def generate_predictions_for_window(window_models, games_df, window_name, max_samples=2000):
    """Generate predictions for a specific window"""
    print(f"[*] Generating predictions for window: {window_name}")
    
    # Sample data for prediction generation
    sample_df = games_df.sample(min(max_samples, len(games_df)), random_state=42)
    
    props = ['points', 'rebounds', 'assists', 'threes']
    prop_cols = {
        'points': 'points',
        'rebounds': 'reboundsTotal', 
        'assists': 'assists',
        'threes': 'threePointersMade'
    }
    
    all_predictions = []
    
    for prop in props:
        if prop_cols[prop] not in sample_df.columns:
            print(f"  ⚠️  Column {prop_cols[prop]} not found, skipping {prop}")
            continue
            
        print(f"  Processing {prop}...")
        predictions_data = []
        
        for idx, (_, game) in enumerate(sample_df.iterrows()):
            actual = game.get(prop_cols[prop])
            if pd.isna(actual) or actual < 0:
                continue
                
            try:
                game_dict = game.to_dict()
                X = pd.DataFrame([{
                    'fieldGoalsAttempted': game_dict.get('fieldGoalsAttempted', 0),
                    'freeThrowsAttempted': game_dict.get('freeThrowsAttempted', 0),
                    'assists': game_dict.get('assists', 0),
                    'reboundsTotal': game_dict.get('reboundsDefensive', 0) + game_dict.get('reboundsOffensive', 0),
                    'threePointersMade': game_dict.get('threePointersMade', 0),
                    'points': game_dict.get('points', 0),
                    'numMinutes': game_dict.get('numMinutes', 0),
                    'fieldGoalsMade': game_dict.get('fieldGoalsMade', 0),
                    'freeThrowsMade': game_dict.get('freeThrowsMade', 0),
                    'turnovers': game_dict.get('turnovers', 0),
                }])
                
                pred = predict_with_window(window_models, X, prop)
                if isinstance(pred, np.ndarray):
                    pred = pred[0] if len(pred) > 0 else 0.0
                
                predictions_data.append({
                    'player_id': game_dict.get('playerId', idx),
                    'game_date': game_dict.get('gameDate', ''),
                    'prop_type': prop,
                    'actual_value': actual,
                    'predicted_value': pred if pred is not None else 0.0,
                    'window_name': window_name
                })
                
            except Exception as e:
                # Skip problematic predictions
                continue
            
            if idx % 500 == 0:
                print(f"    Processed {idx}/{len(sample_df)} games...")
        
        if predictions_data:
            pred_df = pd.DataFrame(predictions_data)
            all_predictions.append(pred_df)
            print(f"  ✅ {prop}: {len(predictions_data)} predictions")
        else:
            print(f"  ❌ {prop}: no predictions generated")
    
    if all_predictions:
        return pd.concat(all_predictions, ignore_index=True)
    else:
        return None


def main():
    parser = argparse.ArgumentParser(description='Generate window predictions for meta-learner training')
    parser.add_argument('--all_windows', action='store_true', help='Generate predictions for all windows')
    parser.add_argument('--window', type=str, help='Generate predictions for specific window')
    parser.add_argument('--max_samples', type=int, default=2000, help='Maximum samples per window')
    args = parser.parse_args()
    
    print("="*70)
    print("WINDOW PREDICTION GENERATOR")
    print("="*70)
    
    # Setup Kaggle credentials
    setup_kaggle_credentials()
    
    # Download data if needed
    csv_path = Path("PlayerStatistics.csv")
    if not csv_path.exists():
        csv_path = download_kaggle_data()
        if csv_path is None:
            print("❌ Failed to download data")
            return False
    
    # Load and process data
    games_df = load_and_process_data(csv_path)
    
    # Load window models
    print(f"\n[*] Loading window models...")
    cache_dir = Path("player_models")
    if not cache_dir.exists():
        print("❌ ERROR: player_models directory not found")
        return False
    
    window_models = load_all_window_models(str(cache_dir))
    print(f"✅ Loaded {len(window_models)} windows")
    
    # Create output directory
    output_dir = Path("artifacts/window_predictions")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate predictions
    if args.all_windows:
        windows_to_process = list(window_models.keys())
        print(f"[*] Processing all {len(windows_to_process)} windows")
    elif args.window:
        if args.window not in window_models:
            print(f"❌ Window {args.window} not found")
            return False
        windows_to_process = [args.window]
        print(f"[*] Processing single window: {args.window}")
    else:
        print("❌ Must specify --all_windows or --window <name>")
        return False
    
    total_predictions = 0
    
    for window_name in windows_to_process:
        print(f"\n{'='*60}")
        print(f"PROCESSING WINDOW: {window_name}")
        print(f"{'='*60}")
        
        try:
            predictions_df = generate_predictions_for_window(
                window_models[window_name], 
                games_df, 
                window_name,
                args.max_samples
            )
            
            if predictions_df is not None and len(predictions_df) > 0:
                output_file = output_dir / f"window_{window_name}_predictions.parquet"
                predictions_df.to_parquet(output_file, index=False)
                print(f"✅ Saved {len(predictions_df)} predictions to {output_file}")
                total_predictions += len(predictions_df)
            else:
                print(f"❌ No predictions generated for window {window_name}")
                
        except Exception as e:
            print(f"❌ Failed to process window {window_name}: {e}")
            continue
    
    print(f"\n{'='*70}")
    print("PREDICTION GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total windows processed: {len(windows_to_process)}")
    print(f"Total predictions generated: {total_predictions:,}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}")
    
    return True


if __name__ == "__main__":
    main()
