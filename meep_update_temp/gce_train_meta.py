#!/usr/bin/env python
"""
Train Meta-Learner V4 on Google Compute Engine

This is a standalone version of modal_train_meta.py optimized for GCE.
Requirements:
- 16 CPU cores, 32GB RAM (n2-highmem-8 or similar)
- Kaggle credentials as environment variables
- Python 3.8+ with required packages

Usage:
    export KAGGLE_USERNAME="your_username"
    export KAGGLE_KEY="your_key"
    python gce_train_meta.py
"""

import os
import sys
import pickle
import shutil
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error

# CRITICAL: Force CPU mode BEFORE any torch imports
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Add local modules
sys.path.insert(0, ".")
from ensemble_predictor import load_all_window_models, predict_with_window
from train_meta_learner_v4 import MetaLearnerV4, ExperimentConfig


def setup_kaggle_credentials():
    """Setup Kaggle credentials from environment variables"""
    kaggle_username = os.getenv("KAGGLE_USERNAME")
    kaggle_key = os.getenv("KAGGLE_KEY")
    
    if not kaggle_username or not kaggle_key:
        print("❌ ERROR: Set KAGGLE_USERNAME and KAGGLE_KEY environment variables")
        print("   export KAGGLE_USERNAME='your_username'")
        print("   export KAGGLE_KEY='your_key'")
        return False
    
    os.environ['KAGGLE_USERNAME'] = kaggle_username
    os.environ['KAGGLE_KEY'] = kaggle_key
    print("✅ Kaggle credentials configured")
    return True


def download_kaggle_data():
    """Download PlayerStatistics.csv from Kaggle"""
    print("\n[*] Downloading PlayerStatistics.csv from Kaggle...")
    
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
    print(f"\n[*] Processing data from {csv_path}...")
    
    games_df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Total records: {len(games_df):,}")
    
    # Debug: Print actual column names from Kaggle dataset
    print(f"\n[*] Dataset columns ({len(games_df.columns)} total):")
    for i, col in enumerate(games_df.columns):
        print(f"  {i+1:2d}. {col}")
        if i >= 30:  # Show first 30 columns
            print(f"  ... and {len(games_df.columns) - 30} more")
            break
    
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


def setup_model_cache():
    """Setup model cache from local models directory"""
    print("\n[*] Setting up model cache...")
    
    cache_dir = Path("model_cache")
    cache_dir.mkdir(exist_ok=True)
    
    model_source = Path(".")
    models_found = 0
    
    for model_file in model_source.glob("player_models_*.pkl"):
        dest = cache_dir / model_file.name
        if not dest.exists():
            shutil.copy(model_file, dest)
        models_found += 1
    
    for meta_file in model_source.glob("player_models_*_meta.json"):
        dest = cache_dir / meta_file.name
        if not dest.exists():
            shutil.copy(meta_file, dest)
    
    print(f"✅ Found {models_found} window model files")
    
    if models_found == 0:
        print("❌ ERROR: No player_models_*.pkl files found in current directory")
        print("   Make sure window model files are in the same directory as this script")
        return None
    
    return cache_dir


def collect_v4_predictions(df, window_models, max_samples=1500):
    """Collect V4 predictions from window models"""
    print(f"\n[*] Collecting V4 predictions (max {max_samples} samples)...")
    
    sample_df = df.sample(min(max_samples, len(df)), random_state=42)
    
    props = ['points', 'rebounds', 'assists', 'threes']
    prop_cols = {
        'points': 'points',
        'rebounds': 'reboundsTotal', 
        'assists': 'assists',
        'threes': 'threePointersMade'
    }
    
    window_predictions = {}
    actuals = {}
    
    for prop in props:
        if prop_cols[prop] not in sample_df.columns:
            print(f"  ⚠️  Column {prop_cols[prop]} not found, skipping {prop}")
            continue
            
        print(f"  Processing {prop}...")
        preds_list = []
        actuals_list = []
        
        for idx, (_, game) in enumerate(sample_df.iterrows()):
            actual = game.get(prop_cols[prop])
            if pd.isna(actual) or actual < 0:
                continue
                
            window_preds = []
            for window_name, models in window_models.items():
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
                    
                    pred = predict_with_window(models, X, prop)
                    if isinstance(pred, np.ndarray):
                        pred = pred[0] if len(pred) > 0 else 0.0
                    window_preds.append(pred if pred is not None else 0.0)
                except Exception:
                    window_preds.append(0.0)
            
            if len(window_preds) >= 20:
                while len(window_preds) < 27:
                    window_preds.append(np.mean(window_preds))
                
                preds_list.append(window_preds[:27])
                actuals_list.append(actual)
            
            if idx % 500 == 0:
                non_zero = sum(1 for p in window_preds if p != 0.0)
                print(f"    Processed {idx}/{len(sample_df)} games... (non-zero preds: {non_zero}/27)")
        
        if len(actuals_list) >= 100:
            window_predictions[prop] = np.array(preds_list)
            actuals[prop] = np.array(actuals_list)
            print(f"  ✅ {prop}: {len(actuals_list)} samples")
        else:
            print(f"  ❌ {prop}: insufficient samples ({len(actuals_list)})")
    
    return window_predictions, actuals


def prepare_player_stats(games_df):
    """Prepare player statistics for V4 training"""
    print("\n[*] Preparing player statistics...")
    
    # Check what player column exists in the dataset
    player_col = None
    for col in ['playerName', 'player', 'Player', 'player_name']:
        if col in games_df.columns:
            player_col = col
            break
    
    # If no single player column, create one from firstName + lastName
    if not player_col and 'firstName' in games_df.columns and 'lastName' in games_df.columns:
        games_df['playerName'] = games_df['firstName'] + ' ' + games_df['lastName']
        player_col = 'playerName'
        print(f"  Created playerName column from firstName + lastName")
    
    if not player_col:
        print(f"  Available columns: {list(games_df.columns)[:20]}")
        raise ValueError("No player name column found in dataset")
    
    print(f"  Using player column: {player_col}")
    
    player_stats = games_df.groupby(player_col).agg({
        'points': ['mean', 'std'],
        'reboundsDefensive': ['mean', 'std'],
        'reboundsOffensive': ['mean', 'std'],
        'assists': ['mean', 'std'],
        'threePointersMade': ['mean', 'std'],
        'numMinutes': ['mean', 'count']
    }).round(3)
    
    player_stats.columns = ['_'.join(col).strip() for col in player_stats.columns]
    player_stats = player_stats[player_stats[('numMinutes', 'count')] >= 50]
    
    print(f"  ✅ Player stats: {len(player_stats)} players")
    return player_stats


def train_meta_learner_v4_gce(config_path="experiments/v4_full.yaml"):
    """Main training function for GCE"""
    print("="*70)
    print("META-LEARNER V4 TRAINING ON GOOGLE COMPUTE ENGINE")
    print("="*70)
    print(f"  Config: {config_path}")
    print(f"  Training: 2019-2020 + 2020-2021")
    print(f"  Backtest: 2022-2023 + 2023-2024")
    print(f"  Resources: Local CPU cores, RAM")
    print("="*70)
    
    # 1. Setup Kaggle credentials
    if not setup_kaggle_credentials():
        return {"status": "error", "message": "Kaggle credentials not configured"}
    
    # 2. Download data
    csv_path = download_kaggle_data()
    if csv_path is None:
        return {"status": "error", "message": "Failed to download data"}
    
    # 3. Load and process data
    games_df = load_and_process_data(csv_path)
    
    # 4. Setup model cache
    cache_dir = setup_model_cache()
    if cache_dir is None:
        return {"status": "error", "message": "No window models found"}
    
    # 5. Load window models
    print(f"\n[*] Loading window models...")
    window_models = load_all_window_models(str(cache_dir))
    print(f"✅ Loaded {len(window_models)} windows")
    
    # 6. Filter to training seasons
    training_years = [2019, 2020]  # 2019-2020 + 2020-2021
    training_df = games_df[games_df['season_year'].isin(training_years)].copy()
    print(f"\n[*] Training data: {len(training_df):,} records")
    
    # 7. Prepare player stats
    player_stats = prepare_player_stats(games_df)
    
    # 8. Load V4 experiment configuration
    if not Path(config_path).exists():
        return {"status": "error", "message": f"Config not found: {config_path}"}
    
    config = ExperimentConfig(config_path)
    print(f"\n[*] Experiment: {config.config['experiment']['name']}")
    print(f"    Components enabled: {list(config.config['feature_flags'].keys())}")
    
    # 9. Collect V4 predictions
    window_predictions, actuals = collect_v4_predictions(training_df, window_models)
    
    if not window_predictions:
        return {"status": "error", "message": "No training data collected"}
    
    # 10. Train V4 meta-learner
    print(f"\n[*] Training V4 meta-learner...")
    meta_learner = MetaLearnerV4(config)
    
    v4_results = meta_learner.fit_v4(
        window_predictions,
        actuals,
        training_df,
        player_stats
    )
    
    print(f"\n✅ V4 Training Complete!")
    
    # 11. Save V4 model
    output_file = "meta_learner_v4_all_components.pkl"
    print(f"\n[*] Saving model to {output_file}...")
    
    with open(output_file, 'wb') as f:
        pickle.dump(meta_learner, f)
    
    print(f"✅ Model saved: {output_file}")
    
    return {
        "status": "success",
        "config": config_path,
        "training_results": v4_results,
        "model_file": output_file,
        "windows_loaded": len(window_models),
        "training_samples": {prop: len(arr) for prop, arr in actuals.items()}
    }


def main():
    """Main entry point"""
    print("NBA META-LEARNER V4 TRAINING FOR GOOGLE COMPUTE ENGINE")
    print("="*70)
    
    # Check environment
    print("\n[*] Environment check:")
    print(f"  Python: {sys.version}")
    print(f"  Working directory: {os.getcwd()}")
    print(f"  CPU cores: {os.cpu_count()}")
    
    # Check required files
    required_files = [
        "ensemble_predictor.py",
        "train_meta_learner_v4.py", 
        "experiments/v4_full.yaml"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ ERROR: Missing required files:")
        for file in missing_files:
            print(f"   {file}")
        return
    
    print(f"✅ All required files found")
    
    # Run training
    result = train_meta_learner_v4_gce()
    
    # Print results
    print("\n" + "="*70)
    print("TRAINING RESULT")
    print("="*70)
    print(f"Status: {result['status']}")
    
    if result['status'] == 'success':
        print(f"\n✅ V4 Meta-learner trained successfully!")
        print(f"  Model: {result['model_file']}")
        print(f"  Windows loaded: {result['windows_loaded']}")
        print(f"  Training samples: {result['training_samples']}")
        print(f"\nNext steps:")
        print(f"  1. Test the model: python predict_today.py")
        print(f"  2. Upload to production if needed")
    else:
        print(f"\n❌ Training failed: {result.get('message', 'Unknown error')}")
    
    print("="*70)


if __name__ == "__main__":
    main()
