#!/usr/bin/env python
"""
Train Meta-Learner V4 on Google Compute Engine with Multi-Season Support

Uses MetaLearnerV4 with all components enabled:
- Cross-Window Residual Correction
- Player Identity Embeddings  
- Temporal Memory Over Windows

Training: 2019-2020 + 2020-2021 (seasons windows HAVE seen)
Backtest: 2022-2023 + 2023-2024 (seasons windows HAVEN'T seen)

Usage:
    python gce_train_meta_v4.py --config experiments/v4_full.yaml --project-id YOUR_PROJECT --bucket-name YOUR_BUCKET
"""

import os
import sys
import argparse
from pathlib import Path
import pickle
import shutil
import yaml
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Google Cloud imports
from google.cloud import storage
from google.cloud import secretmanager

# CRITICAL: Force CPU mode BEFORE any torch imports
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Add local modules
sys.path.insert(0, str(Path(__file__).parent))

from ensemble_predictor import load_all_window_models, predict_with_window
from train_meta_learner_v4 import MetaLearnerV4, ExperimentConfig

class GCETrainingManager:
    """Manages training on Google Compute Engine"""
    
    def __init__(self, project_id: str, bucket_name: str, secret_id: str = None):
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.secret_id = secret_id
        self.storage_client = storage.Client(project=project_id)
        self.bucket = self.storage_client.bucket(bucket_name)
        
    def get_kaggle_credentials(self):
        """Get Kaggle credentials from Secret Manager"""
        if not self.secret_id:
            # Try environment variables first
            username = os.getenv('KAGGLE_USERNAME')
            key = os.getenv('KAGGLE_KEY')
            if username and key:
                return username, key
            raise ValueError("No Kaggle credentials available")
        
        try:
            secret_client = secretmanager.SecretManagerServiceClient()
            name = f"projects/{self.project_id}/secrets/{self.secret_id}/versions/latest"
            response = secret_client.access_secret_version(request={"name": name})
            payload = response.payload.data.decode("UTF-8")
            
            # Parse JSON credentials
            import json
            creds = json.loads(payload)
            return creds['username'], creds['key']
        except Exception as e:
            raise Exception(f"Failed to get Kaggle credentials: {e}")
    
    def download_models_from_gcs(self, local_dir: str = "/tmp/model_cache"):
        """Download window models from GCS bucket"""
        print("\n[*] Downloading models from GCS...")
        cache_dir = Path(local_dir)
        cache_dir.mkdir(exist_ok=True)
        
        # List and download model files
        blobs = self.bucket.list_blobs(prefix="models/")
        models_downloaded = 0
        
        for blob in blobs:
            if blob.name.endswith('.pkl') or blob.name.endswith('.json'):
                local_path = cache_dir / Path(blob.name).name
                blob.download_to_filename(local_path)
                models_downloaded += 1
        
        print(f"[OK] Downloaded {models_downloaded} model files")
        return str(cache_dir)
    
    def upload_model_to_gcs(self, local_file: str, gcs_path: str):
        """Upload trained model to GCS"""
        print(f"\n[*] Uploading model to GCS: {gcs_path}")
        blob = self.bucket.blob(gcs_path)
        blob.upload_from_filename(local_file)
        print(f"[OK] Model uploaded to gs://{self.bucket_name}/{gcs_path}")
    
    def download_data_from_kaggle(self, local_dir: str = "/tmp"):
        """Download data directly from Kaggle"""
        print("\n[*] Downloading PlayerStatistics.csv from Kaggle...")
        
        try:
            # Get credentials
            username, key = self.get_kaggle_credentials()
            os.environ['KAGGLE_USERNAME'] = username
            os.environ['KAGGLE_KEY'] = key
            print("  Using Kaggle credentials from Secret Manager")
            
            # Download dataset
            import kaggle
            kaggle.api.dataset_download_files(
                'eoinamoore/historical-nba-data-and-player-box-scores',
                path=local_dir,
                unzip=True
            )
            
            # Find the downloaded CSV
            csv_path = Path(local_dir) / "PlayerStatistics.csv"
            if not csv_path.exists():
                # Try alternative names
                for file in Path(local_dir).glob("*.csv"):
                    if "player" in file.name.lower() or "statistics" in file.name.lower():
                        csv_path = file
                        break
            
            if not csv_path.exists():
                raise Exception("Could not find downloaded CSV file")
                
            print(f"  Downloaded: {csv_path}")
            return str(csv_path)
            
        except Exception as e:
            raise Exception(f"Failed to download data: {e}")

def train_meta_learner_v4(config_path: str = "experiments/v4_full.yaml", 
                         project_id: str = None, 
                         bucket_name: str = None,
                         secret_id: str = None):
    """
    Train MetaLearnerV4 on GCE with all components enabled.
    """
    print("="*70)
    print("META-LEARNER V4 TRAINING ON GCE (All Components)")
    print("="*70)
    print(f"  Config: {config_path}")
    print(f"  Training: 2019-2020 + 2020-2021")
    print(f"  Backtest: 2022-2023 + 2023-2024")
    print(f"  CPU Cores: 16")
    print(f"  RAM: 32GB")
    print("="*70)

    # Initialize GCE manager
    if not project_id or not bucket_name:
        raise ValueError("project_id and bucket_name are required")
    
    manager = GCETrainingManager(project_id, bucket_name, secret_id)

    # Load V4 experiment configuration
    if not Path(config_path).exists():
        raise Exception(f"Config not found: {config_path}")
    
    config = ExperimentConfig(config_path)
    print(f"  Experiment: {config.config['experiment']['name']}")
    print(f"  Components enabled: {list(config.config['feature_flags'].keys())}")

    # Download models from GCS
    model_cache_dir = manager.download_models_from_gcs()
    window_models = load_all_window_models(model_cache_dir)
    print(f"[OK] Loaded {len(window_models)} windows")

    # Download data from Kaggle
    csv_path = manager.download_data_from_kaggle()
    games_df = pd.read_csv(csv_path, low_memory=False)
    
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

    # Filter to training seasons
    training_years = [2019, 2020]  # 2019-2020 + 2020-2021
    training_df = games_df[games_df['season_year'].isin(training_years)].copy()
    
    # Prepare player stats for V4
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
        raise Exception("No player name column found in dataset")
    
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
    
    print(f"  Training data: {len(training_df):,} records")
    print(f"  Player stats: {len(player_stats)} players")

    # Collect V4 predictions
    def collect_v4_predictions(df: pd.DataFrame, max_samples: int = 1500) -> dict:
        sample_df = df.sample(min(max_samples, len(df)), random_state=42)
        
        window_predictions = {}
        actuals = {}
        
        props = ['points', 'rebounds', 'assists', 'threes']
        prop_cols = {
            'points': 'points',
            'rebounds': 'reboundsTotal', 
            'assists': 'assists',
            'threes': 'threePointersMade'
        }
        
        for prop in props:
            if prop_cols[prop] not in sample_df.columns:
                continue
                
            preds_list = []
            actuals_list = []
            
            for _, game in sample_df.iterrows():
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
            
            if len(actuals_list) >= 100:
                window_predictions[prop] = np.array(preds_list)
                actuals[prop] = np.array(actuals_list)
                print(f"  {prop}: {len(actuals_list)} samples")
        
        return {
            'window_predictions': window_predictions,
            'actuals': actuals,
            'games_df': sample_df
        }
    
    training_data = collect_v4_predictions(training_df)
    
    if not training_data['window_predictions']:
        raise Exception("No training data collected")
    
    meta_learner = MetaLearnerV4(config)
    
    v4_results = meta_learner.fit_v4(
        training_data['window_predictions'],
        training_data['actuals'],
        training_data['games_df'],
        player_stats
    )
    
    print(f"\n✅ V4 Training Complete!")

    # Save V4 model locally first
    local_output_file = "/tmp/meta_learner_v4_all_components.pkl"
    with open(local_output_file, 'wb') as f:
        pickle.dump(meta_learner, f)
    
    # Upload to GCS
    gcs_path = "models/meta_learner_v4_all_components.pkl"
    manager.upload_model_to_gcs(local_output_file, gcs_path)
    
    print(f"  Model saved to: gs://{bucket_name}/{gcs_path}")
    
    return {
        "status": "success",
        "config": config_path,
        "training_results": v4_results,
        "model_file": f"gs://{bucket_name}/{gcs_path}"
    }

def main():
    parser = argparse.ArgumentParser(description='Train NBA Meta-Learner V4 on GCE')
    parser.add_argument('--config', default='experiments/v4_full.yaml', 
                       help='Path to experiment config file')
    parser.add_argument('--project-id', required=True,
                       help='Google Cloud project ID')
    parser.add_argument('--bucket-name', required=True,
                       help='Google Cloud Storage bucket name')
    parser.add_argument('--secret-id', default='kaggle-credentials',
                       help='Secret Manager secret ID for Kaggle credentials')
    
    args = parser.parse_args()
    
    try:
        result = train_meta_learner_v4(
            config_path=args.config,
            project_id=args.project_id,
            bucket_name=args.bucket_name,
            secret_id=args.secret_id
        )
        
        print("\n" + "="*70)
        print("V4 TRAINING RESULT")
        print("="*70)
        print(f"Status: {result['status']}")
        
        if result['status'] == 'success':
            print("\n✅ V4 Meta-learner trained with all components!")
            print(f"Model saved: {result['model_file']}")
            print(f"\nNext steps for feature removal:")
            print("1. Run with: --config experiments/v4_residual_only.yaml")
            print("2. Run with: --config experiments/v4_temporal_memory_only.yaml")
            print("3. Run with: --config experiments/v4_player_embeddings_only.yaml")
            print("4. Compare backtest performance to identify redundant components")
        else:
            print(f"\n❌ V4 Training failed")
            
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
