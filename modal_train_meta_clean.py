import modal
import os
from pathlib import Path

# Modal app setup
app = modal.App("nba-meta-learner-v4")

# Volumes
model_volume = modal.Volume.from_name("nba-models", create_if_missing=True)
data_volume = modal.Volume.from_name("nba-data", create_if_missing=True)

# Image with all dependencies
image = (
    modal.Image.debian_slim()
    .pip_install([
        "pandas",
        "numpy", 
        "scikit-learn",
        "lightgbm",
        "pytorch-tabnet",
        "torch",
        "pyarrow",
        "pyyaml",
        "scipy",
        "kaggle",
        "requests",
        "dill"
    ])
    .add_local_file("ensemble_predictor.py", remote_path="/root/ensemble_predictor.py")
    .add_local_file("train_meta_learner_v4.py", remote_path="/root/train_meta_learner_v4.py")
    .add_local_file("hybrid_multi_task.py", remote_path="/root/hybrid_multi_task.py")
    .add_local_file("optimization_features.py", remote_path="/root/optimization_features.py")
    .add_local_file("phase7_features.py", remote_path="/root/phase7_features.py")
    .add_local_file("rolling_features.py", remote_path="/root/rolling_features.py")
    .add_local_file("meta_learner_ensemble.py", remote_path="/root/meta_learner_ensemble.py")
    .add_local_dir("experiments", remote_path="/root/experiments")
    .add_local_dir("shared", remote_path="/root/shared")
)

@app.function(
    image=image,
    gpu="a10g",
    secrets=[
        modal.Secret.from_name("KAGGLE_USERNAME"),
        modal.Secret.from_name("KAGGLE_KEY")
    ],
    timeout=7200,
    cpu=16,
    memory=32768,
    volumes={
        "/models": model_volume,
        "/data": data_volume
    }
)
def backtest_meta_learner(config_path: str):
    """Backtest V4 meta-learner on different seasons"""
    import yaml
    import pickle
    import pandas as pd
    import numpy as np
    from pathlib import Path
    from train_meta_learner_v4 import ExperimentConfig
    
    print("[*] Loading configuration...")
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = ExperimentConfig(config_dict)
    
    # Load window models (same as training)
    print("[*] Loading window models...")
    from ensemble_predictor import load_all_window_models, predict_with_window
    window_models = load_all_window_models("/models")
    print(f"[OK] Loaded {len(window_models)} windows")
    
    # Load the trained model
    model_file = "/models/meta_learner_v4_all_components.pkl"
    try:
        import dill as pickle
        with open(model_file, 'rb') as f:
            meta_learner = pickle.load(f)
        print(f"[✓] Loaded trained model from {model_file}")
    except FileNotFoundError:
        return {"status": "error", "message": f"No trained model found at {model_file}. Run training first."}
    
    # Download and prepare data (same as training)
    print("\n[*] Downloading PlayerStatistics.csv from Kaggle...")
    
    try:
        # Set up Kaggle credentials
        kaggle_username = os.getenv("username")
        kaggle_key = os.getenv("key")
        
        if kaggle_username and kaggle_key:
            os.environ['KAGGLE_USERNAME'] = kaggle_username
            os.environ['KAGGLE_KEY'] = kaggle_key
        
        import kaggle
        
        kaggle.api.dataset_download_files(
            'eoinamoore/historical-nba-data-and-player-box-scores',
            path='/root/',
            unzip=True
        )
        
        csv_path = Path("/root/PlayerStatistics.csv")
        games_df = pd.read_csv(csv_path, low_memory=False)
        
    except Exception as e:
        return {"status": "error", "message": f"Failed to download data: {e}"}
    
    # Process dates (same as training)
    if 'gameDate' in games_df.columns:
        games_df['gameDate'] = pd.to_datetime(games_df['gameDate'], format='mixed', utc=True)
        games_df['gameDate'] = games_df['gameDate'].dt.tz_localize(None)
        games_df['year'] = games_df['gameDate'].dt.year
        games_df['month'] = games_df['gameDate'].dt.month
        games_df['season_year'] = games_df.apply(
            lambda row: row['year'] if row['month'] >= 10 else row['year'] - 1,
            axis=1
        )
    
    # Create playerName (same as training)
    if 'firstName' in games_df.columns and 'lastName' in games_df.columns:
        games_df['playerName'] = games_df['firstName'] + ' ' + games_df['lastName']
        player_col = 'playerName'
    
    # Filter to BACKTEST seasons (different from training)
    backtest_years = [2022, 2023]  # 2022-2023 + 2023-2024 seasons
    backtest_df = games_df[games_df['season_year'].isin(backtest_years)].copy()
    
    print(f"  Backtest data: {len(backtest_df):,} records")
    print(f"  Backtest seasons: {backtest_years}")
    
    # Prepare player stats (same as training)
    player_stats = games_df.groupby(player_col).agg({
        'points': ['mean', 'std'],
        'reboundsDefensive': ['mean', 'std'],
        'reboundsOffensive': ['mean', 'std'],
        'assists': ['mean', 'std'],
        'threePointersMade': ['mean', 'std'],
        'numMinutes': ['mean', 'count']
    }).round(3).reset_index()
    
    player_stats[('reboundsTotal', 'mean')] = player_stats[('reboundsDefensive', 'mean')] + player_stats[('reboundsOffensive', 'mean')]
    player_stats[('reboundsTotal', 'std')] = player_stats[('reboundsDefensive', 'std')] + player_stats[('reboundsOffensive', 'std')]
    
    player_stats.columns = ['_'.join(col).strip() for col in player_stats.columns]
    player_stats = player_stats.rename(columns={f'{player_col}_': player_col})
    
    # Collect backtest predictions (reuse function from training)
    def collect_v4_predictions(df: pd.DataFrame, max_samples: int = None) -> dict:
        if max_samples is None:
            sample_df = df
        else:
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
                            'steals': game_dict.get('steals', 0),
                            'blocks': game_dict.get('blocks', 0),
                            'reboundsDefensive': game_dict.get('reboundsDefensive', 0),
                            'reboundsOffensive': game_dict.get('reboundsOffensive', 0),
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
                actuals[prop] = np.array(actuals_list).flatten()
        
        return {
            'window_predictions': window_predictions,
            'actuals': actuals,
            'games_df': sample_df
        }
    
    # Collect backtest data
    print(f"\n{'='*70}")
    print(f"COLLECTING BACKTEST DATA")
    print(f"{'='*70}")
    
    backtest_data = collect_v4_predictions(backtest_df, max_samples=5000)
    
    if not backtest_data['window_predictions']:
        return {"status": "error", "message": "No backtest data"}
    
    # Run backtesting (evaluation only, no training)
    print(f"\n{'='*70}")
    print(f"RUNNING BACKTEST EVALUATION")
    print(f"{'='*70}")
    
    # Use the loaded meta-learner to make predictions
    try:
        # Temporarily disable experiment saving for backtest
        import train_meta_learner_v4
        original_save = train_meta_learner_v4.save_experiment_results
        train_meta_learner_v4.save_experiment_results = lambda *args, **kwargs: None
        
        backtest_results = meta_learner.fit_v4(
            backtest_data['window_predictions'],
            backtest_data['actuals'],
            backtest_data['games_df'],
            player_stats
        )
        
        # Restore original save function
        train_meta_learner_v4.save_experiment_results = original_save
        
        print(f"\n✅ Backtesting Complete!")
        
        return {
            "status": "success",
            "config": config_path,
            "backtest_results": backtest_results,
            "backtest_seasons": backtest_years,
            "sample_size": len(backtest_data['games_df'])
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Backtesting failed: {e}"}

@app.function(
    image=image,
    cpu=16.0,
    memory=32768,
    timeout=7200,
    secrets=[
        modal.Secret.from_name("KAGGLE_USERNAME"),
        modal.Secret.from_name("KAGGLE_KEY")
    ],
    volumes={
        "/models": model_volume,
        "/data": data_volume
    }
)
def train_meta_learner_v4(config_path: str = "experiments/v4_full.yaml"):
    """
    Train MetaLearnerV4 on Modal with all components enabled.
    Uses Kaggle dataset with lowercase column names.
    """
    import sys
    import os
    from pathlib import Path
    import yaml
    import pandas as pd
    import numpy as np
    import pickle
    import shutil
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    # CRITICAL: Force CPU mode BEFORE any torch imports
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

    sys.path.insert(0, "/root")
    os.chdir("/root")

    from ensemble_predictor import load_all_window_models, predict_with_window
    from train_meta_learner_v4 import MetaLearnerV4, ExperimentConfig

    print("="*70)
    print("META-LEARNER V4 TRAINING ON MODAL (All Components)")
    print("="*70)
    print(f"  Config: {config_path}")
    print(f"  Training: 2019-2020 + 2020-2021")
    print(f"  Backtest: 2022-2023 + 2023-2024")
    print("="*70)

    # Load V4 experiment configuration
    full_config_path = f"/root/{config_path}"
    if not Path(full_config_path).exists():
        print(f"[!] Config not found: {full_config_path}")
        return {"status": "error", "message": f"Config not found: {config_path}"}
    
    config = ExperimentConfig(full_config_path)
    print(f"  Experiment: {config.config['experiment']['name']}")
    print(f"  Components enabled: {list(config.config['feature_flags'].keys())}")

    # Setup model cache
    cache_dir = Path("/root/model_cache")
    cache_dir.mkdir(exist_ok=True)
    
    model_source = Path("/models")
    for model_file in model_source.glob("player_models_*.pkl"):
        dest = cache_dir / model_file.name
        if not dest.exists():
            shutil.copy(model_file, dest)
    
    window_models = load_all_window_models(str(cache_dir))
    print(f"[OK] Loaded {len(window_models)} windows")

    # Download data directly from Kaggle
    print("\n[*] Downloading PlayerStatistics.csv from Kaggle...")
    
    try:
        # Set up Kaggle credentials from Modal secrets FIRST
        kaggle_username = os.getenv("username")
        kaggle_key = os.getenv("key")
        
        if kaggle_username and kaggle_key:
            os.environ['KAGGLE_USERNAME'] = kaggle_username
            os.environ['KAGGLE_KEY'] = kaggle_key
            print("  Using Kaggle credentials from Modal secrets")
        else:
            print("  [!] No Kaggle credentials found in Modal secrets")
            raise Exception("Kaggle credentials not found")
        
        # Now import kaggle (it will authenticate automatically)
        import kaggle
        
        # Download dataset
        kaggle.api.dataset_download_files(
            'eoinamoore/historical-nba-data-and-player-box-scores',
            path='/root/',
            unzip=True
        )
        
        # Find the downloaded CSV
        csv_path = Path("/root/PlayerStatistics.csv")
        if not csv_path.exists():
            # Try alternative names
            for file in Path("/root").glob("*.csv"):
                if "player" in file.name.lower() or "statistics" in file.name.lower():
                    csv_path = file
                    break
        
        if not csv_path.exists():
            return {"status": "error", "message": "Could not find downloaded CSV file"}
            
        print(f"  Downloaded: {csv_path}")
        games_df = pd.read_csv(csv_path, low_memory=False)
        
    except Exception as e:
        return {"status": "error", "message": f"Failed to download data: {e}"}
    
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

    # Create playerName from firstName + lastName (Kaggle dataset structure)
    if 'firstName' in games_df.columns and 'lastName' in games_df.columns:
        games_df['playerName'] = games_df['firstName'] + ' ' + games_df['lastName']
        player_col = 'playerName'
        print(f"  Created playerName column from firstName + lastName")
    else:
        return {"status": "error", "message": "Could not create player name column"}

    # Filter to training seasons
    training_years = [2019, 2020]  # 2019-2020 + 2020-2021
    training_df = games_df[games_df['season_year'].isin(training_years)].copy()
    
    # Prepare player stats for V4 using lowercase Kaggle columns
    player_stats = games_df.groupby(player_col).agg({
        'points': ['mean', 'std'],
        'reboundsDefensive': ['mean', 'std'],
        'reboundsOffensive': ['mean', 'std'],
        'assists': ['mean', 'std'],
        'threePointersMade': ['mean', 'std'],
        'numMinutes': ['mean', 'count']
    }).round(3).reset_index()
    
    # Create total rebounds from defensive + offensive
    player_stats[('reboundsTotal', 'mean')] = player_stats[('reboundsDefensive', 'mean')] + player_stats[('reboundsOffensive', 'mean')]
    player_stats[('reboundsTotal', 'std')] = player_stats[('reboundsDefensive', 'std')] + player_stats[('reboundsOffensive', 'std')]
    
    player_stats.columns = ['_'.join(col).strip() for col in player_stats.columns]
    # Fix the player column name after flattening (it becomes 'playerName_' with empty string)
    player_stats = player_stats.rename(columns={f'{player_col}_': player_col})
    # Filter by the count column (now flattened)
    count_col = 'numMinutes_count'
    if count_col in player_stats.columns:
        player_stats = player_stats[player_stats[count_col] >= 50]
    else:
        print(f"  Available columns: {list(player_stats.columns)}")
        return {"status": "error", "message": f"Count column {count_col} not found"}
    
    print(f"  Training data: {len(training_df):,} records")
    print(f"  Player stats: {len(player_stats)} players")

    # Collect V4 predictions
    def collect_v4_predictions(df: pd.DataFrame, max_samples: int = None) -> dict:
        # Use all data if no limit specified
        if max_samples is None:
            sample_df = df
        else:
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
                        # Use lowercase Kaggle columns for features
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
                            'steals': game_dict.get('steals', 0),
                            'blocks': game_dict.get('blocks', 0),
                            'reboundsDefensive': game_dict.get('reboundsDefensive', 0),
                            'reboundsOffensive': game_dict.get('reboundsOffensive', 0),
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
                actuals[prop] = np.array(actuals_list).flatten()  # Ensure 1D array
                print(f"  {prop}: {len(actuals_list)} samples (shape: {actuals[prop].shape})")
        
        return {
            'window_predictions': window_predictions,
            'actuals': actuals,
            'games_df': sample_df
        }

    # Train V4
    print(f"\n{'='*70}")
    print(f"TRAINING V4 META-LEARNER")
    print(f"{'='*70}")
    
    training_data = collect_v4_predictions(training_df, max_samples=5000)
    
    if not training_data['window_predictions']:
        return {"status": "error", "message": "No training data"}
    
    meta_learner = MetaLearnerV4(config)
    
    v4_results = meta_learner.fit_v4(
        training_data['window_predictions'],
        training_data['actuals'],
        training_data['games_df'],
        player_stats
    )
    
    print(f"\n✅ V4 Training Complete!")

    # Save V4 model
    import dill as pickle
    output_file = "/models/meta_learner_v4_all_components.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(meta_learner, f)
    
    model_volume.commit()
    
    print(f"  Model saved: {output_file}")
    
    return {
        "status": "success",
        "config": config_path,
        "training_results": v4_results,
        "model_file": output_file
    }

@app.local_entrypoint()
def main_v4(config: str = "experiments/v4_full.yaml", backtest: bool = False):
    """Main V4 training or backtesting entry point"""
    
    if backtest:
        print("="*70)
        print("BACKTESTING V4 META-LEARNER")
        print("="*70)
        print(f"Config: {config}")
        print(f"Mode: Backtesting (using pre-trained model)")
        print("="*70)
        
        result = backtest_meta_learner.remote(config_path=config)
    else:
        print("="*70)
        print("TRAINING V4 META-LEARNER")
        print("="*70)
        print(f"Config: {config}")
        print(f"Mode: Training (creating new model)")
        print("="*70)
        
        result = train_meta_learner_v4.remote(config_path=config)
    
    print(f"Training: 2019-2020 + 2020-2021")
    print(f"Backtest: 2022-2023 + 2023-2024")
    print(f"Resources: 16 CPU cores, 32GB RAM")
    print("="*70)
    
    
    print("\n" + "="*70)
    print("V4 TRAINING RESULT")
    print("="*70)
    print(f"Status: {result['status']}")
    
    if result['status'] == 'success':
        print("\n✅ V4 Meta-learner trained with all components!")
        print(f"\nNext steps for feature removal:")
        print("1. Run: modal run modal_train_meta_clean.py::main_v4 --config experiments/v4_residual_only.yaml")
        print("2. Run: modal run modal_train_meta_clean.py::main_v4 --config experiments/v4_temporal_memory_only.yaml")
        print("3. Run: modal run modal_train_meta_clean.py::main_v4 --config experiments/v4_player_embeddings_only.yaml")
        print("4. Compare backtest performance to identify redundant components")
    else:
        print(f"\n❌ V4 Training failed: {result.get('message', 'Unknown error')}")
