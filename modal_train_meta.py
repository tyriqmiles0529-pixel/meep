#!/usr/bin/env python
"""
Train Meta-Learner V4 on Modal with Multi-Season Support

Uses MetaLearnerV4 with all components enabled:
- Cross-Window Residual Correction
- Player Identity Embeddings  
- Temporal Memory Over Windows

Training: 2019-2020 + 2020-2021 (seasons windows HAVE seen)
Backtest: 2022-2023 + 2023-2024 (seasons windows HAVEN'T seen)

Usage:
    modal run modal_train_meta.py::main_v4
    modal run modal_train_meta.py::main_v4 --config experiments/v4_full.yaml
"""

import modal

app = modal.App("nba-meta-learner-v4-training")

# Volumes
model_volume = modal.Volume.from_name("nba-models")
data_volume = modal.Volume.from_name("nba-data", create_if_missing=True)

# Image with all dependencies including V4 requirements
image = (
    modal.Image.debian_slim()
    .pip_install(
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
        "requests"
    )
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
    cpu=16.0,  # 16 cores for faster training
    memory=32768,  # 32GB RAM
    timeout=7200,  # 2 hours
    volumes={
        "/models": model_volume,
        "/data": data_volume
    }
)
def train_meta_learner(training_season: str = "2024-2025"):
    """
    Train meta-learner on Modal with high resources.

    Args:
        training_season: Season to train on (default: 2024-2025)
    """
    import sys
    import os
    from pathlib import Path

    # CRITICAL: Force CPU mode BEFORE any torch imports
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    import numpy as np
    import pandas as pd
    import pickle
    import shutil

    sys.path.insert(0, "/root")
    os.chdir("/root")

    # Set default seasons if not provided
    if training_season is None:
        training_season = ["2019-2020", "2020-2021"]
    if backtest_season is None:
        backtest_season = ["2022-2023", "2023-2024"]

    print("="*70)
    print("META-LEARNER V4 TRAINING ON MODAL (All Components)")
    print("="*70)
    print(f"  Training Seasons: {training_season}")
    print(f"  Backtest Seasons: {backtest_season}")
    print(f"  CPU Cores: 16")
    print(f"  RAM: 32GB")
    print("="*70)
    
    # Load V4 experiment configuration
    config_path = "/root/experiments/v4_full.yaml"
    config = ExperimentConfig(config_path)
    print(f"  Experiment: {config.config['experiment']['name']}")
    print(f"  Components enabled: {list(config.config['feature_flags'].keys())}")

    # 1. Copy models from volume to local cache
    print("\n[*] Setting up model cache...")
    cache_dir = Path("/root/model_cache")
    cache_dir.mkdir(exist_ok=True)

    model_source = Path("/models")
    for model_file in model_source.glob("player_models_*.pkl"):
        dest = cache_dir / model_file.name
        if not dest.exists():
            shutil.copy(model_file, dest)

    for meta_file in model_source.glob("player_models_*_meta.json"):
        dest = cache_dir / meta_file.name
        if not dest.exists():
            shutil.copy(meta_file, dest)

    models_count = len(list(cache_dir.glob("player_models_*.pkl")))
    print(f"[OK] {models_count} window models ready")

    # 2. Load window models
    print("\n[*] Loading window models...")
    window_models = load_all_window_models(str(cache_dir))
    print(f"[OK] Loaded {len(window_models)} windows")

    # 3. Load training data from Kaggle PlayerStatistics.csv
    print(f"\n[*] Loading training data: {training_season}...")
    print(f"    Source: Kaggle PlayerStatistics.csv")

    # Try volume path first, then local
    csv_paths = [
        Path("/data/PlayerStatistics.csv"),
        Path("PlayerStatistics.csv"),
        Path("data/PlayerStatistics.csv"),
    ]

    csv_path = None
    for path in csv_paths:
        if path.exists():
            csv_path = path
            break

    if csv_path is None:
        print(f"[!] ERROR: PlayerStatistics.csv not found!")
        print(f"    Download from: https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores/")
        print(f"    Upload with: modal volume put nba-data PlayerStatistics.csv PlayerStatistics.csv")
        return {"status": "error", "message": "PlayerStatistics.csv not found"}

    print(f"  Reading from: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)

    print(f"  Total records: {len(df):,}")

    # Extract season from gameDate (mixed formats: ISO8601 or "YYYY-MM-DD HH:MM:SS")
    if 'gameDate' in df.columns:
        # Parse with utc=True to handle mixed timezones
        df['gameDate'] = pd.to_datetime(df['gameDate'], format='mixed', utc=True)

        # Convert to timezone-naive for easier processing
        df['gameDate'] = df['gameDate'].dt.tz_localize(None)

        df['year'] = df['gameDate'].dt.year
        df['month'] = df['gameDate'].dt.month

        # NBA season spans Oct-June, so games from Oct-Dec are start of season
        # e.g., Nov 2024 game is part of 2024-2025 season
        df['season_year'] = df.apply(
            lambda row: row['year'] if row['month'] >= 10 else row['year'] - 1,
            axis=1
        )

        # Filter to training seasons (extract start years: ["2019-2020", "2020-2021"] -> [2019, 2020])
        training_years = [int(season.split('-')[0]) for season in training_seasons]
        df = df[df['season_year'].isin(training_years)]
        print(f"  Filtered to {training_seasons}: {len(df):,} records")
    else:
        print(f"  [!] No 'gameDate' column found, using all data")

    if len(df) < 1000:
        print(f"[!] WARNING: Only {len(df)} records - may not be enough for training")

    # 4. Prepare column mappings (Kaggle CSV uses lowercase names)
    prop_col_map = {
        'points': ['points', 'PTS', 'POINTS'],
        'rebounds': ['reboundsTotal', 'REB', 'REBOUNDS'],
        'assists': ['assists', 'AST', 'ASSISTS'],
        'threes': ['threePointersMade', 'FG3M', '3PM']
    }

    # Print available columns for debugging
    print(f"\n[*] Available columns in data:")
    print(f"    {list(df.columns)[:20]}...")  # Show first 20 columns

    # Find which columns exist
    actual_cols = {}
    for prop, candidates in prop_col_map.items():
        for col in candidates:
            if col in df.columns:
                actual_cols[prop] = col
                break

    print(f"\n[*] Data columns mapped:")
    for prop, col in actual_cols.items():
        print(f"    {prop}: {col}")

    # 5. Collect window predictions for each prop
    def collect_predictions(prop_name: str):
        """Collect window predictions and actuals for a prop"""
        print(f"\n{'='*70}")
        print(f"COLLECTING PREDICTIONS: {prop_name.upper()}")
        print(f"{'='*70}")

        if prop_name not in actual_cols:
            print(f"  [!] No data column for {prop_name}, skipping")
            return None

        actual_col = actual_cols[prop_name]
        window_preds = []
        contexts = []
        actuals = []

        # Sample to speed up training (use 5000 games max)
        sample_df = df.sample(min(5000, len(df)), random_state=42)

        for idx, (_, game) in enumerate(sample_df.iterrows(), 1):
            # Get actual value
            actual = game.get(actual_col)
            if pd.isna(actual) or actual < 0:
                continue

            # Get predictions from each window
            preds = []
            for window_idx, (window_name, models) in enumerate(window_models.items()):
                try:
                    # Create simple feature row (Kaggle CSV column names)
                    game_dict = game.to_dict() if hasattr(game, 'to_dict') else game

                    X = pd.DataFrame([{
                        'fieldGoalsAttempted': game_dict.get('fieldGoalsAttempted', 0),
                        'freeThrowsAttempted': game_dict.get('freeThrowsAttempted', 0),
                        'assists': game_dict.get('assists', 0),
                        'rebounds': game_dict.get('reboundsTotal', 0),
                        'threes': game_dict.get('threePointersMade', 0),
                        'points': game_dict.get('points', 0),
                        'numMinutes': game_dict.get('numMinutes', 0),
                    }])

                    pred = predict_with_window(models, X, prop_name)
                    if isinstance(pred, np.ndarray):
                        pred = pred[0]

                    preds.append(pred if pred is not None else 0.0)

                    # Debug first window
                    if idx == 1 and window_idx == 0:
                        print(f"    [DEBUG] First prediction sample:")
                        print(f"      Window: {window_name}")
                        print(f"      Input features: {X.columns.tolist()}")
                        print(f"      Prediction: {pred}")
                except Exception as e:
                    if idx == 1:
                        print(f"    [!] Window {window_name} failed: {e}")
                    preds.append(0.0)

            if len(preds) < 20:
                continue

            # Pad to 27
            while len(preds) < 27:
                preds.append(np.mean(preds))

            window_preds.append(preds[:27])

            # Extract context (Kaggle CSV has 'home' column: 1=home, 0=away)
            contexts.append({
                'position_encoded': 2,
                'usage_rate': 0.20,  # Not available in Kaggle CSV
                'minutes_avg': game_dict.get('numMinutes', 30),
                'is_home': int(game_dict.get('home', 0)),
            })

            actuals.append(actual)

            if idx % 500 == 0:
                # Check if we're getting real predictions
                non_zero = sum(1 for p in preds if p != 0.0)
                print(f"  Processed {idx}/{len(sample_df)} games... (non-zero preds: {non_zero}/27)")

        print(f"  ✓ Collected {len(actuals):,} samples")

        if len(actuals) < 100:
            print(f"  [!] Not enough samples, skipping")
            return None

        return {
            'window_predictions': np.array(window_preds),
            'player_context': pd.DataFrame(contexts),
            'actuals': np.array(actuals)
        }

    def load_season_data(seasons: list) -> pd.DataFrame:
        """Load and filter data for specific seasons"""
        print(f"\n[*] Loading data for seasons: {seasons}")
        
        # Load fresh data
        df_full = pd.read_csv(csv_path, low_memory=False)
        
        # Process dates and extract seasons
        if 'gameDate' in df_full.columns:
            df_full['gameDate'] = pd.to_datetime(df_full['gameDate'], format='mixed', utc=True)
            df_full['gameDate'] = df_full['gameDate'].dt.tz_localize(None)
            df_full['year'] = df_full['gameDate'].dt.year
            df_full['month'] = df_full['gameDate'].dt.month
            df_full['season_year'] = df_full.apply(
                lambda row: row['year'] if row['month'] >= 10 else row['year'] - 1,
                axis=1
            )
        
        # Filter to requested seasons
        season_years = [int(season.split('-')[0]) for season in seasons]
        df_filtered = df_full[df_full['season_year'].isin(season_years)]
        
        print(f"  Loaded {len(df_filtered):,} records for {seasons}")
        return df_filtered

    def backtest_on_season(season: str, window_models: Dict, meta_learner, actual_cols: Dict) -> Dict:
        """Backtest meta-learner on a single season"""
        print(f"\n[*] Backtesting on {season}...")
        
        # Load season data
        season_data = load_season_data([season])
        
        if len(season_data) < 100:
            print(f"  [!] Insufficient data: {len(season_data)} records")
            return {"status": "insufficient_data"}
        
        season_results = {}
        
        for prop in ['points', 'rebounds', 'assists', 'threes']:
            if prop not in actual_cols:
                season_results[prop] = "no_data_column"
                continue
            
            if prop not in meta_learner.meta_models:
                season_results[prop] = "not_trained"
                continue
            
            print(f"    Testing {prop}...")
            
            # Sample for speed
            sample_df = season_data.sample(min(2000, len(season_data)), random_state=42)
            
            window_preds = []
            contexts = []
            actuals = []
            
            actual_col = actual_cols[prop]
            
            for _, game in sample_df.iterrows():
                actual = game.get(actual_col)
                if pd.isna(actual) or actual < 0:
                    continue
                
                # Get window predictions
                preds = []
                for window_name, models in window_models.items():
                    try:
                        game_dict = game.to_dict() if hasattr(game, 'to_dict') else game
                        X = pd.DataFrame([{
                            'fieldGoalsAttempted': game_dict.get('fieldGoalsAttempted', 0),
                            'freeThrowsAttempted': game_dict.get('freeThrowsAttempted', 0),
                            'assists': game_dict.get('assists', 0),
                            'rebounds': game_dict.get('reboundsTotal', 0),
                            'threes': game_dict.get('threePointersMade', 0),
                            'points': game_dict.get('points', 0),
                            'numMinutes': game_dict.get('numMinutes', 0),
                        }])
                        
                        pred = predict_with_window(models, X, prop)
                        if isinstance(pred, np.ndarray):
                            pred = pred[0] if len(pred) > 0 else 0.0
                        preds.append(pred if pred is not None else 0.0)
                    except Exception:
                        preds.append(0.0)
                
                if len(preds) < 20:
                    continue
                
                # Pad to 27
                while len(preds) < 27:
                    preds.append(np.mean(preds))
                
                window_preds.append(preds[:27])
                contexts.append({
                    'position_encoded': 2,
                    'usage_rate': 0.20,
                    'minutes_avg': game_dict.get('numMinutes', 30),
                    'is_home': int(game_dict.get('home', 0)),
                })
                actuals.append(actual)
            
            if len(actuals) < 50:
                season_results[prop] = "insufficient_samples"
                continue
            
            # Get meta-learner predictions
            try:
                window_predictions = np.array(window_preds)
                player_context = pd.DataFrame(contexts)
                actuals_array = np.array(actuals)
                
                meta_preds = meta_learner.predict(
                    window_predictions=window_predictions,
                    player_context=player_context,
                    prop_name=prop
                )
                
                # Calculate metrics
                meta_rmse = np.sqrt(mean_squared_error(actuals_array, meta_preds))
                meta_mae = mean_absolute_error(actuals_array, meta_preds)
                
                # Baseline: simple average
                avg_preds = np.mean(window_predictions, axis=1)
                avg_rmse = np.sqrt(mean_squared_error(actuals_array, avg_preds))
                avg_mae = mean_absolute_error(actuals_array, avg_preds)
                
                # Improvement
                rmse_improvement = ((avg_rmse - meta_rmse) / avg_rmse) * 100
                mae_improvement = ((avg_mae - meta_mae) / avg_mae) * 100
                
                season_results[prop] = {
                    'samples': len(actuals),
                    'meta_rmse': meta_rmse,
                    'avg_rmse': avg_rmse,
                    'rmse_improvement_pct': rmse_improvement,
                    'meta_mae': meta_mae,
                    'avg_mae': avg_mae,
                    'mae_improvement_pct': mae_improvement
                }
                
                print(f"      RMSE: {meta_rmse:.3f} vs {avg_rmse:.3f} ({rmse_improvement:+.1f}%)")
                
            except Exception as e:
                season_results[prop] = f"prediction_failed: {str(e)}"
        
        return season_results
        """Collect predictions and evaluate on backtest data"""
        print(f"    Backtesting {prop}...")

        if prop not in actual_cols:
            print(f"      [!] No data column for {prop}")
            return None

        actual_col = actual_cols[prop]
        window_preds = []
        contexts = []
        actuals = []
        meta_preds = []

        # Sample to speed up backtesting (use 2000 games max)
        sample_df = backtest_df.sample(min(2000, len(backtest_df)), random_state=42)

        for idx, (_, game) in enumerate(sample_df.iterrows(), 1):
            # Get actual value
            actual = game.get(actual_col)
            if pd.isna(actual) or actual < 0:
                continue

            # Get predictions from each window
            preds = []
            for window_idx, (window_name, models) in enumerate(window_models.items()):
                try:
                    game_dict = game.to_dict() if hasattr(game, 'to_dict') else game

                    X = pd.DataFrame([{
                        'fieldGoalsAttempted': game_dict.get('fieldGoalsAttempted', 0),
                        'freeThrowsAttempted': game_dict.get('freeThrowsAttempted', 0),
                        'assists': game_dict.get('assists', 0),
                        'rebounds': game_dict.get('reboundsTotal', 0),
                        'threes': game_dict.get('threePointersMade', 0),
                        'points': game_dict.get('points', 0),
                        'numMinutes': game_dict.get('numMinutes', 0),
                    }])

                    pred = predict_with_window(models, X, prop)
                    if isinstance(pred, np.ndarray):
                        pred = pred[0] if len(pred) > 0 else 0.0
                    preds.append(pred if pred is not None else 0.0)

                except Exception:
                    preds.append(0.0)

            if len(preds) < 20:
                continue

            # Pad to 27
            while len(preds) < 27:
                preds.append(np.mean(preds))

            window_preds.append(preds[:27])

            # Extract context
            contexts.append({
                'position_encoded': 2,
                'usage_rate': 0.20,
                'minutes_avg': game_dict.get('numMinutes', 30),
                'is_home': int(game_dict.get('home', 0)),
            })

            actuals.append(actual)

        if len(actuals) < 50:
            print(f"      [!] Not enough samples: {len(actuals)}")
            return None

        # Convert to arrays
        window_predictions = np.array(window_preds)
        player_context = pd.DataFrame(contexts)
        actuals_array = np.array(actuals)

        # Get meta-learner predictions
        try:
            meta_preds = meta_learner.predict(
                window_predictions=window_predictions,
                player_context=player_context,
                prop_name=prop
            )
        except Exception as e:
            print(f"      [!] Meta-learner prediction failed: {e}")
            return None

        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error

        # Meta-learner metrics
        meta_rmse = np.sqrt(mean_squared_error(actuals_array, meta_preds))
        meta_mae = mean_absolute_error(actuals_array, meta_preds)

        # Simple average baseline
        avg_preds = np.mean(window_predictions, axis=1)
        avg_rmse = np.sqrt(mean_squared_error(actuals_array, avg_preds))
        avg_mae = mean_absolute_error(actuals_array, avg_preds)

        # Improvement
        rmse_improvement = ((avg_rmse - meta_rmse) / avg_rmse) * 100
        mae_improvement = ((avg_mae - meta_mae) / avg_mae) * 100

        print(f"      Meta RMSE: {meta_rmse:.3f} vs Avg: {avg_rmse:.3f} ({rmse_improvement:+.1f}%)")

        return {
            'samples': len(actuals),
            'meta_rmse': meta_rmse,
            'avg_rmse': avg_rmse,
            'rmse_improvement_pct': rmse_improvement,
            'meta_mae': meta_mae,
            'avg_mae': avg_mae,
            'mae_improvement_pct': mae_improvement
        }

    # 6. Train meta-learner
    print(f"\n{'='*70}")
    print(f"TRAINING META-LEARNER")
    print(f"{'='*70}")

    meta_learner = ContextAwareMetaLearner(n_windows=27, cv_folds=3)  # 3 folds for speed

    results = {}
    for prop in ['points', 'rebounds', 'assists', 'threes']:
        data = collect_predictions(prop)

        if data is None:
            results[prop] = "skipped"
            continue

        # Train with OOF
        metrics = meta_learner.fit_oof(
            window_predictions=data['window_predictions'],
            y_true=data['actuals'],
            player_context=data['player_context'],
            prop_name=prop
        )

        results[prop] = {
            'samples': len(data['actuals']),
            'improvement_rmse': f"{metrics['improvement_rmse_pct']:+.1f}%",
            'oof_rmse': f"{metrics['oof_rmse']:.3f}",
            'baseline_rmse': f"{metrics['baseline_rmse']:.3f}"
        }

    # 7. Save to volume
    output_file = "/models/meta_learner_2025_2026.pkl"
    print(f"\n[*] Saving meta-learner to {output_file}...")

    with open(output_file, 'wb') as f:
        pickle.dump(meta_learner, f)

    # Commit volume
    model_volume.commit()

    print(f"\n{'='*70}")
    print(f"✅ TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Saved: {output_file}")
    print(f"  Props trained: {len(meta_learner.meta_models)}")
    print(f"\nResults:")
    for prop, result in results.items():
        if isinstance(result, dict):
            print(f"  {prop:12s}: {result['samples']:5,} samples, {result['improvement_rmse']} improvement")
        else:
            print(f"  {prop:12s}: {result}")
    print(f"{'='*70}\n")

    return {
        "status": "success",
        "props_trained": len(meta_learner.meta_models),
        "results": results
    }

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
    print(f"  CPU Cores: 16")
    print(f"  RAM: 32GB")
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
        
        # Debug: Print actual column names from Kaggle dataset
        print(f"\n[*] Dataset columns ({len(games_df.columns)} total):")
        for i, col in enumerate(games_df.columns):
            print(f"  {i+1:2d}. {col}")
            if i >= 30:  # Show first 30 columns
                print(f"  ... and {len(games_df.columns) - 30} more")
                break
        
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

    # Filter to training seasons
    training_years = [2019, 2020]  # 2019-2020 + 2020-2021
    training_df = games_df[games_df['season_year'].isin(training_years)].copy()
    
    # Prepare player stats for V4
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
        return {"status": "error", "message": "No player name column found in dataset"}
    
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
def main_v4(config: str = "experiments/v4_full.yaml"):
    """
    Train V4 meta-learner on Modal with all components.
    """
    print("="*70)
    print("NBA META-LEARNER V4 TRAINING (All Components)")
    print("="*70)
    print(f"Config: {config}")
    print(f"Training: 2019-2020 + 2020-2021")
    print(f"Backtest: 2022-2023 + 2023-2024")
    print(f"Resources: 16 CPU cores, 32GB RAM")
    print("="*70)
    
    result = train_meta_learner_v4.remote(config_path=config)
    
    print("\n" + "="*70)
    print("V4 TRAINING RESULT")
    print("="*70)
    print(f"Status: {result['status']}")
    
    if result['status'] == 'success':
        print("\n✅ V4 Meta-learner trained with all components!")
        print(f"\nNext steps for feature removal:")
        print("1. Run: modal run modal_train_meta.py::main_v4 --config experiments/v4_residual_only.yaml")
        print("2. Run: modal run modal_train_meta.py::main_v4 --config experiments/v4_temporal_memory_only.yaml")
        print("3. Run: modal run modal_train_meta.py::main_v4 --config experiments/v4_player_embeddings_only.yaml")
        print("4. Compare backtest performance to identify redundant components")
    else:
        print(f"\n❌ V4 Training failed: {result.get('message', 'Unknown error')}")

@app.local_entrypoint()
def main(season: str = "2024-2025"):
    """
    Train meta-learner on Modal.

    Args:
        season: Training season (default: 2024-2025)
    """
    print("="*70)
    print("NBA META-LEARNER TRAINING")
    print("="*70)
    print(f"Season: {season}")
    print(f"Resources: 16 CPU cores, 32GB RAM")
    print("="*70)

    result = train_meta_learner.remote(training_season=season)

    print("\n" + "="*70)
    print("TRAINING RESULT")
    print("="*70)
    print(f"Status: {result['status']}")
    if 'props_trained' in result:
        print(f"Props trained: {result['props_trained']}")
    print("="*70)

    if result['status'] == 'success':
        print("\n✅ Meta-learner trained and saved to Modal volume!")
        print("\nNext: Run analyzer with meta-learner:")
        print("  modal run modal_analyzer.py")
    else:
        print(f"\n❌ Training failed: {result.get('message', 'Unknown error')}")
