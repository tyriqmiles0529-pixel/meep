#!/usr/bin/env python
"""
Train Meta-Learner on Modal

Trains context-aware meta-learner on 2024-2025 season using Modal's resources.
Saves directly to Modal volume.

Usage:
    modal run modal_train_meta.py
"""

import modal

app = modal.App("nba-meta-learner-training")

# Volumes
model_volume = modal.Volume.from_name("nba-models")
data_volume = modal.Volume.from_name("nba-data", create_if_missing=True)

# Image with all dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        "pandas",
        "numpy",
        "scikit-learn",
        "lightgbm",
        "pytorch-tabnet",
        "torch",
        "pyarrow",  # For parquet support
    )
    .add_local_file("ensemble_predictor.py", remote_path="/root/ensemble_predictor.py")
    .add_local_file("meta_learner_ensemble.py", remote_path="/root/meta_learner_ensemble.py")
    .add_local_file("hybrid_multi_task.py", remote_path="/root/hybrid_multi_task.py")
    .add_local_file("optimization_features.py", remote_path="/root/optimization_features.py")
    .add_local_file("phase7_features.py", remote_path="/root/phase7_features.py")
    .add_local_file("rolling_features.py", remote_path="/root/rolling_features.py")
    .add_local_dir("shared", remote_path="/root/shared")
    .add_local_dir("priors_data", remote_path="/root/priors_data")
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
    import numpy as np
    import pandas as pd
    import pickle
    import shutil

    sys.path.insert(0, "/root")
    os.chdir("/root")

    from ensemble_predictor import load_all_window_models, predict_with_window
    from meta_learner_ensemble import ContextAwareMetaLearner

    print("="*70)
    print("META-LEARNER TRAINING ON MODAL")
    print("="*70)
    print(f"  Training Season: {training_season}")
    print(f"  CPU Cores: 16")
    print(f"  RAM: 32GB")
    print("="*70)

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

        # Filter to training season (extract start year: "2024-2025" -> 2024)
        season_start_year = int(training_season.split('-')[0])
        df = df[df['season_year'] == season_start_year]
        print(f"  Filtered to {training_season}: {len(df):,} records")
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
                        pred = pred[0] if len(pred) > 0 else 0.0
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
