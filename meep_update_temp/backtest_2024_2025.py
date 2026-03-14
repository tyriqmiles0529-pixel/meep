#!/usr/bin/env python
"""
Backtest 2024-2025 Season

Train models on 1947-2024 data, validate on complete 2024-2025 season.
Current season is 2025-2026, so 2024-2025 is the most recent complete season.

This validates model accuracy on unseen recent data.
"""

import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from shared.data_loading import get_year_column


def load_trained_models(model_cache_dir: str = "model_cache") -> Dict:
    """Load all trained player models from cache"""
    cache_path = Path(model_cache_dir)

    if not cache_path.exists():
        raise FileNotFoundError(f"Model cache not found: {model_cache_dir}")

    # Find all player model files
    model_files = sorted(cache_path.glob("player_models_*.pkl"))

    if not model_files:
        raise FileNotFoundError(f"No player models found in {model_cache_dir}")

    print(f"Found {len(model_files)} model windows")

    # Load all windows
    all_models = {}
    for model_file in model_files:
        # Extract window years from filename
        stem = model_file.stem  # "player_models_2022_2024"
        parts = stem.split('_')
        if len(parts) >= 4:
            start_year = int(parts[2])
            end_year = int(parts[3])

            # Skip windows that include 2024 or 2025 (contaminated)
            if end_year >= 2024:
                print(f"  Skipping {start_year}-{end_year} (contains test data)")
                continue

            # Load models with CPU mapping (handles GPU-trained models)
            import torch
            import io

            # Load pickle with custom unpickler that maps CUDA tensors to CPU
            class CPU_Unpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if module == 'torch.storage' and name == '_load_from_bytes':
                        return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
                    else:
                        return super().find_class(module, name)

            with open(model_file, 'rb') as f:
                models = CPU_Unpickler(f).load()

            # Force all TabNet models to CPU and set use_gpu=False
            import torch
            device = torch.device('cpu')

            if isinstance(models, dict):
                for key, model in models.items():
                    # Set use_gpu flag if it exists
                    if hasattr(model, 'use_gpu'):
                        model.use_gpu = False
                    if hasattr(model, 'device_name'):
                        model.device_name = 'cpu'
                    if hasattr(model, 'network') and hasattr(model.network, 'to'):
                        model.network.to(device)
                    # For hybrid multi-task models
                    if hasattr(model, 'correlated_tabnet') and model.correlated_tabnet:
                        if hasattr(model.correlated_tabnet, 'use_gpu'):
                            model.correlated_tabnet.use_gpu = False
                        if hasattr(model.correlated_tabnet, 'device_name'):
                            model.correlated_tabnet.device_name = 'cpu'
                        if hasattr(model.correlated_tabnet, 'network'):
                            model.correlated_tabnet.network.to(device)
                    if hasattr(model, 'independent_models'):
                        for prop, prop_models in model.independent_models.items():
                            if 'tabnet' in prop_models and prop_models['tabnet']:
                                if hasattr(prop_models['tabnet'], 'use_gpu'):
                                    prop_models['tabnet'].use_gpu = False
                                if hasattr(prop_models['tabnet'], 'device_name'):
                                    prop_models['tabnet'].device_name = 'cpu'
                                if hasattr(prop_models['tabnet'], 'network'):
                                    prop_models['tabnet'].network.to(device)

            # Extract feature names from model if available
            feature_names = None
            if isinstance(models, dict):
                # Try to get feature names from multi_task_model
                if 'multi_task_model' in models and hasattr(models['multi_task_model'], 'feature_names'):
                    feature_names = models['multi_task_model'].feature_names
                # Or from individual models
                elif 'points' in models and hasattr(models['points'], 'feature_names'):
                    feature_names = models['points'].feature_names

            all_models[f"{start_year}-{end_year}"] = {
                'models': models,
                'start_year': start_year,
                'end_year': end_year,
                'feature_names': feature_names
            }
            print(f"  ✓ Loaded {start_year}-{end_year} ({len(feature_names) if feature_names else '?'} features)")

    return all_models


def predict_with_hybrid_model(model, X: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Make predictions with hybrid multi-task model.

    Handles both old single-task models and new hybrid multi-task models.
    Ensures proper device handling for GPU inference.
    """
    import torch

    predictions = {}

    # Ensure model is in eval mode if it's a PyTorch model
    if hasattr(model, 'network'):
        model.network.eval()

    # Check if it's a hybrid multi-task model
    if hasattr(model, 'predict'):
        # Hybrid multi-task model - returns dict of predictions
        # The predict method should handle device conversion internally
        # but TabNet sometimes doesn't, so we wrap it
        with torch.no_grad():
            preds = model.predict(X)

        if isinstance(preds, dict):
            # Already a dict with prop names
            predictions = preds
        else:
            # Single prediction - assume points
            predictions['points'] = preds
    else:
        # Old single-task model - just call predict
        with torch.no_grad():
            predictions['points'] = model.predict(X)

    return predictions


def backtest_window(window_models: Dict, test_df: pd.DataFrame, verbose: bool = True) -> Dict:
    """Backtest a single window on 2024-2025 data"""

    # Extract features (remove target columns)
    feature_cols = [c for c in test_df.columns if c not in [
        'points', 'reboundsTotal', 'assists', 'threePointersMade', 'numMinutes',
        'personId', 'gameId', 'gameDate', 'firstName', 'lastName'
    ]]

    X_test = test_df[feature_cols].copy()

    # Convert categorical columns to numeric, skip object (string) columns
    numeric_cols = []
    for col in X_test.columns:
        if X_test[col].dtype.name == 'category':
            try:
                X_test[col] = X_test[col].astype(float)
                numeric_cols.append(col)
            except:
                # Skip if conversion fails (string categories)
                pass
        elif X_test[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            numeric_cols.append(col)
        # Skip object dtype (strings)

    # Keep only numeric columns
    X_test = X_test[numeric_cols].fillna(0)

    # Align features with model's training features
    if 'feature_names' in window_models and window_models['feature_names']:
        model_features = window_models['feature_names']
        # Only use features that model was trained on
        available_features = [f for f in model_features if f in X_test.columns]
        X_test = X_test[available_features]

        # Add missing features as zeros (if model expects features we don't have)
        for feat in model_features:
            if feat not in X_test.columns:
                X_test[feat] = 0

        # Ensure column order matches training
        X_test = X_test[model_features]

    # Get actual values
    y_actual = {
        'points': test_df['points'].fillna(0).values,
        'rebounds': test_df['reboundsTotal'].fillna(0).values,
        'assists': test_df['assists'].fillna(0).values,
        'threes': test_df['threePointersMade'].fillna(0).values,
        'minutes': test_df['numMinutes'].fillna(0).values
    }

    # Get predictions from each model
    results = {}
    models = window_models['models']

    # Check if it's a hybrid multi-task model
    if 'multi_task_model' in models:
        # New hybrid multi-task model
        hybrid_model = models['multi_task_model']
        predictions = predict_with_hybrid_model(hybrid_model, X_test)

        # Calculate MAE for each prop
        for prop in ['points', 'rebounds', 'assists', 'threes', 'minutes']:
            if prop in predictions:
                y_pred = predictions[prop]
                mae = np.mean(np.abs(y_pred - y_actual[prop]))
                results[prop] = {
                    'mae': float(mae),
                    'predictions': y_pred,
                    'actual': y_actual[prop]
                }
    else:
        # Old single-task models
        for prop in ['points', 'rebounds', 'assists', 'threes', 'minutes']:
            if prop in models and models[prop] is not None:
                model = models[prop]
                y_pred = model.predict(X_test)
                mae = np.mean(np.abs(y_pred - y_actual[prop]))
                results[prop] = {
                    'mae': float(mae),
                    'predictions': y_pred,
                    'actual': y_actual[prop]
                }

    return results


def main():
    print("="*70)
    print("BACKTEST: 2024-2025 SEASON")
    print("="*70)
    print("Training data: 1947-2023 (excluding 2024-2025)")
    print("Test data: 2024-2025 complete season")
    print("="*70)

    # Load ONLY 2024-2025 data (memory efficient)
    print("\n[*] Loading data...")

    # Memory-efficient: Read Parquet in batches and filter
    print("- Reading aggregated_nba_data.parquet in batches...")
    parquet_file = pq.ParquetFile("aggregated_nba_data.parquet")

    # Find year column
    test_batch = parquet_file.read_row_group(0, use_threads=True).to_pandas()
    year_col = None
    for col_name in ['season', 'game_year', 'season_end_year', 'year']:
        if col_name in test_batch.columns:
            year_col = col_name
            break

    if not year_col:
        raise ValueError("No year column found in dataset")

    print(f"- Using year column: {year_col}")
    print(f"- Filtering for {year_col} == 2025 (2024-2025 season)")

    # Read row groups and filter
    filtered_chunks = []
    for i in range(parquet_file.num_row_groups):
        batch = parquet_file.read_row_group(i, use_threads=True).to_pandas()
        # Filter to 2024-2025 season only
        batch_filtered = batch[batch[year_col] == 2025]
        if len(batch_filtered) > 0:
            filtered_chunks.append(batch_filtered)

        if (i + 1) % 5 == 0:
            print(f"  Processed {i+1}/{parquet_file.num_row_groups} row groups...")

    # Combine filtered chunks
    agg_df = pd.concat(filtered_chunks, ignore_index=True)
    print(f"- Loaded {len(agg_df):,} rows for 2024-2025 season")
    print(f"- Memory: {agg_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    # Split into train and test
    year_col = get_year_column(agg_df)

    # Test set: 2024-2025 season only
    test_df = agg_df[agg_df[year_col] == 2025].copy()  # 2024-2025 season

    print(f"\n[*] Data split:")
    print(f"  Test (2024-2025): {len(test_df):,} player-games")

    if len(test_df) == 0:
        print("\n[!] No 2024-2025 data found!")
        print("Current dataset ends at:", agg_df[year_col].max())
        return

    # Load models (excluding any with 2024+ data)
    print(f"\n[*] Loading models...")
    all_models = load_trained_models("model_cache")

    if not all_models:
        print("\n[!] No valid models found (all contain 2024+ data)")
        print("You need to train models that end before 2024")
        return

    # Backtest each window
    print(f"\n{'='*70}")
    print(f"BACKTESTING ON 2024-2025 SEASON")
    print(f"{'='*70}\n")

    all_results = {}

    for window_name, window_models in all_models.items():
        print(f"Testing window {window_name}...")

        try:
            results = backtest_window(window_models, test_df, verbose=True)
            all_results[window_name] = results

            # Print results
            for prop, metrics in results.items():
                print(f"  {prop}: MAE = {metrics['mae']:.3f}")

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            all_results[window_name] = {'error': str(e)}

    # META-LEARNER ENSEMBLE (replacing simple averaging)
    print(f"\n{'='*70}")
    print(f"META-LEARNER ENSEMBLE (Context-Aware Stacking)")
    print(f"{'='*70}\n")

    from meta_learner_ensemble import ContextAwareMetaLearner, extract_player_context

    # Collect all window predictions
    ensemble_predictions = {
        'points': [],
        'rebounds': [],
        'assists': [],
        'threes': [],
        'minutes': []
    }

    for window_name, results in all_results.items():
        if 'error' not in results:
            for prop in ensemble_predictions.keys():
                if prop in results:
                    ensemble_predictions[prop].append(results[prop]['predictions'])

    # Actual values
    y_actual = {
        'points': test_df['points'].fillna(0).values,
        'rebounds': test_df['reboundsTotal'].fillna(0).values,
        'assists': test_df['assists'].fillna(0).values,
        'threes': test_df['threePointersMade'].fillna(0).values,
        'minutes': test_df['numMinutes'].fillna(0).values
    }

    # Extract player context
    player_context = extract_player_context(test_df)

    # Train meta-learner for each prop
    meta_learner = ContextAwareMetaLearner(n_windows=len(all_results))
    meta_results = {}

    for prop, pred_list in ensemble_predictions.items():
        if pred_list:
            # Stack predictions: (n_samples, n_windows)
            X_base = np.column_stack(pred_list)

            # Train meta-learner with OOF predictions
            metrics = meta_learner.fit_oof(
                window_predictions=X_base,
                y_true=y_actual[prop],
                player_context=player_context,
                prop_name=prop
            )

            meta_results[prop] = metrics

    # Save meta-learner
    meta_learner.save('meta_learner_2024_2025.pkl')
    print(f"\n[OK] Saved meta-learner: meta_learner_2024_2025.pkl")

    # Save results
    output_path = Path("backtest_results_2024_2025.json")

    # Convert results to JSON-serializable format
    json_results = {}
    for window_name, results in all_results.items():
        if 'error' in results:
            json_results[window_name] = results
        else:
            json_results[window_name] = {
                prop: {'mae': metrics['mae']}
                for prop, metrics in results.items()
            }

    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\n[OK] Results saved to: {output_path}")

    print(f"\n{'='*70}")
    print("BACKTEST COMPLETE")
    print(f"{'='*70}")
    print(f"Windows tested: {len([r for r in all_results.values() if 'error' not in r])}")
    print(f"Test samples: {len(test_df):,}")
    print(f"Results saved: {output_path}")


if __name__ == '__main__':
    main()
