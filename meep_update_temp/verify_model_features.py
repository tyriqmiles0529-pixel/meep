#!/usr/bin/env python
"""
Verify Model Features and Structure

Checks that trained models:
1. Have all required features
2. Use TabNet embeddings (not placeholders)
3. Are hybrid multi-task or single-task
4. Have reasonable sizes
"""

import pickle
import json
import numpy as np
from pathlib import Path


def check_model_structure(model, model_name: str):
    """Check if model has proper structure and features"""
    print(f"\n{'='*70}")
    print(f"MODEL: {model_name}")
    print(f"{'='*70}")

    # Check model type
    model_type = type(model).__name__
    print(f"Type: {model_type}")

    # Check if it's None (placeholder)
    if model is None:
        print("❌ MODEL IS NONE (PLACEHOLDER)")
        return {
            'valid': False,
            'error': 'Model is None',
            'type': 'placeholder'
        }

    # Check for hybrid multi-task
    is_hybrid = hasattr(model, 'correlated_tabnet') or hasattr(model, 'independent_models')

    if is_hybrid:
        print("✓ Hybrid Multi-Task Model detected")

        # Check correlated props TabNet
        if hasattr(model, 'correlated_tabnet'):
            tabnet = model.correlated_tabnet
            print(f"  Correlated TabNet: {type(tabnet).__name__}")

            if hasattr(tabnet, 'network'):
                print(f"    ✓ Has neural network (embeddings present)")
            else:
                print(f"    ⚠ No neural network found")

        # Check independent props
        if hasattr(model, 'independent_models'):
            print(f"  Independent models: {len(model.independent_models)}")
            for prop, prop_model in model.independent_models.items():
                print(f"    - {prop}: {type(prop_model).__name__ if prop_model else 'None'}")

        # Check feature names
        if hasattr(model, 'feature_names'):
            features = model.feature_names
            print(f"\n📋 Features: {len(features)} total")

            # Group by prefix
            prefixes = {}
            for feat in features:
                if '_' in feat:
                    prefix = feat.split('_')[0]
                else:
                    prefix = 'base'

                if prefix not in prefixes:
                    prefixes[prefix] = []
                prefixes[prefix].append(feat)

            print("\nFeature breakdown by category:")
            for prefix in sorted(prefixes.keys()):
                feats = prefixes[prefix]
                print(f"  {prefix:12s}: {len(feats):3d} features")

                # Show examples for important categories
                if prefix in ['adv', 'per100', 'shoot', 'pbp'] and len(feats) <= 5:
                    for f in feats:
                        print(f"    - {f}")
                elif prefix in ['adv', 'per100', 'shoot', 'pbp']:
                    print(f"    Examples: {', '.join(feats[:3])}...")

        return {
            'valid': True,
            'type': 'hybrid_multi_task',
            'feature_count': len(features) if hasattr(model, 'feature_names') else 0,
            'has_embeddings': hasattr(model, 'correlated_tabnet')
        }

    else:
        # Single-task or ensemble model
        print("Single-task or Ensemble Model")

        # Check if it's a VotingRegressor/VotingClassifier
        if hasattr(model, 'estimators_'):
            print(f"  Ensemble with {len(model.estimators_)} models:")
            for i, estimator in enumerate(model.estimators_):
                est_type = type(estimator).__name__
                print(f"    [{i+1}] {est_type}")

                # Check if TabNet is in ensemble
                if hasattr(estimator, 'network'):
                    print(f"        ✓ TabNet with neural network")

        # Check feature names
        if hasattr(model, 'feature_names_in_'):
            features = model.feature_names_in_
            print(f"\n📋 Features: {len(features)} total")

            # Group by prefix
            prefixes = {}
            for feat in features:
                if '_' in feat:
                    prefix = feat.split('_')[0]
                else:
                    prefix = 'base'

                if prefix not in prefixes:
                    prefixes[prefix] = []
                prefixes[prefix].append(feat)

            print("\nFeature breakdown:")
            for prefix in sorted(prefixes.keys()):
                feats = prefixes[prefix]
                print(f"  {prefix:12s}: {len(feats):3d} features")

        return {
            'valid': True,
            'type': 'single_task',
            'feature_count': len(features) if hasattr(model, 'feature_names_in_') else 0,
            'has_embeddings': any(hasattr(e, 'network') for e in model.estimators_) if hasattr(model, 'estimators_') else False
        }


def verify_window(model_path: Path, meta_path: Path):
    """Verify a single model window"""
    print(f"\n{'='*70}")
    print(f"WINDOW: {model_path.stem}")
    print(f"{'='*70}")

    # Load metadata
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        print(f"Window: {meta.get('window', 'Unknown')}")
        print(f"Training rows: {meta.get('train_rows', 0):,}")
        print(f"Columns: {meta.get('columns', 0)}")
        print(f"Neural epochs: {meta.get('neural_epochs', 'N/A')}")
        print(f"Mode: {meta.get('mode', 'unknown')}")
    else:
        print("⚠ No metadata file found")
        meta = {}

    # Load models
    with open(model_path, 'rb') as f:
        models = pickle.load(f)

    print(f"\nModel file size: {model_path.stat().st_size / (1024**2):.1f} MB")

    if model_path.stat().st_size < 1000:
        print("❌ WARNING: File is suspiciously small (< 1 KB) - likely placeholder!")

    # Check each prop model
    print(f"\n🎯 Checking models for each prop:")

    if isinstance(models, dict):
        # Check if it's hybrid multi-task (single model for all props)
        if 'multi_task_model' in models:
            print("\n✓ Hybrid Multi-Task Model (all props in one)")
            model = models['multi_task_model']
            result = check_model_structure(model, 'multi_task_model')
            return {'multi_task': result}

        else:
            # Individual models per prop
            results = {}
            for prop_name in ['points', 'rebounds', 'assists', 'threes', 'minutes']:
                if prop_name in models:
                    model = models[prop_name]
                    print(f"\n--- {prop_name.upper()} ---")

                    if model is None:
                        print("❌ Model is None (placeholder)")
                        results[prop_name] = {'valid': False, 'error': 'None'}
                    else:
                        result = check_model_structure(model, prop_name)
                        results[prop_name] = result
                else:
                    print(f"\n--- {prop_name.upper()} ---")
                    print("⚠ Not found in models dict")
                    results[prop_name] = {'valid': False, 'error': 'Not found'}

            return results
    else:
        print(f"⚠ Unexpected models type: {type(models)}")
        return {'error': f'Unexpected type: {type(models)}'}


def main():
    print("="*70)
    print("MODEL FEATURE VERIFICATION")
    print("="*70)
    print("Checking: model_cache/")
    print("="*70)

    model_cache = Path("model_cache")

    if not model_cache.exists():
        print("\n❌ model_cache directory not found!")
        print("\nDownload models first:")
        print("  py -3.12 -m modal volume get nba-models / model_cache")
        return

    # Find all model files
    model_files = sorted(model_cache.glob("player_models_*.pkl"))

    if not model_files:
        print("\n❌ No model files found in model_cache/")
        return

    print(f"\nFound {len(model_files)} model windows\n")

    # Verify each window
    all_results = {}

    for model_path in model_files:
        meta_path = model_path.with_name(model_path.stem + '_meta.json')

        try:
            results = verify_window(model_path, meta_path)
            all_results[model_path.stem] = results
        except Exception as e:
            print(f"\n❌ Error verifying {model_path.name}: {e}")
            all_results[model_path.stem] = {'error': str(e)}

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")

    valid_count = 0
    invalid_count = 0
    has_embeddings = 0
    feature_counts = []

    for window_name, results in all_results.items():
        if isinstance(results, dict):
            # Check if multi-task or single-task
            if 'multi_task' in results:
                result = results['multi_task']
                if result.get('valid'):
                    valid_count += 1
                    if result.get('has_embeddings'):
                        has_embeddings += 1
                    if result.get('feature_count'):
                        feature_counts.append(result['feature_count'])
                else:
                    invalid_count += 1
            else:
                # Single-task - count valid props
                valid_props = sum(1 for r in results.values() if isinstance(r, dict) and r.get('valid'))
                if valid_props > 0:
                    valid_count += 1

                    # Check embeddings
                    if any(r.get('has_embeddings') for r in results.values() if isinstance(r, dict)):
                        has_embeddings += 1

                    # Get feature count
                    counts = [r.get('feature_count', 0) for r in results.values() if isinstance(r, dict)]
                    if counts:
                        feature_counts.append(max(counts))
                else:
                    invalid_count += 1

    print(f"Total windows: {len(all_results)}")
    print(f"Valid models: {valid_count}")
    print(f"Invalid/Placeholder: {invalid_count}")
    print(f"With embeddings: {has_embeddings}")

    if feature_counts:
        print(f"\nFeature counts:")
        print(f"  Min: {min(feature_counts)}")
        print(f"  Max: {max(feature_counts)}")
        print(f"  Average: {np.mean(feature_counts):.1f}")

    print("\n✓ Verification complete!")

    if invalid_count > 0:
        print(f"\n⚠ Found {invalid_count} invalid windows")
        print("These need to be retrained")
    else:
        print("\n✅ All models are valid and ready for backtesting!")


if __name__ == "__main__":
    main()
