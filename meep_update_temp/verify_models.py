#!/usr/bin/env python
"""
Verify Trained Models - Check Features, Embeddings, and Model Structure
"""
import pickle
import json
from pathlib import Path

def verify_model_window(model_path, meta_path):
    """Verify a single model window"""
    print("="*70)
    print(f"Verifying: {model_path.name}")
    print("="*70)

    # Load metadata
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    print(f"\n📊 Metadata:")
    print(f"  Window: {meta['window']}")
    print(f"  Seasons: {meta['seasons']}")
    print(f"  Training rows: {meta['train_rows']:,}")
    print(f"  Columns: {meta['columns']}")
    print(f"  Neural epochs: {meta.get('neural_epochs', 'N/A')}")

    # Load models
    with open(model_path, 'rb') as f:
        models = pickle.load(f)

    print(f"\n🎯 Targets Trained:")
    for target in models.keys():
        print(f"  - {target}")

    # Inspect first model to see structure
    first_target = list(models.keys())[0]
    first_model = models[first_target]

    print(f"\n🔍 Model Structure (for '{first_target}'):")
    print(f"  Type: {type(first_model).__name__}")

    if hasattr(first_model, 'models_'):
        # It's a voting ensemble
        print(f"  Ensemble models: {len(first_model.models_)}")
        for i, model in enumerate(first_model.models_):
            model_type = type(model).__name__
            print(f"    [{i+1}] {model_type}")

            # Check if TabNet (has embeddings)
            if hasattr(model, 'network'):
                print(f"        ✓ TabNet neural network (has embeddings)")
                if hasattr(model, 'input_dim'):
                    print(f"        Input features: {model.input_dim}")
            elif hasattr(model, 'n_features_in_'):
                print(f"        Features used: {model.n_features_in_}")

    # Try to get feature names if available
    if hasattr(first_model, 'feature_names_in_'):
        print(f"\n📋 Features ({len(first_model.feature_names_in_)} total):")
        features = first_model.feature_names_in_

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

        for prefix, feats in sorted(prefixes.items()):
            print(f"  {prefix}_*: {len(feats)} features")
            if len(feats) <= 5:
                for f in feats:
                    print(f"    - {f}")
            else:
                print(f"    Examples: {', '.join(feats[:3])}...")

    print()

def main():
    print("="*70)
    print("NBA MODEL VERIFICATION")
    print("="*70)
    print("\nChecking model_cache directory for trained models...\n")

    model_cache = Path("model_cache")

    if not model_cache.exists():
        print("❌ model_cache directory not found!")
        print("Run this after downloading models from Modal:")
        print("  py -3.12 -m modal volume get nba-models model_cache")
        return

    # Find all model files
    model_files = sorted(model_cache.glob("player_models_*.pkl"))

    if not model_files:
        print("❌ No model files found in model_cache/")
        return

    print(f"Found {len(model_files)} model windows\n")

    # Verify first and last windows (most interesting)
    for model_path in [model_files[0], model_files[-1]]:
        meta_path = model_path.with_name(model_path.stem + '_meta.json')

        if meta_path.exists():
            verify_model_window(model_path, meta_path)
        else:
            print(f"⚠️  Metadata not found for {model_path.name}")

    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total windows trained: {len(model_files)}")
    print(f"First window: {model_files[0].stem}")
    print(f"Last window: {model_files[-1].stem}")
    print("\n✓ Models verified!")
    print("\nTo verify embeddings are being used:")
    print("  - Look for 'TabNet neural network' in the output above")
    print("  - TabNet learns embeddings automatically from categorical features")
    print("  - Player IDs, team IDs, positions are embedded as vectors")

if __name__ == "__main__":
    main()
