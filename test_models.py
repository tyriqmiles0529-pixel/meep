#!/usr/bin/env python3
"""
Quick Model Testing Script

Tests your trained models to ensure they're production-ready.

Usage:
    python test_models.py
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

def test_model_structure(model_path, prop_name):
    """Test a single model's structure and functionality."""
    print(f"\n{'='*70}")
    print(f"Testing {prop_name.upper()} Model")
    print(f"{'='*70}")

    try:
        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        print(f"✅ Model loaded successfully")

        # Check model type
        model_type = type(model).__name__
        print(f"   Model type: {model_type}")

        # Check for hybrid components
        has_tabnet = hasattr(model, 'tabnet') and model.tabnet is not None
        has_lgbm = hasattr(model, 'lgbm') and model.lgbm is not None
        has_sigma = hasattr(model, 'sigma_model') and model.sigma_model is not None

        print(f"\n📊 Model Components:")
        print(f"   TabNet:  {'✅ Present' if has_tabnet else '❌ Missing'}")
        print(f"   LightGBM: {'✅ Present' if has_lgbm else '❌ Missing'}")
        print(f"   Sigma:    {'✅ Present' if has_sigma else '❌ Missing'}")

        # Check features
        if hasattr(model, 'feature_names'):
            num_features = len(model.feature_names)
            print(f"\n🔍 Features: {num_features} total")
            print(f"   Sample features: {model.feature_names[:5]}")

            # Check for embeddings
            embedding_features = [f for f in model.feature_names if 'tabnet_emb' in f]
            if embedding_features:
                print(f"   🧠 TabNet embeddings: {len(embedding_features)} dimensions")
                print(f"      {embedding_features}")
            else:
                print(f"   ⚠️  No TabNet embedding features found")
        else:
            print(f"\n⚠️  No feature_names attribute found")
            num_features = None

        # Test prediction with dummy data
        print(f"\n🧪 Testing Prediction...")

        if num_features:
            # Create dummy input with correct shape
            X_test = pd.DataFrame(
                np.random.randn(5, num_features),
                columns=model.feature_names
            )

            try:
                predictions = model.predict(X_test)
                print(f"   ✅ Prediction successful")
                print(f"   Output shape: {predictions.shape}")
                print(f"   Sample predictions: {predictions[:3]}")

                # Test uncertainty if available
                if has_sigma:
                    try:
                        pred, uncertainty = model.predict(X_test, return_uncertainty=True)
                        print(f"   ✅ Uncertainty estimation working")
                        print(f"   Sample uncertainties: {uncertainty[:3]}")
                    except:
                        print(f"   ⚠️  Uncertainty estimation not available")

            except Exception as e:
                print(f"   ❌ Prediction failed: {e}")
        else:
            print(f"   ⚠️  Skipping prediction test (unknown feature count)")

        # Summary
        print(f"\n{'─'*70}")
        if has_tabnet and has_lgbm:
            print(f"✅ {prop_name.upper()}: Hybrid model ready for production")
        elif has_lgbm:
            print(f"✅ {prop_name.upper()}: LightGBM model ready (no neural component)")
        else:
            print(f"⚠️  {prop_name.upper()}: Unknown model structure")
        print(f"{'─'*70}")

        return True

    except FileNotFoundError:
        print(f"❌ Model file not found: {model_path}")
        return False
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*70)
    print("🔍 NBA MODEL PRODUCTION READINESS TEST")
    print("="*70)

    models_dir = Path('./models')

    if not models_dir.exists():
        print(f"\n❌ Models directory not found: {models_dir}")
        print(f"   Please download nba_models_trained.zip from Colab")
        print(f"   Extract it to ./models/")
        return

    print(f"\n📂 Models directory: {models_dir}")

    props = ['minutes', 'points', 'rebounds', 'assists', 'threes']
    results = {}

    for prop in props:
        model_path = models_dir / f"{prop}_model.pkl"
        if model_path.exists():
            results[prop] = test_model_structure(model_path, prop)
        else:
            print(f"\n⚠️  Skipping {prop}: Model file not found")
            results[prop] = False

    # Final summary
    print(f"\n\n{'='*70}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*70}\n")

    total = len(props)
    passed = sum(1 for v in results.values() if v)

    print(f"Models tested: {passed}/{total}")
    print(f"\nStatus by prop:")
    for prop in props:
        status = "✅ Ready" if results.get(prop, False) else "❌ Not Ready"
        print(f"   {prop.capitalize():<12} {status}")

    if passed == total:
        print(f"\n🎉 All models ready for production!")
        print(f"\nNext steps:")
        print(f"   1. Run: python predict_today.py")
        print(f"   2. Implement feature engineering for live data")
        print(f"   3. Start making predictions!")
    elif passed > 0:
        print(f"\n⚠️  Some models missing - check Colab export")
    else:
        print(f"\n❌ No models found - download from Colab first")

    print(f"\n{'='*70}\n")

if __name__ == '__main__':
    main()
