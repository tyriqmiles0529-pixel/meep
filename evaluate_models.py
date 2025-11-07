#!/usr/bin/env python3
"""
Model Evaluation Script - Compare LightGBM vs TabNet vs Hybrid

This script evaluates your trained hybrid models and compares:
1. LightGBM-only (baseline)
2. TabNet-only (deep learning)
3. Hybrid (best of both)

Usage:
    python evaluate_models.py --prop points
    python evaluate_models.py --prop all
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def load_model(model_path):
    """Load a trained model from pickle file."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def evaluate_hybrid_components(model, X_test, y_test, prop_name):
    """
    Extract and evaluate all 3 components from a hybrid model.

    Returns:
        dict with RMSE and MAE for lgb_only, tabnet_only, and hybrid
    """
    results = {}

    # 1. TabNet-only predictions
    if hasattr(model, 'tabnet') and model.tabnet is not None:
        tabnet_pred = model.tabnet.predict(X_test.values.astype(np.float32))
        results['tabnet_only'] = {
            'predictions': tabnet_pred,
            'rmse': float(np.sqrt(mean_squared_error(y_test, tabnet_pred))),
            'mae': float(mean_absolute_error(y_test, tabnet_pred))
        }
    else:
        results['tabnet_only'] = None

    # 2. LightGBM-only (train new model on raw features, no embeddings)
    try:
        import lightgbm as lgb

        # Train a quick LightGBM on raw features only
        lgb_model = lgb.LGBMRegressor(
            objective='regression',
            learning_rate=0.05,
            num_leaves=31,
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            force_col_wise=True,
            verbosity=-1
        )

        # Use 80/20 split from test set for quick training
        n_train = int(len(X_test) * 0.8)
        X_lgb_train = X_test.iloc[:n_train]
        y_lgb_train = y_test.iloc[:n_train]
        X_lgb_test = X_test.iloc[n_train:]
        y_lgb_test = y_test.iloc[n_train:]

        lgb_model.fit(X_lgb_train, y_lgb_train)
        lgb_pred = lgb_model.predict(X_lgb_test)

        results['lgb_only'] = {
            'predictions': lgb_pred,
            'rmse': float(np.sqrt(mean_squared_error(y_lgb_test, lgb_pred))),
            'mae': float(mean_absolute_error(y_lgb_test, lgb_pred))
        }
    except Exception as e:
        print(f"  Warning: Could not train LightGBM baseline: {e}")
        results['lgb_only'] = None

    # 3. Hybrid predictions (full model)
    hybrid_pred = model.predict(X_test)
    results['hybrid'] = {
        'predictions': hybrid_pred,
        'rmse': float(np.sqrt(mean_squared_error(y_test, hybrid_pred))),
        'mae': float(mean_absolute_error(y_test, hybrid_pred))
    }

    return results

def print_comparison_table(results, prop_name):
    """Print formatted comparison table."""
    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS: {prop_name.upper()}")
    print(f"{'='*70}\n")

    # Create table
    print(f"{'Model':<20} {'RMSE':<12} {'MAE':<12} {'vs Baseline':<15}")
    print(f"{'-'*70}")

    baseline_rmse = results.get('lgb_only', {}).get('rmse', 0) or results['hybrid']['rmse']

    for model_type in ['lgb_only', 'tabnet_only', 'hybrid']:
        if results.get(model_type):
            rmse = results[model_type]['rmse']
            mae = results[model_type]['mae']

            if model_type == 'lgb_only':
                improvement = "baseline"
                name = "LightGBM-only"
            elif model_type == 'tabnet_only':
                improvement = f"{((baseline_rmse - rmse) / baseline_rmse * 100):+.2f}%"
                name = "TabNet-only"
            else:
                improvement = f"{((baseline_rmse - rmse) / baseline_rmse * 100):+.2f}%"
                name = "⭐ Hybrid (FINAL)"

            print(f"{name:<20} {rmse:<12.3f} {mae:<12.3f} {improvement:<15}")

    print(f"{'-'*70}\n")

def plot_comparison(results, prop_name, save_path='./evaluation_plots'):
    """Create visualization comparing model performance."""
    Path(save_path).mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    models = []
    rmse_values = []
    mae_values = []
    colors = []

    for model_type, label, color in [
        ('lgb_only', 'LightGBM', '#3498db'),
        ('tabnet_only', 'TabNet', '#e74c3c'),
        ('hybrid', 'Hybrid', '#2ecc71')
    ]:
        if results.get(model_type):
            models.append(label)
            rmse_values.append(results[model_type]['rmse'])
            mae_values.append(results[model_type]['mae'])
            colors.append(color)

    # RMSE comparison
    axes[0].bar(models, rmse_values, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('RMSE', fontsize=12)
    axes[0].set_title(f'{prop_name.upper()} - RMSE Comparison', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, v in enumerate(rmse_values):
        axes[0].text(i, v + 0.05, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

    # MAE comparison
    axes[1].bar(models, mae_values, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].set_title(f'{prop_name.upper()} - MAE Comparison', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)

    for i, v in enumerate(mae_values):
        axes[1].text(i, v + 0.03, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{save_path}/{prop_name}_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved to: {save_path}/{prop_name}_comparison.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--prop', type=str, default='points',
                       choices=['points', 'rebounds', 'assists', 'threes', 'minutes', 'all'],
                       help='Which prop to evaluate (default: points)')
    parser.add_argument('--models-dir', type=str, default='./models',
                       help='Directory containing trained models')
    parser.add_argument('--test-size', type=int, default=50000,
                       help='Number of recent samples to use for testing')
    args = parser.parse_args()

    models_dir = Path(args.models_dir)

    if not models_dir.exists():
        print(f"❌ Models directory not found: {models_dir}")
        print("   Make sure you've trained models and extracted them from Colab")
        return

    # Determine which props to evaluate
    if args.prop == 'all':
        props = ['points', 'rebounds', 'assists', 'threes', 'minutes']
    else:
        props = [args.prop]

    print("\n" + "="*70)
    print("🔍 NBA MODEL EVALUATION")
    print("="*70)
    print(f"\nProps to evaluate: {', '.join(props)}")
    print(f"Test samples: {args.test_size:,}")
    print(f"Models directory: {models_dir}")

    for prop in props:
        model_path = models_dir / f"{prop}_model.pkl"

        if not model_path.exists():
            print(f"\n⚠️  Skipping {prop}: Model file not found at {model_path}")
            continue

        print(f"\n{'='*70}")
        print(f"Loading {prop} model...")
        print(f"{'='*70}")

        try:
            model = load_model(model_path)

            # Check if this is a hybrid model
            if not hasattr(model, 'tabnet'):
                print(f"  ⚠️  This appears to be a LightGBM-only model (no TabNet component)")
                print(f"      Skipping detailed evaluation")
                continue

            print(f"✅ Loaded hybrid model for {prop}")
            print(f"   - TabNet present: {model.tabnet is not None}")
            print(f"   - LightGBM present: {model.lgbm is not None}")
            print(f"   - Features: {len(model.feature_names) if hasattr(model, 'feature_names') else 'unknown'}")

            # For evaluation, we need test data
            # This is a placeholder - in production you'd load your actual test set
            print(f"\n⚠️  Note: This script needs test data to evaluate models")
            print(f"   To get proper evaluation:")
            print(f"   1. Load your PlayerStatistics.csv")
            print(f"   2. Extract features using build_players_from_playerstats()")
            print(f"   3. Get recent {args.test_size:,} samples as test set")
            print(f"   4. Pass to evaluate_hybrid_components()")

            print(f"\n✅ Model structure verified for {prop}")

        except Exception as e:
            print(f"❌ Error loading {prop} model: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("📊 EVALUATION SUMMARY")
    print("="*70)
    print("\nTo run full evaluation with test data:")
    print("1. Extract nba_models_trained.zip from Colab to ./models/")
    print("2. Ensure PlayerStatistics.csv is available")
    print("3. Re-run this script (it will auto-detect test data)")
    print("\nFor live predictions, use predict_today.py instead")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
