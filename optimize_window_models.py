"""
Window Model Redundancy Optimization
Analyze trained V4 meta-learner to identify and remove low-contribution windows.
"""

import modal
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Modal setup
app = modal.App("window-optimization")
model_volume = modal.Volume.from_name("nba-models", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .pip_install([
        "pandas", "numpy", "scikit-learn", "matplotlib", 
        "lightgbm", "shap", "seaborn"
    ])
    .add_local_file("train_meta_learner_v4.py", remote_path="/root/train_meta_learner_v4.py")
)

@app.function(
    image=image,
    cpu=8.0,
    memory=16384,
    volumes={"/models": model_volume}
)
def analyze_window_importance(model_path: str = "/models/meta_learner_v4_all_components.pkl"):
    """
    Analyze trained V4 meta-learner to extract window importance scores.
    """
    import sys
    import numpy as np
    import pandas as pd
    from sklearn.inspection import permutation_importance
    import shap
    
    sys.path.insert(0, "/root")
    from train_meta_learner_v4 import MetaLearnerV4
    
    print("="*70)
    print("WINDOW MODEL IMPORTANCE ANALYSIS")
    print("="*70)
    
    # Load trained V4 meta-learner
    with open(model_path, 'rb') as f:
        meta_learner = pickle.load(f)
    
    print(f"✓ Loaded V4 meta-learner from {model_path}")
    
    # Analyze component structures
    window_importance = {}
    
    # 1. Analyze residual correction component
    if 'residual_correction' in meta_learner.components:
        rc = meta_learner.components['residual_correction']
        print(f"\n[*] Analyzing Residual Correction Component:")
        
        for stat, model in rc.residual_models.items():
            if hasattr(model, 'feature_importances_'):
                # Extract window-specific importance
                # Features are structured: [window1_pred, window2_pred, ..., windowN_pred, error_patterns]
                n_windows = len([f for f in range(len(model.feature_importances_)) 
                               if f < 27])  # Assuming 27 windows max
                
                window_scores = model.feature_importances_[:n_windows]
                
                # Normalize to percentages
                window_scores_pct = (window_scores / window_scores.sum() * 100).round(2)
                
                window_importance[f"{stat}_residual"] = window_scores_pct
                
                print(f"  {stat}: Top 5 windows")
                top_idx = np.argsort(window_scores_pct)[-5:][::-1]
                for idx in top_idx:
                    print(f"    Window {idx+1}: {window_scores_pct[idx]:.1f}% contribution")
    
    # 2. Analyze base meta-learner (if exists)
    if hasattr(meta_learner, 'meta_models'):
        print(f"\n[*] Analyzing Base Meta-Learner Models:")
        
        for stat, model in meta_learner.meta_models.items():
            if hasattr(model, 'feature_importances_'):
                window_scores = model.feature_importances_
                window_scores_pct = (window_scores / window_scores.sum() * 100).round(2)
                
                window_importance[f"{stat}_base"] = window_scores_pct
                
                print(f"  {stat}: Top 5 windows")
                top_idx = np.argsort(window_scores_pct)[-5:][::-1]
                for idx in top_idx:
                    print(f"    Window {idx+1}: {window_scores_pct[idx]:.1f}% contribution")
    
    # 3. Aggregate importance across all stats
    print(f"\n[*] Aggregating Window Importance Across Stats:")
    
    # Collect all window scores
    all_window_scores = {}
    for key, scores in window_importance.items():
        for i, score in enumerate(scores):
            window_name = f"Window_{i+1}"
            if window_name not in all_window_scores:
                all_window_scores[window_name] = []
            all_window_scores[window_name].append(score)
    
    # Calculate average importance per window
    avg_importance = {}
    for window_name, scores in all_window_scores.items():
        avg_importance[window_name] = np.mean(scores)
    
    # Sort by importance
    sorted_windows = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n📊 Window Importance Ranking:")
    for i, (window, importance) in enumerate(sorted_windows):
        print(f"  {i+1:2d}. {window}: {importance:.2f}% average contribution")
    
    # 4. Identify low-contribution windows
    threshold = 2.0  # <2% contribution
    low_importance_windows = [(w, imp) for w, imp in sorted_windows if imp < threshold]
    
    print(f"\n⚠️  Low-Contribution Windows (<{threshold}%):")
    for window, importance in low_importance_windows:
        print(f"  {window}: {importance:.2f}%")
    
    print(f"\n📈 Optimization Summary:")
    print(f"  Total windows analyzed: {len(sorted_windows)}")
    print(f"  Low-contribution windows: {len(low_importance_windows)}")
    print(f"  Potential reduction: {len(low_importance_windows)/len(sorted_windows)*100:.1f}%")
    
    # 5. Create optimization recommendations
    print(f"\n🎯 Optimization Recommendations:")
    
    if len(low_importance_windows) > 0:
        windows_to_remove = [w for w, _ in low_importance_windows]
        print(f"  Remove windows: {windows_to_remove}")
        print(f"  Expected compute reduction: {len(low_importance_windows)/len(sorted_windows)*100:.1f}%")
        
        # Create optimized window list
        all_windows = [f"Window_{i+1}" for i in range(len(sorted_windows))]
        optimized_windows = [w for w in all_windows if w not in windows_to_remove]
        
        print(f"  Optimized window set: {optimized_windows}")
        print(f"  New total: {len(optimized_windows)} windows")
    else:
        print(f"  All windows contribute >{threshold}% - no immediate optimization needed")
    
    # Save results
    results = {
        'window_importance': window_importance,
        'avg_importance': avg_importance,
        'sorted_windows': sorted_windows,
        'low_importance_windows': low_importance_windows,
        'optimization_recommendations': {
            'threshold': threshold,
            'windows_to_remove': [w for w, _ in low_importance_windows],
            'potential_reduction': len(low_importance_windows)/len(sorted_windows)*100
        }
    }
    
    results_file = "/models/window_optimization_analysis.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n✓ Analysis saved to {results_file}")
    
    return results

@app.local_entrypoint()
def main():
    """
    Analyze window importance from trained V4 meta-learner.
    """
    print("="*70)
    print("WINDOW MODEL REDUNDANCY OPTIMIZATION")
    print("="*70)
    print("Analyzing trained V4 meta-learner...")
    
    results = analyze_window_importance.remote()
    
    print("\n" + "="*70)
    print("OPTIMIZATION ANALYSIS COMPLETE")
    print("="*70)
    
    recs = results['optimization_recommendations']
    print(f"Windows to remove: {len(recs['windows_to_remove'])}")
    print(f"Potential reduction: {recs['potential_reduction']:.1f}%")
    
    if recs['windows_to_remove']:
        print(f"\nNext steps:")
        print(f"1. Remove low-contribution windows from training pipeline")
        print(f"2. Retrain optimized meta-learner")
        print(f"3. Validate accuracy is maintained")
        print(f"4. Deploy optimized system")
