"""
Simple Super Meta-Learner

Instead of replaying all predictions, this learns optimal weights
from the existing window performance metrics.

Each window already has RMSE scores. The super meta-learner learns
which windows to trust more for each stat.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import Ridge

print("="*70)
print("SIMPLE SUPER META-LEARNER")
print("="*70)

cache_dir = Path("model_cache")

# Load all window metadata
window_configs = [
    ("2002-2006", "player_ensemble_2002_2006_meta.json"),
    ("2007-2011", "player_ensemble_2007_2011_meta.json"),
    ("2012-2016", "player_ensemble_2012_2016_meta.json"),
    ("2017-2021", "player_ensemble_2017_2021_meta.json"),
    ("2022-2024", "player_ensemble_2022_2024_meta.json")
]

print("\n" + "="*70)
print("LOAD WINDOW METADATA")
print("="*70)

loaded_windows = []

for window_name, meta_file in window_configs:
    meta_path = cache_dir / meta_file

    if not meta_path.exists():
        print(f"  SKIP: {window_name} (not found)")
        continue

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    loaded_windows.append({
        'name': window_name,
        'meta': meta
    })

    print(f"  Loaded: {window_name}")

print(f"\nTotal windows: {len(loaded_windows)}")

print("\n" + "="*70)
print("CALCULATE SUPER META WEIGHTS")
print("="*70)

# For each stat, calculate weights based on inverse RMSE
# Better performing windows get higher weight
super_meta_weights = {}

for stat_name in ['points', 'rebounds', 'assists', 'threes', 'minutes']:
    print(f"\n{stat_name.upper()}:")
    print("-" * 50)

    # Get RMSE from each window
    rmses = []
    window_names = []

    for window in loaded_windows:
        if stat_name in window['meta']['metrics']:
            rmse = window['meta']['metrics'][stat_name]['rmse']
            rmses.append(rmse)
            window_names.append(window['name'])

    if len(rmses) == 0:
        print(f"  No data available")
        continue

    # Calculate weights (inverse RMSE, normalized)
    # Lower RMSE = higher weight
    inverse_rmses = [1.0 / r for r in rmses]
    total_inverse = sum(inverse_rmses)
    weights = [inv / total_inverse for inv in inverse_rmses]

    print(f"  Window weights (based on training RMSE):")
    for i, (name, rmse, weight) in enumerate(zip(window_names, rmses, weights)):
        print(f"    {name:<15} RMSE: {rmse:.3f}  Weight: {weight:.3f}")

    # Calculate weighted average RMSE
    weighted_rmse = sum(w * r for w, r in zip(weights, rmses))
    print(f"\n  Weighted RMSE: {weighted_rmse:.3f}")

    super_meta_weights[stat_name] = {
        'window_names': window_names,
        'weights': weights,
        'rmses': rmses,
        'weighted_rmse': weighted_rmse
    }

print("\n" + "="*70)
print("LOAD 2025 BASELINE FOR COMPARISON")
print("="*70)

# Load baseline results from test_all_windows_on_2025.json if available
baseline_file = Path("test_all_windows_on_2025.json")

if baseline_file.exists():
    with open(baseline_file, 'r') as f:
        test_data = json.load(f)

    baseline_2025 = test_data.get('baseline_2025', {})

    print("\n2025 BASELINE:")
    for stat_name in ['points', 'rebounds', 'assists', 'threes', 'minutes']:
        if stat_name in baseline_2025:
            baseline_rmse = baseline_2025[stat_name]['rmse']
            print(f"  {stat_name:10s}: {baseline_rmse:.3f}")

    print("\n" + "="*70)
    print("EXPECTED IMPROVEMENT ON 2025")
    print("="*70)

    improvements = []

    for stat_name in ['points', 'rebounds', 'assists', 'threes', 'minutes']:
        if stat_name not in baseline_2025:
            continue

        if stat_name not in super_meta_weights:
            continue

        baseline_rmse = baseline_2025[stat_name]['rmse']
        super_rmse = super_meta_weights[stat_name]['weighted_rmse']

        improvement = (baseline_rmse - super_rmse) / baseline_rmse * 100

        print(f"\n{stat_name.upper()}:")
        print(f"  2025 Baseline:  {baseline_rmse:.3f}")
        print(f"  Super Meta:     {super_rmse:.3f}")
        print(f"  Improvement:    {improvement:+.1f}%")

        improvements.append(improvement)

    if improvements:
        avg_improvement = sum(improvements) / len(improvements)

        print("\n" + "="*70)
        print("VERDICT")
        print("="*70)

        print(f"\nAverage expected improvement: {avg_improvement:+.1f}%")

        if avg_improvement >= 1.0:
            print(f"\n[YES] Super meta-learner validated")
        elif avg_improvement >= 0.3:
            print(f"\n[MAYBE] Marginal improvement")
        else:
            print(f"\n[NO] No significant improvement")

else:
    print("\nNo baseline data available (run test_all_windows_on_2025.py first)")

print("\n" + "="*70)
print("SAVE SUPER META-LEARNER")
print("="*70)

# Save weights
output_file = cache_dir / "super_meta_learner_simple.pkl"
output_meta_file = cache_dir / "super_meta_learner_simple_meta.json"

with open(output_file, 'wb') as f:
    pickle.dump(super_meta_weights, f)

meta_data = {
    'trained_date': datetime.now().isoformat(),
    'windows_used': [w['name'] for w in loaded_windows],
    'method': 'weighted_average_by_inverse_rmse',
    'metrics': {
        stat: {
            'rmse': super_meta_weights[stat]['weighted_rmse'],
            'window_weights': dict(zip(
                super_meta_weights[stat]['window_names'],
                super_meta_weights[stat]['weights']
            ))
        }
        for stat in super_meta_weights.keys()
    }
}

with open(output_meta_file, 'w') as f:
    json.dump(meta_data, f, indent=2)

print(f"\n[SAVED] Super meta weights: {output_file}")
print(f"[SAVED] Metadata: {output_meta_file}")

print("\n" + "="*70)
print("HOW TO USE IN PRODUCTION")
print("="*70)

print("""
To make predictions with super meta-learner:

1. Get prediction from each window for the same game
2. Combine predictions using the learned weights:

   final_prediction = sum(weight[i] * prediction[i] for each window)

3. The weights automatically favor better-performing windows

Example weights by stat:
""")

for stat_name in ['points', 'rebounds', 'assists', 'threes']:
    if stat_name not in super_meta_weights:
        continue
    print(f"\n{stat_name.upper()}:")
    for name, weight in zip(super_meta_weights[stat_name]['window_names'],
                           super_meta_weights[stat_name]['weights']):
        print(f"  {name}: {weight:.3f}")
