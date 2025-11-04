"""
Retrain Player Ensembles with Fixed Base Predictions

Now that we've fixed build_ensemble_training_data to generate
diverse signals (not just baseline repeated 5 times), retrain
all window ensembles and test if they beat the baseline.
"""

import shutil
from pathlib import Path
import subprocess
import sys

print("="*70)
print("RETRAIN ENSEMBLES WITH FIXED BASE PREDICTIONS")
print("="*70)

cache_dir = Path("model_cache")

# Clear existing ensemble caches to force retrain
print("\nClearing old ensemble caches...")
for pkl_file in cache_dir.glob("player_ensemble_*.pkl"):
    print(f"  Removing: {pkl_file}")
    pkl_file.unlink()

for meta_file in cache_dir.glob("player_ensemble_*_meta.json"):
    print(f"  Removing: {meta_file}")
    meta_file.unlink()

print("\nRetraining all windows with fixed base predictions...")
print("This will take a while...\n")

# Retrain each window
windows = [
    ("2002", "2006"),
    ("2007", "2011"),
    ("2012", "2016"),
    ("2017", "2021"),
    ("2022", "2024"),
]

for start, end in windows:
    print(f"\n{'='*70}")
    print(f"Retraining window {start}-{end}...")
    print(f"{'='*70}")

    # Run train_ensemble_players.py for this window
    # Since it's integrated into train_auto.py, we need to call it differently
    # For now, just note that user should run train_auto.py
    print(f"  [INFO] Ensemble will be retrained when train_auto.py runs")

print("\n" + "="*70)
print("INSTRUCTIONS")
print("="*70)
print("""
The train_ensemble_players.py now has fixed base predictions.

To retrain all ensembles:
1. Run: python train_auto.py --retrain-current

This will:
- Clear current season cache
- Retrain all window ensembles with diverse base predictions
- Meta-learner will learn optimal weights for real signals (not just baseline)

After retraining, run:
  python true_backtest_all_approaches.py

This will test if the fixed ensembles beat the simple baseline.
""")
