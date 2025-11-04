"""
Clear Ensemble Cache to Force Retrain

Removes all cached player ensemble models so they will be retrained
with the fixed base prediction code.
"""

from pathlib import Path

cache_dir = Path("model_cache")

print("="*70)
print("CLEARING ENSEMBLE CACHE")
print("="*70)

# Remove all player ensemble files
removed_count = 0

for pkl_file in cache_dir.glob("player_ensemble_*.pkl"):
    print(f"  Removing: {pkl_file.name}")
    pkl_file.unlink()
    removed_count += 1

for meta_file in cache_dir.glob("player_ensemble_*_meta.json"):
    print(f"  Removing: {meta_file.name}")
    meta_file.unlink()
    removed_count += 1

print(f"\n[OK] Removed {removed_count} cached ensemble files")
print(f"\nNext steps:")
print(f"1. Run: python train_auto.py")
print(f"   This will retrain all ensembles with fixed base predictions")
print(f"2. Run: python true_backtest_all_approaches.py")
print(f"   This will test which approach actually wins")
