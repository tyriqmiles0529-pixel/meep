"""
Retrain current season window with weighted team context.
Quick test to see if weighted context improves performance.
"""

import os
import sys
import pickle
from pathlib import Path

# Clear the current season cache to force retrain
cache_dir = Path("model_cache")

print("="*70)
print("RETRAIN WITH WEIGHTED TEAM CONTEXT")
print("="*70)

# Remove current season cache files
current_cache = cache_dir / "player_ensemble_2022_2026.pkl"
current_meta = cache_dir / "player_ensemble_2022_2026_meta.json"

if current_cache.exists():
    print(f"\nRemoving old cache: {current_cache}")
    current_cache.unlink()

if current_meta.exists():
    print(f"Removing old metadata: {current_meta}")
    current_meta.unlink()

print("\n✅ Cache cleared!")
print("\nNow run:")
print("  python train_ensemble_players_v2.py --verbose")
print("\nThis will retrain the 2022-2026 window with weighted context.")
print("\nThen compare with:")
print("  python compare_ensemble_baseline.py")
