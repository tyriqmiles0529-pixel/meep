"""
Retrain Current Window EXCLUDING 2025 Season

Simple approach:
1. Clear existing 2022-2026 cache
2. Retrain using only 2022-2024 data (exclude 2025)
3. Test on 2025 holdout
"""

import os
from pathlib import Path

cache_dir = Path("model_cache")

print("="*70)
print("RETRAIN CURRENT WINDOW (EXCLUDE 2025 FOR HOLDOUT)")
print("="*70)

# Files to remove
files_to_remove = [
    cache_dir / "player_ensemble_2022_2026.pkl",
    cache_dir / "player_ensemble_2022_2026_meta.json"
]

print("\nStep 1: Clearing existing 2022-2026 cache...")
for file in files_to_remove:
    if file.exists():
        file.unlink()
        print(f"  Removed: {file}")

print("\n✅ Cache cleared!")
print("\nStep 2: Run training with 2025 exclusion...")
print("\nCommands to run:")
print("  1. Modify train_ensemble_players.py to exclude season_end_year == 2025")
print("  2. Run: python train_ensemble_players.py --verbose")
print("  3. Run: python backtest_current_season.py (will test on 2025 holdout)")
print("\nOr just run the true_holdout_test.py script which does all of this:")
print("  python true_holdout_test.py")
