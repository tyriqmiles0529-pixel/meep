#!/usr/bin/env python
"""
Remove only the recent windows (2022-2024 and 2025-2027) from Modal cache.

This allows backtesting on 2024-2025 without retraining everything.
"""

import modal

app = modal.App("remove-recent-windows")
model_volume = modal.Volume.from_name("nba-models")


@app.function(volumes={"/models": model_volume})
def remove_windows():
    """Remove only windows that contain 2024 or 2025 data"""
    import os
    import glob

    print("="*70)
    print("REMOVING RECENT WINDOWS FROM CACHE")
    print("="*70)
    print("Target: Windows containing 2024 or 2025 season data")
    print("="*70)

    # Patterns for windows we want to remove
    # Any window that ends in 2024, 2025, or 2026
    years_to_remove = [2022, 2023, 2024, 2025, 2026]

    files_to_remove = []

    for year in years_to_remove:
        # Pattern 1: Windows ending with this year (e.g., 2022-2024)
        patterns = [
            f"/models/player_models_*_{year}.pkl",
            f"/models/player_models_*_{year}_meta.json",
            f"/models/game_models_*_{year}.pkl",
            f"/models/game_models_*_{year}_meta.json",
        ]

        for pattern in patterns:
            files_to_remove.extend(glob.glob(pattern))

    # Remove duplicates
    files_to_remove = list(set(files_to_remove))

    if not files_to_remove:
        print("\n✓ No recent windows found - cache is clean!")
        return {"removed": 0, "files": []}

    print(f"\nFound {len(files_to_remove)} files to remove:")
    for f in files_to_remove:
        print(f"  - {os.path.basename(f)}")

    print(f"\nRemoving files...")
    removed_files = []

    for file_path in files_to_remove:
        try:
            os.remove(file_path)
            removed_files.append(os.path.basename(file_path))
            print(f"  ✓ Removed: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"  ✗ Error removing {os.path.basename(file_path)}: {e}")

    # Commit changes
    model_volume.commit()

    print(f"\n{'='*70}")
    print(f"CLEANUP COMPLETE")
    print(f"{'='*70}")
    print(f"Removed {len(removed_files)} files")
    print(f"Cache is now clean for backtesting on 2024-2025")

    return {
        "removed": len(removed_files),
        "files": removed_files
    }


@app.local_entrypoint()
def main():
    print("="*70)
    print("REMOVE RECENT WINDOWS")
    print("="*70)
    print("This will remove windows containing 2024/2025 data")
    print("Windows to remove:")
    print("  - 2022-2024")
    print("  - 2023-2025")
    print("  - 2024-2026")
    print("  - 2025-2027")
    print("="*70)

    result = remove_windows.remote()

    print(f"\n✅ Removed {result['removed']} files")
    print("\nNow you can run backtesting:")
    print("  python backtest_2024_2025.py")


if __name__ == "__main__":
    pass
