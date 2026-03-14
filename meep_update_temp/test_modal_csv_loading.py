#!/usr/bin/env python
"""Test CSV loading before running on Modal"""
import sys
sys.path.insert(0, ".")

from shared.data_loading import load_player_data, get_year_column, get_season_range

# Simulate Modal's CSV directory path
csv_dir = "C:/Users/tmiles11/.cache/kagglehub/datasets/eoinamoore/historical-nba-data-and-player-box-scores/versions/257"

print("="*70)
print("TESTING CSV AGGREGATION (MODAL SIMULATION)")
print("="*70)
print(f"CSV directory: {csv_dir}")
print()

try:
    # Test loading CSVs
    print("Loading player data from CSV directory...")
    df = load_player_data(csv_dir, verbose=True)

    # Get season range
    year_col = get_year_column(df)
    min_yr, max_yr = get_season_range(df)

    print()
    print("="*70)
    print("SUCCESS!")
    print("="*70)
    print(f"Loaded: {len(df):,} rows")
    print(f"Columns: {len(df.columns)}")
    print(f"Year column: {year_col}")
    print(f"Season range: {min_yr}-{max_yr}")
    print()

    # Show merge success rates
    print("Column prefixes (showing what merged successfully):")
    prefixes = set()
    for col in df.columns:
        if '_' in col:
            prefix = col.split('_')[0]
            prefixes.add(prefix)

    for prefix in sorted(prefixes):
        cols = [c for c in df.columns if c.startswith(prefix + '_')]
        print(f"  {prefix}_*: {len(cols)} columns")

    print()
    print("Sample columns:")
    for col in list(df.columns)[:20]:
        print(f"  - {col}")

except Exception as e:
    print()
    print("="*70)
    print("ERROR!")
    print("="*70)
    print(f"{type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
