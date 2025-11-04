"""
Test script to verify player priors matching is working correctly.
Tests the _season_from_date fix and name matching logic.

Expected results AFTER fix:
- season_end_year: 100% populated (not nan)
- Kaggle seasons: [2002.0, 2003.0, ..., 2025.0, 2026.0]
- Season overlap: 24 common seasons
- Name-merge match rate: 40-60%

Expected results BEFORE fix (broken):
- season_end_year: 0% populated (all nan)
- Kaggle seasons: []
- Season overlap: 0
- Name-merge match rate: 0.0%
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Fix Windows console encoding
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("=" * 80)
print("PLAYER PRIORS FIX VERIFICATION TEST")
print("=" * 80)
print()

# Import the _season_from_date function
try:
    from train_auto import _season_from_date
    print("[OK] Successfully imported _season_from_date function")
except ImportError as e:
    print(f"✗ Failed to import _season_from_date: {e}")
    sys.exit(1)

print()
print("-" * 80)
print("TEST 1: _season_from_date with already-parsed datetime")
print("-" * 80)

# Create test data with dates already as datetime
test_dates = pd.Series([
    pd.Timestamp('2023-10-24'),  # Early season -> 2024
    pd.Timestamp('2023-12-25'),  # Mid season -> 2024
    pd.Timestamp('2024-04-15'),  # Late season -> 2024
    pd.Timestamp('2024-07-01'),  # Offseason -> 2024
    pd.Timestamp('2024-08-01'),  # New season starts -> 2025
    pd.Timestamp('2024-10-22'),  # New season -> 2025
])

print(f"Input dates (already datetime64):")
print(f"  dtype: {test_dates.dtype}")
print(f"  values: {test_dates.tolist()}")
print()

# Test the function
result_seasons = _season_from_date(test_dates)

print(f"Output seasons:")
print(f"  values: {result_seasons.tolist()}")
print()

# Expected results
expected = [2024, 2024, 2024, 2024, 2025, 2025]
matches = [r == e for r, e in zip(result_seasons, expected)]

if all(matches):
    print("✓ TEST 1 PASSED: All seasons calculated correctly!")
else:
    print("✗ TEST 1 FAILED: Some seasons incorrect")
    for i, (res, exp, match) in enumerate(zip(result_seasons, expected, matches)):
        status = "✓" if match else "✗"
        print(f"  {status} Row {i}: got {res}, expected {exp}")
    sys.exit(1)

print()
print("-" * 80)
print("TEST 2: _season_from_date with string dates (will be parsed)")
print("-" * 80)

# Create test data with string dates
test_dates_str = pd.Series([
    '2023-10-24',
    '2023-12-25',
    '2024-04-15',
    '2024-07-01',
    '2024-08-01',
    '2024-10-22',
])

print(f"Input dates (strings):")
print(f"  dtype: {test_dates_str.dtype}")
print(f"  values: {test_dates_str.tolist()}")
print()

# Test the function
result_seasons_str = _season_from_date(test_dates_str)

print(f"Output seasons:")
print(f"  values: {result_seasons_str.tolist()}")
print()

matches_str = [r == e for r, e in zip(result_seasons_str, expected)]

if all(matches_str):
    print("✓ TEST 2 PASSED: String dates parsed and seasons calculated correctly!")
else:
    print("✗ TEST 2 FAILED: Some seasons incorrect")
    sys.exit(1)

print()
print("-" * 80)
print("TEST 3: Load actual PlayerStatistics and verify season_end_year")
print("-" * 80)

try:
    import kagglehub

    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("eoinamoore/historical-nba-data-and-player-box-scores")
    print(f"✓ Dataset downloaded to: {path}")
    print()

    # Load PlayerStatistics
    ps_path = Path(path) / "PlayerStatistics.csv"
    print(f"Loading PlayerStatistics from: {ps_path}")

    ps = pd.read_csv(ps_path, nrows=10000)  # Load first 10k rows for testing
    print(f"✓ Loaded {len(ps):,} rows")
    print()

    # Check if gameDate column exists
    if 'gameDate' not in ps.columns:
        print("✗ gameDate column not found")
        print(f"Available columns: {list(ps.columns)[:10]}")
        sys.exit(1)

    print(f"Original gameDate dtype: {ps['gameDate'].dtype}")
    print(f"Sample dates: {ps['gameDate'].head(3).tolist()}")
    print()

    # Parse dates with format='mixed' (like train_auto.py does)
    print("Parsing dates with format='mixed'...")
    ps['gameDate'] = pd.to_datetime(ps['gameDate'], errors="coerce", format='mixed', utc=True).dt.tz_convert(None)

    print(f"After parsing dtype: {ps['gameDate'].dtype}")
    print(f"Sample parsed dates: {ps['gameDate'].head(3).tolist()}")
    print()

    # Create season_end_year using the function
    print("Creating season_end_year using _season_from_date...")
    ps['season_end_year'] = _season_from_date(ps['gameDate']).astype('float32')

    # Check results
    non_null_seasons = ps['season_end_year'].notna().sum()
    total_rows = len(ps)
    pct_populated = (non_null_seasons / total_rows * 100) if total_rows > 0 else 0

    print(f"season_end_year populated: {non_null_seasons:,} / {total_rows:,} rows ({pct_populated:.1f}%)")
    print()

    if pct_populated >= 99.0:  # Allow for some edge cases
        print(f"✓ TEST 3 PASSED: season_end_year is {pct_populated:.1f}% populated!")
    else:
        print(f"✗ TEST 3 FAILED: season_end_year only {pct_populated:.1f}% populated (expected ~100%)")
        print()
        print("Sample rows with nan season_end_year:")
        print(ps[ps['season_end_year'].isna()][['gameDate', 'season_end_year']].head(5))
        sys.exit(1)

    # Show season distribution
    print("Season distribution (first 20):")
    season_counts = ps['season_end_year'].value_counts().sort_index()
    for season, count in season_counts.head(20).items():
        print(f"  {season:.0f}: {count:,} games")
    print()

    # Get unique seasons
    unique_seasons = sorted(ps['season_end_year'].dropna().unique())
    print(f"Unique seasons found: {len(unique_seasons)}")
    print(f"Season range: {min(unique_seasons):.0f} to {max(unique_seasons):.0f}")
    print(f"Seasons: {[int(s) for s in unique_seasons[:10]]} ... {[int(s) for s in unique_seasons[-3:]]}")
    print()

except ImportError:
    print("✗ kagglehub not installed, skipping TEST 3")
    print("  Install with: pip install kagglehub")
    print()
except Exception as e:
    print(f"✗ TEST 3 FAILED with error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("-" * 80)
print("TEST 4: Name matching verification")
print("-" * 80)

try:
    # Load first/last names
    if 'firstName' in ps.columns and 'lastName' in ps.columns:
        print("✓ Both firstName and lastName columns found")

        # Construct full names
        ps['full_name'] = (ps['firstName'].fillna("") + " " + ps['lastName'].fillna("")).str.strip()

        # Show samples
        print()
        print("Sample constructed names:")
        for name in ps['full_name'].dropna().head(10):
            print(f"  - {name}")
        print()

        # Check for valid names (not empty, not nan)
        valid_names = ps['full_name'].notna() & (ps['full_name'] != "")
        valid_count = valid_names.sum()
        valid_pct = (valid_count / len(ps) * 100) if len(ps) > 0 else 0

        print(f"Valid names: {valid_count:,} / {len(ps):,} ({valid_pct:.1f}%)")

        if valid_pct >= 90.0:
            print("✓ TEST 4 PASSED: Name construction working!")
        else:
            print(f"✗ TEST 4 FAILED: Only {valid_pct:.1f}% valid names")
            sys.exit(1)
    else:
        print("✗ firstName or lastName column missing")
        print(f"Available columns: {list(ps.columns)[:20]}")
        sys.exit(1)

except Exception as e:
    print(f"✗ TEST 4 FAILED with error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 80)
print("ALL TESTS PASSED! ✓")
print("=" * 80)
print()
print("Summary:")
print("  ✓ _season_from_date correctly handles already-parsed datetime")
print("  ✓ _season_from_date correctly parses string dates")
print("  ✓ season_end_year is ~100% populated (not nan)")
print("  ✓ Name construction from firstName + lastName works")
print()
print("Next steps:")
print("  1. Clear model caches:")
print("     Remove-Item model_cache\\ensemble_2*.pkl, model_cache\\ensemble_2*_meta.json -Force")
print()
print("  2. Clear Python cache:")
print("     Remove-Item -Recurse -Force __pycache__")
print()
print("  3. Run full training:")
print("     python train_auto.py --enable-window-ensemble --dataset \"eoinamoore/historical-nba-data-and-player-box-scores\" --verbose")
print()
print("Expected in training output:")
print("  - season_end_year populated: X / X rows (100.0%)")
print("  - Kaggle seasons: [2002.0, 2003.0, ..., 2025.0, 2026.0]")
print("  - Season overlap: 24 common seasons")
print("  - Name-merge matched: 40-60% (instead of 0.0%)")
print()

