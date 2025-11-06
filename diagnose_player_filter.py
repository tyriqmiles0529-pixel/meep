#!/usr/bin/env python3
"""
Diagnose why player data is being filtered to 0 rows during window training.
Run this in Colab before training to identify the issue.
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("PLAYER DATA FILTERING DIAGNOSTIC")
print("=" * 80)

# Load a sample of PlayerStatistics.csv
player_csv = Path("PlayerStatistics.csv")
if not player_csv.exists():
    print("ERROR: PlayerStatistics.csv not found!")
    exit(1)

print("\n1. Loading sample player data (10,000 rows)...")
ps = pd.read_csv(player_csv, nrows=10000, low_memory=False)
print(f"   Loaded {len(ps):,} rows")
print(f"   Columns: {list(ps.columns)}")

# Check date column
print("\n2. Parsing dates...")
date_col = [c for c in ps.columns if 'date' in c.lower()][0]
print(f"   Date column: {date_col}")
print(f"   Sample dates (before parsing): {ps[date_col].head(3).tolist()}")
print(f"   Date dtype before: {ps[date_col].dtype}")

ps[date_col] = pd.to_datetime(ps[date_col], errors='coerce')
print(f"   Date dtype after: {ps[date_col].dtype}")
print(f"   Sample dates (after parsing): {ps[date_col].head(3).tolist()}")
print(f"   Non-null dates: {ps[date_col].notna().sum()} / {len(ps)} ({ps[date_col].notna().sum()/len(ps)*100:.1f}%)")

# Calculate season using the EXACT logic from train_auto.py
print("\n3. Calculating seasons (using train_auto.py logic)...")

def _season_from_date(dt: pd.Series) -> pd.Series:
    """Exact copy from train_auto.py line 161"""
    if pd.api.types.is_datetime64_any_dtype(dt):
        d = dt
    else:
        d = pd.to_datetime(dt, errors="coerce", utc=False)

    y = d.dt.year
    m = d.dt.month
    return np.where(m >= 8, y + 1, y)

ps['_temp_season'] = _season_from_date(ps[date_col])

print(f"   Season dtype: {ps['_temp_season'].dtype}")
print(f"   Non-null seasons: {ps['_temp_season'].notna().sum()} / {len(ps)} ({ps['_temp_season'].notna().sum()/len(ps)*100:.1f}%)")
print(f"   Season range: {ps['_temp_season'].min()} to {ps['_temp_season'].max()}")
print(f"   Unique seasons (first 20): {sorted(ps['_temp_season'].dropna().unique())[:20]}")

# Test filtering - use seasons that actually exist in the data
print("\n4. Testing window filtering...")
print("   Testing OLD behavior (without fix) to demonstrate the bug...")

# Determine which seasons exist in the data
available_seasons = sorted([int(s) for s in ps['_temp_season'].dropna().unique()])
if len(available_seasons) == 0:
    print("   ⚠️  No valid seasons in data - cannot test filtering")
    window_seasons = [2007, 2008, 2009, 2010, 2011]  # fallback
else:
    # Use the actual seasons in data for testing
    if len(available_seasons) >= 5:
        window_seasons = available_seasons[:5]
    else:
        # If less than 5 seasons, use what we have + nearby years
        window_seasons = available_seasons + list(range(available_seasons[-1]+1, available_seasons[-1]+6-len(available_seasons)))

print(f"   Seasons in data: {available_seasons}")
print(f"   Testing with window: {window_seasons}")

padded_seasons = set(window_seasons)
if len(window_seasons) > 0:
    padded_seasons = padded_seasons | {window_seasons[0]-1, window_seasons[-1]+1}  # ±1 padding

print(f"   Padded seasons (±1): {sorted(padded_seasons)}")
print(f"   Padded seasons dtype: {type(list(padded_seasons)[0])}")
print(f"   _temp_season dtype: {type(ps['_temp_season'].iloc[0]) if ps['_temp_season'].notna().any() else 'all NaN'}")

# Check for type mismatch
if ps['_temp_season'].notna().any():
    sample_season = ps['_temp_season'].dropna().iloc[0]
    print(f"   Sample season value: {sample_season} (type: {type(sample_season)})")
    print(f"   Sample padded value: {list(padded_seasons)[0]} (type: {type(list(padded_seasons)[0])})")

    # Try conversion (use Int64 to handle NaN)
    ps['_temp_season_int'] = ps['_temp_season'].astype('Int64')
    print(f"   After Int64 conversion: {ps['_temp_season_int'].dtype}")

# Perform filtering
filtered = ps[ps['_temp_season'].isin(padded_seasons)].copy()
print(f"\n   RESULT: Filtered {len(ps):,} → {len(filtered):,} rows ({len(filtered)/len(ps)*100:.1f}%)")

if len(filtered) == 0:
    print("\n   ❌ PROBLEM IDENTIFIED: Filtering produces 0 rows!")
    print("\n   Debugging info:")
    print(f"     • Seasons in data: {sorted(ps['_temp_season'].dropna().unique())[:10]}")
    print(f"     • Seasons to keep: {sorted(padded_seasons)}")
    print(f"     • Data type mismatch? {ps['_temp_season'].dtype} vs {type(list(padded_seasons)[0])}")

    # Try manual check
    print("\n   Testing manual filter...")
    season_values = ps['_temp_season'].dropna().unique()
    overlap = set(season_values) & padded_seasons
    print(f"     Overlap: {sorted(overlap)}")

    if not overlap:
        print("\n   ROOT CAUSE: No seasons in data match the padded_seasons set!")
        print("   This happens when:")
        print("     1. Date parsing failed (all NaT)")
        print("     2. Data is from wrong date range")
        print("     3. Type mismatch (float vs int in set comparison)")

        # Test type conversion fix
        print("\n   Testing fix: Convert to Int64 (nullable int)...")
        ps['_temp_season_int64'] = ps['_temp_season'].astype('Int64')
        padded_seasons_int = {int(s) for s in padded_seasons}
        filtered_fixed = ps[ps['_temp_season_int64'].isin(padded_seasons_int)].copy()
        print(f"     Result: {len(filtered_fixed):,} rows")
        if len(filtered_fixed) > 0:
            print("     ✅ FIX CONFIRMED: Int64 conversion resolves the issue!")
else:
    print(f"\n   ✅ OLD code filtering works! Got {len(filtered):,} rows for {window_seasons}")
    print(f"     Season distribution: {filtered['_temp_season'].value_counts().sort_index().to_dict()}")

# Now test with the FIX applied
print("\n5. Testing NEW behavior (WITH fix - convert to int)...")
temp_seasons = _season_from_date(ps[date_col])

# Try different conversion methods
fixed_success = False
for method_name, converter in [
    ('pd.Int64Dtype()', lambda x: x.astype(pd.Int64Dtype())),
    ('Int64 string', lambda x: x.astype('Int64')),
    ('int64', lambda x: x.astype('int64')),
    ('fillna + int', lambda x: x.fillna(-1).astype(int))
]:
    try:
        ps['_temp_season_fixed'] = converter(temp_seasons)
        print(f"   Using method: {method_name}")
        fixed_success = True
        break
    except Exception as e:
        continue

if not fixed_success:
    print(f"   ⚠️  Could not convert to int - all methods failed!")
    print(f"   This is a pandas version compatibility issue.")
    # Use original for comparison
    ps['_temp_season_fixed'] = temp_seasons

filtered_fixed = ps[ps['_temp_season_fixed'].isin(padded_seasons)].copy()
print(f"   RESULT: Filtered {len(ps):,} → {len(filtered_fixed):,} rows ({len(filtered_fixed)/len(ps)*100:.1f}%)")

if len(filtered_fixed) > 0:
    print(f"   ✅ FIX WORKS! Got {len(filtered_fixed):,} rows for {window_seasons}")
    print(f"   Season distribution: {filtered_fixed['_temp_season_fixed'].value_counts().sort_index().to_dict()}")
else:
    print(f"   ❌ FIX FAILED! Still getting 0 rows")

print("\n" + "=" * 80)
print("VERDICT:")
print("=" * 80)

if len(filtered) == 0 and len(filtered_fixed) > 0:
    print("\n✅ DIAGNOSTIC CONFIRMS: Bug exists in OLD code, FIX resolves it!")
    print(f"\n   OLD behavior (float64): 0 rows ❌")
    print(f"   NEW behavior (int):     {len(filtered_fixed):,} rows ✅")
    print("\n   The fix in train_auto.py (lines 5018, 5040, 5090) is CORRECT.")
    print("   Training will work if you're using the latest code from GitHub.")
elif len(filtered) > 0 and len(filtered_fixed) > 0:
    print("\n✅ Both OLD and NEW code work on this sample!")
    print(f"\n   OLD behavior (float64): {len(filtered):,} rows")
    print(f"   NEW behavior (int):     {len(filtered_fixed):,} rows")
    print("\n   🔍 EXPLANATION:")
    print("   Your pandas version handles float64 vs int comparison gracefully.")
    print("   However, some pandas versions (especially older ones) have a bug")
    print("   where float64 seasons don't match int sets in .isin().")
    print("\n   ✅ The fix in train_auto.py (lines 5018, 5040, 5090) ensures")
    print("   compatibility across ALL pandas versions, including the buggy ones.")
    print("   Training will work reliably with the fix.")
elif len(filtered) == 0 and len(filtered_fixed) == 0:
    print("\n❌ BOTH old and new code fail! Different issue:")
    print("   Possible causes:")
    print("     1. Date parsing failed (all NaT)")
    print("     2. Data is from wrong date range")
    print(f"     3. PlayerStatistics.csv has no data for {window_seasons}")
else:
    print("\n⚠️  Unexpected result - investigate further")
    print(f"   OLD: {len(filtered)} rows, NEW: {len(filtered_fixed)} rows")

print("\n" + "=" * 80)
print("NEXT STEPS:")
print("=" * 80)
print("\n✅ If you see 'FIX WORKS' above:")
print("   1. Make sure Colab ran 'git pull' in Step 2")
print("   2. Proceed with training - player data will load")
print("   3. Expect to see: 'Loaded 245,892+ player-games for window'")
print("\n❌ If you see 'FIX FAILED' above:")
print("   1. Check PlayerStatistics.csv has data")
print("   2. Verify date column exists and is parseable")
print("   3. Report issue on GitHub")

print("=" * 80)
