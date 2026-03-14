#!/usr/bin/env python3
"""
Test player data filtering on LOCAL PlayerStatistics.csv
This will show if the fix actually works on your data.
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("LOCAL PLAYER DATA FILTERING TEST")
print("=" * 80)

# Copy _season_from_date from train_auto.py
def _season_from_date(dt):
    """Convert UTC-naive datetime to NBA season end-year."""
    if pd.api.types.is_datetime64_any_dtype(dt):
        d = dt
    else:
        d = pd.to_datetime(dt, errors="coerce", utc=False)
    y = d.dt.year
    m = d.dt.month
    return np.where(m >= 8, y + 1, y)

# Load sample
player_csv = Path("PlayerStatistics.csv")
if not player_csv.exists():
    print("❌ PlayerStatistics.csv not found!")
    exit(1)

print(f"\n1. Loading sample (50,000 rows)...")
ps = pd.read_csv(player_csv, nrows=50000, low_memory=False)
print(f"   Loaded {len(ps):,} rows")

# Find date column
date_col = [c for c in ps.columns if 'date' in c.lower()][0]
print(f"   Date column: {date_col}")

# Parse dates
ps[date_col] = pd.to_datetime(ps[date_col], errors='coerce')
print(f"   Non-null dates: {ps[date_col].notna().sum():,} / {len(ps):,}")

# Test window 2007-2011
window_seasons = [2007, 2008, 2009, 2010, 2011]
start_year = 2007
end_year = 2011
padded_seasons = set(window_seasons) | {start_year-1, end_year+1}

print(f"\n2. Testing window: {window_seasons}")
print(f"   Padded seasons: {sorted(padded_seasons)}")

# OLD METHOD (without fix - float64)
print("\n3. OLD METHOD (float64 - may fail):")
ps['_temp_season_old'] = _season_from_date(ps[date_col])
print(f"   dtype: {ps['_temp_season_old'].dtype}")
filtered_old = ps[ps['_temp_season_old'].isin(padded_seasons)].copy()
print(f"   Result: {len(filtered_old):,} rows")

# NEW METHOD (with fix - int)
print("\n4. NEW METHOD (Int64 - FIXED):")
temp_seasons = pd.Series(_season_from_date(ps[date_col]))
ps['_temp_season_new'] = temp_seasons.fillna(-1).astype(int)
print(f"   dtype: {ps['_temp_season_new'].dtype}")
filtered_new = ps[ps['_temp_season_new'].isin(padded_seasons)].copy()
print(f"   Result: {len(filtered_new):,} rows")

print("\n" + "=" * 80)
print("VERDICT:")
print("=" * 80)

if len(filtered_new) == 0:
    print("\n❌ BOTH methods failed! This means:")
    print("   1. PlayerStatistics.csv has no data for 2007-2011")
    print("   2. OR date column is corrupted/empty")
    print("\n   Checking data range...")
    season_range = ps['_temp_season_new'].replace(-1, np.nan).dropna()
    if len(season_range) > 0:
        print(f"   Available seasons: {int(season_range.min())} to {int(season_range.max())}")
    else:
        print("   ❌ No valid seasons found in data!")
elif len(filtered_old) == 0 and len(filtered_new) > 0:
    print(f"\n✅ FIX CONFIRMED!")
    print(f"   OLD method (float64): 0 rows ❌")
    print(f"   NEW method (int):     {len(filtered_new):,} rows ✅")
    print("\n   The fix in train_auto.py IS working!")
    print("   If you're still seeing 0 rows in Colab, the issue is:")
    print("     1. Colab is using old code (cached)")
    print("     2. OR PlayerStatistics.csv in Colab is corrupted")
else:
    print(f"\n✅ Both methods work on this sample!")
    print(f"   OLD method: {len(filtered_old):,} rows")
    print(f"   NEW method: {len(filtered_new):,} rows")
    print("\n   If still seeing 0 rows in training:")
    print("     1. Check you're uploading the FULL PlayerStatistics.csv (302 MB)")
    print("     2. NOT just a sample or truncated version")

print("\n" + "=" * 80)
