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

# Test filtering for a specific window (2007-2011)
print("\n4. Testing window filtering (2007-2011)...")
window_seasons = [2007, 2008, 2009, 2010, 2011]
padded_seasons = set(window_seasons) | {2006, 2012}  # ±1 padding

print(f"   Window seasons: {window_seasons}")
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
    print(f"\n   ✅ Filtering works! Got {len(filtered):,} rows for 2007-2011")
    print(f"     Season distribution: {filtered['_temp_season'].value_counts().sort_index().to_dict()}")

print("\n" + "=" * 80)
print("RECOMMENDATION:")
print("=" * 80)

if len(filtered) == 0:
    print("\n❌ CRITICAL BUG: Window filtering removes all player data!")
    print("\nFIX NEEDED in train_auto.py around line 5040-5042:")
    print("   Change:")
    print("     hist_players_df['_temp_season'] = _season_from_date(hist_players_df[date_col])")
    print("     padded_seasons = set(window_seasons) | {start_year-1, end_year+1}")
    print("     hist_players_df = hist_players_df[hist_players_df['_temp_season'].isin(padded_seasons)]")
    print("\n   To:")
    print("     hist_players_df['_temp_season'] = _season_from_date(hist_players_df[date_col]).astype('Int64')")
    print("     padded_seasons = set(window_seasons) | {start_year-1, end_year+1}")
    print("     hist_players_df = hist_players_df[hist_players_df['_temp_season'].isin(padded_seasons)]")
else:
    print("\n✅ Filtering works correctly on this sample!")
    print("   If training still fails, the issue may be:")
    print("     1. Different behavior on full dataset")
    print("     2. Memory issues (file too large)")
    print("     3. Issue occurs in build_players_from_playerstats() later")

print("=" * 80)
