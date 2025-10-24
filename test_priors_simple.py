#!/usr/bin/env python3
"""
Minimal test to reproduce and fix the priors loading error.
Run this on Windows: python test_priors_simple.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

def test_duplicate_season_column():
    """Simulate the exact error scenario with duplicate columns"""
    print("\n" + "="*60)
    print("REPRODUCING THE ERROR")
    print("="*60)

    # Create a test dataframe with duplicate 'season' column
    # (this happens when merging multiple CSVs)
    df = pd.DataFrame({
        'player_id': ['1', '2', '3'],
        'season': [2019, 2020, 2021],
        'pts': [20.5, 25.3, 18.2]
    })

    # Simulate adding another 'season' column (from a bad merge)
    df['season'] = [2019, 2020, 2021]  # This creates duplicate

    # Force duplicate by concatenating
    df = pd.concat([df, pd.DataFrame({'season': [2019, 2020, 2021]})], axis=1)

    print(f"\nDataFrame with duplicate columns:")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Has duplicates: {df.columns.duplicated().any()}")
    print(f"\nDataFrame:")
    print(df)

    print("\n" + "="*60)
    print("ATTEMPTING BAD CODE (will fail)")
    print("="*60)

    try:
        # This is what causes the error!
        season_for_game = pd.to_numeric(df["season"], errors="coerce") + 1
        print(f"✗ Should have failed but didn't!")
    except Exception as e:
        print(f"✓ Error reproduced: {e}")
        print(f"  Error type: {type(e).__name__}")

    print("\n" + "="*60)
    print("APPLYING FIX")
    print("="*60)

    # FIX: Remove duplicates before accessing column
    if df.columns.duplicated().any():
        dup_cols = df.columns[df.columns.duplicated()].tolist()
        print(f"  Duplicate columns found: {dup_cols}")
        df_fixed = df.loc[:, ~df.columns.duplicated()]
        print(f"  After deduplication: {list(df_fixed.columns)}")

    # Now try again
    try:
        season_col = df_fixed["season"]

        # Extra safety: ensure it's a Series, not DataFrame
        if isinstance(season_col, pd.DataFrame):
            print(f"  'season' is still a DataFrame, taking first column...")
            season_col = season_col.iloc[:, 0]

        season_for_game = pd.to_numeric(season_col, errors="coerce") + 1
        df_fixed["season_for_game"] = season_for_game

        print(f"\n✓ Fixed successfully!")
        print(f"\nResult:")
        print(df_fixed)

    except Exception as e:
        print(f"✗ Still failed: {e}")


def show_fix_code():
    """Display the exact code needed in train_auto.py"""
    print("\n" + "="*60)
    print("CODE TO ADD TO train_auto.py")
    print("="*60)
    print("""
# Add this BEFORE creating season_for_game:

# Check for duplicate columns before shifting season
if priors_players.columns.duplicated().any():
    dup_cols = priors_players.columns[priors_players.columns.duplicated()].tolist()
    log(f"Warning: Duplicate columns found: {dup_cols}. Removing duplicates.", verbose)
    priors_players = priors_players.loc[:, ~priors_players.columns.duplicated()]

# Shift to next season (with safe Series handling)
if "season" in priors_players.columns:
    try:
        season_series = priors_players["season"]

        # Ensure it's a Series, not DataFrame
        if isinstance(season_series, pd.DataFrame):
            season_series = season_series.iloc[:, 0]

        priors_players["season_for_game"] = pd.to_numeric(season_series, errors="coerce") + 1
        priors_players = priors_players.drop(columns=["season"])

    except Exception as e:
        log(f"Error creating season_for_game: {e}", verbose)
        log(f"season column type: {type(priors_players['season'])}", verbose)
        raise
else:
    log("Warning: No 'season' column found in player priors", verbose)
""")


if __name__ == "__main__":
    print("\n🔧 Priors Loading Error - Diagnostic Test\n")

    # Test 1: Reproduce the error
    test_duplicate_season_column()

    # Test 2: Show the fix
    show_fix_code()

    print("\n" + "="*60)
    print("✅ TEST COMPLETE")
    print("="*60)
    print("\nThe error occurs when merging multiple Basketball Reference CSVs")
    print("creates duplicate 'season' columns. The fix removes duplicates")
    print("before attempting to access the column.\n")
