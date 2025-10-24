#!/usr/bin/env python3
"""
Check if priors are actually being used in games_df
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Sample games_df builder (minimal version)
def check_priors_in_games():
    print("\n" + "="*60)
    print("Checking if priors data is actually being merged")
    print("="*60)

    # Load Team Abbrev
    priors_root = Path("C:/Users/tmiles11/nba_predictor/priors_data")
    abbrev_path = priors_root / "Team Abbrev.csv"

    if not abbrev_path.exists():
        print("✗ Team Abbrev.csv not found")
        return

    abbrev_df = pd.read_csv(abbrev_path, low_memory=False)
    print(f"\n1. Team Abbrev.csv: {len(abbrev_df)} rows")
    print(f"   Columns: {list(abbrev_df.columns)}")
    print(f"   Sample:")
    print(abbrev_df.head())

    # Load Team Summaries
    summaries_path = priors_root / "Team Summaries.csv"
    if not summaries_path.exists():
        print("✗ Team Summaries.csv not found")
        return

    summaries_df = pd.read_csv(summaries_path, low_memory=False)
    print(f"\n2. Team Summaries.csv: {len(summaries_df)} rows")
    print(f"   Columns: {list(summaries_df.columns)}")

    # Filter NBA non-playoff
    if "lg" in summaries_df.columns:
        summaries_df = summaries_df[summaries_df["lg"] == "NBA"]
    if "playoffs" in summaries_df.columns:
        summaries_df = summaries_df[summaries_df["playoffs"] == False]

    print(f"   After NBA filter: {len(summaries_df)} rows")
    print(f"   Sample:")
    print(summaries_df[["season", "abbreviation", "o_rtg", "d_rtg", "pace", "srs"]].head(10))

    # Check season ranges
    if "season" in summaries_df.columns:
        seasons = summaries_df["season"].dropna()
        print(f"\n3. Season coverage:")
        print(f"   Min season: {int(seasons.min())}")
        print(f"   Max season: {int(seasons.max())}")
        print(f"   Unique seasons: {seasons.nunique()}")

    # Check abbreviation coverage
    if "abbreviation" in summaries_df.columns:
        abbrevs = summaries_df["abbreviation"].value_counts()
        print(f"\n4. Team abbreviation coverage:")
        print(f"   Unique teams: {len(abbrevs)}")
        print(f"   Top 10 teams:")
        print(abbrevs.head(10))

    # Simulate what would happen with season shift
    summaries_df["season_for_game"] = pd.to_numeric(summaries_df["season"], errors="coerce") + 1

    print(f"\n5. After season shift (+1):")
    print(f"   Season → season_for_game mapping:")
    sample = summaries_df[["season", "season_for_game", "abbreviation", "o_rtg", "pace"]].head(10)
    print(sample)

    # Check if any recent seasons exist (2020-2025)
    recent_seasons = summaries_df[summaries_df["season_for_game"] >= 2020]
    print(f"\n6. Recent seasons (2020+) for matching:")
    print(f"   Rows with season_for_game >= 2020: {len(recent_seasons)}")
    if len(recent_seasons) > 0:
        print(f"   Sample recent data:")
        print(recent_seasons[["season_for_game", "abbreviation", "o_rtg", "d_rtg", "pace", "srs"]].head(20))

    # KEY INSIGHT: Check what the odds dataset provides
    print("\n" + "="*60)
    print("KEY: What do games need to match priors?")
    print("="*60)
    print("\nGames need:")
    print("  1. home_abbrev (from odds dataset)")
    print("  2. away_abbrev (from odds dataset)")
    print("  3. season_end_year (from game date)")
    print("\nPriors provide:")
    print("  1. abbreviation (team code)")
    print("  2. season_for_game (season priors apply to)")
    print("  3. o_rtg, d_rtg, pace, srs (stats)")
    print("\nMerge logic:")
    print("  games_df.merge(priors, left_on=['home_abbrev', 'season_end_year'],")
    print("                         right_on=['abbreviation', 'season_for_game'])")
    print("\n⚠️  If odds dataset doesn't provide home_abbrev/away_abbrev,")
    print("    then priors CAN'T be merged!")

if __name__ == "__main__":
    check_priors_in_games()
