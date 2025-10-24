#!/usr/bin/env python3
"""
Diagnose what data is actually available for training after all merges
Run after games_df is built but before training starts
"""

import sys
import pandas as pd
import numpy as np

# Add this code snippet to train_auto.py RIGHT BEFORE line "# Train game models + OOF"
# Insert around line 1648 (right after all merges complete)

DIAGNOSTIC_CODE = """
# ============== DIAGNOSTIC: Check what data we actually have ==============
print("\\n" + "="*60)
print("DIAGNOSTIC: Checking priors data availability")
print("="*60)

print(f"\\nTotal games: {len(games_df):,}")
print(f"Columns in games_df: {len(games_df.columns)}")

# Check betting odds columns
odds_cols = ["market_implied_home", "market_implied_away", "market_spread", "market_total", "home_abbrev", "away_abbrev"]
print("\\n1. BETTING ODDS AVAILABILITY:")
for col in odds_cols:
    if col in games_df.columns:
        non_default = games_df[col].notna().sum()
        if col in ["home_abbrev", "away_abbrev"]:
            print(f"   {col}: {non_default:,} games ({non_default/len(games_df)*100:.1f}%)")
        else:
            # Check if not default value
            default_val = GAME_DEFAULTS.get(col, 0.0)
            non_default = (games_df[col] != default_val).sum()
            print(f"   {col}: {non_default:,} games with non-default values ({non_default/len(games_df)*100:.1f}%)")

# Check team priors columns
priors_cols = ["home_o_rtg_prior", "home_d_rtg_prior", "home_pace_prior", "home_srs_prior",
               "away_o_rtg_prior", "away_d_rtg_prior", "away_pace_prior", "away_srs_prior"]
print("\\n2. TEAM PRIORS AVAILABILITY:")
for col in priors_cols:
    if col in games_df.columns:
        default_val = GAME_DEFAULTS.get(col, 0.0)
        non_default = (games_df[col] != default_val).sum()
        print(f"   {col}: {non_default:,} games with non-default values ({non_default/len(games_df)*100:.1f}%)")
    else:
        print(f"   {col}: MISSING")

# Sample games with priors
print("\\n3. SAMPLE GAMES WITH TEAM PRIORS:")
if "home_o_rtg_prior" in games_df.columns:
    with_priors = games_df[games_df["home_o_rtg_prior"] != GAME_DEFAULTS.get("home_o_rtg_prior", 0.0)]
    if len(with_priors) > 0:
        print(f"   Found {len(with_priors):,} games with real priors")
        sample_cols = ["gid", "date", "home_abbrev", "away_abbrev", "season_end_year",
                      "home_o_rtg_prior", "home_d_rtg_prior", "home_pace_prior",
                      "away_o_rtg_prior", "away_d_rtg_prior", "away_pace_prior"]
        sample_cols = [c for c in sample_cols if c in with_priors.columns]
        print("   Sample:")
        print(with_priors[sample_cols].head(5))
    else:
        print("   ⚠️  NO games have real team priors - all are defaults!")
        print("   This means priors are NOT being used in training")

# Check correlation between odds availability and priors availability
if "home_abbrev" in games_df.columns and "home_o_rtg_prior" in games_df.columns:
    print("\\n4. CORRELATION CHECK:")
    has_abbrev = games_df["home_abbrev"].notna()
    has_priors = games_df["home_o_rtg_prior"] != GAME_DEFAULTS.get("home_o_rtg_prior", 0.0)
    both = has_abbrev & has_priors
    print(f"   Games with abbreviations: {has_abbrev.sum():,}")
    print(f"   Games with real priors: {has_priors.sum():,}")
    print(f"   Games with BOTH: {both.sum():,}")
    if has_abbrev.sum() > 0 and has_priors.sum() == 0:
        print("   ⚠️  WARNING: Abbreviations exist but priors didn't merge!")
        print("   This suggests a season mismatch between games and priors")

print("="*60)
# ============== END DIAGNOSTIC ==============
"""

print("="*60)
print("INSTRUCTIONS: Add this diagnostic code to train_auto.py")
print("="*60)
print("\n1. Open train_auto.py")
print("2. Find line 1648 (or search for '# Train game models + OOF')")
print("3. Insert this code RIGHT BEFORE that line:\n")
print(DIAGNOSTIC_CODE)
print("\n4. Re-run: python train_auto.py --verbose")
print("\nThis will show you:")
print("  • How many games have real odds vs defaults")
print("  • How many games have real priors vs defaults")
print("  • Whether the merge is actually working")
