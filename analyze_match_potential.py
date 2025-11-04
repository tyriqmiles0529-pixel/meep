"""
Analyze player priors matching potential to reach 80%+ match rate.
This script investigates:
1. What % of Kaggle players exist in Basketball Reference
2. What's preventing matches (ID format, name format, seasons)
3. How to improve from 40-60% to 80%+
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

print("=" * 80)
print("PLAYER PRIORS MATCH POTENTIAL ANALYSIS")
print("=" * 80)
print()

# Load Kaggle PlayerStatistics
print("Loading Kaggle PlayerStatistics...")
kaggle_path = Path(r"C:\Users\tmiles11\.cache\kagglehub\datasets\eoinamoore\historical-nba-data-and-player-box-scores\versions\257\PlayerStatistics.csv")

# Load sample for analysis
ps = pd.read_csv(kaggle_path, nrows=100000)
print(f"Loaded {len(ps):,} sample rows")
print()

# Parse dates and create season_end_year
print("Parsing dates and creating season_end_year...")
ps['gameDate'] = pd.to_datetime(ps['gameDate'], errors="coerce", format='mixed', utc=True).dt.tz_convert(None)

def _season_from_date(dt):
    y = dt.dt.year
    m = dt.dt.month
    return np.where(m >= 8, y + 1, y)

ps['season_end_year'] = _season_from_date(ps['gameDate']).astype('float32')
print(f"season_end_year populated: {ps['season_end_year'].notna().sum():,} / {len(ps):,}")
print()

# Construct full names
ps['full_name'] = (ps['firstName'].fillna("") + " " + ps['lastName'].fillna("")).str.strip()

# Normalize names
def normalize_name(s):
    return (
        s.str.normalize('NFKD')
         .str.encode('ascii', errors='ignore')
         .str.decode('ascii')
         .str.lower()
         .str.replace(r"[^a-z]+", " ", regex=True)
         .str.strip()
         .str.replace(r"\s+", " ", regex=True)
         .str.replace(r"\s+(jr|sr|ii|iii|iv|v)$", "", regex=True)
    )

ps['norm_name'] = normalize_name(ps['full_name'])

# Load Basketball Reference priors
print("Loading Basketball Reference priors...")
priors_path = Path(r"C:\Users\tmiles11\nba_predictor\priors_data")

# Load Advanced.csv as representative sample
adv = pd.read_csv(priors_path / "Advanced.csv")
print(f"Loaded {len(adv):,} player-season priors")
print()

# Normalize priors names
adv['norm_name'] = normalize_name(adv['player'])

# Get unique players and seasons
kaggle_players = ps['norm_name'].dropna().unique()
kaggle_seasons = ps['season_end_year'].dropna().unique()
priors_players = adv['norm_name'].dropna().unique()
priors_seasons = adv['season_for_game'].dropna().unique()

print("-" * 80)
print("DATA OVERVIEW")
print("-" * 80)
print(f"Kaggle unique players: {len(kaggle_players):,}")
print(f"Kaggle seasons: {sorted([int(s) for s in kaggle_seasons])}")
print(f"Priors unique players: {len(priors_players):,}")
print(f"Priors seasons: {int(priors_seasons.min())} to {int(priors_seasons.max())}")
print()

# Find name overlap
common_players = set(kaggle_players) & set(priors_players)
print(f"Common players (by normalized name): {len(common_players):,}")
print(f"Name match rate: {len(common_players) / len(kaggle_players) * 100:.1f}%")
print()

# Sample common players
print("Sample common players:")
for name in sorted(common_players)[:20]:
    print(f"  - {name}")
print()

# Find Kaggle players NOT in priors
missing_players = set(kaggle_players) - set(priors_players)
print(f"Kaggle players NOT in priors: {len(missing_players):,}")
print(f"Missing rate: {len(missing_players) / len(kaggle_players) * 100:.1f}%")
print()

print("Sample missing players (first 30):")
for name in sorted(missing_players)[:30]:
    # Find original name
    orig_name = ps[ps['norm_name'] == name]['full_name'].iloc[0] if len(ps[ps['norm_name'] == name]) > 0 else name
    print(f"  - {orig_name} (normalized: {name})")
print()

#Find season overlap
common_seasons = set(kaggle_seasons) & set(priors_seasons)
print("-" * 80)
print("SEASON ANALYSIS")
print("-" * 80)
print(f"Common seasons: {len(common_seasons)}")
print(f"Seasons: {sorted([int(s) for s in common_seasons])}")
print()

# Count player-games by season
season_counts = ps.groupby('season_end_year').size()
print("Player-games by season (sample data):")
for season, count in season_counts.sort_index().items():
    in_priors = "✓" if season in priors_seasons else "✗"
    print(f"  {int(season)}: {count:,} games  {in_priors}")
print()

# Estimate match rate for each season
print("-" * 80)
print("MATCH RATE BY SEASON")
print("-" * 80)

for season in sorted(kaggle_seasons):
    if season not in priors_seasons:
        print(f"{int(season)}: 0.0% (season not in priors)")
        continue

    # Get players from this season in Kaggle
    season_players = ps[ps['season_end_year'] == season]['norm_name'].dropna().unique()

    # Get players from this season in priors
    priors_season_players = adv[adv['season_for_game'] == season]['norm_name'].dropna().unique()

    # Find overlap
    season_common = set(season_players) & set(priors_season_players)
    match_rate = len(season_common) / len(season_players) * 100 if len(season_players) > 0 else 0

    print(f"{int(season)}: {match_rate:.1f}% ({len(season_common):,} / {len(season_players):,} players)")

print()

# Analyze ID matching potential
print("-" * 80)
print("PLAYER ID ANALYSIS")
print("-" * 80)

print("Kaggle player IDs (personId):")
print(f"  Format: {ps['personId'].dtype}")
print(f"  Sample: {ps['personId'].head(10).tolist()}")
print(f"  Unique: {ps['personId'].nunique():,}")
print()

print("Basketball Reference player IDs (player_id):")
print(f"  Format: {adv['player_id'].dtype}")
print(f"  Sample: {adv['player_id'].head(10).tolist()}")
print(f"  Unique: {adv['player_id'].nunique():,}")
print()

# Check if any IDs match
kaggle_ids = set(ps['personId'].astype(str).dropna().unique())
priors_ids = set(adv['player_id'].astype(str).dropna().unique())
common_ids = kaggle_ids & priors_ids

print(f"Common player IDs: {len(common_ids)}")
if len(common_ids) > 0:
    print(f"ID match rate: {len(common_ids) / len(kaggle_ids) * 100:.1f}%")
    print(f"Sample common IDs: {list(common_ids)[:10]}")
else:
    print("✗ NO ID OVERLAP - IDs are in different formats")
    print("  Kaggle uses numeric IDs (e.g., '1000', '1005')")
    print("  Basketball Reference uses alphanumeric codes (e.g., 'jamesle01', 'curryst01')")
    print("  → ID matching will NOT work, must rely on name matching")
print()

# RECOMMENDATIONS
print("=" * 80)
print("RECOMMENDATIONS TO REACH 80%+ MATCH RATE")
print("=" * 80)
print()

# Calculate current theoretical max
ps_2002_plus = ps[ps['season_end_year'] >= 2002]
ps_2002_plus_players = ps_2002_plus['norm_name'].dropna().unique()
priors_2002_plus = adv[adv['season_for_game'] >= 2002]
priors_2002_plus_players = priors_2002_plus['norm_name'].dropna().unique()
common_2002_plus = set(ps_2002_plus_players) & set(priors_2002_plus_players)

theoretical_max = len(common_2002_plus) / len(ps_2002_plus_players) * 100 if len(ps_2002_plus_players) > 0 else 0

print(f"Current Theoretical Maximum (2002+ seasons only):")
print(f"  {theoretical_max:.1f}% ({len(common_2002_plus):,} / {len(ps_2002_plus_players):,} players)")
print()

if theoretical_max < 80:
    print(f"⚠️  WARNING: Even with perfect matching, max rate is {theoretical_max:.1f}%")
    print(f"   Missing players: {len(set(ps_2002_plus_players) - common_2002_plus):,}")
    print()
    print("   Reasons for missing players:")
    print("   1. Current season (2025-26) not in Basketball Reference yet")
    print("   2. Rookies with no prior NBA stats")
    print("   3. G-League call-ups (<10 NBA games)")
    print("   4. International players not tracked by Basketball Reference")
    print("   5. Two-way contract players")
    print()

print("ACTIONABLE IMPROVEMENTS:")
print()

print("1. **Filter to historical seasons only (2002-2024)**")
print("   - Exclude 2025-26 season (Basketball Reference doesn't have it yet)")
print("   - Expected impact: +10-20% match rate")
print()

print("2. **Improved name normalization**")
print("   - Handle nicknames (e.g., 'Bob' vs 'Robert')")
print("   - Handle different suffix formats (e.g., 'Jr.' vs 'Jr')")
print("   - Expected impact: +2-5% match rate")
print()

print("3. **Use player_id mapping if available**")
print("   - Check if Kaggle provides nba.com player IDs elsewhere")
print("   - Basketball Reference has nba.com ID mappings in some CSVs")
print("   - Expected impact: +5-10% match rate if IDs available")
print()

print("4. **Fuzzy matching for close names**")
print("   - Use Levenshtein distance for near-matches")
print("   - Example: 'Giannis Antetokounmpo' vs 'Giannis Antetokounmpo' (one letter off)")
print("   - Expected impact: +2-5% match rate")
print("   - ⚠️  Currently disabled due to memory issues - need to optimize")
print()

print("5. **External player ID mapping**")
print("   - Use NBA API to get player_id → basketball_reference_id mapping")
print("   - One-time setup, then 90%+ match rate")
print("   - Expected impact: +30-40% match rate")
print()

print("=" * 80)
print("NEXT STEPS")
print("=" * 80)
print()
print("To reach 80%+ match rate:")
print()
print("Option A: Filter to 2002-2024 only (exclude current season)")
print("  - Quick win, no code changes")
print("  - Run: Use season filter in train_auto.py")
print()
print("Option B: Build external player ID mapping")
print("  - One-time effort, permanent 80%+ match rate")
print("  - Fetch nba.com → basketball-reference mapping from NBA API")
print("  - Store in CSV for future use")
print()
print("Option C: Optimize fuzzy matching")
print("  - Re-enable fuzzy matching with memory optimization")
print("  - Process in chunks to avoid OOM")
print()
print("Recommended: Option B (external mapping) for best long-term results")
print()
