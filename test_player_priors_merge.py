"""
Quick test for player priors merge fix
Tests ONLY the merge logic without running full training
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from train_auto import build_players_from_playerstats, load_basketball_reference_priors, build_games_from_teamstats

print("="*70)
print("PLAYER PRIORS MERGE TEST")
print("="*70)

# 1. Load game context (small sample)
print("\n1. Loading game context...")
teams_path = Path(r"C:\Users\tmiles11\.cache\kagglehub\datasets\eoinamoore\historical-nba-data-and-player-box-scores\versions\258\TeamStatistics.csv")
if not teams_path.exists():
    print(f"ERROR: TeamStatistics not found at {teams_path}")
    sys.exit(1)

games_df, context_map, team_abbrev = build_games_from_teamstats(teams_path, verbose=True, skip_rest=True)
print(f"✓ Loaded {len(games_df):,} games")

# Create OOF dummy data
oof_games = pd.DataFrame({
    'gid': context_map['gid'].unique(),
    'oof_ml_prob': 0.5,
    'oof_spread_pred': 0.0
})

# 2. Load player priors
print("\n2. Loading player priors...")
priors_root = Path(r"C:\Users\tmiles11\nba_predictor\priors_data")
if not priors_root.exists():
    print(f"ERROR: Priors not found at {priors_root}")
    sys.exit(1)

# Get seasons from games
seasons_to_keep = set(int(x) for x in games_df["season_end_year"].dropna().unique())
padded = set()
for s in seasons_to_keep:
    padded.update([s-1, s, s+1])
seasons_to_keep = padded

priors_players, priors_teams = load_basketball_reference_priors(priors_root, verbose=True, seasons_to_keep=seasons_to_keep)
print(f"✓ Loaded {len(priors_players):,} player-season priors")

# 3. Load player data (small sample for speed)
print("\n3. Loading player statistics (sampling for speed)...")
players_path = Path(r"C:\Users\tmiles11\.cache\kagglehub\datasets\eoinamoore\historical-nba-data-and-player-box-scores\versions\258\PlayerStatistics.csv")
if not players_path.exists():
    print(f"ERROR: PlayerStatistics not found at {players_path}")
    sys.exit(1)

# 4. Test the merge with ONLY current season (2022-2026) to save memory
print("\n4. Testing player priors merge (2022-2026 window only)...")
print("-" * 70)

# Filter to just one 5-year window to avoid OOM
test_window_seasons = {2022, 2023, 2024, 2025, 2026}

frames = build_players_from_playerstats(
    players_path,
    context_map,
    oof_games,
    verbose=True,
    priors_players=priors_players,
    window_seasons=test_window_seasons  # ← Use window filtering!
)

print("\n" + "="*70)
print("TEST RESULTS")
print("="*70)

# Check if we got data
if frames:
    print(f"✓ Generated {len(frames)} stat frames")
    for stat_name, df in frames.items():
        if df is not None and not df.empty:
            print(f"  - {stat_name}: {len(df):,} rows")
else:
    print("✗ No frames generated")

print("\n" + "="*70)
print("Look for these key indicators above:")
print("  1. 'Merge path: is_home flag' (NOT 'tid')")
print("  2. 'season_end_year non-null: ~100%'")
print("  3. 'TOTAL matched: 75-85%'")
print("="*70)
