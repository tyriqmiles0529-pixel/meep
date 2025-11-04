#!/usr/bin/env python3
"""
Standalone test script to debug Basketball Reference priors loading.
Loads CSVs from local folder: C:/Users/tmiles11/nba_predictor/priors_data

Run: python test_priors_load.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# No Kaggle credentials needed - loading from local path!

def log(msg):
    print(msg, flush=True)

def load_and_filter_player_csv(path: Path, csv_name: str):
    """Helper to load player CSV with NBA filtering and TOT preference"""
    if not path.exists():
        log(f"  Skipping {csv_name} (not found)")
        return None

    df = pd.read_csv(path, low_memory=False)
    log(f"  Raw {csv_name}: {len(df)} rows, {len(df.columns)} columns")

    # Filter NBA
    if "lg" in df.columns:
        df = df[df["lg"] == "NBA"]
        log(f"    After NBA filter: {len(df)} rows")

    # Prefer TOT rows for multi-team seasons
    if "team" in df.columns and "player_id" in df.columns:
        tot_rows = df[df["team"] == "TOT"]
        non_tot = df[df["team"] != "TOT"]
        has_tot = set(tot_rows["player_id"])
        non_tot = non_tot[~non_tot["player_id"].isin(has_tot)]
        df = pd.concat([tot_rows, non_tot], ignore_index=True)
        log(f"    After TOT preference: {len(df)} rows")

    return df

def load_basketball_reference_priors(priors_root: Path):
    """Load Basketball Reference priors with detailed debugging"""
    log("\n" + "="*60)
    log("Loading Basketball Reference Player Priors")
    log("="*60)

    priors_players = pd.DataFrame()

    # 1. Per 100 Poss - core rate stats
    log("\n1. Loading Per 100 Poss.csv...")
    per100_path = priors_root / "Per 100 Poss.csv"
    per100 = load_and_filter_player_csv(per100_path, "Per 100 Poss.csv")

    if per100 is not None:
        cols = ["season", "player_id", "player", "age", "pos", "g", "mp",
                "pts_per_100_poss", "trb_per_100_poss", "ast_per_100_poss",
                "stl_per_100_poss", "blk_per_100_poss", "tov_per_100_poss",
                "fg_per_100_poss", "fga_per_100_poss", "fg_percent",
                "x3p_per_100_poss", "x3pa_per_100_poss", "x3p_percent",
                "ft_per_100_poss", "fta_per_100_poss", "ft_percent",
                "orb_per_100_poss", "drb_per_100_poss",
                "o_rtg", "d_rtg"]
        cols = [c for c in cols if c in per100.columns]
        priors_players = per100[cols].copy()
        log(f"  ✓ Base dataframe: {len(priors_players)} rows, {len(priors_players.columns)} columns")
        log(f"    Columns: {list(priors_players.columns)}")

    if priors_players.empty:
        log("\n✗ ERROR: Failed to load Per 100 Poss.csv")
        return priors_players

    # 2. Advanced - PER, WS, BPM, TS%, USG%
    log("\n2. Loading Advanced.csv...")
    advanced_path = priors_root / "Advanced.csv"
    advanced = load_and_filter_player_csv(advanced_path, "Advanced.csv")

    if advanced is not None:
        log(f"  Before merge: {len(priors_players.columns)} columns")
        log(f"    Columns: {list(priors_players.columns)}")

        adv_cols = ["season", "player_id", "per", "ts_percent", "usg_percent",
                    "ws", "ws_per_48", "bpm", "obpm", "dbpm", "vorp"]
        adv_cols = [c for c in adv_cols if c in advanced.columns]

        if len(adv_cols) > 2:
            priors_players = priors_players.merge(
                advanced[adv_cols], on=["season", "player_id"], how="left"
            )
            log(f"  After merge: {len(priors_players.columns)} columns")
            log(f"    Columns: {list(priors_players.columns)}")

            # Check for duplicates immediately after merge
            if priors_players.columns.duplicated().any():
                dup_cols = priors_players.columns[priors_players.columns.duplicated()].tolist()
                log(f"  ⚠️  WARNING: Duplicate columns detected after Advanced merge: {dup_cols}")

    # 3. Player Shooting
    log("\n3. Loading Player Shooting.csv...")
    shooting_path = priors_root / "Player Shooting.csv"
    shooting = load_and_filter_player_csv(shooting_path, "Player Shooting.csv")

    if shooting is not None:
        log(f"  Before merge: {len(priors_players.columns)} columns")

        shoot_cols = ["season", "player_id", "avg_dist_fga",
                      "percent_fga_from_x2p_range", "percent_fga_from_x0_3_range",
                      "percent_fga_from_x3_10_range", "percent_fga_from_x10_16_range",
                      "percent_fga_from_x16_3p_range", "percent_fga_from_x3p_range",
                      "fg_percent_from_x2p_range", "fg_percent_from_x0_3_range",
                      "fg_percent_from_x3_10_range", "fg_percent_from_x10_16_range",
                      "fg_percent_from_x16_3p_range", "fg_percent_from_x3p_range",
                      "percent_assisted_x2p_fg", "percent_assisted_x3p_fg",
                      "percent_dunks_of_fga", "num_of_dunks",
                      "percent_corner_3s_of_3pa", "corner_3_point_percent"]
        shoot_cols = [c for c in shoot_cols if c in shooting.columns]

        if len(shoot_cols) > 2:
            priors_players = priors_players.merge(
                shooting[shoot_cols], on=["season", "player_id"], how="left"
            )
            log(f"  After merge: {len(priors_players.columns)} columns")

            if priors_players.columns.duplicated().any():
                dup_cols = priors_players.columns[priors_players.columns.duplicated()].tolist()
                log(f"  ⚠️  WARNING: Duplicate columns detected after Shooting merge: {dup_cols}")

    # 4. Player Play By Play
    log("\n4. Loading Player Play By Play.csv...")
    pbp_path = priors_root / "Player Play By Play.csv"
    pbp = load_and_filter_player_csv(pbp_path, "Player Play By Play.csv")

    if pbp is not None:
        log(f"  Before merge: {len(priors_players.columns)} columns")

        pbp_cols = ["season", "player_id",
                    "pg_percent", "sg_percent", "sf_percent", "pf_percent", "c_percent",
                    "on_court_plus_minus_per_100_poss", "net_plus_minus_per_100_poss",
                    "bad_pass_turnover", "lost_ball_turnover",
                    "shooting_foul_committed", "offensive_foul_committed",
                    "shooting_foul_drawn", "offensive_foul_drawn",
                    "points_generated_by_assists", "and1"]
        pbp_cols = [c for c in pbp_cols if c in pbp.columns]

        if len(pbp_cols) > 2:
            priors_players = priors_players.merge(
                pbp[pbp_cols], on=["season", "player_id"], how="left"
            )
            log(f"  After merge: {len(priors_players.columns)} columns")

            if priors_players.columns.duplicated().any():
                dup_cols = priors_players.columns[priors_players.columns.duplicated()].tolist()
                log(f"  ⚠️  WARNING: Duplicate columns detected after PBP merge: {dup_cols}")

    # Final duplicate check
    log("\n" + "="*60)
    log("POST-MERGE DUPLICATE CHECK")
    log("="*60)

    if priors_players.columns.duplicated().any():
        dup_cols = priors_players.columns[priors_players.columns.duplicated()].tolist()
        log(f"✗ Duplicate columns found: {dup_cols}")
        log(f"\nRemoving duplicates (keeping first occurrence)...")
        priors_players = priors_players.loc[:, ~priors_players.columns.duplicated()]
        log(f"✓ After deduplication: {len(priors_players.columns)} columns")
    else:
        log("✓ No duplicate columns detected")

    # Normalize percent columns
    log("\n" + "="*60)
    log("NORMALIZING PERCENT COLUMNS")
    log("="*60)

    for col in priors_players.columns:
        if "percent" in col.lower() or "_pct" in col.lower():
            vals = pd.to_numeric(priors_players[col], errors="coerce")
            if vals.notna().any() and vals.max() > 1.5:
                log(f"  Normalizing {col}: max={vals.max():.2f} → {vals.max()/100:.2f}")
                priors_players[col] = vals / 100.0

    # Shift to next season
    log("\n" + "="*60)
    log("CREATING season_for_game (SHIFT +1)")
    log("="*60)

    if "season" not in priors_players.columns:
        log("✗ ERROR: No 'season' column found!")
        return priors_players

    log(f"'season' column info:")
    log(f"  Type: {type(priors_players['season'])}")
    log(f"  Shape: {priors_players['season'].shape if hasattr(priors_players['season'], 'shape') else 'N/A'}")
    log(f"  Sample values: {priors_players['season'].head().tolist()}")

    try:
        season_col = priors_players["season"]

        # Check if it's a DataFrame (happens with duplicate columns)
        if isinstance(season_col, pd.DataFrame):
            log(f"  ⚠️  WARNING: 'season' is a DataFrame with {len(season_col.columns)} columns!")
            log(f"  Column names: {season_col.columns.tolist()}")
            log(f"  Taking first column...")
            season_col = season_col.iloc[:, 0]

        # Convert to numeric
        season_numeric = pd.to_numeric(season_col, errors="coerce")
        log(f"  After numeric conversion: {season_numeric.dtype}, shape={season_numeric.shape}")

        # Create season_for_game
        priors_players["season_for_game"] = season_numeric + 1
        log(f"✓ Created season_for_game successfully")
        log(f"  Sample: season={priors_players['season'].head().tolist()} → season_for_game={priors_players['season_for_game'].head().tolist()}")

        # Drop original season column
        priors_players = priors_players.drop(columns=["season"])
        log(f"✓ Dropped original 'season' column")

    except Exception as e:
        log(f"✗ ERROR creating season_for_game: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

    log("\n" + "="*60)
    log("FINAL RESULT")
    log("="*60)
    log(f"✓ Player priors loaded: {len(priors_players):,} player-seasons")
    log(f"✓ Features: {len(priors_players.columns)} columns")
    log(f"  Columns: {list(priors_players.columns)[:10]}...")

    return priors_players


def main():
    print("\n🏀 Basketball Reference Priors Loader Test\n")

    # Use local priors path
    priors_root = Path("C:/Users/tmiles11/nba_predictor/priors_data")

    if not priors_root.exists():
        print(f"✗ ERROR: Priors folder not found at {priors_root}")
        print("\nPlease ensure you have downloaded the CSVs to:")
        print("  C:/Users/tmiles11/nba_predictor/priors_data/")
        print("\nRequired files:")
        print("  - Team Summaries.csv")
        print("  - Team Abbrev.csv")
        print("  - Per 100 Poss.csv")
        print("  - Advanced.csv")
        print("  - Player Shooting.csv")
        print("  - Player Play By Play.csv")
        return

    print(f"Loading from local path: {priors_root}")
    csv_files = list(priors_root.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  - {f.name}")
    print()

    # Load priors
    priors_players = load_basketball_reference_priors(priors_root)

    if not priors_players.empty:
        print("\n" + "="*60)
        print("✅ SUCCESS! Priors loaded successfully.")
        print("="*60)
        print(f"\nShape: {priors_players.shape}")
        print(f"\nFirst 5 rows:")
        print(priors_players.head())
    else:
        print("\n" + "="*60)
        print("❌ FAILED to load priors")
        print("="*60)


if __name__ == "__main__":
    main()