#!/usr/bin/env python
"""
CSV Aggregation Module

Aggregates all 7 Basketball Reference CSV tables to recreate the full
aggregated dataset with advanced stats, per-100, PBP, and shooting data.

Maintains the same high merge rate achieved in original aggregation.
"""

import pandas as pd
import gc
from pathlib import Path
from typing import Optional, Dict


def load_and_merge_csvs(
    data_dir: str,
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load all 9 CSVs from both Kaggle datasets and merge them.

    Expected files in data_dir:
    1. PlayerStatistics.csv - Base box scores (game_id, player_id, points, assists, etc.)
    2. Player Advanced.csv - Advanced stats (PER, TS%, BPM, VORP, etc.)
    3. Player Per 100 Poss.csv - Per-100 possession stats
    4. Player Play-By-Play.csv - PBP stats (plus/minus, turnovers, fouls)
    5. Player Shooting.csv - Shooting zones and percentages
    6. Players.csv - Biographical data (height, weight, position, birth date)
    7. TeamStatistics.csv - Team box scores
    8. Games.csv - Game metadata (arena, attendance)
    9. Team Summaries.csv - Team advanced stats

    Args:
        data_dir: Directory containing all CSV files
        min_year: Optional minimum season year
        max_year: Optional maximum season year
        verbose: Print progress

    Returns:
        Merged DataFrame with all features
    """
    data_path = Path(data_dir)

    if verbose:
        print("="*70)
        print("AGGREGATING ALL 9 CSVs (FULL FEATURE SET)")
        print("="*70)
        print(f"Data directory: {data_dir}")

    # =====================================================================
    # Step 1: Load base player statistics (game-level box scores)
    # =====================================================================
    player_stats_path = data_path / "PlayerStatistics.csv"
    if not player_stats_path.exists():
        raise FileNotFoundError(f"PlayerStatistics.csv not found in {data_dir}")

    if verbose:
        print("\n[1/5] Loading PlayerStatistics.csv (base box scores)...")

    df = pd.read_csv(player_stats_path, low_memory=False)

    if verbose:
        print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Extract season from gameDate if needed
    if 'gameDate' in df.columns and 'season' not in df.columns:
        df['gameDate'] = pd.to_datetime(df['gameDate'], format='mixed', utc=True, errors='coerce')
        # Convert to timezone-naive for easier processing
        df['gameDate'] = df['gameDate'].dt.tz_localize(None)
        # NBA season spans two years, games from Oct-Dec are start of season, Jan+ are end
        df['season'] = df['gameDate'].dt.year
        # Adjust for games in Oct-Dec (they belong to next year's season)
        df.loc[df['gameDate'].dt.month >= 10, 'season'] += 1

        if verbose:
            print(f"  Created 'season' column from gameDate")

    # Create Player name column for merging with Basketball Reference CSVs
    if 'firstName' in df.columns and 'lastName' in df.columns and 'Player' not in df.columns:
        df['Player'] = df['firstName'].fillna('') + ' ' + df['lastName'].fillna('')
        df['Player'] = df['Player'].str.strip()
        if verbose:
            print(f"  Created 'Player' column from firstName + lastName")

    # Apply year filter early
    if min_year or max_year:
        year_col = None
        for col_name in ['season', 'season_end_year', 'game_year', 'year']:
            if col_name in df.columns:
                year_col = col_name
                break

        if year_col:
            rows_before = len(df)
            if min_year:
                df = df[df[year_col] >= min_year]
            if max_year:
                df = df[df[year_col] <= max_year]

            if verbose:
                print(f"  Filtered by year: {rows_before:,} → {len(df):,} rows")

    # Optimize dtypes
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() < len(df) * 0.5:  # Only categorize if < 50% unique
            df[col] = df[col].astype('category')

    # =====================================================================
    # Step 2: Load and merge Basketball Reference advanced stats
    # =====================================================================

    # Detect merge keys based on available columns
    merge_successful = {}

    # Try to find the right column names for merging
    # Prefer 'Player' (created from firstName+lastName) for Basketball Reference merges
    player_col = None
    season_col = None

    for col_name in ['Player', 'player_id', 'personId', 'playerID', 'PLAYER_ID']:
        if col_name in df.columns:
            player_col = col_name
            break

    for col_name in ['season', 'season_end_year', 'SEASON', 'Year']:
        if col_name in df.columns:
            season_col = col_name
            break

    if not player_col or not season_col:
        raise ValueError(f"Could not find player/season columns. Available: {list(df.columns[:20])}")

    merge_keys = [player_col, season_col]

    if verbose:
        print(f"\nUsing merge keys: {merge_keys}")

    # Advanced stats (PER, BPM, VORP, etc.)
    advanced_path = data_path / "Player Advanced.csv"
    if advanced_path.exists():
        if verbose:
            print("\n[2/5] Loading Player Advanced.csv...")

        adv_df = pd.read_csv(advanced_path, low_memory=False)

        # Detect merge keys in this CSV
        # NOTE: Basketball Reference CSVs have BOTH 'player_id' (row index) and 'Player' (name)
        # We MUST use 'Player' (name) to match with main df's 'Player' column
        adv_player_col = None
        adv_season_col = None

        # Only check for 'Player' column (ignore player_id which is just a row index)
        if 'Player' in adv_df.columns:
            adv_player_col = 'Player'

        for col_name in ['Season', 'season', 'season_end_year', 'SEASON', 'Year']:
            if col_name in adv_df.columns:
                adv_season_col = col_name
                break

        if not adv_player_col or not adv_season_col:
            if verbose:
                print(f"  WARNING: Could not find merge keys. Columns: {list(adv_df.columns[:10])}")
                print("  Skipping Advanced stats merge")
        else:
            # Rename merge keys to match main df
            adv_df = adv_df.rename(columns={adv_player_col: player_col, adv_season_col: season_col})

            # Rename columns with adv_ prefix
            adv_cols = {col: f'adv_{col.lower().replace(" ", "_")}'
                       for col in adv_df.columns if col not in [player_col, season_col]}
            adv_df = adv_df.rename(columns=adv_cols)

            if verbose:
                print(f"  Loaded {len(adv_df):,} rows, {len(adv_df.columns)} columns")
                print(f"  Using merge keys: {adv_player_col} -> {player_col}, {adv_season_col} -> {season_col}")

            # Merge
            before_merge = len(df)
            df = df.merge(adv_df, on=merge_keys, how='left', suffixes=('', '_adv_dup'))

            # Calculate merge rate
            non_null_rate = df[[c for c in df.columns if c.startswith('adv_')]].notna().any(axis=1).mean()
            merge_successful['advanced'] = non_null_rate * 100

            if verbose:
                print(f"  Merged: {before_merge:,} rows → {len(df):,} rows")
                print(f"  Match rate: {non_null_rate*100:.1f}%")

            del adv_df
            gc.collect()
    else:
        if verbose:
            print("\n[2/5] Player Advanced.csv not found - skipping")

    # Per-100 possession stats
    per100_path = data_path / "Player Per 100 Poss.csv"
    if per100_path.exists():
        if verbose:
            print("\n[3/5] Loading Player Per 100 Poss.csv...")

        per100_df = pd.read_csv(per100_path, low_memory=False)

        # Detect merge keys in this CSV
        # Basketball Reference CSVs have 'Player' (name) - use this for matching
        per100_player_col = None
        per100_season_col = None

        if 'Player' in per100_df.columns:
            per100_player_col = 'Player'

        for col_name in ['Season', 'season', 'season_end_year', 'SEASON', 'Year']:
            if col_name in per100_df.columns:
                per100_season_col = col_name
                break

        if not per100_player_col or not per100_season_col:
            if verbose:
                print(f"  WARNING: Could not find merge keys. Columns: {list(per100_df.columns[:10])}")
                print("  Skipping Per 100 Poss merge")
        else:
            # Rename merge keys to match main df
            per100_df = per100_df.rename(columns={per100_player_col: player_col, per100_season_col: season_col})

            # Rename columns with per100_ prefix
            per100_cols = {col: f'per100_{col.lower().replace(" ", "_").replace("/", "_")}'
                          for col in per100_df.columns if col not in [player_col, season_col]}
            per100_df = per100_df.rename(columns=per100_cols)

            if verbose:
                print(f"  Loaded {len(per100_df):,} rows, {len(per100_df.columns)} columns")
                print(f"  Using merge keys: {per100_player_col} -> {player_col}, {per100_season_col} -> {season_col}")

            # Merge
            before_merge = len(df)
            df = df.merge(per100_df, on=merge_keys, how='left', suffixes=('', '_per100_dup'))

            non_null_rate = df[[c for c in df.columns if c.startswith('per100_')]].notna().any(axis=1).mean()
            merge_successful['per100'] = non_null_rate * 100

            if verbose:
                print(f"  Merged: {before_merge:,} rows → {len(df):,} rows")
                print(f"  Match rate: {non_null_rate*100:.1f}%")

            del per100_df
            gc.collect()
    else:
        if verbose:
            print("\n[3/5] Player Per 100 Poss.csv not found - skipping")

    # Play-by-Play stats (plus/minus, turnovers, fouls)
    pbp_path = data_path / "Player Play-By-Play.csv"
    if pbp_path.exists():
        if verbose:
            print("\n[4/5] Loading Player Play-By-Play.csv...")

        pbp_df = pd.read_csv(pbp_path, low_memory=False)

        # Detect merge keys in this CSV
        pbp_player_col = None
        pbp_season_col = None
        for col_name in ['Player', 'player_id', 'personId', 'playerID', 'PLAYER_ID']:
            if col_name in pbp_df.columns:
                pbp_player_col = col_name
                break
        for col_name in ['Season', 'season', 'season_end_year', 'SEASON', 'Year']:
            if col_name in pbp_df.columns:
                pbp_season_col = col_name
                break

        if not pbp_player_col or not pbp_season_col:
            if verbose:
                print(f"  WARNING: Could not find merge keys. Columns: {list(pbp_df.columns[:10])}")
                print("  Skipping Play-By-Play merge")
        else:
            # Rename merge keys to match main df
            pbp_df = pbp_df.rename(columns={pbp_player_col: player_col, pbp_season_col: season_col})

            # Rename columns with pbp_ prefix
            pbp_cols = {col: f'pbp_{col.lower().replace(" ", "_").replace("/", "_").replace("+", "plus").replace("-", "_")}'
                       for col in pbp_df.columns if col not in [player_col, season_col]}
            pbp_df = pbp_df.rename(columns=pbp_cols)

            if verbose:
                print(f"  Loaded {len(pbp_df):,} rows, {len(pbp_df.columns)} columns")
                print(f"  Using merge keys: {pbp_player_col} -> {player_col}, {pbp_season_col} -> {season_col}")

            # Merge
            before_merge = len(df)
            df = df.merge(pbp_df, on=merge_keys, how='left', suffixes=('', '_pbp_dup'))

            non_null_rate = df[[c for c in df.columns if c.startswith('pbp_')]].notna().any(axis=1).mean()
            merge_successful['pbp'] = non_null_rate * 100

            if verbose:
                print(f"  Merged: {before_merge:,} rows → {len(df):,} rows")
                print(f"  Match rate: {non_null_rate*100:.1f}%")

            del pbp_df
            gc.collect()
    else:
        if verbose:
            print("\n[4/5] Player Play-By-Play.csv not found - skipping")

    # Shooting stats (zones, percentages)
    shoot_path = data_path / "Player Shooting.csv"
    if shoot_path.exists():
        if verbose:
            print("\n[5/5] Loading Player Shooting.csv...")

        shoot_df = pd.read_csv(shoot_path, low_memory=False)

        # Detect merge keys in this CSV
        shoot_player_col = None
        shoot_season_col = None
        for col_name in ['Player', 'player_id', 'personId', 'playerID', 'PLAYER_ID']:
            if col_name in shoot_df.columns:
                shoot_player_col = col_name
                break
        for col_name in ['Season', 'season', 'season_end_year', 'SEASON', 'Year']:
            if col_name in shoot_df.columns:
                shoot_season_col = col_name
                break

        if not shoot_player_col or not shoot_season_col:
            if verbose:
                print(f"  WARNING: Could not find merge keys. Columns: {list(shoot_df.columns[:10])}")
                print("  Skipping Shooting merge")
        else:
            # Rename merge keys to match main df
            shoot_df = shoot_df.rename(columns={shoot_player_col: player_col, shoot_season_col: season_col})

            # Rename columns with shoot_ prefix
            shoot_cols = {col: f'shoot_{col.lower().replace(" ", "_").replace("%", "percent").replace("/", "_")}'
                         for col in shoot_df.columns if col not in [player_col, season_col]}
            shoot_df = shoot_df.rename(columns=shoot_cols)

            if verbose:
                print(f"  Loaded {len(shoot_df):,} rows, {len(shoot_df.columns)} columns")
                print(f"  Using merge keys: {shoot_player_col} -> {player_col}, {shoot_season_col} -> {season_col}")

            # Merge
            before_merge = len(df)
            df = df.merge(shoot_df, on=merge_keys, how='left', suffixes=('', '_shoot_dup'))

            non_null_rate = df[[c for c in df.columns if c.startswith('shoot_')]].notna().any(axis=1).mean()
            merge_successful['shooting'] = non_null_rate * 100

            if verbose:
                print(f"  Merged: {before_merge:,} rows → {len(df):,} rows")
                print(f"  Match rate: {non_null_rate*100:.1f}%")

            del shoot_df
            gc.collect()
    else:
        if verbose:
            print("\n[5/5] Player Shooting.csv not found - skipping")

    # =====================================================================
    # Step 6: Add player biographical data (Players.csv)
    # =====================================================================
    players_path = data_path / "Players.csv"
    if players_path.exists():
        if verbose:
            print("\n[6/9] Loading Players.csv (biographical data)...")

        players_df = pd.read_csv(players_path, low_memory=False)

        # Detect merge key (Players.csv has player_id or personId)
        players_merge_col = None
        for col_name in ['player_id', 'personId', 'playerID', 'PLAYER_ID']:
            if col_name in players_df.columns and col_name in df.columns:
                players_merge_col = col_name
                break

        if players_merge_col:
            if verbose:
                print(f"  Loaded {len(players_df):,} rows")
                print(f"  Merging on: {players_merge_col}")

            before_merge = len(df)
            df = df.merge(players_df, on=players_merge_col, how='left', suffixes=('', '_bio_dup'))

            if verbose:
                print(f"  Merged: {before_merge:,} rows → {len(df):,} rows")
        else:
            if verbose:
                print(f"  WARNING: No matching ID column found for Players.csv")
                print(f"  Available in Players.csv: {list(players_df.columns[:10])}")
                print(f"  Skipping Players.csv merge")

        del players_df
        gc.collect()
    else:
        if verbose:
            print("\n[6/9] Players.csv not found - skipping")

    # =====================================================================
    # Step 7: Add team context (TeamStatistics.csv)
    # =====================================================================
    team_stats_path = data_path / "TeamStatistics.csv"
    if team_stats_path.exists():
        if verbose:
            print("\n[7/9] Loading TeamStatistics.csv (team context)...")

        team_df = pd.read_csv(team_stats_path, low_memory=False)

        # Auto-detect game_id and team_id columns
        game_id_col = None
        team_id_col = None

        for col_name in ['game_id', 'gameId', 'GAME_ID', 'game_ID']:
            if col_name in df.columns and col_name in team_df.columns:
                game_id_col = col_name
                break

        for col_name in ['team_id', 'teamId', 'TEAM_ID', 'team_ID']:
            if col_name in df.columns and col_name in team_df.columns:
                team_id_col = col_name
                break

        if game_id_col and team_id_col:
            team_cols = {col: f'team_{col.lower().replace(" ", "_")}'
                        for col in team_df.columns if col not in [game_id_col, team_id_col]}
            team_df = team_df.rename(columns=team_cols)

            if verbose:
                print(f"  Loaded {len(team_df):,} rows")
                print(f"  Merging on: [{game_id_col}, {team_id_col}]")

            before_merge = len(df)
            df = df.merge(team_df, on=[game_id_col, team_id_col], how='left', suffixes=('', '_team_dup'))

            if verbose:
                print(f"  Merged: {before_merge:,} rows → {len(df):,} rows")
        else:
            if verbose:
                print(f"  WARNING: Could not find matching game_id/team_id columns")
                print(f"  Available in main df: {[c for c in df.columns if 'game' in c.lower() or 'team' in c.lower()][:5]}")
                print(f"  Available in TeamStatistics: {[c for c in team_df.columns if 'game' in c.lower() or 'team' in c.lower()][:5]}")
                print(f"  Skipping TeamStatistics merge")

        del team_df
        gc.collect()
    else:
        if verbose:
            print("\n[7/9] TeamStatistics.csv not found - skipping")

    # =====================================================================
    # Step 8: Add game metadata (Games.csv)
    # =====================================================================
    games_path = data_path / "Games.csv"
    if games_path.exists():
        if verbose:
            print("\n[8/9] Loading Games.csv (game metadata)...")

        games_df = pd.read_csv(games_path, low_memory=False)

        # Auto-detect game_id column
        game_id_col = None
        for col_name in ['game_id', 'gameId', 'GAME_ID', 'game_ID']:
            if col_name in df.columns and col_name in games_df.columns:
                game_id_col = col_name
                break

        if game_id_col:
            if verbose:
                print(f"  Loaded {len(games_df):,} rows")
                print(f"  Merging on: {game_id_col}")

            before_merge = len(df)
            df = df.merge(games_df, on=game_id_col, how='left', suffixes=('', '_game_dup'))

            if verbose:
                print(f"  Merged: {before_merge:,} rows → {len(df):,} rows")
        else:
            if verbose:
                print(f"  WARNING: Could not find matching game_id column")
                print(f"  Skipping Games.csv merge")

        del games_df
        gc.collect()
    else:
        if verbose:
            print("\n[8/9] Games.csv not found - skipping")

    # =====================================================================
    # Step 9: Add team advanced stats (Team Summaries.csv)
    # =====================================================================
    team_summ_path = data_path / "Team Summaries.csv"
    if team_summ_path.exists():
        if verbose:
            print("\n[9/9] Loading Team Summaries.csv (team advanced)...")

        team_summ_df = pd.read_csv(team_summ_path, low_memory=False)

        # Auto-detect team_id and season columns
        # Team Summaries uses different column names than PlayerStatistics
        team_summ_team_col = None
        team_summ_season_col = None

        for col_name in ['Team', 'team_id', 'teamId', 'TEAM_ID', 'team_ID']:
            if col_name in team_summ_df.columns:
                team_summ_team_col = col_name
                break

        for col_name in ['Season', 'season', 'season_end_year', 'SEASON', 'Year']:
            if col_name in team_summ_df.columns:
                team_summ_season_col = col_name
                break

        # Find matching columns in main dataframe
        main_team_col = None
        main_season_col = season_col  # Already detected earlier

        for col_name in ['team_id', 'teamId', 'TEAM_ID', 'team_ID']:
            if col_name in df.columns:
                main_team_col = col_name
                break

        if team_summ_team_col and team_summ_season_col and main_team_col and main_season_col:
            # Rename Team Summaries columns to match main df
            team_summ_df = team_summ_df.rename(columns={
                team_summ_team_col: main_team_col,
                team_summ_season_col: main_season_col
            })

            team_summ_cols = {col: f'team_adv_{col.lower().replace(" ", "_")}'
                             for col in team_summ_df.columns if col not in [main_team_col, main_season_col]}
            team_summ_df = team_summ_df.rename(columns=team_summ_cols)

            if verbose:
                print(f"  Loaded {len(team_summ_df):,} rows")
                print(f"  Merging on: [{main_team_col}, {main_season_col}]")

            before_merge = len(df)
            df = df.merge(team_summ_df, on=[main_team_col, main_season_col], how='left', suffixes=('', '_teamadv_dup'))

            if verbose:
                print(f"  Merged: {before_merge:,} rows → {len(df):,} rows")
        else:
            if verbose:
                print(f"  WARNING: Could not find matching team_id/season columns")
                print(f"  Skipping Team Summaries merge")

        del team_summ_df
        gc.collect()
    else:
        if verbose:
            print("\n[9/9] Team Summaries.csv not found - skipping")

    # =====================================================================
    # Final optimization
    # =====================================================================
    if verbose:
        print("\n" + "="*70)
        print("AGGREGATION COMPLETE")
        print("="*70)
        print(f"Final dataset: {len(df):,} rows, {len(df.columns)} columns")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

        if merge_successful:
            print("\nMerge Success Rates:")
            for table, rate in merge_successful.items():
                print(f"  {table:12s}: {rate:.1f}%")
            avg_rate = sum(merge_successful.values()) / len(merge_successful)
            print(f"  {'Average':12s}: {avg_rate:.1f}%")

        # Show year range
        try:
            year_col = None
            for col_name in ['season', 'season_end_year', 'game_year', 'year']:
                if col_name in df.columns:
                    year_col = col_name
                    break
            if year_col:
                min_yr = int(df[year_col].min())
                max_yr = int(df[year_col].max())
                print(f"\nYear range: {min_yr}-{max_yr}")
        except:
            pass

        print("="*70)

    return df
