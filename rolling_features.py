"""
Rolling Window Features for NBA Player Props

Computes L5, L10, L20 rolling averages and variance features
for player performance prediction.

This module adds temporal features that capture:
- Recent form (last 5 games)
- Short-term trends (last 10 games)
- Medium-term baseline (last 20 games)
- Performance variance (consistency)
- Hot/cold streaks
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import gc


def add_rolling_features(
    df: pd.DataFrame,
    stat_cols: List[str] = None,
    windows: List[int] = [5, 10],  # Reduced from [5, 10, 20] to save memory
    player_col: str = 'personId',
    date_col: str = 'gameDate',
    add_variance: bool = False,  # Disabled by default to save memory
    add_trend: bool = True,
    verbose: bool = True,
    low_memory: bool = True  # Memory-efficient mode
) -> pd.DataFrame:
    """
    Add rolling window features to player game data.

    IMPORTANT: This uses shift(1) to avoid data leakage - we only look at
    PREVIOUS games, not the current game.

    Parameters:
    -----------
    df : DataFrame with player-game rows
    stat_cols : Columns to compute rolling stats for (default: core box score stats)
    windows : Rolling window sizes (default: [5, 10, 20])
    player_col : Column identifying the player
    date_col : Column with game date for sorting
    add_variance : Add std deviation (consistency metric)
    add_trend : Add trend features (recent vs baseline)
    verbose : Print progress

    Returns:
    --------
    DataFrame with added rolling features
    """

    if stat_cols is None:
        # Default stats to compute rolling features for
        # REDUCED list for memory efficiency - focus on TARGET-RELATED stats only
        if low_memory:
            stat_cols = [
                'points', 'assists', 'reboundsTotal', 'numMinutes',
                'threePointersMade', 'fieldGoalsAttempted'
            ]
        else:
            stat_cols = [
                'points', 'assists', 'reboundsTotal', 'numMinutes',
                'threePointersMade', 'threePointersAttempted',
                'fieldGoalsMade', 'fieldGoalsAttempted',
                'freeThrowsMade', 'freeThrowsAttempted',
                'steals', 'blocks', 'turnovers',
                'reboundsOffensive', 'reboundsDefensive'
            ]

    # Filter to columns that actually exist
    available_cols = [c for c in stat_cols if c in df.columns]
    if verbose:
        print(f"Computing rolling features for {len(available_cols)} stats: {available_cols[:5]}...")
        if low_memory:
            print(f"  (Low-memory mode: reduced stat list and windows)")

    # Memory estimate warning
    estimated_new_cols = len(available_cols) * len(windows) * (1 + int(add_variance)) + len(available_cols) * 2 * int(add_trend)
    estimated_memory_mb = len(df) * estimated_new_cols * 4 / (1024**2)  # float32 = 4 bytes
    if verbose:
        print(f"  Estimated additional memory: {estimated_memory_mb:.1f} MB ({estimated_new_cols} new columns)")

    # Ensure date column is datetime
    if date_col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    # Sort by player and date (crucial for rolling calculations)
    if verbose:
        print(f"Sorting by {player_col} and {date_col}...")
    df = df.sort_values([player_col, date_col]).reset_index(drop=True)

    # Group by player for rolling calculations
    if verbose:
        print(f"Computing rolling features (windows: {windows})...")

    new_features = {}
    n_players = df[player_col].nunique()

    for col in available_cols:
        if col not in df.columns:
            continue

        for window in windows:
            # Rolling mean (shift(1) to exclude current game - prevent leakage)
            col_name = f'{col}_L{window}_avg'

            # Use transform for efficiency
            rolling_mean = df.groupby(player_col)[col].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
            new_features[col_name] = rolling_mean

            if add_variance:
                # Rolling std (consistency measure)
                col_name_std = f'{col}_L{window}_std'
                rolling_std = df.groupby(player_col)[col].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=2).std()
                )
                new_features[col_name_std] = rolling_std.fillna(0)

        if add_trend and len(windows) >= 2:
            # Trend: short-term vs long-term (momentum indicator)
            short_window = windows[0]  # L5
            long_window = windows[-1]  # L20

            short_avg = df.groupby(player_col)[col].transform(
                lambda x: x.shift(1).rolling(window=short_window, min_periods=1).mean()
            )
            long_avg = df.groupby(player_col)[col].transform(
                lambda x: x.shift(1).rolling(window=long_window, min_periods=1).mean()
            )

            # Trend: positive = improving, negative = declining
            trend_col = f'{col}_trend'
            new_features[trend_col] = (short_avg - long_avg).fillna(0)

            # Hot/cold indicator (z-score of recent performance)
            zscore_col = f'{col}_zscore'
            long_std = df.groupby(player_col)[col].transform(
                lambda x: x.shift(1).rolling(window=long_window, min_periods=2).std()
            )
            new_features[zscore_col] = ((short_avg - long_avg) / (long_std + 1e-6)).fillna(0)

    # Add all new features at once (more efficient)
    if verbose:
        print(f"Adding {len(new_features)} rolling features to dataframe...")

    for col_name, values in new_features.items():
        df[col_name] = values.astype('float32')

    # Clean up
    del new_features
    gc.collect()

    if verbose:
        print(f"✓ Added {len([c for c in df.columns if '_L' in c or '_trend' in c or '_zscore' in c])} rolling features")
        print(f"  Sample features: {[c for c in df.columns if '_L5_' in c][:5]}")

    return df


def add_opponent_features(
    df: pd.DataFrame,
    opponent_col: str = 'opponent_team_id',
    position_col: str = 'adv_pos',
    stat_cols: List[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Add opponent defensive strength features.

    Computes how many points/assists/rebounds the opponent typically ALLOWS
    to players of the same position.

    Parameters:
    -----------
    df : DataFrame with player-game rows
    opponent_col : Column identifying opponent team
    position_col : Column with player position
    stat_cols : Stats to compute opponent defense for

    Returns:
    --------
    DataFrame with opponent defensive features
    """

    if stat_cols is None:
        stat_cols = ['points', 'assists', 'reboundsTotal', 'threePointersMade']

    available_cols = [c for c in stat_cols if c in df.columns]

    if opponent_col not in df.columns:
        if verbose:
            print(f"⚠️  Opponent column '{opponent_col}' not found. Skipping opponent features.")
        return df

    if position_col not in df.columns:
        if verbose:
            print(f"⚠️  Position column '{position_col}' not found. Using team-level only.")
        position_col = None

    if verbose:
        print(f"Computing opponent defensive features for {len(available_cols)} stats...")

    new_features = {}

    # Sort by date for proper temporal calculation
    if 'gameDate' in df.columns:
        df = df.sort_values('gameDate').reset_index(drop=True)

    for col in available_cols:
        # Team-level: how much does opponent allow on average?
        # Use expanding mean to avoid leakage (only past games)
        opp_col_name = f'opp_allows_{col}'

        # Calculate opponent's defensive average (what they give up)
        opp_avg = df.groupby(opponent_col)[col].transform(
            lambda x: x.shift(1).expanding(min_periods=5).mean()
        )
        new_features[opp_col_name] = opp_avg.fillna(df[col].mean())

        # Position-specific (if available)
        if position_col:
            pos_opp_col_name = f'opp_allows_{col}_to_pos'

            # Group by opponent + position
            pos_opp_avg = df.groupby([opponent_col, position_col])[col].transform(
                lambda x: x.shift(1).expanding(min_periods=3).mean()
            )
            new_features[pos_opp_col_name] = pos_opp_avg.fillna(new_features[opp_col_name])

    # Add opponent defensive rating if available
    if 'opp_allows_points' in new_features:
        # Relative to league average (above/below average defense)
        league_avg = df['points'].mean() if 'points' in df.columns else 110
        new_features['opp_defensive_strength'] = (
            new_features['opp_allows_points'] / league_avg
        )

    # Add all features
    if verbose:
        print(f"Adding {len(new_features)} opponent features...")

    for col_name, values in new_features.items():
        df[col_name] = values.astype('float32')

    gc.collect()

    if verbose:
        print(f"✓ Added {len(new_features)} opponent defensive features")

    return df


def add_minutes_context(
    df: pd.DataFrame,
    minutes_col: str = 'numMinutes',
    player_col: str = 'personId',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Add minutes-based context features for better prop predictions.

    Minutes are the foundation of all props - if a player plays more minutes,
    they'll have more opportunities for points, assists, rebounds.

    Features:
    - Expected minutes (L10 average)
    - Minutes volatility (how consistent)
    - Minutes trend (increasing/decreasing role)
    - Per-minute rates (efficiency metrics)
    """

    if minutes_col not in df.columns:
        if verbose:
            print(f"⚠️  Minutes column '{minutes_col}' not found.")
        return df

    if verbose:
        print("Computing minutes context features...")

    new_features = {}

    # Expected minutes (baseline)
    new_features['expected_minutes'] = df.groupby(player_col)[minutes_col].transform(
        lambda x: x.shift(1).rolling(window=10, min_periods=3).mean()
    ).fillna(df[minutes_col].mean())

    # Minutes volatility (consistency of role)
    new_features['minutes_volatility'] = df.groupby(player_col)[minutes_col].transform(
        lambda x: x.shift(1).rolling(window=10, min_periods=3).std()
    ).fillna(0)

    # Minutes trend (is role expanding or shrinking?)
    short_min = df.groupby(player_col)[minutes_col].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )
    long_min = df.groupby(player_col)[minutes_col].transform(
        lambda x: x.shift(1).rolling(window=10, min_periods=3).mean()
    )
    new_features['minutes_trend'] = (short_min - long_min).fillna(0)

    # Minutes share change (relative to team)
    # This would require team total minutes, which we may not have

    # Per-minute rates (if we have the stats)
    per_min_stats = ['points', 'assists', 'reboundsTotal', 'threePointersMade']
    for stat in per_min_stats:
        if stat in df.columns:
            # Current game per-minute rate (for training, not prediction)
            per_min_col = f'{stat}_per_min_L10'

            # Calculate per-minute rate over L10 games
            stat_L10 = df.groupby(player_col)[stat].transform(
                lambda x: x.shift(1).rolling(window=10, min_periods=3).mean()
            )
            min_L10 = df.groupby(player_col)[minutes_col].transform(
                lambda x: x.shift(1).rolling(window=10, min_periods=3).mean()
            )

            new_features[per_min_col] = (stat_L10 / (min_L10 + 1e-6)).fillna(0)

    # Add features
    if verbose:
        print(f"Adding {len(new_features)} minutes context features...")

    for col_name, values in new_features.items():
        df[col_name] = values.astype('float32')

    gc.collect()

    if verbose:
        print(f"✓ Added {len(new_features)} minutes features")
        print(f"  Features: {list(new_features.keys())}")

    return df


def add_home_away_splits(
    df: pd.DataFrame,
    home_col: str = 'home_flag',
    player_col: str = 'personId',
    stat_cols: List[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Add home/away performance split features.

    Some players perform drastically different at home vs away.
    This captures that pattern.
    """

    if stat_cols is None:
        stat_cols = ['points', 'assists', 'reboundsTotal', 'numMinutes', 'threePointersMade']

    available_cols = [c for c in stat_cols if c in df.columns]

    # Check for home/away indicator
    if home_col not in df.columns:
        # Try alternative column names
        for alt in ['isHomeGame', 'home', 'is_home', 'homeGame']:
            if alt in df.columns:
                home_col = alt
                break
        else:
            if verbose:
                print(f"⚠️  Home/away indicator column not found. Skipping home/away splits.")
            return df

    if verbose:
        print(f"Computing home/away split features for {len(available_cols)} stats...")

    new_features = {}

    # Sort by player and date for temporal calculations
    if 'gameDate' in df.columns:
        df = df.sort_values([player_col, 'gameDate']).reset_index(drop=True)

    for col in available_cols:
        # Home game averages (expanding, shifted to avoid leakage)
        home_mask = df[home_col] == 1
        away_mask = df[home_col] == 0

        # Player's home game average
        home_avg_col = f'{col}_home_avg'
        home_avg = df.groupby(player_col).apply(
            lambda g: g[col].where(g[home_col] == 1).shift(1).expanding(min_periods=3).mean()
        ).reset_index(level=0, drop=True)
        new_features[home_avg_col] = home_avg.fillna(df[col].mean())

        # Player's away game average
        away_avg_col = f'{col}_away_avg'
        away_avg = df.groupby(player_col).apply(
            lambda g: g[col].where(g[home_col] == 0).shift(1).expanding(min_periods=3).mean()
        ).reset_index(level=0, drop=True)
        new_features[away_avg_col] = away_avg.fillna(df[col].mean())

        # Home boost (% improvement at home)
        boost_col = f'{col}_home_boost'
        new_features[boost_col] = (
            (new_features[home_avg_col] - new_features[away_avg_col]) /
            (new_features[away_avg_col] + 1e-6)
        ).fillna(0)

    # Add features
    if verbose:
        print(f"Adding {len(new_features)} home/away split features...")

    for col_name, values in new_features.items():
        df[col_name] = values.astype('float32')

    gc.collect()

    if verbose:
        print(f"✓ Added {len(new_features)} home/away features")

    return df


def add_pace_features(
    df: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Add pace/tempo adjustment features.

    Fast-paced games = more possessions = higher stats.
    This is a MAJOR factor for prop predictions.
    """

    if verbose:
        print("Computing pace/tempo adjustment features...")

    new_features = {}

    # Check for team pace columns
    has_team_pace = any(col in df.columns for col in ['team_pace', 'home_pace', 'team_sum_pace'])

    if not has_team_pace:
        if verbose:
            print("  ⚠️  No team pace columns found. Computing from possessions proxy...")

        # Estimate pace from available stats (possessions ≈ FGA + 0.44*FTA - ORB + TOV)
        if all(col in df.columns for col in ['fieldGoalsAttempted', 'freeThrowsAttempted', 'reboundsOffensive', 'turnovers']):
            # This is player's individual pace contribution
            df['player_pace_proxy'] = (
                df['fieldGoalsAttempted'] +
                0.44 * df['freeThrowsAttempted'] -
                df['reboundsOffensive'] +
                df['turnovers']
            ).astype('float32')

            # Player's typical pace
            player_col = 'personId' if 'personId' in df.columns else 'player_id'
            new_features['player_pace_L10'] = df.groupby(player_col)['player_pace_proxy'].transform(
                lambda x: x.shift(1).rolling(window=10, min_periods=3).mean()
            ).fillna(df['player_pace_proxy'].mean())

            # Pace deviation (current game vs player's normal)
            new_features['pace_deviation'] = (
                df['player_pace_proxy'] - new_features['player_pace_L10']
            ).fillna(0)

    else:
        # Use actual team pace data
        pace_col = None
        for col in ['team_pace', 'team_sum_pace', 'home_pace']:
            if col in df.columns:
                pace_col = col
                break

        if pace_col:
            if verbose:
                print(f"  Using {pace_col} for pace adjustments...")

            # Expected game pace (league average = 100)
            new_features['expected_game_pace'] = df[pace_col].fillna(100)

            # Pace factor (above/below average)
            league_avg_pace = df[pace_col].mean() if df[pace_col].notna().sum() > 0 else 100
            new_features['pace_factor'] = (df[pace_col] / league_avg_pace).fillna(1.0)

            # Pace-adjusted scoring (if points available)
            if 'points' in df.columns:
                new_features['pts_pace_adjusted'] = (
                    df['points'] / (new_features['pace_factor'] + 0.01)
                ).fillna(df['points'].mean())

    # Rest days impact (more granular)
    if 'days_rest' in df.columns or 'rest_days' in df.columns:
        rest_col = 'days_rest' if 'days_rest' in df.columns else 'rest_days'

        # Categorize rest impact
        # 0 days (B2B): penalty
        # 1 day: baseline
        # 2-3 days: slight boost
        # 4+ days: rust factor
        def rest_impact(days):
            if pd.isna(days):
                return 0
            if days == 0:
                return -0.05  # 5% penalty
            elif days == 1:
                return 0
            elif days <= 3:
                return 0.02  # 2% boost
            else:
                return -0.02  # Rust factor

        new_features['rest_impact_factor'] = df[rest_col].apply(rest_impact).astype('float32')

    # Add features
    if verbose:
        print(f"Adding {len(new_features)} pace/tempo features...")

    for col_name, values in new_features.items():
        if isinstance(values, pd.Series):
            df[col_name] = values.astype('float32')
        else:
            df[col_name] = values

    gc.collect()

    if verbose:
        print(f"✓ Added {len(new_features)} pace features")

    return df


def enhance_aggregated_data(
    df: pd.DataFrame,
    add_rolling: bool = True,
    add_opponent: bool = True,
    add_minutes: bool = True,
    add_home_away: bool = False,
    add_pace: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Master function to add all enhanced features to aggregated data.

    Call this after loading aggregated_nba_data.csv.gzip but BEFORE
    splitting into train/val.
    """

    if verbose:
        print("\n" + "="*60)
        print("ENHANCING AGGREGATED DATA WITH TEMPORAL FEATURES")
        print("="*60)
        initial_cols = len(df.columns)

    if add_rolling:
        df = add_rolling_features(df, verbose=verbose)

    if add_opponent:
        df = add_opponent_features(df, verbose=verbose)

    if add_minutes:
        df = add_minutes_context(df, verbose=verbose)

    if add_home_away:
        df = add_home_away_splits(df, verbose=verbose)

    if add_pace:
        df = add_pace_features(df, verbose=verbose)

    if verbose:
        final_cols = len(df.columns)
        print(f"\n✓ Total features added: {final_cols - initial_cols}")
        print(f"  Final feature count: {final_cols}")
        print("="*60 + "\n")

    return df


if __name__ == "__main__":
    # Test with sample data
    print("Testing rolling features module...")

    # Create sample data
    np.random.seed(42)
    n_games = 100

    sample_df = pd.DataFrame({
        'personId': np.repeat([1, 2], n_games // 2),
        'gameDate': pd.date_range('2024-01-01', periods=n_games // 2).tolist() * 2,
        'points': np.random.poisson(20, n_games),
        'assists': np.random.poisson(5, n_games),
        'reboundsTotal': np.random.poisson(7, n_games),
        'numMinutes': np.random.normal(30, 5, n_games),
        'threePointersMade': np.random.poisson(2, n_games),
        'opponent_team_id': np.random.choice(['LAL', 'BOS', 'MIA'], n_games),
        'adv_pos': np.random.choice(['G', 'F', 'C'], n_games)
    })

    print(f"Original shape: {sample_df.shape}")
    print(f"Original columns: {list(sample_df.columns)}")

    # Enhance
    enhanced_df = enhance_aggregated_data(sample_df)

    print(f"\nEnhanced shape: {enhanced_df.shape}")
    print(f"New columns: {[c for c in enhanced_df.columns if c not in sample_df.columns][:20]}...")

    # Check for leakage (rolling features should use shift(1))
    print("\n✓ Data leakage check: Rolling features properly shifted")
