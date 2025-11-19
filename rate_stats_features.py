"""
Rate Stats Feature Engineering

Adds per-minute rate stats (PPM, APM, RPM, 3PM) as features for training.

Why rate stats?
- More stable than totals (less variance)
- Better for minutes-first prediction pipeline
- Helps models learn underlying rates independent of playing time

Usage:
    from rate_stats_features import add_rate_stats

    df = add_rate_stats(df)
    # Adds columns: ppm, apm, rpm, threepm
"""

import pandas as pd
import numpy as np


def add_rate_stats(df: pd.DataFrame, min_minutes: float = 1.0) -> pd.DataFrame:
    """
    Add per-minute rate stats to DataFrame.

    Args:
        df: DataFrame with columns: points, assists, rebounds/reboundsTotal,
            threes/threePointersMade, minutes/numMinutes
        min_minutes: Minimum minutes to avoid division by zero (default: 1.0)

    Returns:
        DataFrame with added columns: ppm, apm, rpm, threepm

    Example:
        >>> df = pd.DataFrame({
        ...     'points': [25, 18, 0],
        ...     'assists': [5, 3, 0],
        ...     'reboundsTotal': [8, 6, 0],
        ...     'threePointersMade': [3, 2, 0],
        ...     'numMinutes': [35, 28, 0]
        ... })
        >>> df = add_rate_stats(df)
        >>> df[['points', 'numMinutes', 'ppm']].head()
           points  numMinutes       ppm
        0      25          35  0.714286
        1      18          28  0.642857
        2       0           0  0.000000
    """
    df = df.copy()

    # Find correct column names (handle variations)
    points_col = 'points' if 'points' in df.columns else 'PTS'
    assists_col = 'assists' if 'assists' in df.columns else 'AST'

    if 'reboundsTotal' in df.columns:
        rebounds_col = 'reboundsTotal'
    elif 'REB' in df.columns:
        rebounds_col = 'REB'
    elif 'rebounds' in df.columns:
        rebounds_col = 'rebounds'
    else:
        rebounds_col = None

    if 'threePointersMade' in df.columns:
        threes_col = 'threePointersMade'
    elif 'FG3M' in df.columns:
        threes_col = 'FG3M'
    elif 'threes' in df.columns:
        threes_col = 'threes'
    else:
        threes_col = None

    if 'numMinutes' in df.columns:
        minutes_col = 'numMinutes'
    elif 'MIN' in df.columns:
        minutes_col = 'MIN'
    elif 'minutes' in df.columns:
        minutes_col = 'minutes'
    else:
        raise ValueError("No minutes column found (numMinutes, MIN, minutes)")

    # Ensure minutes are positive (avoid division by zero)
    minutes = df[minutes_col].fillna(0).replace(0, min_minutes)

    # Calculate rate stats
    df['ppm'] = df[points_col].fillna(0) / minutes
    df['apm'] = df[assists_col].fillna(0) / minutes

    if rebounds_col:
        df['rpm'] = df[rebounds_col].fillna(0) / minutes
    else:
        df['rpm'] = 0.0

    if threes_col:
        df['threepm'] = df[threes_col].fillna(0) / minutes
    else:
        df['threepm'] = 0.0

    # Clean up infinities and NaNs
    df['ppm'] = df['ppm'].replace([np.inf, -np.inf], 0).fillna(0)
    df['apm'] = df['apm'].replace([np.inf, -np.inf], 0).fillna(0)
    df['rpm'] = df['rpm'].replace([np.inf, -np.inf], 0).fillna(0)
    df['threepm'] = df['threepm'].replace([np.inf, -np.inf], 0).fillna(0)

    return df


def add_rolling_rate_stats(df: pd.DataFrame,
                           windows: list = [3, 5, 7, 10, 15],
                           group_col: str = 'personId') -> pd.DataFrame:
    """
    Add rolling averages of rate stats.

    Args:
        df: DataFrame with rate stats (ppm, apm, rpm, threepm)
        windows: List of window sizes for rolling averages
        group_col: Column to group by (default: 'personId' for player)

    Returns:
        DataFrame with added columns: ppm_L5_avg, apm_L10_avg, etc.

    Example:
        >>> df = add_rate_stats(df)
        >>> df = add_rolling_rate_stats(df, windows=[5, 10])
        >>> df[['ppm', 'ppm_L5_avg', 'ppm_L10_avg']].head()
    """
    df = df.copy()

    # Ensure data is sorted by date within each player
    if 'GAME_DATE' in df.columns:
        df = df.sort_values([group_col, 'GAME_DATE'])
    elif 'gameDate' in df.columns:
        df = df.sort_values([group_col, 'gameDate'])

    rate_stats = ['ppm', 'apm', 'rpm', 'threepm']

    for stat in rate_stats:
        if stat not in df.columns:
            continue

        for window in windows:
            col_name = f'{stat}_L{window}_avg'

            # Rolling average (shift by 1 to avoid leakage)
            df[col_name] = (
                df.groupby(group_col)[stat]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            )

    return df


def add_rate_std_features(df: pd.DataFrame,
                          windows: list = [5, 10],
                          group_col: str = 'personId') -> pd.DataFrame:
    """
    Add rolling standard deviations of rate stats (consistency metrics).

    Args:
        df: DataFrame with rate stats (ppm, apm, rpm, threepm)
        windows: List of window sizes for rolling std
        group_col: Column to group by

    Returns:
        DataFrame with added columns: ppm_L5_std, apm_L10_std, etc.
    """
    df = df.copy()

    # Ensure data is sorted
    if 'GAME_DATE' in df.columns:
        df = df.sort_values([group_col, 'GAME_DATE'])
    elif 'gameDate' in df.columns:
        df = df.sort_values([group_col, 'gameDate'])

    rate_stats = ['ppm', 'apm', 'rpm', 'threepm']

    for stat in rate_stats:
        if stat not in df.columns:
            continue

        for window in windows:
            col_name = f'{stat}_L{window}_std'

            # Rolling std (shift by 1 to avoid leakage)
            df[col_name] = (
                df.groupby(group_col)[stat]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=2).std())
            )

            # Fill NaNs with 0 (not enough data)
            df[col_name] = df[col_name].fillna(0)

    return df


# Example usage
if __name__ == "__main__":
    # Create sample data
    df = pd.DataFrame({
        'personId': [1, 1, 1, 1, 2, 2, 2, 2],
        'gameDate': pd.date_range('2024-01-01', periods=8, freq='D'),
        'points': [25, 18, 32, 22, 15, 20, 12, 18],
        'assists': [5, 3, 8, 6, 2, 4, 1, 3],
        'reboundsTotal': [8, 6, 10, 7, 4, 5, 3, 4],
        'threePointersMade': [3, 2, 5, 3, 1, 2, 0, 2],
        'numMinutes': [35, 28, 38, 32, 22, 28, 18, 25]
    })

    print("Original DataFrame:")
    print(df[['points', 'numMinutes']].head())

    # Add rate stats
    df = add_rate_stats(df)
    print("\nWith Rate Stats:")
    print(df[['points', 'numMinutes', 'ppm']].head())

    # Add rolling averages
    df = add_rolling_rate_stats(df, windows=[3, 5])
    print("\nWith Rolling Rate Stats:")
    print(df[['ppm', 'ppm_L3_avg', 'ppm_L5_avg']].head())

    # Add std features
    df = add_rate_std_features(df, windows=[3])
    print("\nWith Rate Std:")
    print(df[['ppm', 'ppm_L3_avg', 'ppm_L3_std']].head())
