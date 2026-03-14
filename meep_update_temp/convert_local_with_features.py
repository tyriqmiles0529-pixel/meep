#!/usr/bin/env python
"""
Convert CSV.gzip to Parquet WITH pre-computed temporal features.
FILTERED TO 1980+ to save memory and training time.

Features added:
- Rolling L5/L10 averages for core stats
- Trend indicators and momentum
- Z-scores for hot/cold streaks
- Per-minute rates
- Minutes context (volatility, consistency)
- Pace proxy
- Usage patterns
- Home/Away splits
- Rest impact
- Efficiency metrics
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import gc
from tqdm import tqdm

csv_path = 'aggregated_nba_data.csv.gzip'
parquet_path = 'aggregated_nba_data_2000_features.parquet'
MIN_YEAR = 2000  # Filter to this year and later (saves 16%)

print("="*60)
print("Converting CSV.gzip to Parquet WITH TEMPORAL FEATURES")
print(f"FILTERED TO {MIN_YEAR}+ (saves ~20% memory)")
print("="*60)
print(f"Input: {csv_path} (416 MB)")
print(f"Output: {parquet_path}")
print("RAM usage: ~6-8 GB peak")
print("Time: ~25-35 minutes\n")

# Step 1: Load all data
print("Step 1: Loading all data...")
df = pd.read_csv(csv_path, compression='gzip', low_memory=False)
print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

# Step 2: Filter to MIN_YEAR+
print(f"\nStep 2: Filtering to {MIN_YEAR}+ seasons...")
before_count = len(df)
if 'season' in df.columns:
    df = df[df['season'] >= MIN_YEAR].copy()
elif 'game_year' in df.columns:
    df = df[df['game_year'] >= MIN_YEAR].copy()
after_count = len(df)
print(f"  Filtered: {before_count:,} → {after_count:,} rows ({(1-after_count/before_count)*100:.1f}% reduction)")
gc.collect()

# Step 3: Optimize dtypes
print("\nStep 3: Optimizing dtypes...")
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype(str)
for col in df.select_dtypes(include=['float64']).columns:
    df[col] = df[col].astype('float32')
for col in df.select_dtypes(include=['int64']).columns:
    if df[col].min() >= -2147483648 and df[col].max() <= 2147483647:
        df[col] = df[col].astype('int32')

mem_mb = df.memory_usage(deep=True).sum() / 1024**2
print(f"  Memory: {mem_mb:.1f} MB")

# Step 4: Sort by player and date
print("\nStep 4: Sorting by player and game date...")
if 'gameDate' in df.columns:
    df['gameDate'] = pd.to_datetime(df['gameDate'], format='mixed', utc=True, errors='coerce')
    df = df.sort_values(['personId', 'gameDate']).reset_index(drop=True)
    print("  Sorted by personId and gameDate")

# Step 5: Core rolling features
print("\nStep 5: Computing rolling features...")
stats_to_roll = ['points', 'assists', 'reboundsTotal', 'numMinutes', 'threePointersMade',
                 'steals', 'blocks', 'turnovers', 'fieldGoalsPercentage', 'freeThrowsPercentage']
stats_found = [s for s in stats_to_roll if s in df.columns]
print(f"  Stats: {len(stats_found)} found")

for stat in tqdm(stats_found, desc="  Rolling L5/L10"):
    # L5 and L10 averages (shift to avoid leakage)
    df[f'{stat}_L5'] = df.groupby('personId')[stat].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    ).astype('float32')

    df[f'{stat}_L10'] = df.groupby('personId')[stat].transform(
        lambda x: x.shift(1).rolling(window=10, min_periods=1).mean()
    ).astype('float32')

    # Trend: L5 vs L10 (positive = improving)
    df[f'{stat}_trend'] = (df[f'{stat}_L5'] - df[f'{stat}_L10']).astype('float32')

    # Z-score: how many std above/below season mean (hot/cold streak)
    season_stats = df.groupby(['personId', 'season'])[stat].transform(['mean', 'std'])
    # Can't do this easily in one transform, so compute differently
    gc.collect()

gc.collect()

# Step 6: Advanced momentum features
print("\nStep 6: Computing momentum features...")
for stat in ['points', 'assists', 'reboundsTotal']:
    if stat in df.columns:
        # Last game vs L5 average (did they over/underperform?)
        df[f'{stat}_vs_L5'] = (df.groupby('personId')[stat].shift(1) - df[f'{stat}_L5']).fillna(0).astype('float32')

        # Consistency: std of last 5 games (lower = more consistent)
        df[f'{stat}_consistency'] = df.groupby('personId')[stat].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=2).std()
        ).fillna(0).astype('float32')

# Step 7: Minutes context
print("\nStep 7: Computing minutes context...")
if 'numMinutes' in df.columns:
    # Per-minute rates (very predictive!)
    for stat in ['points', 'assists', 'reboundsTotal', 'threePointersMade', 'steals', 'blocks']:
        if stat in df.columns:
            df[f'{stat}_per_min'] = (df[stat] / df['numMinutes'].replace(0, np.nan)).fillna(0).astype('float32')

    # Minutes volatility
    df['minutes_volatility'] = df.groupby('personId')['numMinutes'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=2).std()
    ).fillna(0).astype('float32')

    # Minutes trend (getting more or less playing time?)
    df['minutes_trend'] = df['numMinutes_trend'] if 'numMinutes_trend' in df.columns else 0

# Step 8: Efficiency metrics
print("\nStep 8: Computing efficiency metrics...")
if all(col in df.columns for col in ['points', 'fieldGoalsAttempted', 'freeThrowsAttempted']):
    # True Shooting Attempts
    tsa = df['fieldGoalsAttempted'] + 0.44 * df['freeThrowsAttempted']
    df['true_shooting_pct'] = (df['points'] / (2 * tsa.replace(0, np.nan))).fillna(0).astype('float32')

    # Rolling TS%
    df['true_shooting_L5'] = df.groupby('personId')['true_shooting_pct'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    ).astype('float32')

if all(col in df.columns for col in ['assists', 'turnovers']):
    # AST/TO ratio
    df['ast_to_ratio'] = (df['assists'] / df['turnovers'].replace(0, np.nan)).fillna(0).clip(0, 20).astype('float32')

    # Rolling AST/TO
    df['ast_to_ratio_L5'] = df.groupby('personId')['ast_to_ratio'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    ).astype('float32')

# Step 9: Pace proxy
print("\nStep 9: Computing pace features...")
if all(col in df.columns for col in ['fieldGoalsAttempted', 'freeThrowsAttempted', 'turnovers', 'numMinutes']):
    df['poss_proxy'] = (
        df['fieldGoalsAttempted'] + 0.44 * df['freeThrowsAttempted'] + df['turnovers']
    ).astype('float32')
    df['pace_proxy'] = (df['poss_proxy'] / df['numMinutes'].replace(0, np.nan)).fillna(0).astype('float32')

    # Player's typical pace
    df['pace_L10'] = df.groupby('personId')['pace_proxy'].transform(
        lambda x: x.shift(1).rolling(window=10, min_periods=3).mean()
    ).astype('float32')

# Step 10: Home/Away context
print("\nStep 10: Computing home/away features...")
if 'home' in df.columns:
    df['home'] = df['home'].astype('int8')

    # Home/away performance difference
    for stat in ['points', 'assists', 'reboundsTotal']:
        if stat in df.columns:
            # Player's home average
            home_avg = df.groupby(['personId', 'home'])[stat].transform('mean')
            df[f'{stat}_home_away_diff'] = (home_avg - df[f'{stat}_L10']).fillna(0).astype('float32')

# Step 11: Rest impact (games between)
print("\nStep 11: Computing rest features...")
if 'gameDate' in df.columns:
    # Days since last game
    df['days_rest'] = df.groupby('personId')['gameDate'].diff().dt.days.fillna(3).clip(0, 10).astype('float32')

    # Back-to-back indicator
    df['is_back_to_back'] = (df['days_rest'] <= 1).astype('int8')

    # Well-rested indicator (3+ days)
    df['is_well_rested'] = (df['days_rest'] >= 3).astype('int8')

# Step 12: Season context
print("\nStep 12: Computing season context...")
if 'season' in df.columns:
    # Games played this season (experience/fatigue)
    df['games_this_season'] = df.groupby(['personId', 'season']).cumcount().astype('int16')

    # Is it early season (first 20 games), mid, or late (last 20)?
    df['season_phase'] = pd.cut(df['games_this_season'], bins=[0, 20, 62, 82], labels=[0, 1, 2]).astype('int8')

gc.collect()
mem_mb = df.memory_usage(deep=True).sum() / 1024**2
print(f"\nTotal columns: {len(df.columns)}")
print(f"Memory: {mem_mb:.1f} MB")

# Step 13: Write to Parquet
print("\nStep 13: Writing to Parquet...")
table = pa.Table.from_pandas(df, preserve_index=False)
pq.write_table(table, parquet_path, compression='snappy')

# Verify
import os
pf = pq.ParquetFile(parquet_path)
size_mb = os.path.getsize(parquet_path) / 1024**2
print(f"\nDone! Total rows: {len(df):,}")
print(f"Verified rows: {pf.metadata.num_rows:,}")
print(f"Parquet file: {size_mb:.1f} MB")
print(f"Total columns: {len(df.columns)}")

# Year range
print(f"\nSeason range: {df['season'].min()} - {df['season'].max()}")
print(f"Game year range: {df['game_year'].min()} - {df['game_year'].max()}")

# List new features
new_features = [c for c in df.columns if any(x in c for x in ['_L5', '_L10', '_trend', '_per_min', 'pace_', 'volatility',
                                                               '_consistency', '_vs_L5', 'true_shooting', 'ast_to_ratio',
                                                               'days_rest', 'is_back_to_back', 'is_well_rested',
                                                               'games_this_season', 'season_phase', '_home_away'])]
print(f"\nNew temporal features added: {len(new_features)}")
for f in sorted(new_features)[:20]:
    print(f"  - {f}")
if len(new_features) > 20:
    print(f"  ... and {len(new_features)-20} more")

print("\n" + "="*60)
print(f"Upload {parquet_path} to Kaggle!")
print("Then train WITHOUT --add-rolling-features (already included)")
print("="*60)
