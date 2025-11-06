#!/usr/bin/env python3
"""Quick diagnostic to check data preparation issues."""

import pandas as pd
from pathlib import Path

print("=" * 60)
print("DATA PREPARATION DIAGNOSTICS")
print("=" * 60)

# Check window CSV
window_csv = Path(".window_2002_2006_players.csv")
if window_csv.exists():
    print(f"\n1. Window CSV: {window_csv}")
    df = pd.read_csv(window_csv, nrows=100)
    print(f"   Rows (sample): {len(df):,}")
    print(f"   Columns: {list(df.columns)[:15]}...")
    
    # Check key columns
    if 'teamId' in df.columns:
        print(f"   ✓ teamId exists")
        print(f"     Non-null: {df['teamId'].notna().sum()} / {len(df)}")
        print(f"     Sample: {df['teamId'].head(10).tolist()}")
    else:
        print(f"   ✗ teamId MISSING")
    
    if 'home' in df.columns:
        print(f"   ✓ home exists")
        print(f"     Non-null: {df['home'].notna().sum()} / {len(df)}")
        print(f"     Unique values: {df['home'].unique()[:10]}")
        print(f"     Sample: {df['home'].head(10).tolist()}")
    else:
        print(f"   ✗ home MISSING")
    
    if 'gameDate' in df.columns or 'date' in df.columns:
        date_col = 'gameDate' if 'gameDate' in df.columns else 'date'
        print(f"   ✓ {date_col} exists")
        print(f"     Non-null: {df[date_col].notna().sum()} / {len(df)}")
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        print(f"     Date range: {df[date_col].min()} to {df[date_col].max()}")
    else:
        print(f"   ✗ date column MISSING")
else:
    print(f"\n✗ Window CSV not found: {window_csv}")

# Check current season CSV
current_csv = Path(".current_season_players_temp.csv")
if current_csv.exists():
    print(f"\n2. Current Season CSV: {current_csv}")
    df = pd.read_csv(current_csv, nrows=100)
    print(f"   Rows (sample): {len(df):,}")
    print(f"   Has teamId: {'teamId' in df.columns}")
    print(f"   Has home: {'home' in df.columns}")
    if 'teamId' in df.columns:
        print(f"     teamId sample: {df['teamId'].head(5).tolist()}")
    if 'home' in df.columns:
        print(f"     home sample: {df['home'].head(5).tolist()}")
else:
    print(f"\n✗ Current season CSV not found: {current_csv}")

# Check main PlayerStatistics.csv
player_csv = Path("PlayerStatistics.csv")
if player_csv.exists():
    print(f"\n3. PlayerStatistics.csv")
    df = pd.read_csv(player_csv, nrows=100)
    print(f"   Columns: {list(df.columns)[:15]}...")
    print(f"   Has teamId: {'teamId' in df.columns}")
    print(f"   Has home: {'home' in df.columns}")
    if 'home' in df.columns:
        print(f"     home values: {df['home'].value_counts().to_dict()}")
else:
    print(f"\n✗ PlayerStatistics.csv not found")

print("\n" + "=" * 60)
print("RECOMMENDATIONS:")
print("=" * 60)

recommendations = []
if window_csv.exists():
    df = pd.read_csv(window_csv, nrows=100)
    if 'teamId' not in df.columns:
        recommendations.append("• Window CSV missing teamId - will use home flag for merge")
    if 'home' not in df.columns or df['home'].isna().all():
        recommendations.append("• Window CSV missing/empty home column - merge will fail!")

if recommendations:
    for rec in recommendations:
        print(rec)
else:
    print("No issues detected in sample data")

print("=" * 60)
