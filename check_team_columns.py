#!/usr/bin/env python3
"""
Check what columns are actually available in TeamStatistics.csv
"""

import pandas as pd
from pathlib import Path
import kagglehub

# Download main dataset
dataset = "eoinamoore/historical-nba-data-and-player-box-scores"
print(f"Downloading {dataset}...")
ds_root = Path(kagglehub.dataset_download(dataset))
print(f"Downloaded to: {ds_root}\n")

# Find TeamStatistics CSV
teams_csv = None
for p in ds_root.glob("*.csv"):
    if "teamstatistics" in p.name.lower() or "team_statistics" in p.name.lower():
        teams_csv = p
        break

if not teams_csv:
    print("ERROR: TeamStatistics CSV not found!")
    exit(1)

print(f"Found: {teams_csv.name}\n")

# Read just the header
df_header = pd.read_csv(teams_csv, nrows=0)
print(f"AVAILABLE COLUMNS IN {teams_csv.name}:")
print("="*60)
for i, col in enumerate(df_header.columns, 1):
    print(f"{i:3d}. {col}")

print("\n" + "="*60)

# Check for team abbreviation/name columns
abbrev_cols = [c for c in df_header.columns if any(word in c.lower() for word in ['abbrev', 'tricode', 'code', 'abbr'])]
name_cols = [c for c in df_header.columns if any(word in c.lower() for word in ['name', 'city'])]
id_cols = [c for c in df_header.columns if any(word in c.lower() for word in ['teamid', 'team_id'])]

print("\nRELEVANT COLUMNS FOR TEAM IDENTIFICATION:")
print(f"  Team ID columns: {id_cols}")
print(f"  Team abbreviation columns: {abbrev_cols}")
print(f"  Team name columns: {name_cols}")

# Sample data
if abbrev_cols or name_cols:
    cols_to_read = id_cols + abbrev_cols + name_cols
    sample = pd.read_csv(teams_csv, nrows=100, usecols=cols_to_read)

    print("\nSAMPLE DATA (first 20 unique teams):")
    print("="*60)
    sample_unique = sample.drop_duplicates(subset=id_cols if id_cols else name_cols)
    print(sample_unique.head(20))
