#!/usr/bin/env python3
"""
Build a team ID → abbreviation mapping from the main NBA dataset
This allows us to merge team priors for ALL games, not just games with odds

Strategy:
1. TeamStatistics has teamTricode column (3-letter abbreviation like "LAL", "BOS")
2. We can map teamId → teamTricode
3. Then use teamTricode to match with Basketball Reference abbreviations
"""

import pandas as pd
from pathlib import Path
import kagglehub

# Download main dataset
dataset = "eoinamoore/historical-nba-data-and-player-box-scores"
print(f"Downloading {dataset}...")
ds_root = Path(kagglehub.dataset_download(dataset))
print(f"Downloaded to: {ds_root}")

# Find TeamStatistics CSV
teams_csv = None
for p in ds_root.glob("*.csv"):
    if "teamstatistics" in p.name.lower() or "team_statistics" in p.name.lower():
        teams_csv = p
        break

if not teams_csv:
    print("ERROR: TeamStatistics CSV not found!")
    exit(1)

print(f"\nReading {teams_csv.name}...")

# Read TeamStatistics with relevant columns
cols = ["teamId", "teamTricode", "teamName", "teamCity", "gameDate"]
df = pd.read_csv(teams_csv, low_memory=False, usecols=[c for c in cols if c in pd.read_csv(teams_csv, nrows=0).columns])

print(f"Loaded {len(df):,} rows")
print(f"Columns: {list(df.columns)}")

# Check what team ID columns exist
if "teamTricode" not in df.columns:
    print("\nERROR: teamTricode not found! Checking available columns...")
    print(list(pd.read_csv(teams_csv, nrows=0).columns))
    exit(1)

# Build team ID → abbreviation mapping
print("\nBuilding team ID → abbreviation mapping...")

# Convert team IDs to string
df["teamId"] = df["teamId"].astype(str).str.strip()
df["teamTricode"] = df["teamTricode"].astype(str).str.strip().str.upper()

# Get most common abbreviation for each team ID (handles rebrands)
team_map = df.groupby("teamId")["teamTricode"].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]).to_dict()

print(f"\nCreated mapping for {len(team_map)} teams:")
for tid, abbr in sorted(team_map.items(), key=lambda x: x[1]):
    # Get team name for display
    team_name = df[df["teamId"] == tid]["teamName"].iloc[0] if "teamName" in df.columns else ""
    print(f"  {tid}: {abbr} ({team_name})")

# Save to CSV for reference
output = pd.DataFrame([
    {"team_id": tid, "abbreviation": abbr}
    for tid, abbr in team_map.items()
])
output.to_csv("team_id_to_abbrev.csv", index=False)
print(f"\n✓ Saved mapping to team_id_to_abbrev.csv")

# Show sample of team data
print("\nSample team data:")
print(df[["teamId", "teamTricode", "teamName", "teamCity"]].drop_duplicates("teamId").head(15))
