"""
Upload Aggregated NBA Dataset to Kaggle Datasets

Run this script in your Kaggle notebook (where aggregated_nba_data.csv.gzip exists)
to publish it as a downloadable dataset.

Usage (in Kaggle notebook):
    !python upload_aggregated_to_kaggle.py
"""

import os
import json
from pathlib import Path

# Configuration
DATASET_SLUG = "aggregated-nba-data"  # URL-friendly name
DATASET_TITLE = "Aggregated NBA Player Statistics (2002-2026)"
DATASET_FILE = "aggregated_nba_data.csv.gzip"

# Check if file exists
if not os.path.exists(DATASET_FILE):
    print(f"❌ Error: {DATASET_FILE} not found in current directory!")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files available: {os.listdir('.')}")
    exit(1)

file_size_mb = os.path.getsize(DATASET_FILE) / (1024 * 1024)
print(f"✓ Found {DATASET_FILE} ({file_size_mb:.2f} MB)")

# Create dataset metadata
metadata = {
    "title": DATASET_TITLE,
    "id": f"tyriqmiles/{DATASET_SLUG}",
    "licenses": [{"name": "CC0-1.0"}],
    "keywords": ["basketball", "nba", "sports", "statistics", "player-stats"],
    "resources": [
        {
            "path": DATASET_FILE,
            "description": "Pre-aggregated NBA player statistics with rolling averages, team context, opponent matchups, and Basketball Reference priors. Optimized for player prop predictions."
        }
    ],
    "description": """# Aggregated NBA Player Statistics Dataset

This dataset contains pre-processed NBA player statistics from 2002-2026 with:

## Features Included:
- **Player Stats**: Points, rebounds, assists, 3PM, minutes, shooting percentages
- **Rolling Averages**: L3, L5, L10 game averages for all stats
- **Per-Minute Rates**: Position-agnostic efficiency metrics
- **Team Context**: Pace, offensive/defensive strength, recent form
- **Opponent Matchups**: Defensive ratings, matchup edges
- **Basketball Reference Priors**: Historical O-Rtg, D-Rtg, Four Factors
- **Temporal Features**: Season, era, home/away splits
- **Advanced Stats**: Usage rate, TS%, eFG%, true shooting

## Use Cases:
- NBA player prop predictions (points, rebounds, assists)
- Daily fantasy sports (DFS) modeling
- Sports betting analytics
- Player performance analysis

## File Format:
- Gzipped CSV for efficient storage/transfer
- 150+ engineered features per player-game
- ~1.6M rows (all player-games from 2002-2026)

## Data Source:
Aggregated from [Historical NBA Data and Player Box Scores](https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores)
with Basketball Reference statistical priors.

## Created By:
Generated using memory-optimized aggregation pipeline for NBA prediction models.
"""
}

# Write metadata file
metadata_path = "dataset-metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Created {metadata_path}")
print("\n" + "="*70)
print("Dataset Metadata:")
print("="*70)
print(json.dumps(metadata, indent=2))
print("="*70)

# Create/update dataset
print("\n📤 Uploading dataset to Kaggle...")
print("   This may take several minutes depending on file size...\n")

try:
    # Try to create new dataset
    os.system(f'kaggle datasets create -p . -q')
    print("\n✅ Dataset created successfully!")
    print(f"\n🔗 View your dataset at:")
    print(f"   https://www.kaggle.com/datasets/tyriqmiles/{DATASET_SLUG}")

except Exception as e:
    print(f"\n⚠️  Dataset may already exist. Trying to update...")
    try:
        os.system(f'kaggle datasets version -p . -m "Updated aggregated dataset" -q')
        print("\n✅ Dataset updated successfully!")
        print(f"\n🔗 View your dataset at:")
        print(f"   https://www.kaggle.com/datasets/tyriqmiles/{DATASET_SLUG}")
    except Exception as e2:
        print(f"\n❌ Error: {e2}")
        print("\nManual steps:")
        print("1. Go to https://www.kaggle.com/datasets")
        print("2. Click 'New Dataset'")
        print("3. Upload aggregated_nba_data.csv.gzip")
        print(f"4. Set title to: {DATASET_TITLE}")

print("\n" + "="*70)
print("NEXT STEPS:")
print("="*70)
print("1. Visit the dataset URL above")
print("2. Verify the upload completed successfully")
print("3. Download locally using:")
print(f"   kaggle datasets download -d tyriqmiles/{DATASET_SLUG}")
print("="*70)
