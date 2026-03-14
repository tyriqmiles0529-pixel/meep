#!/usr/bin/env python
"""
Upload NBA Data to Modal

Downloads Kaggle dataset and uploads to Modal volume for training.
Run this ONCE before training.

Usage:
    modal run modal_upload_data.py
"""

import modal

# Create Modal app
app = modal.App("nba-data-upload")

# Create data volume (persistent storage)
data_volume = modal.Volume.from_name("nba-data", create_if_missing=True)

# Define image with Kaggle CLI
image = modal.Image.debian_slim().pip_install("kaggle")


@app.function(
    image=image,
    volumes={"/data": data_volume},
    secrets=[modal.Secret.from_name("kaggle-secret")],  # Kaggle credentials
    timeout=7200,  # 2 hours for multiple downloads
)
def download_kaggle_datasets():
    """Download BOTH Kaggle NBA datasets and upload to Modal volume"""
    import os
    import subprocess

    print("="*70)
    print("DOWNLOADING KAGGLE NBA DATASETS")
    print("="*70)

    # Set Kaggle credentials from Modal secret
    os.environ["KAGGLE_USERNAME"] = os.environ.get("KAGGLE_USERNAME", "")
    os.environ["KAGGLE_KEY"] = os.environ.get("KAGGLE_KEY", "")

    # Both datasets needed for all CSVs
    datasets = [
        ("eoinamoore/historical-nba-data-and-player-box-scores", "dataset1"),  # PlayerStatistics.csv
        ("sumitrodatta/nba-aba-baa-stats", "dataset2"),  # Advanced, Per 100, PBP, Shooting
    ]

    download_dir = "/data/kaggle_download"
    os.makedirs(download_dir, exist_ok=True)

    # Download each dataset
    for dataset, name in datasets:
        print(f"\n[{name}] Downloading: {dataset}")
        dataset_dir = os.path.join(download_dir, name)
        os.makedirs(dataset_dir, exist_ok=True)

        subprocess.run([
            "kaggle", "datasets", "download",
            "-d", dataset,
            "-p", dataset_dir,
            "--unzip"
        ], check=True)

        print(f"OK Downloaded to {dataset_dir}")

    # List downloaded files
    print("\nDownloaded files:")
    for root, dirs, files in os.walk(download_dir):
        for f in files:
            filepath = os.path.join(root, f)
            size_mb = os.path.getsize(filepath) / 1024 / 1024
            print(f"  {f}: {size_mb:.1f} MB")

    # Organize CSVs into csv_dir for easy access
    csv_dir = "/data/csv_dir"
    os.makedirs(csv_dir, exist_ok=True)

    import shutil

    # Map CSV files to their source datasets
    csv_mapping = [
        # From dataset1 (eoinamoore/historical-nba-data-and-player-box-scores)
        ("PlayerStatistics.csv", "dataset1"),  # Game-level box scores
        ("Players.csv", "dataset1"),            # Player biographical data (height, weight, position)
        ("TeamStatistics.csv", "dataset1"),     # Team-level box scores
        ("Games.csv", "dataset1"),              # Game context (arena, attendance)

        # From dataset2 (sumitrodatta/nba-aba-baa-stats)
        ("Advanced.csv", "dataset2", "Player Advanced.csv"),    # PER, BPM, VORP, TS%, usage%
        ("Per 100 Poss.csv", "dataset2", "Player Per 100 Poss.csv"), # Pace-adjusted stats
        ("Player Play By Play.csv", "dataset2", "Player Play-By-Play.csv"), # Plus/minus, turnovers, fouls
        ("Player Shooting.csv", "dataset2"),    # Shooting zones and percentages
        ("Team Summaries.csv", "dataset2"),     # Team context
    ]

    print("\n" + "="*70)
    print("ORGANIZING CSVs")
    print("="*70)

    for mapping in csv_mapping:
        # Handle tuples with 2 or 3 elements (source_file, dataset, [dest_file])
        if len(mapping) == 2:
            csv_file, source_dataset = mapping
            dest_file = csv_file
        else:
            csv_file, source_dataset, dest_file = mapping

        # Look in the specific dataset directory
        dataset_dir = os.path.join(download_dir, source_dataset)

        # Try to find the file in the dataset directory
        found = False
        for root, dirs, files in os.walk(dataset_dir):
            for f in files:
                if f == csv_file or f.lower() == csv_file.lower():
                    src = os.path.join(root, f)
                    dst = os.path.join(csv_dir, dest_file)
                    shutil.copy(src, dst)
                    size_mb = os.path.getsize(src) / 1024 / 1024
                    print(f"OK [{source_dataset}] {csv_file} -> {dest_file}: {size_mb:.1f} MB")
                    found = True
                    break
            if found:
                break

        if not found:
            print(f"⚠ [{source_dataset}] {csv_file}: NOT FOUND")

    # Commit volume to persist changes
    data_volume.commit()

    print("\n" + "="*70)
    print("DATA UPLOAD COMPLETE")
    print("="*70)
    print(f"Location: Modal volume 'nba-data'")
    print(f"CSV files: /data/csv_dir/")
    print(f"Ready for training!")


@app.function(
    volumes={"/data": data_volume},
)
def list_data():
    """List files in the data volume"""
    import os

    print("="*70)
    print("DATA VOLUME CONTENTS")
    print("="*70)

    csv_dir = "/data/csv_dir"
    if os.path.exists(csv_dir):
        print(f"\n{csv_dir}:")
        for f in os.listdir(csv_dir):
            filepath = os.path.join(csv_dir, f)
            size_mb = os.path.getsize(filepath) / 1024 / 1024
            print(f"  {f}: {size_mb:.1f} MB")
    else:
        print("❌ csv_dir/ not found. Run download_kaggle_dataset() first.")

    print("="*70)


@app.local_entrypoint()
def main():
    """Main entry point"""
    print("Choose an option:")
    print("1. Download Kaggle dataset")
    print("2. List data volume contents")

    choice = input("\nEnter choice (1 or 2): ").strip()

    if choice == "1":
        print("\nStarting download...")
        download_kaggle_datasets.remote()
    elif choice == "2":
        list_data.remote()
    else:
        print("Invalid choice")


# Alternative: Run directly
if __name__ == "__main__":
    # Uncomment one:

    # Option 1: Download data
    with app.run():
        download_kaggle_datasets.remote()

    # Option 2: List data
    # with app.run():
    #     list_data.remote()
