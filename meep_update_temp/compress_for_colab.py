#!/usr/bin/env python3
"""
Compress PlayerStatistics.csv for Colab upload
Creates a valid ZIP file with maximum compression
"""

import zipfile
import os
from pathlib import Path

# Use the full Kaggle dataset
kaggle_csv = Path(r"C:\Users\tmiles11\.cache\kagglehub\datasets\eoinamoore\historical-nba-data-and-player-box-scores\versions\263\PlayerStatistics.csv")
output_zip = Path("PlayerStatistics.csv.zip")

print(f"Source: {kaggle_csv}")
print(f"Size: {kaggle_csv.stat().st_size / 1024 / 1024:.1f} MB")

# Remove old zip if exists
if output_zip.exists():
    print(f"\nRemoving old {output_zip}...")
    output_zip.unlink()

print(f"\nCompressing to {output_zip}...")
print("This will take ~2 minutes...\n")

with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
    # Write with the simple name (not full path)
    zf.write(kaggle_csv, arcname='PlayerStatistics.csv')

final_size = output_zip.stat().st_size / 1024 / 1024
original_size = kaggle_csv.stat().st_size / 1024 / 1024
compression_ratio = (1 - final_size/original_size) * 100

print(f"✅ Created {output_zip}")
print(f"   Original: {original_size:.1f} MB")
print(f"   Compressed: {final_size:.1f} MB")
print(f"   Compression: {compression_ratio:.1f}%")
print(f"\n📤 Upload this file to Colab!")
