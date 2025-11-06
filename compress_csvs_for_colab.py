#!/usr/bin/env python3
"""
Compress all CSV files for faster Colab upload.
This will create .zip files for each large CSV to dramatically reduce upload time.
"""

from pathlib import Path
import zipfile
import os

print("=" * 70)
print("CSV COMPRESSION FOR COLAB UPLOAD")
print("=" * 70)

# Find all CSV files in current directory
csv_files = list(Path(".").glob("*.csv"))

# Filter out temp files and small files
csv_files = [f for f in csv_files if not f.name.startswith('.') and f.stat().st_size > 1_000_000]  # > 1MB

print(f"\nFound {len(csv_files)} CSV files to compress:\n")

total_before = 0
total_after = 0

for csv_file in sorted(csv_files):
    size_mb = csv_file.stat().st_size / 1024 / 1024
    zip_file = csv_file.with_suffix('.csv.zip')
    
    # Skip if already compressed
    if zip_file.exists():
        zip_size_mb = zip_file.stat().st_size / 1024 / 1024
        print(f"⏭️  {csv_file.name}")
        print(f"    Already compressed: {zip_file.name} ({zip_size_mb:.1f} MB)")
        total_before += size_mb
        total_after += zip_size_mb
        continue
    
    print(f"📦 {csv_file.name} ({size_mb:.1f} MB)")
    
    # Compress
    try:
        with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            zf.write(csv_file, csv_file.name)
        
        zip_size_mb = zip_file.stat().st_size / 1024 / 1024
        ratio = (1 - zip_size_mb / size_mb) * 100
        
        print(f"    ✓ Created {zip_file.name} ({zip_size_mb:.1f} MB, {ratio:.0f}% smaller)")
        
        total_before += size_mb
        total_after += zip_size_mb
        
    except Exception as e:
        print(f"    ✗ Error: {e}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Total CSV size: {total_before:.1f} MB")
print(f"Total ZIP size: {total_after:.1f} MB")
print(f"Space saved: {total_before - total_after:.1f} MB ({(1-total_after/total_before)*100:.0f}% reduction)")
print(f"\nUpload time estimate:")
print(f"  Original CSVs: ~{total_before / 5:.0f} seconds (at 5 MB/s)")
print(f"  Compressed ZIPs: ~{total_after / 5:.0f} seconds (at 5 MB/s)")
print(f"  Time saved: ~{(total_before - total_after) / 5:.0f} seconds")

print("\n" + "=" * 70)
print("NEXT STEPS FOR COLAB:")
print("=" * 70)
print("1. Upload the .zip files to Colab (not the .csv files)")
print("2. In Colab, run:")
print("   !unzip '*.csv.zip'")
print("   !rm *.csv.zip  # Clean up after extraction")
print("3. Continue with training as normal")
print("=" * 70)
