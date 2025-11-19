#!/usr/bin/env python
"""Upload Parquet file to Modal"""
import modal
from pathlib import Path

app = modal.App("upload-parquet")
data_volume = modal.Volume.from_name("nba-data")

# Create image that includes the Parquet file
image = modal.Image.debian_slim().add_local_file(
    "aggregated_nba_data.parquet",
    remote_path="/root/aggregated_nba_data.parquet"
)

@app.function(
    image=image,
    volumes={"/data": data_volume},
    timeout=3600
)
def upload_parquet():
    """Upload local Parquet to Modal volume"""
    import shutil
    import os

    src = "/root/aggregated_nba_data.parquet"
    dst = "/data/aggregated_nba_data.parquet"

    if os.path.exists(src):
        src_size_mb = os.path.getsize(src) / 1024 / 1024
        print(f"Uploading {src} ({src_size_mb:.1f} MB) to Modal volume...")
        shutil.copy(src, dst)
        dst_size_mb = os.path.getsize(dst) / 1024 / 1024
        print(f"Uploaded: {dst_size_mb:.1f} MB")
        data_volume.commit()
        print("Done! Parquet file is now on Modal volume at /data/aggregated_nba_data.parquet")
        return {"size_mb": dst_size_mb, "path": dst}
    else:
        print(f"ERROR: File not found: {src}")
        print("Make sure aggregated_nba_data.parquet exists in the current directory")
        return {"error": "File not found"}

@app.local_entrypoint()
def main():
    print("="*70)
    print("UPLOADING PARQUET TO MODAL")
    print("="*70)

    # Check if local file exists
    local_file = Path("aggregated_nba_data.parquet")
    if not local_file.exists():
        print(f"ERROR: {local_file} not found in current directory")
        print("Please ensure aggregated_nba_data.parquet exists before uploading")
        return

    size_mb = local_file.stat().st_size / 1024 / 1024
    print(f"Local file: {local_file} ({size_mb:.1f} MB)")
    print("Starting upload...")

    result = upload_parquet.remote()

    print("="*70)
    print("UPLOAD COMPLETE!")
    print("="*70)
    print(f"Result: {result}")
    print("\nNext step: Train models")
    print("  py -3.12 -m modal run modal_train.py --window-start 2022 --window-end 2024")
