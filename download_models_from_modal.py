#!/usr/bin/env python
"""
Download all window models from Modal volume to local directory for Kaggle upload.

Usage:
    python download_models_from_modal.py
"""

import subprocess
import sys
from pathlib import Path

def download_models():
    """Download all player model files from Modal volume"""

    # Create output directory
    output_dir = Path("kaggle_models")
    output_dir.mkdir(exist_ok=True)

    print("="*70)
    print("DOWNLOADING WINDOW MODELS FROM MODAL")
    print("="*70)
    print(f"Output directory: {output_dir.absolute()}")
    print()

    # Get list of files in Modal volume
    print("[*] Fetching file list from Modal...")
    result = subprocess.run(
        ["modal", "volume", "ls", "nba-models"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"[!] Error fetching file list: {result.stderr}")
        return False

    # Parse file list
    files = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]

    # Filter to just .pkl files (exclude meta.json)
    model_files = [f for f in files if f.endswith('.pkl')]

    print(f"[OK] Found {len(model_files)} model files")
    print()

    # Download each file
    downloaded = 0
    skipped = 0

    for i, filename in enumerate(model_files, 1):
        dest_path = output_dir / filename

        # Skip if already exists
        if dest_path.exists():
            print(f"[{i}/{len(model_files)}] SKIP: {filename} (already exists)")
            skipped += 1
            continue

        print(f"[{i}/{len(model_files)}] Downloading: {filename}...", end=" ", flush=True)

        # Download file
        result = subprocess.run(
            ["modal", "volume", "get", "nba-models", filename, str(dest_path)],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore'  # Ignore unicode errors from Modal CLI
        )

        if result.returncode == 0:
            # Check file size
            size_mb = dest_path.stat().st_size / (1024 * 1024)
            print(f"OK ({size_mb:.1f} MB)")
            downloaded += 1
        else:
            print(f"FAILED")
            print(f"    Error: {result.stderr}")

    print()
    print("="*70)
    print(f"DOWNLOAD COMPLETE")
    print("="*70)
    print(f"  Downloaded: {downloaded}")
    print(f"  Skipped:    {skipped}")
    print(f"  Total:      {len(model_files)}")
    print()

    if downloaded > 0 or skipped == len(model_files):
        print("✓ All models ready in:", output_dir.absolute())
        print()
        print("Next steps:")
        print("1. Go to https://www.kaggle.com/datasets")
        print("2. Create new dataset: 'nba-window-models'")
        print(f"3. Upload all files from: {output_dir.absolute()}")
        print("4. Make it private (models are proprietary)")
        print("5. Follow KAGGLE_META_TRAINING.md for full workflow")
        return True
    else:
        print("⚠ Some downloads failed. Please check errors above.")
        return False


if __name__ == "__main__":
    try:
        success = download_models()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n[!] Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[!] Error: {e}")
        sys.exit(1)
