"""
Prepare for Backtesting

Automates the workflow:
1. Download models from Modal
2. Delete windows that overlap with test period
3. Ready for backtest

Usage:
    python prepare_backtest.py --test-start 2024 --test-end 2025
    python prepare_backtest.py --test-start 2024 --test-end 2025 --download
"""

import argparse
from pathlib import Path
import re
import json
import subprocess


def list_local_models(cache_dir='model_cache'):
    """List all locally cached player models"""
    cache_path = Path(cache_dir)

    if not cache_path.exists():
        cache_path.mkdir(parents=True)
        print(f"Created directory: {cache_path}")
        return []

    models = []
    for f in cache_path.glob('player_models_*.pkl'):
        match = re.search(r'player_models_(\d{4})_(\d{4})', f.name)
        if match:
            start_year = int(match.group(1))
            end_year = int(match.group(2))

            # Check for metadata
            meta_file = f.with_suffix('.json').with_name(f.stem.replace('player_models_', 'player_models_') + '_meta.json')
            has_meta = meta_file.exists()

            models.append({
                'file': f,
                'start_year': start_year,
                'end_year': end_year,
                'has_meta': has_meta,
                'meta_file': meta_file
            })

    return sorted(models, key=lambda x: x['start_year'])


def download_models_from_modal(cache_dir='model_cache'):
    """Download all models from Modal volume"""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("DOWNLOADING MODELS FROM MODAL")
    print("="*70)

    try:
        # List models on Modal
        print("\nListing models on Modal volume...")
        result = subprocess.run(
            ['modal', 'volume', 'ls', 'nba-models', '/models/'],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"Error listing Modal volume: {result.stderr}")
            return False

        print(result.stdout)

        # Download recursively
        print(f"\nDownloading to {cache_dir}...")
        result = subprocess.run(
            ['modal', 'volume', 'get', 'nba-models', '/models/', str(cache_path), '--recursive'],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"Error downloading: {result.stderr}")
            return False

        print("✓ Download complete!")
        return True

    except FileNotFoundError:
        print("Error: Modal CLI not found. Install with: pip install modal")
        return False


def delete_overlapping_windows(models, test_start_year, test_end_year):
    """Delete models that overlap with test period"""
    deleted = []
    kept = []

    for model in models:
        # Check if window overlaps with test period
        overlaps = not (model['end_year'] < test_start_year or model['start_year'] > test_end_year)

        if overlaps:
            print(f"  ❌ Deleting: {model['file'].name} (overlaps with test period)")
            model['file'].unlink()
            deleted.append(model)

            if model['has_meta']:
                model['meta_file'].unlink()
                print(f"     Deleted: {model['meta_file'].name}")
        else:
            print(f"  ✓ Keeping: {model['file'].name}")
            kept.append(model)

    return deleted, kept


def main():
    parser = argparse.ArgumentParser(description='Prepare for backtesting')
    parser.add_argument('--test-start', type=int, required=True, help='Start year of test period (e.g., 2024)')
    parser.add_argument('--test-end', type=int, required=True, help='End year of test period (e.g., 2025)')
    parser.add_argument('--cache-dir', default='model_cache', help='Model cache directory')
    parser.add_argument('--download', action='store_true', help='Download models from Modal first')
    parser.add_argument('--skip-delete', action='store_true', help='Skip deletion (dry run)')

    args = parser.parse_args()

    print("="*70)
    print("BACKTEST PREPARATION")
    print("="*70)
    print(f"Test period: {args.test_start}-{args.test_end}")
    print(f"Cache directory: {args.cache_dir}")

    # Step 1: Download models (optional)
    if args.download:
        success = download_models_from_modal(args.cache_dir)
        if not success:
            print("\n⚠ Download failed, continuing with local models...")

    # Step 2: List local models
    print("\n" + "="*70)
    print("LOCAL MODELS")
    print("="*70)

    models = list_local_models(args.cache_dir)

    if not models:
        print("No models found in cache!")
        print("\nNext steps:")
        print(f"  1. Download models: python prepare_backtest.py --test-start {args.test_start} --test-end {args.test_end} --download")
        print(f"  2. Or train models: modal run modal_train.py")
        return

    print(f"\nFound {len(models)} model(s):")
    for model in models:
        meta_str = "✓ meta" if model['has_meta'] else "✗ no meta"
        print(f"  • {model['start_year']}-{model['end_year']}: {model['file'].name} ({meta_str})")

    # Step 3: Check for overlaps
    print("\n" + "="*70)
    print("CHECKING FOR DATA LEAKAGE")
    print("="*70)

    overlapping = []
    safe = []

    for model in models:
        overlaps = not (model['end_year'] < args.test_start or model['start_year'] > args.test_end)
        if overlaps:
            overlapping.append(model)
        else:
            safe.append(model)

    if not overlapping:
        print("✓ No overlapping windows found - safe to backtest!")
        print(f"\nSafe models ({len(safe)}):")
        for model in safe:
            print(f"  • {model['start_year']}-{model['end_year']}")
        return

    print(f"⚠ Found {len(overlapping)} overlapping window(s):")
    for model in overlapping:
        print(f"  • {model['start_year']}-{model['end_year']}: {model['file'].name}")

    # Step 4: Delete overlapping windows
    if args.skip_delete:
        print("\n(Dry run - not deleting)")
        print("\nTo delete, run without --skip-delete")
    else:
        print("\n" + "="*70)
        print("DELETING OVERLAPPING WINDOWS")
        print("="*70)

        deleted, kept = delete_overlapping_windows(models, args.test_start, args.test_end)

        print(f"\n✓ Deleted {len(deleted)} window(s)")
        print(f"✓ Kept {len(kept)} window(s)")

    # Step 5: Suggest retraining
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)

    if overlapping and not args.skip_delete:
        # Find the most recent safe window
        if kept:
            latest = max(kept, key=lambda x: x['end_year'])
            latest_end = latest['end_year']
        else:
            latest_end = args.test_start - 1

        # Suggest new window
        new_start = max(latest_end - 4, 2002)  # 5-year window
        new_end = args.test_start - 1

        print(f"\n1. Retrain most recent window (without test data):")
        print(f"   modal run modal_train.py --window-start {new_start} --window-end {new_end}")

        print(f"\n2. Download retrained model:")
        print(f"   modal volume get nba-models /models/player_models_{new_start}_{new_end}.pkl {args.cache_dir}/player_models_{new_start}_{new_end}.pkl")
        print(f"   modal volume get nba-models /models/player_models_{new_start}_{new_end}_meta.json {args.cache_dir}/player_models_{new_start}_{new_end}_meta.json")

    print(f"\n3. Run backtest:")
    print(f"   python backtest_2024_2025.py")

    print(f"\n4. Or use backtest engine:")
    print(f"   python backtest_engine.py --start-date {args.test_start}-10-01 --end-date {args.test_end}-04-15")


if __name__ == '__main__':
    main()
