"""Download models from Modal and delete recent windows"""
import subprocess
import sys
from pathlib import Path

# Models to download (safe for 2024-2025 backtest)
safe_models = [
    'player_models_2019_2021',
    'player_models_2016_2018',
    'player_models_2013_2015',
    'player_models_2010_2012',
    'player_models_2007_2009',
    'player_models_2004_2006',
    'player_models_2001_2003',
]

# Models to skip (contain 2024-2025 data)
skip_models = [
    'player_models_2022_2024',  # Has 2024 data
    'player_models_2025_2026',  # Has 2025 data
]

cache_dir = Path('model_cache')
cache_dir.mkdir(exist_ok=True)

print("="*70)
print("DOWNLOADING MODELS FROM MODAL")
print("="*70)

for model in safe_models:
    print(f"\nDownloading {model}...")

    # Download .pkl file
    result = subprocess.run([
        'py', '-3.12', '-m', 'modal', 'volume', 'get',
        'nba-models', f'{model}.pkl', f'model_cache/{model}.pkl',
        '--force'
    ], capture_output=True, text=True, encoding='utf-8', errors='ignore')

    if result.returncode == 0:
        print(f"  OK Downloaded {model}.pkl")
    else:
        print(f"  FAILED: {result.stderr[:100]}")

    # Download _meta.json file
    result = subprocess.run([
        'py', '-3.12', '-m', 'modal', 'volume', 'get',
        'nba-models', f'{model}_meta.json', f'model_cache/{model}_meta.json',
        '--force'
    ], capture_output=True, text=True, encoding='utf-8', errors='ignore')

    if result.returncode == 0:
        print(f"  OK Downloaded {model}_meta.json")

print("\n" + "="*70)
print("CHECKING FOR RECENT WINDOWS TO DELETE")
print("="*70)

# Check if any skip_models exist locally
for model in skip_models:
    pkl_file = cache_dir / f'{model}.pkl'
    meta_file = cache_dir / f'{model}_meta.json'

    if pkl_file.exists():
        print(f"\nWARNING Found {model}.pkl - DELETE THIS (has 2024-2025 data)")
        pkl_file.unlink()
        print(f"  OK Deleted {model}.pkl")

    if meta_file.exists():
        meta_file.unlink()
        print(f"  OK Deleted {model}_meta.json")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

models = list(cache_dir.glob('player_models_*.pkl'))
print(f"\nLocal models ({len(models)}):")
for m in sorted(models):
    print(f"  • {m.name}")

print("\nOK Ready for backtest!")
print("\nNext step:")
print("  python backtest_2024_2025.py")
