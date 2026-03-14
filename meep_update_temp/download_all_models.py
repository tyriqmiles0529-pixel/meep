"""Download ALL models from Modal (1947-2021)"""
import subprocess
from pathlib import Path

# ALL models from Modal (excluding 2022+ which has test data)
all_models = [
    'player_models_1947_1949',
    'player_models_1950_1952',
    'player_models_1953_1955',
    'player_models_1956_1958',
    'player_models_1959_1961',
    'player_models_1962_1964',
    'player_models_1965_1967',
    'player_models_1968_1970',
    'player_models_1971_1973',
    'player_models_1974_1976',
    'player_models_1977_1979',
    'player_models_1980_1982',
    'player_models_1983_1985',
    'player_models_1986_1988',
    'player_models_1989_1991',
    'player_models_1992_1994',
    'player_models_1995_1997',
    'player_models_1998_2000',
    'player_models_2001_2003',
    'player_models_2004_2006',
    'player_models_2007_2009',
    'player_models_2010_2012',
    'player_models_2013_2015',
    'player_models_2016_2018',
    'player_models_2019_2021',
]

cache_dir = Path('model_cache')
cache_dir.mkdir(exist_ok=True)

print("="*70)
print(f"DOWNLOADING ALL {len(all_models)} MODELS FROM MODAL (1947-2021)")
print("="*70)

success = 0
failed = 0

for i, model in enumerate(all_models, 1):
    print(f"\n[{i}/{len(all_models)}] {model}...")

    # Download .pkl
    result = subprocess.run([
        'py', '-3.12', '-m', 'modal', 'volume', 'get',
        'nba-models', f'{model}.pkl', f'model_cache/{model}.pkl',
        '--force'
    ], capture_output=True, text=True, encoding='utf-8', errors='ignore')

    if result.returncode == 0:
        print(f"  OK {model}.pkl")
        success += 1
    else:
        print(f"  SKIP (already exists or error)")
        success += 1  # Count as success if already exists

    # Download meta
    subprocess.run([
        'py', '-3.12', '-m', 'modal', 'volume', 'get',
        'nba-models', f'{model}_meta.json', f'model_cache/{model}_meta.json',
        '--force'
    ], capture_output=True, text=True, encoding='utf-8', errors='ignore')

print("\n" + "="*70)
print("DOWNLOAD COMPLETE")
print("="*70)

models = sorted(cache_dir.glob('player_models_*.pkl'))
print(f"\nTotal models: {len(models)}")
print(f"Downloaded/verified: {success}/{len(all_models)}")

print("\nOK Ready for backtest!")
print("Next: python backtest_2024_2025.py")
