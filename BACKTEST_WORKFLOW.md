# NBA Predictor: Backtest Workflow

Complete workflow for downloading models, clearing recent data, and running backtests.

---

## Step 1: Download Models from Modal

### List Available Models

```bash
# See what's in the model volume
modal volume ls nba-models /models/
```

### Download Player Models

```bash
# Download specific window (e.g., 2022-2026)
modal volume get nba-models /models/player_models_2022_2026.pkl model_cache/player_models_2022_2026.pkl
modal volume get nba-models /models/player_models_2022_2026_meta.json model_cache/player_models_2022_2026_meta.json

# Download all windows if you have multiple
modal volume get nba-models /models/player_models_2017_2021.pkl model_cache/player_models_2017_2021.pkl
modal volume get nba-models /models/player_models_2012_2016.pkl model_cache/player_models_2012_2016.pkl
modal volume get nba-models /models/player_models_2007_2011.pkl model_cache/player_models_2007_2011.pkl
modal volume get nba-models /models/player_models_2002_2006.pkl model_cache/player_models_2002_2006.pkl
```

### Download Game Models

```bash
# Download game models
modal volume get nba-models /models/game_models_2002_2026.pkl model_cache/game_models_2002_2026.pkl
modal volume get nba-models /models/game_models_2002_2026_meta.json model_cache/game_models_2002_2026_meta.json
```

### Download All Models at Once

```bash
# Download entire model directory
modal volume get nba-models /models/ model_cache/ --recursive
```

---

## Step 2: Verify Downloaded Models

```bash
# Check what you downloaded
dir model_cache

# View metadata for a window
type model_cache\player_models_2022_2026_meta.json

# Quick Python check
python -c "import pickle; m = pickle.load(open('model_cache/player_models_2022_2026.pkl', 'rb')); print(f'Loaded: {type(m)}')"
```

---

## Step 3: Delete Recent Windows (Prevent Data Leakage)

### Why Delete Recent Windows?

Backtesting on 2024-2025 season requires models trained WITHOUT seeing 2024-2025 data. You need to:
1. Delete any windows that include 2024-2025
2. Keep only models trained on historical data (e.g., 2002-2023)

### Check What Windows You Have

```bash
# List all cached models
dir model_cache\player_models_*.pkl
```

### Delete Recent Windows Manually

```powershell
# Delete 2022-2026 window (includes 2024-2025 data)
del model_cache\player_models_2022_2026.pkl
del model_cache\player_models_2022_2026_meta.json

# Delete 2017-2021 if it overlaps your test period
del model_cache\player_models_2017_2021.pkl
del model_cache\player_models_2017_2021_meta.json
```

### Or Use Python Script

```bash
# Delete windows that include seasons after 2023
python -c "
from pathlib import Path
import re

for f in Path('model_cache').glob('player_models_*.pkl'):
    match = re.search(r'player_models_(\d{4})_(\d{4})', f.name)
    if match:
        end_year = int(match.group(2))
        if end_year >= 2024:
            print(f'Deleting: {f.name}')
            f.unlink()
            meta = f.with_suffix('.json').with_stem(f.stem + '_meta')
            if meta.exists():
                meta.unlink()
"
```

### Clear Cache on Modal Volume (if needed)

```bash
# Delete from Modal volume too
modal volume rm nba-models /models/player_models_2022_2026.pkl
modal volume rm nba-models /models/player_models_2022_2026_meta.json
```

---

## Step 4: Retrain Windows for Backtest (If Needed)

If you deleted recent windows, retrain with data ending BEFORE your test period.

### Train Windows Ending at 2023

```bash
# Window 1: 2002-2006
modal run modal_train.py --window-start 2002 --window-end 2006

# Window 2: 2007-2011
modal run modal_train.py --window-start 2007 --window-end 2011

# Window 3: 2012-2016
modal run modal_train.py --window-start 2012 --window-end 2016

# Window 4: 2017-2021
modal run modal_train.py --window-start 2017 --window-end 2021

# Window 5: 2019-2023 (does NOT include 2024-2025)
modal run modal_train.py --window-start 2019 --window-end 2023
```

### Download Retrained Models

```bash
# Download the new 2019-2023 window
modal volume get nba-models /models/player_models_2019_2023.pkl model_cache/player_models_2019_2023.pkl
modal volume get nba-models /models/player_models_2019_2023_meta.json model_cache/player_models_2019_2023_meta.json
```

---

## Step 5: Run Backtest

### Option A: Use Existing Backtest Script

```bash
# Backtest on 2024-2025 season
python backtest_2024_2025.py
```

### Option B: Use Backtest Engine

```bash
# Full backtest with betting simulation
python backtest_engine.py --start-date 2024-10-01 --end-date 2025-04-15 --bankroll 10000
```

### Option C: Custom Backtest

```python
# custom_backtest.py
from backtest_engine import BacktestEngine
import pandas as pd

# Initialize backtest
backtest = BacktestEngine(
    model_cache_dir='model_cache',
    data_path='aggregated_nba_data.parquet',
    start_date='2024-10-01',
    end_date='2025-04-15',
    initial_bankroll=10000
)

# Run backtest
results = backtest.run()

# View results
print(f"Total bets: {results['total_bets']}")
print(f"Win rate: {results['win_rate']:.2%}")
print(f"ROI: {results['roi']:.2%}")
print(f"Final bankroll: ${results['final_bankroll']:,.2f}")

# Export ledger
results['ledger'].to_csv('backtest_ledger.csv', index=False)
```

---

## Complete Workflow Example

### Scenario: Backtest 2024-2025 Season

```bash
# 1. Download models from Modal
modal volume ls nba-models /models/
modal volume get nba-models /models/ model_cache/ --recursive

# 2. Check what you have
dir model_cache\player_models_*.pkl

# 3. Delete windows that include 2024-2025
del model_cache\player_models_2022_2026.pkl
del model_cache\player_models_2022_2026_meta.json

# 4. Retrain most recent window (if needed)
modal run modal_train.py --window-start 2019 --window-end 2023

# 5. Download retrained model
modal volume get nba-models /models/player_models_2019_2023.pkl model_cache/player_models_2019_2023.pkl
modal volume get nba-models /models/player_models_2019_2023_meta.json model_cache/player_models_2019_2023_meta.json

# 6. Run backtest
python backtest_2024_2025.py

# 7. View results
type backtest_ledger.csv
```

---

## Step 6: Analyze Backtest Results

### View Summary Metrics

```bash
# If backtest_engine.py creates summary
type backtest_summary.json
```

### Load Ledger in Python

```python
import pandas as pd

# Load ledger
ledger = pd.read_csv('backtest_ledger.csv')

# Summary stats
print(f"Total bets: {len(ledger)}")
print(f"Wins: {(ledger['result'] == 'win').sum()}")
print(f"Losses: {(ledger['result'] == 'loss').sum()}")
print(f"Win rate: {(ledger['result'] == 'win').mean():.2%}")
print(f"Total profit: ${ledger['profit'].sum():,.2f}")
print(f"ROI: {(ledger['profit'].sum() / ledger['stake'].sum()):.2%}")

# Best bets
print("\nTop 10 profitable bets:")
print(ledger.nlargest(10, 'profit')[['date', 'player', 'prop', 'line', 'profit']])

# Worst bets
print("\nTop 10 losing bets:")
print(ledger.nsmallest(10, 'profit')[['date', 'player', 'prop', 'line', 'profit']])

# By prop type
print("\nPerformance by prop:")
print(ledger.groupby('prop').agg({
    'result': lambda x: (x == 'win').mean(),
    'profit': 'sum'
}))
```

---

## Troubleshooting

### Issue: "Model not found for window"

**Cause:** Missing model files in `model_cache/`

**Solution:**
```bash
# Download missing windows
modal volume get nba-models /models/player_models_2017_2021.pkl model_cache/player_models_2017_2021.pkl
```

### Issue: "Data leakage detected"

**Cause:** Model was trained on data that includes your backtest period

**Solution:**
```bash
# Delete the window and retrain with earlier end date
del model_cache\player_models_2022_2026.pkl
modal run modal_train.py --window-start 2019 --window-end 2023
```

### Issue: "No predictions made"

**Cause:** No games in backtest period, or date filter too restrictive

**Solution:**
```python
# Check data availability
import pandas as pd
df = pd.read_parquet('aggregated_nba_data.parquet')
df['gameDate'] = pd.to_datetime(df['gameDate'])
print(df[(df['gameDate'] >= '2024-10-01') & (df['gameDate'] <= '2025-04-15')].shape)
```

### Issue: Modal volume access denied

**Cause:** Not authenticated or wrong volume name

**Solution:**
```bash
# Re-authenticate
modal token new

# List volumes
modal volume list

# Check volume name
modal volume ls nba-models
```

---

## Quick Reference Commands

```bash
# Download all models
modal volume get nba-models /models/ model_cache/ --recursive

# Delete recent windows (PowerShell)
Remove-Item -Path "model_cache\player_models_2022_2026*" -Force

# Retrain window
modal run modal_train.py --window-start 2019 --window-end 2023

# Download retrained
modal volume get nba-models /models/player_models_2019_2023.pkl model_cache/player_models_2019_2023.pkl

# Run backtest
python backtest_2024_2025.py

# View results
python -c "import pandas as pd; df = pd.read_csv('backtest_ledger.csv'); print(df.describe())"
```

---

## Data Leakage Prevention Checklist

- [ ] Identified backtest period (e.g., 2024-10-01 to 2025-04-15)
- [ ] Listed all cached models
- [ ] Deleted models with `end_year >= 2024`
- [ ] Retrained most recent window with `end_year = 2023`
- [ ] Downloaded retrained models
- [ ] Verified no overlap between training and test data
- [ ] Ran backtest
- [ ] Analyzed results

---

Generated: 2025-11-18
