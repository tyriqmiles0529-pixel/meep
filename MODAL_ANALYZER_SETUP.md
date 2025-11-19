# Run RIQ Analyzer on Modal

## Setup (One-Time)

### 1. Add API Key to Modal Secrets

```bash
# Set your API Sports key as a Modal secret
modal secret create api-sports-key API_SPORTS_KEY=your_key_here
```

Or via Modal dashboard:
1. Go to https://modal.com/secrets
2. Create new secret named `api-sports-key`
3. Add key: `API_SPORTS_KEY=your_actual_key`

### 2. Verify Models are on Modal

```bash
# Check window models
modal volume ls nba-models player_models_*.pkl

# Should see all 25+ windows
```

## Usage

### Run Analyzer with Ensemble (Recommended):
```bash
modal run modal_analyzer.py
```

### Run Analyzer without Ensemble (Faster):
```bash
modal run modal_analyzer.py --use-ensemble=False
```

## Benefits of Running on Modal

| Feature | Local | Modal |
|---------|-------|-------|
| **RAM** | Limited (~8GB) | 16GB (configurable) |
| **CPU** | 4-8 cores | 8 cores |
| **Ensemble** | May crash | No issues |
| **Speed** | Depends on machine | Consistent |
| **Cost** | Free | ~$0.50-1/run |

## How It Works

```
Your Laptop
    ↓
modal run modal_analyzer.py
    ↓
Modal Cloud (16GB RAM, 8 CPU)
    ↓
    ├─ Copy models from Modal volume → /root/model_cache/
    ├─ Load ensemble (25 windows + meta-learner)
    ├─ Fetch games from API
    ├─ Build features (150-218 features)
    ├─ Get ensemble predictions
    └─ Generate betting recommendations
    ↓
Results returned to your laptop
```

## Cost Estimate

- CPU: 8 cores @ $0.10/hour
- RAM: 16GB included
- **Typical run: 10-15 minutes**
- **Cost per run: ~$0.15-0.25**

For daily use: **~$5-7/month**

## Advantages

### 1. No Memory Issues
Local ensemble can crash with 25 windows. Modal has 16GB RAM - no problem!

### 2. Faster
- 8 CPU cores
- Modal's network is faster for API calls
- Parallel model loading

### 3. Consistent Environment
- Same Python version
- Same dependencies
- No local setup issues

### 4. Can Run from Anywhere
- Run from laptop, phone, or any computer
- Just need Modal CLI installed

## Downloading Results Locally (Optional)

If you want to save results locally:

```python
# Add to modal_analyzer.py before return:

# Save results to volume
import shutil
shutil.copy("/root/bets_ledger.pkl", "/models/latest_bets_ledger.pkl")
model_volume.commit()
```

Then download:
```bash
modal volume get nba-models latest_bets_ledger.pkl bets_ledger.pkl
```

## Scheduling (Optional)

Run analyzer automatically every day:

```python
# Add to modal_analyzer.py

from modal import Cron

@app.function(
    schedule=Cron("0 9 * * *"),  # 9 AM daily
    # ... other params
)
def scheduled_analyzer():
    return run_analyzer(use_ensemble=True)
```

## Comparison

### Local:
```bash
python riq_analyzer.py --use-ensemble
```
- ✅ Free
- ❌ May crash (memory)
- ❌ Slower (limited CPU)
- ❌ Depends on local setup

### Modal:
```bash
modal run modal_analyzer.py
```
- ❌ Costs ~$0.25/run
- ✅ Never crashes (16GB RAM)
- ✅ Faster (8 CPU cores)
- ✅ Consistent environment

## Troubleshooting

### Error: "Secret not found: api-sports-key"
**Fix:** Create the secret first:
```bash
modal secret create api-sports-key API_SPORTS_KEY=your_key
```

### Error: "Volume not found: nba-models"
**Fix:** Train models first or verify volume exists:
```bash
modal volume ls
```

### Slow startup
**Fix:** Normal on first run (downloads models). Subsequent runs are faster.

---

**Recommended workflow:**
1. **Daily production:** `modal run modal_analyzer.py` (reliable, fast)
2. **Testing locally:** `python riq_analyzer.py` (free, quick iteration)
3. **Heavy analysis:** Modal (handles ensemble without issues)
