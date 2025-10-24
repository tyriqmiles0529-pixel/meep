# ML Model Training Guide

## 🚀 Quick Start (Fully Automated)

### Step 1: One-time setup
```bash
python setup_kaggle.py
```

This will:
- Check if you have Kaggle credentials
- Guide you through authentication
- Test the connection

**First time?** You'll need to:
1. Go to https://www.kaggle.com/settings
2. Click "Create New API Token"
3. Save the downloaded `kaggle.json` to `~/.kaggle/`

### Step 2: Train models
```bash
python train_auto.py
```

**That's it!** Just run this ONE command and the script automatically:

1. ✅ **Downloads dataset** from Kaggle (cached after first run)
2. ✅ **Auto-detects columns** (player names, stats, dates)
3. ✅ **Engineers features**:
   - Rolling averages (3g, 5g, 10g)
   - Rolling std dev
   - Rolling max
   - Trends (short-term vs long-term)
   - Home/away indicator
   - Game number in season
4. ✅ **Trains LightGBM models**:
   - Points (PTS)
   - Assists (AST)
   - Rebounds (REB)
   - 3-Pointers (3PM)
5. ✅ **Saves everything**:
   - Models → `models/{stat}_model.pkl`
   - Registry → `models/model_registry.json`

**Runtime:** 5-10 minutes (first run may take longer for download)

---

## What You Get

### Models (`models/` directory)
```
models/
├── points_model.pkl          # PTS prediction model
├── assists_model.pkl         # AST prediction model
├── rebounds_model.pkl        # REB prediction model
├── threepoint_goals_model.pkl # 3PM prediction model
└── model_registry.json       # Metadata and metrics
```

### Model Registry Example
```json
{
  "points": {
    "version": "v1.0",
    "trained_date": "2025-10-24T01:23:45",
    "test_mae": 4.82,
    "test_rmse": 6.45,
    "train_size": 45230,
    "test_size": 11308,
    "features": ["points_avg_3g", "points_avg_5g", ...],
    "model_file": "points_model.pkl"
  }
}
```

---

## Expected Performance

Based on typical NBA prop models:

| Stat | Expected MAE | Expected RMSE | Notes |
|------|--------------|---------------|-------|
| PTS  | 4-6 points   | 6-8 points    | Most predictable |
| AST  | 1.5-2.5      | 2.5-3.5       | Moderate variance |
| REB  | 2-3          | 3-4           | Matchup-dependent |
| 3PM  | 0.8-1.2      | 1.2-1.8       | High variance |

**If your metrics are better:** Great! You have a good dataset.
**If your metrics are worse:** The dataset might be small or noisy.

---

## Troubleshooting

### "Kaggle credentials not found"
```bash
python setup_kaggle.py
```
Follow the interactive prompts.

### "Dataset download failed"
1. Check internet connection
2. Verify Kaggle account is active
3. Make sure you've accepted dataset terms at:
   https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores

### "Missing required columns"
The dataset structure may have changed. The script auto-detects columns, but if it fails:
1. Run: `python explore_dataset.py`
2. Check the actual column names
3. Manually edit `train_auto.py` column mappings if needed

### Models underperforming
- Check dataset size (need 1000+ games per player ideally)
- Verify data quality (missing values, outliers)
- Check feature importance (are features relevant?)
- Consider adding more features (see advanced section)

---

## Advanced: Customization

### Adjust training parameters

Edit `train_auto.py`:

```python
# Line ~270: LightGBM parameters
params = {
    'objective': 'regression',
    'metric': 'mae',
    'num_leaves': 31,        # ← More = complex model (try 50-100)
    'learning_rate': 0.05,   # ← Lower = slower but better (try 0.01-0.03)
    'feature_fraction': 0.8, # ← Feature sampling
    'bagging_fraction': 0.8, # ← Row sampling
}
```

### Add more features

```python
# Custom features (add before training)

# Minutes played (if available)
if 'minutes' in df.columns:
    df['minutes_avg_5g'] = df.groupby('player_id')['minutes'].transform(
        lambda x: x.shift(1).rolling(5).mean()
    )

# Days rest
df['days_rest'] = df.groupby('player_id')['date'].diff().dt.days

# Opponent strength (if you have opponent stats)
# ... add your custom features
```

### Change test/train split

```python
# Line ~250: Change test seasons
TEST_SEASONS = ['2023-24', '2022-23']  # Hold out 2 seasons
```

---

## Integration with RIQ Analyzer

Once models are trained, they're automatically ready to use.

### Future: Enable ML mode (coming soon)
```bash
export USE_ML_MODELS=true
python nba_prop_analyzer_fixed.py
```

This will:
- Use trained LightGBM predictions instead of EWMA
- Keep all ELG/Kelly logic the same
- Improve projection accuracy

---

## Retraining

### When to retrain:
- **Weekly:** to include latest games
- **After major roster changes:** trades, injuries
- **Start of new season:** fresh data

### How to retrain:
Just run the same command again:
```bash
python train_auto.py
```

The script will:
- Download latest data
- Retrain all models
- Overwrite old models
- Update model registry

---

## Dataset Info

**Source:** Kaggle - Historical NBA Data and Player Box Scores
**Creator:** eoinamoore
**URL:** https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores

**Contains:**
- Player game-by-game stats (multiple seasons)
- Points, assists, rebounds, 3PM, and more
- Historical data for training

**License:** Check Kaggle dataset page for terms

---

## Summary

**Before:** Heuristic EWMA projections (decent but simple)
**After:** LightGBM with 50+ features (better accuracy)

**The best part:** ELG/Kelly framework doesn't change - we just improved the input projections!

**One command to train:**
```bash
python train_auto.py
```

That's it! 🎉
