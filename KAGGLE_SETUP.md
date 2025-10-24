# Kaggle Dataset Setup

## Quick Setup (3 steps)

### 1. Get your Kaggle API credentials

1. Go to https://www.kaggle.com/settings
2. Scroll to "API" section
3. Click "Create New API Token"
4. This downloads `kaggle.json` to your computer

### 2. Set up credentials in this environment

**Option A: Upload kaggle.json**
```bash
mkdir -p ~/.kaggle
# Upload your kaggle.json file to ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**Option B: Set environment variables**
```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
```

### 3. Test the connection

```bash
python explore_dataset.py
```

---

## Alternative: Download Dataset Manually

If you prefer not to use API:

1. Go to: https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores
2. Click "Download" (requires Kaggle login)
3. Extract the ZIP file to `/home/user/meep/data/`
4. Run the training script with local path

---

## Next Steps

Once you have the dataset, I'll build:
1. ✅ Feature engineering pipeline
2. ✅ LightGBM training script
3. ✅ Integration with RIQ MEEPING MACHINE
4. ✅ Backtesting framework

Let me know once you have the credentials set up or the data downloaded!
