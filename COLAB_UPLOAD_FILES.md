# 📤 Files to Upload to Google Colab

For full training with historical data (2002-2026), you need to upload these files:

## 1. priors_data.zip (4.8 MB) ✅
**Location:** `C:\Users\tmiles11\nba_predictor\priors_data.zip`

Contains Basketball Reference statistical priors (6 CSV files):
- Team Summaries.csv
- Team Abbrev.csv  
- Per 100 Poss.csv
- Advanced.csv
- Player Shooting.csv
- Player Play By Play.csv

## 2. PlayerStatistics.csv (302.8 MB) ✅
**Location:** `C:\Users\tmiles11\nba_predictor\PlayerStatistics.csv`

Contains 20+ years of player game logs (2002-2026) from Kaggle dataset.

**Why needed:** The Kaggle API download in Colab doesn't include this file (only downloads TeamStatistics). Without it, player models only train on current season data from nba_api.

---

## Quick Upload Guide

1. Open `NBA_COLAB_COMPLETE.ipynb` in Google Colab
2. Run **STEP 1** cell - it will prompt for uploads:
   - First: Upload `priors_data.zip` 
   - Second: Upload `PlayerStatistics.csv`
3. Continue with STEP 2 and 3 as normal

---

## What Gets Trained

### With both files uploaded:
- ✅ Game models: 2002-2026 (32k+ games)
- ✅ Player models: 2002-2026 (800k+ player-games)
- ✅ All features: Team priors, Player priors, Phases 1-7, Neural network

### With only priors_data.zip:
- ✅ Game models: 2002-2026 (32k+ games)  
- ⚠️ Player models: 2025-2026 only (current season from nba_api)
- ⚠️ Limited historical context for player props

---

**Total upload size:** ~307 MB  
**Upload time:** 3-5 minutes on typical connection
