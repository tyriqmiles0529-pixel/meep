# 📦 How to Upload Priors Data to Colab

## Quick Steps:

### 1. Zip Your Priors Folder

**On Windows:**
1. Navigate to: `C:\Users\tmiles11\nba_predictor\`
2. Right-click on `priors_data` folder
3. Click "Send to" → "Compressed (zipped) folder"
4. Name it: `priors_data.zip`

**Or use command line:**
```bash
cd C:\Users\tmiles11\nba_predictor
powershell Compress-Archive -Path priors_data -DestinationPath priors_data.zip
```

---

### 2. Upload to Colab

When the Colab script runs, it will ask for priors upload:

```
📊 OPTIONAL: Upload priors_data folder
   If you have it, upload the ZIP file now.
   If not, just skip this and press the cancel/X button.
   Training works fine without priors!
```

**Click "Choose Files"** and select `priors_data.zip`

---

### 3. What This Does:

**With Priors (recommended):**
- ✅ Advanced Basketball Reference stats
- ✅ Better baseline predictions
- ✅ ~5-10% accuracy improvement
- ✅ Player shooting zones, position %, usage rates

**Without Priors (still works):**
- ✅ Uses statistical defaults
- ✅ Still trains all models
- ✅ Slightly lower accuracy but functional

---

## 📁 What's in Priors Data:

Your `priors_data` folder contains 7 CSVs:

**Team Stats:**
1. `Team Summaries.csv` - O/D ratings, pace, four factors
2. `Team Abbrev.csv` - Team name mappings

**Player Stats:**
3. `Per 100 Poss.csv` - Rate stats
4. `Advanced.csv` - PER, TS%, USG%, BPM, VORP
5. `Player Shooting.csv` - Shot zones, corner 3%
6. `Player Play By Play.csv` - Position %, on-court +/-

---

## 🚀 Ready?

1. Zip your `priors_data` folder
2. Open the updated `PASTE_INTO_COLAB.py` 
3. Copy everything
4. Paste into Colab
5. Run and upload when prompted!

**Upload priors = better models!** 📊
