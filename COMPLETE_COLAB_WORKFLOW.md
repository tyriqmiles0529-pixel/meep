# 🚀 Complete Google Colab Workflow

## Three-Notebook Production Pipeline

Your NBA prediction system now has a **complete cloud-based workflow** with three specialized notebooks:

---

## 📚 The Three Notebooks

### 1️⃣ **NBA_COLAB_SIMPLE.ipynb** - Model Training
**Purpose**: Train neural hybrid models with GPU acceleration

**What it does**:
- Loads PlayerStatistics.csv (full NBA history 1947-2025)
- Loads priors_data.zip (Basketball Reference career stats)
- Generates 150-218 features (7-phase engineering)
- Trains 5 TabNet + LightGBM hybrid models (one per stat type)
- Exports trained models as `nba_models_trained.zip`

**Time**: ~1.5 hours (L4 GPU) or 20-30 min (A100 GPU)

**Workflow**:
1. Upload to Colab
2. Enable GPU (Runtime → Change runtime type)
3. Upload `priors_data.zip`
4. Run all cells
5. Download `nba_models_trained.zip`

---

### 2️⃣ **Riq_Machine.ipynb** - Live Predictions
**Purpose**: Get daily predictions with Safe Mode and auto-download

**What it does**:
- Loads trained models from `nba_models_trained.zip`
- Fetches today's NBA games and betting lines
- Generates predictions with full feature engineering
- Compares to betting lines (finds value)
- **Optional**: Safe Mode (adds margin buffer for conservative picks)
- Saves predictions to `bets_ledger.pkl`
- **NEW**: Auto-downloads results!

**Time**: 5-10 minutes

**Workflow**:
1. Upload to Colab
2. Upload 3 files:
   - `nba_models_trained.zip`
   - `priors_data.zip`
   - `PlayerStatistics.csv`
3. Configure API keys (The Odds API)
4. **Optional**: Enable Safe Mode
5. Run Full Analysis
6. **CRITICAL**: Run Download Results cell
7. Save `bets_ledger.pkl` locally

**Safe Mode Example**:
- Projection: 2.8 assists
- Regular line: 3.5 (UNDER)
- Safe Mode margin: 1.0
- **Safe line required**: 4.5+ (extra buffer)

---

### 3️⃣ **Evaluate_Predictions.ipynb** - Performance & Calibration
**Purpose**: Settle bets, track performance, recalibrate models

**What it does**:
- Uploads your `bets_ledger.pkl`
- Fetches actual game results from NBA API
- Settles all pending predictions (won/lost)
- Calculates win rate, ROI, performance by stat type
- Generates 4-panel calibration plots
- Trains isotonic regression for better calibration
- Exports `calibration.pkl` for production use

**Time**: 5-10 minutes

**Workflow**:
1. Upload to Colab
2. Upload `bets_ledger.pkl` (from Riq_Machine)
3. Run Setup
4. Fetch Results & Settle
5. Evaluate Performance
6. Generate Calibration Plots
7. Recalibrate Models
8. Download files:
   - `calibration.pkl` → Use in production
   - `calibration_analysis.png` → Performance plots
   - `evaluation_report.txt` → Metrics summary

**Output Example**:
```
======================================================================
OVERALL PERFORMANCE
======================================================================

Win Rate: 52.34% (156W-142L)
Total Staked: $2,980.00
Total Profit: +$143.50
ROI: +4.81%

======================================================================
CALIBRATION ANALYSIS
======================================================================

Predicted% | Actual% | Count | Calibration
----------------------------------------------------------------------
50-60%  |  51.2% |    87 | +1.2% ✅
60-70%  |  62.8% |    95 | +2.8% ✅
70-80%  |  68.4% |    73 | -1.6% ✅
```

---

## 🔄 Complete Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ WEEKLY: Train New Models                                     │
│ ├─ NBA_COLAB_SIMPLE.ipynb                                    │
│ ├─ Upload priors_data.zip                                    │
│ ├─ Train for 1.5 hours                                       │
│ └─ Download nba_models_trained.zip                           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ DAILY: Get Predictions                                       │
│ ├─ Riq_Machine.ipynb                                         │
│ ├─ Upload models + data                                      │
│ ├─ Configure Safe Mode (optional)                            │
│ ├─ Run analysis                                              │
│ └─ Download bets_ledger.pkl ⚠️ IMPORTANT                     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ WEEKLY: Evaluate & Calibrate                                │
│ ├─ Evaluate_Predictions.ipynb                               │
│ ├─ Upload bets_ledger.pkl                                   │
│ ├─ Fetch actual results                                     │
│ ├─ Generate calibration plots                               │
│ └─ Download calibration.pkl                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Key Features

### Safe Mode (New!)
- **Purpose**: Conservative betting with extra margin
- **How it works**: Adds buffer to required line
- **Example**: 
  - Without: Proj 2.8, Line 3.5 → Bet UNDER
  - With: Proj 2.8, Line 3.5, Margin 1.0 → Requires 4.5+ to bet
- **Enable**: Set `SAFE_MODE=true` and `SAFE_MARGIN=1.0` in Riq_Machine

### Auto-Download (New!)
- **Purpose**: Prevents losing predictions when Colab session expires
- **How it works**: Download Results cell at end of Riq_Machine
- **Downloads**:
  - `bets_ledger.pkl` (all predictions)
  - `prop_analysis_*.json` (latest run)
- **CRITICAL**: Always run this cell after analysis!

### Isotonic Calibration
- **Purpose**: Fixes overconfident predictions
- **Example**: Model says 90% → Actually wins 47% → Recalibrate!
- **How**: Trains isotonic regression on settled bets
- **Output**: `calibration.pkl` → Auto-loaded by Riq_Machine next time

---

## 📊 Performance Tracking

### What Gets Tracked
- Every prediction saved to `bets_ledger.pkl`
- Includes: player, stat, line, pick, odds, predicted probability
- Settlement: Actual result, won/lost status
- Metadata: Game date, bookmaker, recorded timestamp

### Evaluation Metrics
- **Win Rate**: % of predictions that win
- **ROI**: Return on investment (profit / staked)
- **Calibration**: Do 70% predictions actually win 70%?
- **By Stat Type**: Performance breakdown (points, assists, etc)
- **By Confidence**: Win rate at each probability bucket

---

## ⚠️ Critical Reminders

### 1. Always Download After Predictions!
❌ **DON'T FORGET**: Colab doesn't auto-save to your computer
✅ **DO THIS**: Run Download Results cell after every analysis
📁 **Save**: `bets_ledger.pkl` to `C:\Users\tmiles11\nba_predictor\`

### 2. Evaluate Weekly
- Upload latest `bets_ledger.pkl` to Evaluate_Predictions.ipynb
- Settle bets, generate calibration
- Download `calibration.pkl`
- Copy to project root → Auto-loads next prediction run

### 3. Retrain Models Monthly
- NBA meta changes throughout season
- New players, injuries, role changes
- Run NBA_COLAB_SIMPLE.ipynb monthly
- Replace old models with new ones

---

## 🚨 Your Current Situation

**Problem**: You ran analysis tonight (11:31 PM) but files are in Colab cloud
**Solution**: Run this in Colab NOW before session expires:

```python
from google.colab import files
files.download('/content/meep/bets_ledger.pkl')
files.download('/content/meep/prop_analysis_20251107_233159.json')
```

**Next Steps**:
1. Download files from Colab
2. Save `bets_ledger.pkl` to project folder
3. Upload to `Evaluate_Predictions.ipynb` to evaluate
4. Generate calibration for better future predictions

---

## 📖 Additional Resources

- **START_HERE_COLAB.md** - Detailed setup guide
- **DOWNLOAD_FROM_COLAB.py** - Download code snippet
- **README.md** - Complete system documentation
- **SAFE_MODE_GUIDE.md** - Safe Mode configuration

---

## 🎉 Summary

You now have a **complete production pipeline**:
✅ Train models with GPU acceleration
✅ Get daily predictions with Safe Mode
✅ Auto-download results (no more lost predictions!)
✅ Evaluate performance and track ROI
✅ Generate calibration curves
✅ Recalibrate models for better accuracy

**Everything runs in Google Colab - no local setup required!** 🚀
