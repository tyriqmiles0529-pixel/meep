# NBA Predictor - Execution Guide
**After Cache Cleanup - What to Run, When, and Why**

Last Updated: 2025-11-04

---

## 🎯 Quick Start (After Cleanup)

Since you've deleted caches (`model_cache/`, `player_cache.pkl`, etc.), you need to rebuild everything:

### **STEP 1: Full Model Training (REQUIRED FIRST)**
```powershell
# This rebuilds ALL models from scratch (2002-2026 data)
# Runtime: ~30-60 minutes depending on your machine
python train_auto.py --verbose --fresh --lgb-log-period 50

# What it does:
# ✓ Downloads latest data from Kaggle
# ✓ Trains game models (moneyline, spread)
# ✓ Trains player models (points, assists, rebounds, threes, minutes)
# ✓ Saves to models/ directory
# ✓ Creates training_metadata.json with RMSE metrics
```

**Why this is first:** All other steps depend on having trained models. Without `models/*.pkl`, nothing else will work.

---

## 📅 Daily Workflow (During NBA Season)

### **Morning - Get Today's Predictions**
```powershell
# Run the analyzer to get today's prop bets
python riq_analyzer.py

# What it does:
# ✓ Fetches today's NBA games from API-Sports
# ✓ Fetches betting lines from The Odds API
# ✓ Uses trained models to predict player props
# ✓ Calculates Kelly sizing and EV
# ✓ Saves results to prop_analysis_YYYYMMDD_HHMMSS.json
```

**When:** Every morning before games start (or whenever lines are posted)

**Output:** JSON file with recommended bets ranked by ELG (Expected Log Growth)

---

### **Evening - Update Results (After Games Finish)**
```powershell
# Fetch actual game results and update ledger
python fetch_bet_results_incremental.py

# What it does:
# ✓ Looks up unsettled bets in bets_ledger.pkl
# ✓ Fetches actual player stats from nba_api
# ✓ Updates win/loss for each bet
# ✓ Trains calibration curves (isotonic regression)
```

**When:** After games finish (usually 10pm-midnight ET)

**How often:** Daily during season, or whenever you want to update results

**Progress:** Run multiple times until it says "No more unsettled bets" (it processes in batches)

---

### **Weekly - Check Performance**
```powershell
# Analyze betting performance and model accuracy
python analyze_ledger.py

# What it does:
# ✓ Shows overall win rate by prop type
# ✓ Identifies overconfidence issues
# ✓ Plots calibration curves
# ✓ Calculates ROI and Sharpe ratio
```

**When:** Weekly, or after accumulating 50+ settled bets

**Look for:**
- Win rates by confidence bucket
- Calibration curve alignment (predicted vs actual)
- Which prop types are performing best

---

## 🔄 Periodic Maintenance

### **Monthly - Recalibrate Models**
```powershell
# Rebuild calibration curves with latest settled predictions
python recalibrate_models.py

# What it does:
# ✓ Trains isotonic regression on settled bets
# ✓ Corrects for systematic over/under-confidence
# ✓ Updates calibration.pkl
```

**When:** After 500+ settled predictions (check with `analyze_ledger.py`)

**Why:** Models drift over time - recalibration keeps them honest

---

### **Seasonal - Full Retrain**
```powershell
# Rebuild models with fresh data (new season start)
python train_auto.py --verbose --fresh --lgb-log-period 50

# Then rebuild ensemble components
python train_ensemble_enhanced.py

# Then rebuild dynamic window selector
python train_dynamic_selector_enhanced.py
```

**When:**
- Start of new NBA season (October)
- After major rule changes
- After accumulating 2-3 months of new data

**Why:** Incorporates latest season data and team changes

---

## 🎛️ Advanced Workflows

### **Backtest New Features (Before Production)**
```powershell
# Test new feature engineering on historical data
python train_auto.py --verbose --fresh --player-season-cutoff 2020

# What this does:
# ✓ Trains on 2020-2026 data only
# ✓ Faster iteration for testing
# ✓ Compare metrics to baseline
```

**When:** Testing Phase 1/2/3 feature engineering improvements

---

### **Multi-Window Ensemble Training**
```powershell
# Train separate models for different time windows
python train_auto.py --verbose --fresh --enable-window-ensemble

# What it does:
# ✓ Trains 5-year window models (2002-2006, 2007-2011, etc.)
# ✓ Caches to model_cache/ensemble_YYYY_YYYY.pkl
# ✓ Dynamic selector chooses best window per player
```

**When:** After initial training, for advanced ensemble performance

**Runtime:** 2-3 hours (trains 5 separate model sets)

---

## 📊 File Outputs Explained

| File | Created By | Purpose |
|------|-----------|---------|
| `models/*.pkl` | train_auto.py | Trained LightGBM models |
| `training_metadata.json` | train_auto.py | RMSE, feature names, training config |
| `bets_ledger.pkl` | riq_analyzer.py | All recommended bets + results |
| `calibration.pkl` | fetch_bet_results_incremental.py | Isotonic calibration curves |
| `prop_analysis_*.json` | riq_analyzer.py | Daily predictions output |
| `equity_curve.pkl` | (future) | Bankroll tracking |
| `player_cache.pkl` | riq_analyzer.py | Cached player stats (speeds up API calls) |
| `model_cache/*.pkl` | train_auto.py (--enable-window-ensemble) | Window-specific ensembles |

---

## 🚨 Error Recovery

### "Models not found" Error
```powershell
# You deleted caches - need to retrain
python train_auto.py --verbose --fresh
```

### "API rate limit exceeded"
```powershell
# Wait 60 seconds between fetch runs
# Or reduce DAYS_TO_FETCH in riq_analyzer.py
```

### "Calibration failed - not enough data"
```powershell
# Normal! Need 200+ settled predictions
# Keep running fetch_bet_results_incremental.py
python analyze_ledger.py  # Check progress
```

### "Player not found in nba_api"
- Player name mismatch between APIs
- Check spelling in riq_analyzer.py debug output
- Fallback to API-Sports data (automatic)

---

## 📈 Performance Metrics to Track

After running `analyze_ledger.py`, monitor:

1. **Overall Win Rate**: Target 52%+ (you're at 49.1% currently)
2. **By Prop Type**:
   - Points: 50.0%
   - Assists: 52.8% ✅ **(profitable!)**
   - Rebounds: 46.5%
   - Threes: 50.0%
3. **Calibration**: Predicted probabilities should match actual win rates
4. **ROI**: Return on investment per bet category
5. **Sharpe Ratio**: Risk-adjusted returns

---

## 🎯 Next Steps to Improve (Prioritized)

### **Priority 1: Get More Data (This Week)**
```powershell
# Run daily until 500+ settled predictions
python fetch_bet_results_incremental.py
python analyze_ledger.py  # Check progress
```

**Goal:** 500-1000 settled predictions for robust recalibration

---

### **Priority 2: Implement Phase Features (Next Week)**

**Phase 1 - Shot Volume** (in train_auto.py):
- Add FGA (field goal attempts) rolling stats
- Add 3PA rolling stats
- Add FTA rolling stats
- Per-minute rates for each

**Phase 2 - Matchup Context** (already partially done):
- Opponent defensive rating
- Pace adjustments
- ✅ Already using real-time team stats from nba_api

**Phase 3 - Advanced Rates**:
- Usage rate (FGA + 0.44*FTA)
- Rebound rate
- Assist rate
- ✅ Now calculated from actual volume stats in riq_analyzer.py

**Expected Improvement:** +3-5% accuracy on predictions

---

### **Priority 3: Full Retrain with New Features**
```powershell
# After implementing phases
python train_auto.py --verbose --fresh
python train_ensemble_enhanced.py
python train_dynamic_selector_enhanced.py
```

---

## 🔍 Monitoring Commands

```powershell
# Check current model status
Get-ChildItem models/*.pkl | Select-Object Name, LastWriteTime

# Check ledger size
python -c "import pickle; print(len(pickle.load(open('bets_ledger.pkl','rb'))['bets']), 'total predictions')"

# Check calibration status
python -c "import pickle; cal=pickle.load(open('calibration.pkl','rb')); print('Calibrated types:', list(cal.keys()))"

# View latest predictions
Get-ChildItem prop_analysis_*.json | Sort-Object LastWriteTime -Descending | Select-Object -First 1
```

---

## 💡 Pro Tips

1. **Don't overtrain:** Stick to the scheduled retrains (monthly/seasonal)
2. **Track everything:** The ledger is your ground truth
3. **Recalibrate often:** Models drift - use isotonic regression
4. **Monitor assists:** Currently your best-performing category (52.8%)
5. **Be patient:** Need 500+ predictions for statistical significance
6. **Use --verbose:** Helps debug issues during training
7. **Use --fresh:** Ensures clean training runs (copies CSVs to new folder)

---

## 🎤 Meeting Prep Talking Points

**Technical Stack:**
- 23 years historical data (2002-2026)
- Ensemble ML: LightGBM + Ridge + Elo + Four Factors
- Bayesian calibration with isotonic regression
- Real-time data from nba_api + The Odds API

**Current Performance:**
- 1,728 predictions tracked
- 49.1% overall win rate (improving to 52%+ target)
- Assists model profitable at 52.8%
- Live production system with daily predictions

**Innovation:**
- Dynamic window selection (chooses best historical period per player)
- Phase-based feature engineering (shot volume, efficiency, advanced rates)
- Kelly criterion sizing with drawdown scaling
- Continuous learning from settled predictions

---

## 📞 Support / Questions

Check these files for more detail:
- `README.md` - Comprehensive technical documentation
- `COMMANDS_TO_RUN.md` - Original cleanup checklist
- Training logs: `logs/pipeline_*.log`
- Model metadata: `models/training_metadata.json`

**Debug Mode:**
Set `DEBUG_MODE = True` in riq_analyzer.py for detailed output

---

*This guide assumes you're starting fresh after cache cleanup. Adjust frequencies based on your usage pattern.*
