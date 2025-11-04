# Post-Cleanup Checklist
**What to run RIGHT NOW after deleting caches**

Date: 2025-11-04

---

## ☑️ IMMEDIATE STEPS (Do These First)

### 1. Retrain All Models (~30-60 min)
```powershell
python train_auto.py --verbose --fresh --lgb-log-period 50
```
✅ **Status:** [ ] Not Started | [ ] Running | [ ] Complete  
⏱️ **ETA:** 30-60 minutes  
📍 **Output:** `models/*.pkl`, `training_metadata.json`

**Why:** You deleted model_cache and player_cache - need fresh models

---

### 2. Fetch Today's Props
```powershell
python riq_analyzer.py
```
✅ **Status:** [ ] Not Started | [ ] Running | [ ] Complete  
⏱️ **ETA:** 2-5 minutes  
📍 **Output:** `prop_analysis_YYYYMMDD_HHMMSS.json`

**Why:** Get today's betting recommendations

---

### 3. Update Historical Results (Run Multiple Times)
```powershell
python fetch_bet_results_incremental.py
```
✅ **Status:** [ ] Not Started | [ ] Running | [ ] Complete  
⏱️ **ETA:** 5-10 minutes per run  
📍 **Output:** Updates `bets_ledger.pkl`, `calibration.pkl`

**Why:** Fetch actual outcomes for past predictions

**Note:** Run until it says "Settled 0 bets" (no more data available)

---

### 4. Check Performance
```powershell
python analyze_ledger.py
```
✅ **Status:** [ ] Not Started | [ ] Running | [ ] Complete  
⏱️ **ETA:** <1 minute  
📍 **Output:** Console summary + plots

**Why:** See how models are performing

---

## 📋 OPTIONAL (If You Want Advanced Features)

### 5. Train Ensemble Components
```powershell
python train_ensemble_enhanced.py
```
✅ **Status:** [ ] Not Started | [ ] Running | [ ] Complete  
⏱️ **ETA:** 10-15 minutes

**Why:** Adds Ridge + Elo + Four Factors ensemble (improves accuracy)

---

### 6. Train Dynamic Window Selector
```powershell
python train_dynamic_selector_enhanced.py
```
✅ **Status:** [ ] Not Started | [ ] Running | [ ] Complete  
⏱️ **ETA:** 15-20 minutes

**Why:** Context-aware window selection for player predictions

---

## 🔍 VERIFICATION STEPS

### Check Models Exist
```powershell
Get-ChildItem models/*.pkl
```
**Expected:** 10+ .pkl files (points_model.pkl, assists_model.pkl, etc.)

---

### Check Ledger Size
```powershell
python -c "import pickle; data = pickle.load(open('bets_ledger.pkl', 'rb')); print(f'Total predictions: {len(data.get(\"bets\", []))}')"
```
**Expected:** 1,728+ predictions

---

### Check Today's Props
```powershell
# View latest analysis file
Get-ChildItem prop_analysis_*.json | Sort-Object LastWriteTime -Descending | Select-Object -First 1
```
**Expected:** JSON file from today with betting recommendations

---

## 🚨 TROUBLESHOOTING

### If Step 1 fails:
- Check internet connection (needs to download from Kaggle)
- Verify API keys in keys.py: `KAGGLE_KEY`, `KAGGLE_USERNAME`
- Try without `--fresh` flag

### If Step 2 fails:
- Check `API_KEY` in riq_analyzer.py (API-Sports key)
- Check `THEODDS_API_KEY` (The Odds API key)
- Models must exist first (run Step 1)

### If Step 3 fails:
- Normal if no games finished yet
- Needs bets_ledger.pkl to exist (run Step 2 first)
- Player names must match between APIs

### If Step 4 fails:
- Needs bets_ledger.pkl with settled bets
- Run Step 3 first to populate results

---

## 📊 SUCCESS METRICS

After completing all steps, you should have:

- ✅ 10+ model files in `models/`
- ✅ `training_metadata.json` with RMSE metrics
- ✅ Today's prop analysis JSON
- ✅ Updated `bets_ledger.pkl` with recent results
- ✅ `calibration.pkl` with isotonic curves
- ✅ Performance summary from `analyze_ledger.py`

---

## 📅 DAILY ROUTINE (Going Forward)

**Morning:**
```powershell
python riq_analyzer.py  # Get today's props
```

**Evening (after games):**
```powershell
python fetch_bet_results_incremental.py  # Update results
```

**Weekly:**
```powershell
python analyze_ledger.py  # Check performance
```

**Monthly (if 500+ settled bets):**
```powershell
python recalibrate_models.py  # Recalibrate
```

**Seasonal (October start of season):**
```powershell
python train_auto.py --verbose --fresh  # Full retrain
```

---

## 🎯 CURRENT STATUS

Based on your last `analyze_ledger.py` output:

- Total Predictions: 1,728
- Settled: ~750
- Overall Win Rate: 49.1%
- Best Category: Assists (52.8% ✅)
- Needs Improvement: Rebounds (46.5%)

**Goal:** Get to 52%+ overall win rate via:
1. More data (run fetch daily)
2. Recalibration (after 500+ settled)
3. Phase features (implement in train_auto.py)

---

## 📞 NEXT QUESTIONS?

1. Do models need retraining? → See EXECUTION_GUIDE.md
2. How often to run each script? → See "Daily Routine" above
3. What do the outputs mean? → See README.md
4. How to improve accuracy? → Implement Phase 1/2/3 features

---

*Mark each checkbox as you complete steps. Update Status column to track progress.*
