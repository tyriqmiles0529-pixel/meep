# NBA Predictor - Consolidated Workflow Guide

**Date:** 2025-11-04  
**Status:** ✅ FULLY CONSOLIDATED

---

## 🎯 ONE FILE PER STEP

The entire NBA prediction pipeline is now **4 simple files**:

| Step | File | Purpose | Frequency |
|------|------|---------|-----------|
| **1. Train** | `train_auto.py` | Train all models (base + ensemble + selector) | Monthly/Seasonal |
| **2. Predict** | `riq_analyzer.py` | Get today's betting recommendations | Daily (morning) |
| **3. Evaluate** | `evaluate.py` | Fetch results + recalibrate + analyze | Daily (evening) |
| **4. Monitor** | *(built into evaluate.py)* | Performance analysis | Weekly |

---

## 📋 COMPLETE WORKFLOW

### **STEP 1: TRAIN MODELS**

```powershell
# Standard training (base + enhanced ensemble)
python train_auto.py --verbose --fresh

# Advanced training (adds 5-year windows + selector)
python train_auto.py --verbose --fresh --enable-window-ensemble
```

**What it does:**
- ✅ Downloads latest data from Kaggle (2002-2026)
- ✅ Trains game models (moneyline, spread)
- ✅ Trains player models (points, assists, rebounds, threes, minutes)
- ✅ Trains enhanced ensemble (Ridge + Elo + Four Factors + Meta-learner)
- ✅ Trains window ensembles (optional, with `--enable-window-ensemble`)
- ✅ Trains dynamic selector (automatic if windows exist)

**Output:** `models/*.pkl` (19 files), `training_metadata.json`

**Runtime:** 45-60 min (standard), 3-4 hours (with windows, first time)

**When to run:**
- Initial setup (once)
- Monthly during season
- Start of new season (October)

---

### **STEP 2: GET PREDICTIONS**

```powershell
# Run daily before games start
python riq_analyzer.py
```

**What it does:**
- ✅ Fetches today's NBA games (API-Sports)
- ✅ Fetches betting lines (The Odds API)
- ✅ Uses trained models to predict player props
- ✅ Calculates Kelly sizing and Expected Value
- ✅ Ranks bets by ELG (Expected Log Growth)
- ✅ Records predictions to ledger

**Output:** `prop_analysis_YYYYMMDD_HHMMSS.json`, updates `bets_ledger.pkl`

**Runtime:** 2-5 minutes

**When to run:**
- Every morning during NBA season
- When betting lines are posted

---

### **STEP 3: EVALUATE RESULTS**

```powershell
# Run all steps (fetch + recalibrate + analyze)
python evaluate.py

# Or run individual steps:
python evaluate.py --fetch-only      # Just fetch results
python evaluate.py --analyze-only    # Just analyze performance
python evaluate.py --recalibrate     # Fetch + recalibrate only
```

**What it does:**
- ✅ **Fetch:** Gets actual player stats from nba_api
- ✅ **Settle:** Updates win/loss for each bet
- ✅ **Recalibrate:** Trains isotonic regression (if 200+ settled)
- ✅ **Analyze:** Shows win rates, calibration, performance by prop type

**Output:** Updates `bets_ledger.pkl`, `calibration.pkl`, console report

**Runtime:** 5-10 minutes

**When to run:**
- Every evening after games finish
- Weekly for performance review

---

## 📊 BEFORE vs AFTER

### **Before Consolidation (6+ commands)**

```powershell
# Training (3 commands)
python train_auto.py --verbose --fresh
python train_ensemble_enhanced.py
python train_dynamic_selector_enhanced.py

# Prediction (1 command)
python riq_analyzer.py

# Evaluation (3 commands)
python fetch_bet_results_incremental.py
python recalibrate_models.py
python analyze_ledger.py
```

### **After Consolidation (3 commands)**

```powershell
# Training (1 command)
python train_auto.py --verbose --fresh

# Prediction (1 command)
python riq_analyzer.py

# Evaluation (1 command)
python evaluate.py
```

**Reduction:** 7 commands → 3 commands (57% fewer!)

---

## 🗂️ FILE STRUCTURE

### **Core Pipeline (3 files)**

```
nba_predictor/
├── train_auto.py              # STEP 1: Training
├── riq_analyzer.py            # STEP 2: Predictions
└── evaluate.py                # STEP 3: Evaluation (NEW!)
```

### **Supporting Files (imported by core)**

```
├── train_ensemble_enhanced.py       # Ensemble logic (imported)
├── train_dynamic_selector_enhanced.py  # Selector logic (called)
├── ensemble_models_enhanced.py      # Model classes (library)
└── player_ensemble_enhanced.py      # Player ensemble (library)
```

### **Legacy/Deprecated (can delete)**

```
├── fetch_bet_results_incremental.py  # ❌ NOW IN evaluate.py
├── recalibrate_models.py             # ❌ NOW IN evaluate.py
├── analyze_ledger.py                 # ❌ NOW IN evaluate.py
└── keys.py                           # ❌ Use .env instead
```

---

## 🔄 DAILY WORKFLOW

### **Morning Routine (5 minutes)**

```powershell
# Get today's predictions
python riq_analyzer.py

# Review output
Get-Content prop_analysis_*.json | Select-String "stake|win_prob|ev"
```

### **Evening Routine (10 minutes)**

```powershell
# Evaluate yesterday's bets
python evaluate.py

# Check win rate
# (displayed automatically in console)
```

### **Weekly Review (5 minutes)**

```powershell
# Full analysis
python evaluate.py --analyze-only
```

---

## 📈 MONTHLY WORKFLOW

### **Mid-Month Check**

```powershell
# Check if recalibration needed
python evaluate.py --analyze-only

# If 500+ settled bets, recalibrate
python evaluate.py --recalibrate
```

### **Month-End Update**

```powershell
# Retrain with latest data
python train_auto.py --verbose

# Evaluate performance
python evaluate.py
```

---

## 🏀 SEASONAL WORKFLOW

### **Pre-Season (September-October)**

```powershell
# Full retrain with fresh data
python train_auto.py --verbose --fresh --enable-window-ensemble

# Takes 3-4 hours first time
# Subsequent runs use cached windows (~1 hour)
```

### **Mid-Season (January)**

```powershell
# Quick retrain with new data
python train_auto.py --verbose --enable-window-ensemble
```

### **Playoffs (April-June)**

```powershell
# Daily workflow continues
# Models adapt to playoff intensity automatically
```

---

## 🎛️ EVALUATE.PY OPTIONS

### **All Steps (Default)**

```powershell
python evaluate.py
```

Runs: Fetch → Recalibrate → Analyze

### **Fetch Results Only**

```powershell
python evaluate.py --fetch-only
```

Use when: Just want to update ledger, skip analysis

### **Analyze Only**

```powershell
python evaluate.py --analyze-only
```

Use when: Want to see stats without fetching new data

### **Fetch + Recalibrate**

```powershell
python evaluate.py --recalibrate
```

Use when: Want to update calibration without full analysis

### **Change Min Samples for Recalibration**

```powershell
python evaluate.py --min-samples 500
```

Default: 200 samples. Higher = more conservative recalibration.

### **Quiet Mode**

```powershell
python evaluate.py --quiet
```

Suppresses all output (useful for cron jobs)

---

## 🔍 VERIFICATION

After consolidation, verify everything works:

```powershell
# 1. Check core files exist
Test-Path train_auto.py, riq_analyzer.py, evaluate.py

# 2. Test evaluate.py
python evaluate.py --analyze-only

# 3. Verify no import errors
python -c "import evaluate; print('✅ evaluate.py OK')"

# 4. Check file sizes
Get-ChildItem train_auto.py, riq_analyzer.py, evaluate.py | Select-Object Name, Length
```

**Expected:**
- `train_auto.py`: ~239 KB
- `riq_analyzer.py`: ~180 KB
- `evaluate.py`: ~17 KB

---

## 🗑️ CLEANUP (OPTIONAL)

Safe to delete after consolidation:

```powershell
# These are now in evaluate.py
Remove-Item fetch_bet_results_incremental.py
Remove-Item recalibrate_models.py
Remove-Item analyze_ledger.py

# Move credentials to .env
# Remove-Item keys.py  # After creating .env
```

**Keep these (used by pipeline):**
- `train_ensemble_enhanced.py` (imported)
- `train_dynamic_selector_enhanced.py` (called as subprocess)
- `ensemble_models_enhanced.py` (library)
- `player_ensemble_enhanced.py` (library)

---

## 📊 OUTPUTS SUMMARY

| File | Created By | Purpose |
|------|-----------|---------|
| `models/*.pkl` | train_auto.py | Trained models (19 files) |
| `training_metadata.json` | train_auto.py | RMSE, features, config |
| `prop_analysis_*.json` | riq_analyzer.py | Daily predictions |
| `bets_ledger.pkl` | riq_analyzer.py | All predictions + results |
| `calibration.pkl` | evaluate.py | Isotonic calibration curves |
| `player_cache.pkl` | riq_analyzer.py | Cached player stats |
| `model_cache/*.pkl` | train_auto.py | Window ensembles (optional) |

---

## ⏱️ TIME SAVINGS

| Task | Before | After | Savings |
|------|--------|-------|---------|
| Training | 3 commands, manual | 1 command, automatic | 66% fewer steps |
| Evaluation | 3 commands, 15 min | 1 command, 10 min | 5 minutes saved |
| Daily workflow | 4+ commands | 2 commands | 50% faster |

---

## 💡 BEST PRACTICES

1. **Daily predictions:** Run `riq_analyzer.py` every morning
2. **Daily evaluation:** Run `evaluate.py` every evening
3. **Weekly review:** Check win rates and calibration
4. **Monthly retrain:** Update models with latest data
5. **Use --fresh sparingly:** Only for clean runs (seasonal)
6. **Monitor calibration:** Recalibrate when drift detected
7. **Keep ledger safe:** Backup `bets_ledger.pkl` regularly

---

## 🎯 QUICK REFERENCE

```powershell
# TRAINING (Monthly/Seasonal)
python train_auto.py --verbose --fresh

# PREDICTION (Daily Morning)
python riq_analyzer.py

# EVALUATION (Daily Evening)
python evaluate.py

# ANALYSIS (Weekly)
python evaluate.py --analyze-only
```

---

## 🚀 AUTOMATION OPTION

### **Windows Task Scheduler**

```powershell
# Morning task (8 AM)
python C:\path\to\nba_predictor\riq_analyzer.py

# Evening task (11 PM)
python C:\path\to\nba_predictor\evaluate.py --quiet
```

### **PowerShell Script**

```powershell
# daily_workflow.ps1
cd C:\path\to\nba_predictor

# Morning
python riq_analyzer.py

# Evening (after 11 PM)
if ((Get-Date).Hour -ge 23) {
    python evaluate.py
}
```

---

## 📞 TROUBLESHOOTING

### **"No bets in ledger"**
**Fix:** Run `riq_analyzer.py` first to create predictions

### **"Not enough settled bets for recalibration"**
**Fix:** Normal! Need 200+ settled predictions. Keep running `evaluate.py` daily.

### **"nba_api not installed"**
**Fix:** `pip install nba_api`

### **"Models not found"**
**Fix:** Run `train_auto.py` first

---

## 📖 RELATED DOCUMENTATION

- `INTEGRATION_SUMMARY.md` - Training consolidation details
- `EXECUTION_GUIDE.md` - Full workflow documentation
- `CHECKLIST.md` - Post-cleanup steps
- `README.md` - Project overview

---

**Bottom Line:** The entire NBA prediction workflow is now **3 simple commands**! Train → Predict → Evaluate. 🚀
