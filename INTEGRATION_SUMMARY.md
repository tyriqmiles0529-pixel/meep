# Training Pipeline Integration Summary

**Date:** 2025-11-04  
**Status:** ✅ FULLY INTEGRATED

---

## 🎯 What Changed

All ensemble training is now **integrated into one command**:

```powershell
python train_auto.py --verbose --fresh --enable-window-ensemble
```

This single command now trains:
1. ✅ Base LightGBM models (game + player)
2. ✅ Enhanced ensemble (Ridge + Elo + Four Factors + Meta-learner)
3. ✅ Window ensembles (5-year windows, if `--enable-window-ensemble` flag used)
4. ✅ Dynamic selector (automatically if 2+ windows exist)

---

## 📋 Integration Details

### **Before (3 separate commands)**
```powershell
# Step 1: Train base models
python train_auto.py --verbose --fresh

# Step 2: Train ensemble components
python train_ensemble_enhanced.py

# Step 3: Train dynamic selector
python train_dynamic_selector_enhanced.py
```

### **After (1 unified command)**
```powershell
# Standard training (base + enhanced ensemble)
python train_auto.py --verbose --fresh

# Advanced training (adds window ensembles + selector)
python train_auto.py --verbose --fresh --enable-window-ensemble
```

---

## 🔧 How It Works

### **Phase 1: Base Models** (Always runs)
- Game models: moneyline, spread
- Player models: points, assists, rebounds, threes, minutes
- **Location:** Lines 1384-1568 in train_auto.py

### **Phase 2: Enhanced Ensemble** (Always runs)
- Ridge regression
- Dynamic Elo rating
- Four Factors model
- Meta-learner (combines all 4)
- **Location:** Lines 4262-4301 in train_auto.py
- **Import:** `from train_ensemble_enhanced import train_all_ensemble_components`

### **Phase 3: Window Ensembles** (Optional - needs flag)
- Trains separate models for 5-year windows (2002-2006, 2007-2011, etc.)
- Caches results to avoid retraining
- **Location:** Lines 4090-4256 in train_auto.py
- **Flag:** `--enable-window-ensemble`
- **Runtime:** ~2-3 hours (trains multiple windows)

### **Phase 4: Dynamic Selector** (Automatic if windows exist)
- Trains context-aware window selection
- Only runs if 2+ window ensembles found
- **Location:** Lines 4630-4657 in train_auto.py (NEW!)
- **Method:** Calls `train_dynamic_selector_enhanced.py` via subprocess
- **Auto-detection:** Checks for `model_cache/player_ensemble_*.pkl` files

---

## 📊 File Outputs

### **Always Created (Standard Training)**
| File | Description |
|------|-------------|
| `models/moneyline_model.pkl` | Game moneyline predictor |
| `models/spread_model.pkl` | Game spread predictor |
| `models/points_model.pkl` | Player points predictor |
| `models/assists_model.pkl` | Player assists predictor |
| `models/rebounds_model.pkl` | Player rebounds predictor |
| `models/threes_model.pkl` | Player threes predictor |
| `models/minutes_model.pkl` | Player minutes predictor |
| `models/*_sigma_model.pkl` | Uncertainty estimators |
| `models/ridge_model_enhanced.pkl` | Enhanced ensemble - Ridge |
| `models/elo_model_enhanced.pkl` | Enhanced ensemble - Elo |
| `models/four_factors_model_enhanced.pkl` | Enhanced ensemble - Four Factors |
| `models/ensemble_meta_learner_enhanced.pkl` | Enhanced ensemble - Meta-learner |
| `models/training_metadata.json` | RMSE metrics, features, config |

### **Created with --enable-window-ensemble**
| File | Description |
|------|-------------|
| `model_cache/player_ensemble_2002_2006.pkl` | Window 1 ensemble |
| `model_cache/player_ensemble_2007_2011.pkl` | Window 2 ensemble |
| `model_cache/player_ensemble_2012_2016.pkl` | Window 3 ensemble |
| `model_cache/player_ensemble_2017_2021.pkl` | Window 4 ensemble |
| `model_cache/player_ensemble_2022_2026.pkl` | Window 5 ensemble (current) |
| `model_cache/player_ensemble_*_meta.json` | Metadata for each window |
| `model_cache/dynamic_selector_enhanced.pkl` | Context-aware selector |
| `model_cache/dynamic_selector_enhanced_meta.json` | Selector metadata |

---

## ⏱️ Training Times

### **Standard (no --enable-window-ensemble)**
- Base models: ~30-45 minutes
- Enhanced ensemble: ~10-15 minutes
- **Total:** ~45-60 minutes

### **Advanced (with --enable-window-ensemble)**
- Base models: ~30-45 minutes
- Enhanced ensemble: ~10-15 minutes
- Window ensembles: ~2-3 hours (first run)
- Window ensembles: ~15-20 minutes (subsequent runs - uses cache)
- Dynamic selector: ~5-10 minutes
- **Total first run:** ~3-4 hours
- **Total subsequent:** ~1 hour (cached windows)

---

## 🚀 Usage Guide

### **Quick Start (Standard)**
```powershell
# Most common usage - trains everything you need
python train_auto.py --verbose --fresh
```

### **Advanced (With Window Ensembles)**
```powershell
# First time - trains all windows (slow)
python train_auto.py --verbose --fresh --enable-window-ensemble

# Subsequent runs - uses cached windows (faster)
python train_auto.py --verbose --enable-window-ensemble
```

### **Update Just Current Season**
```powershell
# Fast update without --fresh (reuses cached data)
python train_auto.py --verbose --enable-window-ensemble
```

---

## 🔄 Automatic Features

### **Smart Caching**
- Window ensembles cache to `model_cache/`
- Cached windows are reused if:
  - File exists
  - Metadata matches expected seasons
  - Not the current season window (always retrained)

### **Auto-Detection**
- Dynamic selector training is automatic
- Only runs if 2+ window ensembles detected
- Skips gracefully if not available

### **Error Handling**
- Each phase has try/except blocks
- Failures don't stop subsequent phases
- Warnings printed for debugging

---

## 📝 Configuration Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--verbose` | False | Print detailed progress |
| `--fresh` | False | Copy CSVs to new folder (clean run) |
| `--enable-window-ensemble` | False | Train 5-year window ensembles |
| `--lgb-log-period` | 0 | LightGBM logging interval (0=silent) |
| `--n-jobs` | -1 | CPU threads (-1=all cores) |
| `--game-season-cutoff` | 2002 | Minimum season for game data |
| `--player-season-cutoff` | 2002 | Minimum season for player data |

---

## 🎯 Recommended Workflow

### **Initial Setup (One Time)**
```powershell
# Full training with all features
python train_auto.py --verbose --fresh --enable-window-ensemble
```

### **Monthly Updates (During Season)**
```powershell
# Quick retrain with new data
python train_auto.py --verbose --enable-window-ensemble
```

### **Seasonal Update (October)**
```powershell
# Fresh retrain at season start
python train_auto.py --verbose --fresh --enable-window-ensemble
```

### **Quick Test (Development)**
```powershell
# Faster training for testing changes
python train_auto.py --verbose --player-season-cutoff 2020
```

---

## ✅ Verification

After training, verify all components:

```powershell
# Check base models
Get-ChildItem models/*.pkl | Measure-Object
# Expected: 19 files

# Check window ensembles (if --enable-window-ensemble used)
Get-ChildItem model_cache/player_ensemble_*.pkl | Measure-Object
# Expected: 5 files (one per window)

# Check dynamic selector (if windows exist)
Test-Path model_cache/dynamic_selector_enhanced.pkl
# Expected: True

# Check metadata
Get-Content models/training_metadata.json | ConvertFrom-Json | Select-Object -ExpandProperty player_metrics
# Should show RMSE for all 5 stats
```

---

## 🔧 Troubleshooting

### **"Dynamic selector training failed"**
**Cause:** Window ensembles don't exist  
**Fix:** Run with `--enable-window-ensemble` first

### **"Window ensemble not found"**
**Cause:** Cache directory missing  
**Fix:** First run needs `--enable-window-ensemble` (takes 3-4 hours)

### **"ImportError: train_ensemble_enhanced"**
**Cause:** Missing ensemble training script  
**Fix:** Ensure `train_ensemble_enhanced.py` exists in project root

### **Training takes too long**
**Cause:** Window ensembles training from scratch  
**Fix:** Subsequent runs reuse cache (much faster)

---

## 🎤 Key Benefits

1. **Simplicity:** One command instead of three
2. **Safety:** Error-tolerant (one failure doesn't stop pipeline)
3. **Efficiency:** Smart caching for window ensembles
4. **Automatic:** Selector training auto-detects windows
5. **Flexible:** Flags control which components run

---

## 📖 Related Files

- `train_auto.py` - Main training pipeline (NOW INCLUDES ALL)
- `train_ensemble_enhanced.py` - Ensemble training logic (imported)
- `train_dynamic_selector_enhanced.py` - Selector training (called as subprocess)
- `EXECUTION_GUIDE.md` - Full workflow documentation
- `CHECKLIST.md` - Step-by-step post-cleanup actions

---

**Bottom Line:** You can now delete the old manual multi-step instructions. Just run `train_auto.py` and everything happens automatically! 🚀
