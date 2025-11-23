# NBA Meta-Learner V4 Project - Migration Context

## 🎯 PROJECT OVERVIEW

**Goal**: Train a V4 meta-learner ensemble that combines predictions from 27 window models (1947-2026) to predict NBA player stats (points, rebounds, assists, threes).

**Current Status**: 🟡 **INCOMPLETE** - 26/27 windows trained, meta-learner trained with incomplete ensemble

**Critical Issue**: Previous agent wasted credits by training meta-learner before completing all window training.

---

## 📊 CURRENT TRAINING STATUS

### ✅ COMPLETED:
- **26 window models trained** (1947-1949 through 2025-2026)
- **V4 meta-learner trained** with 26 windows (saved to `/models/meta_learner_v4_all_components.pkl`)
- **All V4 components working**: residual correction, player embeddings, temporal memory

### ❌ MISSING:
- **2022-2024 window model** - NOT trained yet
- **Complete meta-learner** - needs retraining with all 27 windows

### 💳 BILLING STATUS:
- **Credits exhausted** due to wasteful training sequence
- **Billing reset**: December 1, 2025
- **Estimated cost to complete**: $6-8 (train missing window + retrain meta-learner)

---

## 🚨 CRITICAL NEXT STEPS (IN ORDER)

1. **WAIT for billing reset** (December 1, 2025)
2. **Train 2022-2024 window**:
   ```bash
   modal run train_2022_2024_only.py
   ```
   - Time: ~4 hours
   - Cost: ~$4.40
3. **Verify ALL 27 windows exist**:
   ```bash
   modal run check_window_models.py
   ```
4. **Retrain meta-learner with complete ensemble**:
   ```bash
   modal run modal_train_meta_clean.py::train_meta_learner_v4 --config-path experiments/v4_full.yaml
   ```
   - Time: ~2 hours
   - Cost: ~$2.20
5. **Run backtesting**:
   ```bash
   modal run modal_train_meta_clean.py::main_v4 --config experiments/v4_full.yaml --backtest
   ```

---

## 🏗️ TECHNICAL ARCHITECTURE

### Modal Infrastructure:
- **Volumes**: 
  - `nba-models`: Stores all window models and meta-learner
  - `nba-data`: Stores training data
- **GPU**: A10G ($1.10/hour)
- **Image**: Debian slim with ML dependencies (see `modal_requirements.txt`)

### V4 Meta-Learner Components:
1. **Cross-Window Residual Correction**: Corrects systematic biases across windows
2. **Player Identity Embeddings**: Learns player-specific patterns (475 players)
3. **Temporal Memory (Transformer)**: Captures time-based patterns

### Window Model Structure:
- **25 windows**: 1947-1949 through 2019-2021 (70-182 features each)
- **2 recent windows**: 2025-2026 trained, 2022-2024 missing
- **Features**: Rolling averages, TabNet embeddings, feature interactions

---

## 💰 COST CONSIDERATIONS & WARNINGS

### ⚠️ LESSONS LEARNED:
- **ALWAYS verify prerequisites before expensive training**
- **Check window count before meta-learner training**
- **Monitor billing usage in real-time**
- **Use CPU for non-critical tasks when possible**

### 💡 COST OPTIMIZATION:
- **Window training**: ~$4-5 per window (GPU required)
- **Meta-learner training**: ~$2-3 (GPU required)
- **Backtesting**: ~$1-2 (GPU optional)
- **Data processing**: ~$0.50 (CPU only)

### 🚨 HIGH-COST MISTAKES TO AVOID:
1. Training meta-learner with incomplete windows (wastes $2-3)
2. Running multiple training sessions in parallel
3. Not checking billing limits before starting
4. Using GPU for data processing (use CPU instead)

---

## 📁 KEY FILES & SCRIPTS

### Training Scripts:
- `train_2022_2024_only.py` - Trains missing 2022-2024 window (USE THIS FIRST)
- `modal_train_meta_clean.py` - Main meta-learner training script
- `retrain_2022_plus.py` - Trains 2022-2024 and 2025-2026 (already used)
- `retrain_remaining.py` - Trains 8 windows (2004-2026) - AVOID (too expensive)

### Verification Scripts:
- `check_window_models.py` - Lists all trained windows
- `download_model.py` - Downloads trained models locally

### Local Scripts (No Modal Cost):
- `local_backtest.py` - Local backtesting without Modal
- `production_pipeline.py` - Complex pipeline (AVOID - caused issues)

### Configuration:
- `experiments/v4_full.yaml` - V4 training configuration
- `modal_requirements.txt` - Python dependencies

### Core Modules:
- `train_meta_learner_v4.py` - V4 meta-learner implementation
- `ensemble_predictor.py` - Window model loading and prediction
- `hybrid_multi_task.py` - Multi-task learning architecture

---

## 🎯 PRODUCTION INTEGRATION

### Current Model Location:
- **Modal**: `/models/meta_learner_v4_all_components.pkl` (26 windows)
- **Local**: Download with `modal volume get nba-models meta_learner_v4_all_components.pkl ./meta_learner_v4.pkl`

### Integration Steps:
1. **Download trained meta-learner** locally
2. **Load with dill** (not pickle due to lambda functions)
3. **Integrate into analyzer** for predictions
4. **Update when complete 27-window model is ready**

### Prediction Pipeline:
```python
import dill as pickle
with open('meta_learner_v4.pkl', 'rb') as f:
    meta_learner = pickle.load(f)

# Use meta_learner.predict() for new predictions
```

---

## 🔧 TECHNICAL SPECIFICATIONS

### Model Performance (26-window version):
- **Points MAE**: ~0.069 (2025-2026 window)
- **Assists MAE**: ~0.112
- **Rebounds MAE**: ~0.015
- **Threes MAE**: ~0.041

### Data Requirements:
- **Input**: Player statistics with rolling features
- **Window predictions**: 27 individual model predictions per stat
- **Player embeddings**: 475 players in training set
- **Temporal data**: Game sequences for transformer

### Dependencies:
- **Core**: pandas, numpy, scikit-learn, torch
- **Models**: pytorch-tabnet, lightgbm, xgboost
- **Modal**: modal, dill (for lambda serialization)

---

## 🚨 CRITICAL WARNINGS FOR NEXT AGENT

1. **NEVER train meta-learner without verifying ALL 27 windows exist**
2. **ALWAYS check billing limits before starting any training**
3. **USE `train_2022_2024_only.py` for missing window - NOT the expensive scripts**
4. **VERIFY window count with `check_window_models.py` before meta-learner training**
5. **MONITOR training costs in real-time via Modal dashboard**
6. **DOWNLOAD existing 26-window model before any new training** (backup)

---

## 📞 CONTACT & HANDOFF

**Previous Agent Performance**: ❌ Wasteful, ignored instructions, cost user credits
**User Priority**: Cost-conscious, complete training sequence, production integration

**Success Criteria**:
1. Train 2022-2024 window efficiently
2. Retrain meta-learner with all 27 windows
3. Provide production-ready model
4. Stay within budget ($6-8 remaining cost)

**Immediate Action**: Wait for December 1st billing reset, then train missing window first.

---

## 📋 QUICK REFERENCE CHEAT SHEET

```bash
# 1. Check windows (do this FIRST)
modal run check_window_models.py

# 2. Train missing window (do this SECOND)
modal run train_2022_2024_only.py

# 3. Train meta-learner (do this THIRD - only after all windows exist)
modal run modal_train_meta_clean.py::train_meta_learner_v4 --config-path experiments/v4_full.yaml

# 4. Backtest (do this FOURTH)
modal run modal_train_meta_clean.py::main_v4 --config experiments/v4_full.yaml --backtest

# 5. Download final model
modal volume get nba-models meta_learner_v4_all_components.pkl ./final_model.pkl
```

**⚠️ REMEMBER**: 26 windows = incomplete model, 27 windows = production ready
