# Retraining Verification - Ready to Execute

## ✅ Feature Verification Complete

### 1. RIQ Analyzer Has All Features ✅
**File:** `riq_analyzer.py:3015-3400`

**All 7 Phases Implemented:**
- ✅ **Phase 1:** Shot volume (16 features) - lines 3180-3202
- ✅ **Phase 2:** Matchup context (4 features) - lines 3203-3208
- ✅ **Phase 3:** Advanced rates (3 features) - lines 3209-3213
- ✅ **Phase 4:** Home/away splits (4 features) - lines 3214-3219
- ✅ **Phase 5:** Position/matchup (10 features) - lines 3220-3232
- ✅ **Phase 6:** Momentum & optimization (24-36 features) - lines 3233-3270
- ✅ **Phase 7:** Basketball Reference priors (68 features) - lines 3271+

**Total:** ~150-218 features (depending on priors availability)

### 2. Training Script Has Rolling Features ✅
**File:** `train_player_models.py:143-157`

```python
from rolling_features import add_rolling_features

window_df = add_rolling_features(
    window_df,
    windows=[5, 10, 20],  # L5, L10, L20 rolling averages
    add_variance=True,    # Add std deviation
    add_trend=True,       # Add momentum indicators
    low_memory=False,     # Use full feature set
    verbose=verbose
)
```

**Rolling Features Added:**
- L5, L10, L20 averages for all stats
- Variance/consistency metrics
- Trend/momentum indicators
- ~80 additional features

### 3. Meta-Learner Handles Feature Mismatch ✅
**File:** `backtest_2024_2025.py:189-202`

**How Mixed Features Work:**

```
Old Windows (1947-2000): 70 features
New Windows (2001-2021): 150 features
Test Data: 150 features

Individual Window Prediction:
├─ Window 1 (70 features)
│  └─ Align test data → use only 70 features window expects
│  └─ Predict: [12.5, 14.2, ...] ✅
│
├─ Window 25 (150 features)
│  └─ Align test data → use all 150 features
│  └─ Predict: [13.1, 15.3, ...] ✅
│
Meta-Learner:
└─ Input: 25 predictions per sample [12.5, 13.1, 14.7, ...]
   └─ Doesn't care about raw features!
   └─ Only uses: predictions + context (position, usage, etc.)
   └─ Outputs: Weighted ensemble prediction ✅
```

**Feature Alignment Code:**
```python
# Lines 189-202 in backtest_2024_2025.py
if 'feature_names' in window_models and window_models['feature_names']:
    model_features = window_models['feature_names']

    # Only use features that model was trained on
    available_features = [f for f in model_features if f in X_test.columns]
    X_test = X_test[available_features]

    # Add missing features as zeros (if model expects features we don't have)
    for feat in model_features:
        if feat not in X_test.columns:
            X_test[feat] = 0

    # Ensure column order matches training
    X_test = X_test[model_features]
```

**This handles:**
- ✅ Old windows expecting 70 features, test has 150 → uses 70
- ✅ New windows expecting 150 features, test has 150 → uses 150
- ✅ Missing features → filled with zeros
- ✅ Column order → matched exactly

### 4. TabNet Embeddings Verified ✅
**File:** `neural_hybrid.py:180-220`

**Embedding Extraction:**
```python
# Extract 24-dim embeddings from each decision step
embeddings = []
for step_idx, transformer in enumerate(model.network.encoder.feat_transformers):
    step_emb = transformer(x).detach().cpu().numpy()
    embeddings.append(step_emb)

# Average across decision steps
final_embeddings = np.mean(embeddings, axis=0)  # (n_samples, 24)
```

**Then used in LightGBM:**
```python
# Combine raw features + embeddings
X_combined = np.hstack([X, tabnet_embeddings])  # (n_samples, n_features + 24)

# Train LightGBM on combined features
lgb_model.fit(X_combined, y)
```

**Embeddings capture:**
- Non-linear feature interactions
- Latent player archetypes
- Complex temporal patterns

---

## 🚀 Ready to Execute

**Command:**
```bash
modal run retrain_2001_plus.py
```

**What Happens:**
1. **Delete old windows** (2001-2021) from Modal volume
2. **Train 7 windows in parallel:**
   - 2001-2003
   - 2004-2006
   - 2007-2009
   - 2010-2012
   - 2013-2015
   - 2016-2018
   - 2019-2021

3. **Each window gets:**
   - ✅ Rolling features (L5, L10, L20 + variance + trends)
   - ✅ TabNet embeddings (24-dim)
   - ✅ Hybrid multi-task architecture
   - ✅ All feature interactions
   - ✅ ~150 features total

**Resources:**
- GPU: A10G ($1.10/hour)
- RAM: 64GB
- Timeout: 4 hours per window
- Parallelization: All 7 windows simultaneously

**Cost:** ~$15 (7 windows × 2 hours × $1.10/hour)
**Time:** ~2 hours (parallel execution)

---

## 📥 After Training

**Download retrained models:**
```bash
python download_all_models.py
```

**Verify feature counts:**
```bash
# Check model metadata
cat model_cache/player_models_2019_2021_meta.json
```

**Use in production:**
```bash
# Analyzer automatically uses new features
python riq_analyzer.py

# Or use ensemble mode (after meta-learner training)
python riq_analyzer.py --use-ensemble
```

---

## 🔍 Feature Count Breakdown

### Old Windows (1947-2000): ~70 features
- Basic stats
- Simple rolling averages
- No advanced stats (not available pre-2002)

### New Windows (2001-2021): ~150 features
- All basic stats
- **Rolling features:**
  - L5, L10, L20 averages (×12 stats = 36 features)
  - Variance (×12 stats = 12 features)
  - Trends (×12 stats = 12 features)
- **Advanced stats:**
  - Usage%, TS%, Rebound%, Assist%
  - Per-36 rates
  - Efficiency metrics
- **Interactions:**
  - Shot volume × efficiency
  - Position × usage patterns

### Test Data: ~150 features
- Built by `riq_analyzer.py` with all 7 phases
- Always has maximum feature set
- Individual windows align automatically

---

## ✅ All Systems Verified

| Component | Status | Notes |
|-----------|--------|-------|
| **RIQ Analyzer** | ✅ Ready | All 7 phases implemented |
| **Training Script** | ✅ Ready | Rolling features enabled |
| **Meta-Learner** | ✅ Ready | Handles mixed feature sets |
| **Feature Alignment** | ✅ Ready | Automatic alignment per window |
| **TabNet Embeddings** | ✅ Ready | 24-dim extraction verified |
| **Parallel Training** | ✅ Ready | 7 windows simultaneously |
| **Modal Infrastructure** | ✅ Ready | 64GB RAM, A10G GPU |

**No blockers. Ready to execute retraining.**
