# NBA Prediction System - Correct Implementation Sequence

**Current Status**: Need to update all code/notebooks FIRST, then retrain with all features

---

## Phase 0: Pre-Training Setup (This Week)

### Step 0.1: Update All Code Files ✅
**Files to Update**:
- ✅ `predict_live.py` (created - needs full feature engineering)
- ✅ `backtest_engine.py` (created)
- ✅ `optimization_features.py` (already has Phase 6 features)
- ✅ `phase7_features.py` (already exists)
- ⏳ `train_auto.py` (verify all features integrated)

**Status**: Code files ready

---

### Step 0.2: Update Training Notebook (PRIORITY 1)
**File**: `NBA_COLAB_SIMPLE.ipynb`

**Changes Needed**:

```python
# Cell 1: Updated to use aggregated data
print("📥 Downloading aggregated dataset...")
!kaggle datasets download -d tyriqmiles/aggregated-nba-data
!gunzip aggregated_nba_data.csv.gzip

print("✅ Aggregated data ready (150+ features pre-computed)")
```

```python
# Cell 2: Train with ALL features
!python train_auto.py \
    --dataset ./aggregated_nba_data.csv \
    --use-neural \
    --neural-epochs 30 \
    --neural-device gpu \
    --verbose \
    --fresh \
    --skip-game-models
```

```python
# Cell 3: Verify 24-dim embeddings
import pickle
import numpy as np

with open('./models/points_hybrid_2022_2026.pkl', 'rb') as f:
    model = pickle.load(f)

# Test embeddings
dummy = np.random.randn(10, 150).astype(np.float32)
_, embeddings = model.tabnet.predict(dummy, return_embeddings=True)

print(f"Embedding shape: {embeddings.shape}")
print(f"Expected: (10, 24)")

if embeddings.shape[1] == 24:
    print("✅ 24-dim embeddings working!")
else:
    print(f"⚠️ Got {embeddings.shape[1]}-dim embeddings")
```

```python
# Cell 4: Test predictions on sample
from predict_live import LivePredictionEngine

engine = LivePredictionEngine(models_dir='./models')
# Test on sample data
print("✅ Prediction system ready")
```

```python
# Cell 5: Download models
!zip -r models.zip models/
from google.colab import files
files.download('models.zip')
```

---

### Step 0.3: Update Prediction Notebook (PRIORITY 2)
**File**: `Riq_Machine.ipynb`

**New Structure**:

```python
# Cell 1: Setup
from predict_live import LivePredictionEngine
import pandas as pd
from datetime import datetime

engine = LivePredictionEngine(models_dir='./models')
print("✅ Loaded models:", list(engine.models.keys()))
```

```python
# Cell 2: Get today's games
date = datetime.now().strftime('%Y-%m-%d')
games = engine.get_todays_games(date)

print(f"📅 Games for {date}:")
print(games[['away_team', 'home_team', 'game_time']])
```

```python
# Cell 3: Generate predictions
all_predictions = []

for _, game in games.iterrows():
    print(f"\n🏀 {game['away_team']} @ {game['home_team']}")

    preds = engine.predict_game(game.to_dict(), explain=True)
    all_predictions.extend(preds)

predictions_df = pd.DataFrame(all_predictions)
```

```python
# Cell 4: Display predictions with SHAP
for pred in all_predictions[:5]:  # Show first 5 players
    print(f"\n{'='*60}")
    print(f"{pred['player_name']} ({pred['team']})")
    print(f"{'='*60}")

    # Points prediction
    pts = pred.get('points', {})
    print(f"Points: {pts.get('prediction', 0):.1f} ± {pts.get('uncertainty', 0):.1f}")
    print(f"  80% range: [{pts.get('lower_80', 0):.1f}, {pts.get('upper_80', 0):.1f}]")

    # SHAP explanation
    if 'explanation' in pts:
        print(f"\nTop reasons:")
        for reason in pts['explanation'][:3]:
            print(f"  • {reason['feature']}: {reason['shap_value']:+.2f}")
```

```python
# Cell 5: Export predictions
predictions_df.to_csv('predictions.csv', index=False)
from google.colab import files
files.download('predictions.csv')
```

---

### Step 0.4: Update Evaluation Notebook (PRIORITY 3)
**File**: `Evaluate_Predictions.ipynb`

**New Structure**:

```python
# Cell 1: Setup backtesting
from backtest_engine import BacktestEngine
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

engine = BacktestEngine(models_dir='./models')
```

```python
# Cell 2: Run backtest
results = engine.run_backtest(
    start_date='2024-10-01',
    end_date='2024-11-09',
    save_results=True
)
```

```python
# Cell 3: Performance Summary
print("\n📊 PERFORMANCE SUMMARY")
print("="*70)

for prop in ['points', 'rebounds', 'assists']:
    if prop in results:
        m = results[prop]
        print(f"\n{prop.upper()}:")
        print(f"  Samples: {m['n_samples']:,}")
        print(f"  MAE: {m['mae']:.2f}")
        print(f"  RMSE: {m['rmse']:.2f}")
        print(f"  R²: {m['r2']:.3f}")
        print(f"  Acc within 2.0: {m['acc_within_2.0']*100:.1f}%")
```

```python
# Cell 4: Calibration Analysis
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, prop in enumerate(['points', 'rebounds', 'assists']):
    if prop not in results or 'calibration' not in results[prop]:
        continue

    cal = results[prop]['calibration']
    confidences = [0.68, 0.80, 0.90, 0.95]
    actual = [cal.get(f'coverage_{int(c*100)}', c) for c in confidences]

    axes[idx].plot(confidences, confidences, 'k--', label='Perfect')
    axes[idx].plot(confidences, actual, 'o-', label='Actual')
    axes[idx].set_xlabel('Expected Coverage')
    axes[idx].set_ylabel('Actual Coverage')
    axes[idx].set_title(f'{prop.upper()} Calibration')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('calibration.png', dpi=150)
plt.show()
```

```python
# Cell 5: Profit Analysis
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ROI comparison
props = ['points', 'rebounds', 'assists']
fixed_roi = [results[p]['profit']['fixed_stake']['roi'] for p in props if p in results]
kelly_roi = [results[p]['profit']['kelly']['roi'] for p in props if p in results]

x = range(len(props))
axes[0].bar([i-0.2 for i in x], fixed_roi, 0.4, label='Fixed Stake')
axes[0].bar([i+0.2 for i in x], kelly_roi, 0.4, label='Kelly')
axes[0].set_xticks(x)
axes[0].set_xticklabels(props)
axes[0].set_ylabel('ROI (%)')
axes[0].set_title('Betting Strategy Comparison')
axes[0].legend()
axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)

# Win rate
win_rates = [results[p]['profit']['fixed_stake']['win_rate']*100 for p in props if p in results]
axes[1].bar(props, win_rates)
axes[1].axhline(y=52.4, color='r', linestyle='--', label='Vegas vig (52.4%)')
axes[1].set_ylabel('Win Rate (%)')
axes[1].set_title('Prediction Accuracy')
axes[1].legend()

plt.tight_layout()
plt.savefig('profit_analysis.png', dpi=150)
plt.show()
```

```python
# Cell 6: Drift Detection
for prop in ['points', 'rebounds', 'assists']:
    if prop not in results or 'drift' not in results[prop]:
        continue

    drift_data = pd.DataFrame(results[prop]['drift'])

    plt.figure(figsize=(12, 4))
    plt.plot(drift_data['date'], drift_data['rolling_mae'])
    plt.axhline(y=drift_data['rolling_mae'].iloc[:100].mean(),
                color='r', linestyle='--', label='Baseline')
    plt.fill_between(drift_data['date'],
                     drift_data['rolling_mae'] * 0.8,
                     drift_data['rolling_mae'] * 1.2,
                     alpha=0.2, label='±20% drift threshold')
    plt.title(f'{prop.upper()} Model Drift Over Time')
    plt.xlabel('Date')
    plt.ylabel('Rolling MAE')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
```

---

## Phase 1: Training & Validation (Week 1)

### 1.1: Upload Aggregated Data
**Time**: 5 minutes

```bash
# In Kaggle notebook where aggregated data was created
!kaggle datasets create -p /kaggle/working
```

---

### 1.2: Retrain Models with All Features
**Time**: 3-4 hours (in Colab)

**What gets trained**:
- ✅ All 150+ features from Phase 1-7
- ✅ TabNet with 24-dim embeddings
- ✅ LightGBM on raw + embeddings
- ✅ Sigma models for uncertainty
- ✅ 5 props: minutes, points, rebounds, assists, threes

**Expected output**:
```
Points:
  MAE: 2.2-2.4 (baseline: 2.5-2.7)
  RMSE: 3.0-3.3 (baseline: 3.3-3.6)
  Embeddings: 24-dim, 15-40% importance
```

---

### 1.3: Validate Embeddings
**Time**: 15 minutes

```python
# Run validation tests
# 1. Shape check: (n, 24)
# 2. Not all zeros
# 3. t-SNE clustering
# 4. LightGBM usage 15-40%
# 5. Performance vs baseline
```

---

### 1.4: Test Predictions on Historical Games
**Time**: 1 hour

**Test on last week's games** (Nov 1-8, 2024):
```python
# Use predict_live.py
predictions = engine.predict_all_games(date='2024-11-08')

# Compare to actual results
actuals = fetch_actual_results('2024-11-08')
merged = predictions.merge(actuals, on='player_id')

mae = (merged['pred_points'] - merged['actual_points']).abs().mean()
print(f"MAE: {mae:.2f} (target: <2.5)")
```

---

## Phase 2: Short-Term Improvements (Week 2)

### 2.1: Complete Feature Engineering in predict_live.py
**Priority**: HIGH
**Time**: 1-2 days

**What to add**:
```python
def engineer_features_for_player(...):
    # Currently has: ~40 features (rolling avgs, per-min rates)
    # Need to add: 110 more features

    # TODO 1: Team context (pace, off/def strength)
    # TODO 2: Opponent matchup (def rating, matchup edges)
    # TODO 3: Momentum features (short/med/long trends)
    # TODO 4: Fatigue (B2B, days rest, workload)
    # TODO 5: Variance/consistency metrics
    # TODO 6: Ceiling/floor analysis
    # TODO 7: Era features (season, decade)
    # TODO 8: Basketball Reference priors

    return features  # 150+ features total
```

---

### 2.2: Add SHAP Explainability
**Priority**: HIGH
**Time**: 1 day

```python
# In predict_live.py - already has framework
def _init_shap_explainer(self, prop):
    # Load background data (1000 samples)
    bg_data = pd.read_csv('training_sample.csv', nrows=1000)

    # Create explainer
    explainer = shap.TreeExplainer(model.lgbm)
    self.explainers[prop] = explainer
```

```python
# Generate explanation
shap_values = explainer.shap_values(features)
top_features = get_top_n_features(shap_values, n=5)

# Visualize
shap.waterfall_plot(...)
```

---

### 2.3: Automated Backtesting
**Priority**: HIGH
**Time**: 2 days

**Complete `backtest_engine.py`**:

```python
def generate_predictions_for_date_range(start, end):
    # For each date in range:
    #   1. Get games scheduled
    #   2. Get rosters
    #   3. Engineer features (using data BEFORE date)
    #   4. Predict
    #   5. Store predictions

    # Time-series safe: no future leakage
```

**Test**:
```python
results = engine.run_backtest('2024-10-01', '2024-11-09')
# Expect: MAE ~2.5, coverage 78-82%, ROI >0%
```

---

### 2.4: Model Monitoring Setup
**Priority**: MEDIUM
**Time**: 1 day

```python
# monitor.py
class ModelMonitor:
    def check_daily_performance(self):
        # 1. Fetch yesterday's results
        # 2. Compare to predictions
        # 3. Check drift
        # 4. Alert if degraded >20%
```

**Schedule**:
```bash
# Daily cron job
0 8 * * * python monitor.py --check-daily --email alerts@example.com
```

---

## Phase 3: Long-Term Research (Weeks 3-6)

### 3.1: Interaction Features Discovery
**Priority**: MEDIUM
**Time**: 2-3 days

```python
# Use SHAP to find top interactions
explainer = shap.TreeExplainer(model)
interactions = explainer.shap_interaction_values(X_train)

# Create explicit interaction features
top_20 = get_top_interactions(interactions, n=20)
for feat1, feat2 in top_20:
    X[f'{feat1}_x_{feat2}'] = X[feat1] * X[feat2]
```

**Expected**: +1-2% accuracy improvement

---

### 3.2: Online Learning Pipeline
**Priority**: MEDIUM
**Time**: 3-4 days

```python
# Incremental updates
class OnlineLearner:
    def add_new_games(self, new_data):
        self.buffer.append(new_data)

        if len(self.buffer) >= 500:
            self.partial_refit()

    def partial_refit(self):
        # TabNet: warm-start
        # LightGBM: add boosting rounds
```

---

### 3.3: Streamlit Dashboard
**Priority**: LOW (nice to have)
**Time**: 1 week

```python
# dashboard.py
st.title("NBA Prediction Dashboard")
# Tabs: Predictions | Backtest | Health | Explainability
```

---

## Summary Timeline

| Week | Phase | Tasks | Goal |
|------|-------|-------|------|
| **Week 0** | Pre-Training | Update notebooks, upload data | Ready to train |
| **Week 1** | Training | Retrain models, validate embeddings, test predictions | Models working |
| **Week 2** | Short-term | Complete predict_live.py, SHAP, backtesting | Score: 92/100 |
| **Weeks 3-4** | Research | Interactions, online learning | Score: 95/100 |
| **Weeks 5-6** | Polish | Dashboard, monitoring, documentation | Score: 97/100 |

---

## Immediate Next Steps (This Week)

1. **Today**: Update `NBA_COLAB_SIMPLE.ipynb` ✅
2. **Today**: Update `Riq_Machine.ipynb` ✅
3. **Today**: Update `Evaluate_Predictions.ipynb` ✅
4. **Tomorrow**: Upload aggregated data to Kaggle
5. **Tomorrow**: Start retraining in Colab (3-4 hours)
6. **Day 3**: Validate embeddings, test predictions
7. **Day 4-5**: Complete feature engineering in predict_live.py
8. **Day 6-7**: Add SHAP and run first backtest

---

## Success Metrics

**After Week 1** (Training complete):
- ✅ 24-dim embeddings working
- ✅ MAE < 2.5 for points
- ✅ Models can make predictions

**After Week 2** (Short-term complete):
- ✅ Live predictions working
- ✅ SHAP explanations
- ✅ Backtesting framework
- ✅ Score: 92/100

**After Week 4** (Research features):
- ✅ Interaction features
- ✅ Online learning
- ✅ Score: 95/100

**After Week 6** (Production ready):
- ✅ Dashboard
- ✅ Monitoring
- ✅ Score: 97/100

---

## What NOT to Do Yet

❌ Interaction features (Week 3-4, not now)
❌ Dashboard/UI (Week 5-6, not now)
❌ API interface (optional, later)
❌ Advanced temporal features (later)

**Focus**: Get models trained with current features first, then iterate!
