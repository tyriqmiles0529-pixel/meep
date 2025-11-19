# Research-Grade NBA Prediction System - Implementation Roadmap

**Goal**: Surpass commercial-grade models, achieve 95+ rating

**Current Score**: 88/100 (A-)
**Target Score**: 95/100 (A+)
**Timeline**: 4-6 weeks

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA INGESTION LAYER                          │
├─────────────────────────────────────────────────────────────────┤
│  • Kaggle (Historical 2002-2026)                                │
│  • nba_api (Live games, rosters, stats)                         │
│  • Basketball Reference (Statistical priors)                     │
│  • The Odds API (Betting lines - optional)                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  FEATURE ENGINEERING PIPELINE                    │
├─────────────────────────────────────────────────────────────────┤
│  Phase 1-7 Features (150+):                                      │
│  • Shot volume + efficiency (TS%, eFG%, FT%)                    │
│  • Rolling averages (L3, L5, L10)                               │
│  • Per-minute rates                                              │
│  • Team/opponent context                                         │
│  • Momentum + acceleration                                       │
│  • Fatigue + variance analysis                                   │
│  • Basketball Reference priors                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    MODELING LAYER (HYBRID)                       │
├─────────────────────────────────────────────────────────────────┤
│  TabNet (Deep Learning)                                          │
│    ↓                                                             │
│  24-dim Embeddings                                               │
│    ↓                                                             │
│  LightGBM (Raw + Embeddings) → Predictions + Sigma              │
│                                                                  │
│  Ensemble Meta-Learner:                                          │
│    • Ridge Regression                                            │
│    • Dynamic Elo                                                 │
│    • Four Factors                                                │
│    • LightGBM                                                    │
│    → Logistic Meta-Learner → Final Predictions                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   PREDICTION & ANALYSIS LAYER                    │
├─────────────────────────────────────────────────────────────────┤
│  • Live Predictions (predict_live.py)                            │
│  • SHAP Explainability                                           │
│  • Uncertainty Quantification (80%, 95% intervals)               │
│  • Automated Backtesting                                         │
│  • Model Drift Detection                                         │
│  • Calibration Monitoring                                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     INTERFACE & MONITORING                       │
├─────────────────────────────────────────────────────────────────┤
│  • Research Dashboard (Streamlit)                                │
│  • Performance Tracking                                          │
│  • Alert System (drift, performance degradation)                │
│  • Export (CSV, JSON, API)                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Critical Infrastructure (Weeks 1-2) → Score: 92/100

### ✅ COMPLETED
- [x] Elite feature engineering (Phase 1-7)
- [x] Hybrid TabNet + LightGBM architecture
- [x] Ensemble meta-learner system
- [x] Leakage-safe data pipeline
- [x] Uncertainty quantification (sigma models)

### 🚧 IN PROGRESS

#### 1.1 Complete Live Prediction System
**File**: `predict_live.py` (CREATED - needs completion)

**TODOs**:
- [ ] Implement full feature engineering pipeline (match train_auto.py features exactly)
- [ ] Add team context fetching (pace, offensive/defensive ratings)
- [ ] Add opponent matchup features
- [ ] Integrate Basketball Reference priors lookup
- [ ] Add proper error handling and retries for nba_api
- [ ] Test on historical games first (validate against known results)

**Dependencies**:
```bash
pip install nba-api shap
```

**Integration Points**:
- Uses models from `./models/`
- Fetches live data via `nba_api`
- Exports to CSV/JSON
- **Notebook Integration**: Modify `Riq_Machine.ipynb` to call `predict_live.py`

---

#### 1.2 Add SHAP Explainability
**Status**: Framework added to `predict_live.py`, needs completion

**TODOs**:
- [ ] Create background dataset for SHAP (sample 1000 rows from training data)
- [ ] Implement SHAP waterfall plots for top predictions
- [ ] Add feature importance summary
- [ ] Create explanation export (JSON format for dashboard)
- [ ] Add to notebooks: SHAP plots in `Riq_Machine.ipynb`

**Example Usage**:
```python
# In predict_live.py
explainer = shap.TreeExplainer(model.lgbm)
shap_values = explainer.shap_values(features)

# Top 5 reasons for prediction
shap.waterfall_plot(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=features.iloc[0],
    feature_names=features.columns
))
```

---

#### 1.3 Automated Backtesting Engine
**File**: `backtest_engine.py` (CREATED - needs completion)

**TODOs**:
- [ ] Implement `generate_predictions_for_date_range()` (time-series safe)
- [ ] Add calibration plots (matplotlib/seaborn)
- [ ] Create profit tracking dashboard
- [ ] Add per-team, per-prop, per-season breakdowns
- [ ] Integrate with `Evaluate_Predictions.ipynb`
- [ ] Schedule daily backtest runs (cron/Task Scheduler)

**Metrics to Track**:
```python
{
    'accuracy': {
        'mae': 2.35,
        'rmse': 3.12,
        'mape': 15.2,
        'r2': 0.78,
        'acc_within_2': 0.65
    },
    'calibration': {
        'coverage_80': 0.82,  # 80% intervals contain 82% (good!)
        'coverage_95': 0.94,
        'brier_score': 0.15
    },
    'profit': {
        'roi_fixed': +12.5,  # 12.5% ROI
        'roi_kelly': +18.3,
        'win_rate': 0.56,
        'sharpe_ratio': 1.45
    },
    'drift': {
        'drift_score': 0.05,  # 5% drift (acceptable)
        'alert': False
    }
}
```

**Notebook Integration**:
- `Evaluate_Predictions.ipynb` → Call `backtest_engine.run_backtest()`
- Add calibration plots, profit curves, drift charts

---

## Phase 2: Research-Grade Features (Weeks 3-4) → Score: 95/100

### 2.1 Interaction Features Discovery
**File**: `interaction_features.py` (NEW)

**Approach**:
```python
"""
Automatic interaction feature discovery using SHAP.

Process:
1. Train baseline model
2. Calculate SHAP interaction values
3. Identify top 20 interactions
4. Create explicit interaction features
5. Retrain and compare performance
"""

import shap

# Get SHAP interaction values
explainer = shap.TreeExplainer(model)
shap_interaction = explainer.shap_interaction_values(X_train)

# Find top interactions
interactions = []
for i in range(len(feature_names)):
    for j in range(i+1, len(feature_names)):
        interaction_strength = np.abs(shap_interaction[:, i, j]).mean()
        interactions.append((feature_names[i], feature_names[j], interaction_strength))

# Sort by strength
top_interactions = sorted(interactions, key=lambda x: x[2], reverse=True)[:20]

# Create interaction features
for feat1, feat2, _ in top_interactions:
    X_train[f'{feat1}_x_{feat2}'] = X_train[feat1] * X_train[feat2]
    X_train[f'{feat1}_div_{feat2}'] = X_train[feat1] / (X_train[feat2] + 1e-6)
```

**Expected Interactions**:
- `pts_per_min * usage_rate` (volume + efficiency)
- `opp_def_strength * player_shooting_pct` (matchup quality)
- `days_rest * minutes_L5_avg` (fatigue + workload)
- `is_home * team_pace` (home advantage in fast games)

**Expected Improvement**: +1-2% accuracy

---

### 2.2 Online Learning Pipeline
**File**: `online_learning.py` (NEW)

**Architecture**:
```python
"""
Incremental model updates without full retraining.

Approaches:
1. LightGBM: partial_fit on new data (not natively supported - use refit_tree)
2. TabNet: Warm-start training with new data
3. Rolling window: Keep last 2 seasons, drop older data

Update Schedule:
- Daily: Add yesterday's games to buffer
- Weekly: Partial refit when buffer reaches 500 games
- Monthly: Full retrain with updated Basketball Reference priors
"""

class OnlineLearner:
    def __init__(self, base_model, buffer_size=500):
        self.model = base_model
        self.buffer = []
        self.buffer_size = buffer_size

    def add_new_data(self, X_new, y_new):
        """Add new game data to buffer."""
        self.buffer.append((X_new, y_new))

        if len(self.buffer) >= self.buffer_size:
            self.partial_refit()

    def partial_refit(self):
        """Incrementally update model with buffered data."""
        X_buffer = pd.concat([x for x, y in self.buffer])
        y_buffer = np.concatenate([y for x, y in self.buffer])

        # For TabNet: warm-start
        if hasattr(self.model, 'tabnet'):
            self.model.tabnet.fit(
                X_buffer, y_buffer,
                max_epochs=10,  # Quick update
                from_unsupervised=self.model.tabnet  # Warm start
            )

        # For LightGBM: add boosting rounds
        self.model.lgbm.booster_.refit(X_buffer, y_buffer)

        self.buffer = []  # Clear buffer
```

**Benefits**:
- Always use most recent data
- No manual retraining needed
- Adapt to meta changes (rule changes, playstyle shifts)

---

### 2.3 Advanced Temporal Features
**Add to Phase 6 features** (train_auto.py)

**New Features**:
```python
# 1. Meta trends (league-wide pace/scoring changes)
df['league_pace_trend'] = df.groupby('season_end_year')['team_pace'].transform('mean')
df['player_pace_vs_league'] = df['team_recent_pace'] / df['league_pace_trend']

# 2. Clutch time performance (if play-by-play data available)
df['pts_4th_quarter_L5'] = ...  # Points in 4th quarter (last 5 games)
df['clutch_usage_rate'] = ...   # Usage in close games (<5 pts)

# 3. Playoff mode indicator
df['is_playoffs'] = (df['game_date'] >= playoffs_start_date).astype(int)
df['playoff_experience'] = df.groupby('player_id')['is_playoffs'].cumsum()

# 4. Rest advantage/disadvantage
df['rest_advantage'] = df['player_days_rest'] - df['opp_avg_days_rest']

# 5. Travel distance (if GPS data available)
df['travel_miles'] = calculate_distance(prev_city, current_city)
df['timezone_change'] = abs(prev_tz - current_tz)
```

**Expected Improvement**: +0.5-1% accuracy

---

## Phase 3: Production Infrastructure (Weeks 5-6) → Score: 97/100

### 3.1 Research Dashboard (Streamlit)
**File**: `dashboard.py` (NEW)

**Features**:
```python
import streamlit as st
import plotly.express as px

st.title("🏀 NBA Research-Grade Prediction System")

# Sidebar: Date selector
date = st.sidebar.date_input("Select Date")

# Tab 1: Live Predictions
with st.tabs(["Predictions", "Backtest", "Model Health", "Explainability"]):
    # Predictions tab
    st.header("Today's Predictions")
    predictions = engine.predict_all_games(date=str(date))

    # Display predictions with confidence intervals
    for _, pred in predictions.iterrows():
        with st.expander(f"{pred['player_name']} ({pred['team']})"):
            col1, col2, col3 = st.columns(3)
            col1.metric("Points", f"{pred['points']['prediction']:.1f}")
            col2.metric("Rebounds", f"{pred['rebounds']['prediction']:.1f}")
            col3.metric("Assists", f"{pred['assists']['prediction']:.1f}")

            # Uncertainty plot
            fig = px.bar(
                x=['80% Low', 'Prediction', '80% High'],
                y=[pred['points']['lower_80'], pred['points']['prediction'], pred['points']['upper_80']]
            )
            st.plotly_chart(fig)

    # Backtest tab
    st.header("Historical Performance")
    backtest_results = load_backtest_results()
    st.line_chart(backtest_results['mae_over_time'])

    # Model Health tab
    st.header("Model Monitoring")
    drift = detect_drift()
    st.metric("Drift Score", f"{drift['score']:.2%}", delta=f"{drift['vs_baseline']:+.1%}")

    # Explainability tab
    st.header("Why this prediction?")
    shap_plot = generate_shap_waterfall(pred['player_id'])
    st.pyplot(shap_plot)
```

**Run**:
```bash
streamlit run dashboard.py
```

---

### 3.2 Model Monitoring & Alerts
**File**: `monitor.py` (NEW)

**Alerts**:
```python
class ModelMonitor:
    def __init__(self, alert_thresholds):
        self.thresholds = alert_thresholds

    def check_performance(self, recent_mae, baseline_mae):
        """Alert if MAE degrades >20%."""
        degradation = (recent_mae - baseline_mae) / baseline_mae

        if degradation > self.thresholds['mae_degradation']:
            self.send_alert(
                level='WARNING',
                message=f'MAE degraded by {degradation*100:.1f}%'
            )

    def check_calibration(self, coverage_80, expected=0.80):
        """Alert if calibration is off by >10%."""
        error = abs(coverage_80 - expected)

        if error > self.thresholds['calibration_error']:
            self.send_alert(
                level='WARNING',
                message=f'80% intervals covering {coverage_80*100:.1f}% (expected 80%)'
            )

    def send_alert(self, level, message):
        """Send alert via email/Slack/Discord."""
        # Email
        import smtplib
        # ... send email

        # Or Slack webhook
        import requests
        requests.post(SLACK_WEBHOOK_URL, json={'text': message})
```

**Scheduled Monitoring**:
```bash
# Linux cron
0 8 * * * python monitor.py --check-daily

# Windows Task Scheduler
schtasks /create /tn "NBA Model Monitor" /tr "python monitor.py" /sc daily /st 08:00
```

---

### 3.3 API Interface (Optional)
**File**: `api.py` (NEW)

**FastAPI REST API**:
```python
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/predict")
def predict(player_id: int, date: str):
    """Get predictions for a player on a specific date."""
    engine = LivePredictionEngine()
    pred = engine.predict_player_props(player_id, date)
    return pred

@app.get("/backtest")
def backtest(start_date: str, end_date: str):
    """Run backtest for date range."""
    engine = BacktestEngine()
    results = engine.run_backtest(start_date, end_date)
    return results

# Run: uvicorn api:app --reload
```

---

## Notebook Integration Plan

### 1. NBA_COLAB_SIMPLE.ipynb (Training)
**Purpose**: Model training in Colab

**Updates Needed**:
```python
# Cell 1: Install dependencies
!pip install pytorch-tabnet shap nba-api

# Cell 2: Download aggregated data
!kaggle datasets download -d tyriqmiles/aggregated-nba-data
!unzip aggregated-nba-data.zip

# Cell 3: Train models
!python train_auto.py --dataset ./aggregated_nba_data.csv.gzip \
    --use-neural --neural-epochs 30 --verbose

# Cell 4: Upload trained models
from google.colab import files
!zip -r models.zip models/
files.download('models.zip')

# Cell 5: Test predictions on sample
!python predict_live.py --date 2024-11-08 --explain
```

---

### 2. Riq_Machine.ipynb (Predictions)
**Purpose**: Generate live predictions

**Updates Needed**:
```python
# Cell 1: Load models
from predict_live import LivePredictionEngine
engine = LivePredictionEngine(models_dir='./models')

# Cell 2: Get today's games
games = engine.get_todays_games()
display(games)

# Cell 3: Predict all games with SHAP
predictions = engine.predict_all_games(explain=True)

# Cell 4: Display predictions with explanations
for pred in predictions:
    print(f"\n{pred['player_name']} ({pred['team']})")
    print(f"  Points: {pred['points']['prediction']:.1f} "
          f"± {pred['points']['uncertainty']:.1f}")

    # Show SHAP explanation
    if 'explanation' in pred['points']:
        print("  Top reasons:")
        for reason in pred['points']['explanation'][:3]:
            print(f"    - {reason['feature']}: {reason['shap_value']:+.2f}")

# Cell 5: Export to CSV
predictions.to_csv('predictions.csv')
from google.colab import files
files.download('predictions.csv')
```

---

### 3. Evaluate_Predictions.ipynb (Settle & Calibrate)
**Purpose**: Backtest and calibrate

**Updates Needed**:
```python
# Cell 1: Load backtest engine
from backtest_engine import BacktestEngine
engine = BacktestEngine()

# Cell 2: Run backtest
results = engine.run_backtest(
    start_date='2024-10-01',
    end_date='2024-11-09'
)

# Cell 3: Calibration plots
import matplotlib.pyplot as plt

for prop in ['points', 'rebounds', 'assists']:
    plt.figure(figsize=(10, 6))

    # Plot expected vs actual coverage
    confidences = [0.68, 0.80, 0.90, 0.95]
    actual_coverage = [results[prop]['calibration'][f'coverage_{int(c*100)}']
                      for c in confidences]

    plt.plot(confidences, confidences, 'k--', label='Perfect calibration')
    plt.plot(confidences, actual_coverage, 'o-', label=f'{prop} actual')
    plt.xlabel('Expected Coverage')
    plt.ylabel('Actual Coverage')
    plt.title(f'{prop.upper()} Calibration Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

# Cell 4: Profit simulation
profit = results['points']['profit']
print(f"ROI (Fixed Stake): {profit['fixed_stake']['roi']:+.1f}%")
print(f"ROI (Kelly): {profit['kelly']['roi']:+.1f}%")

# Cell 5: Drift detection
drift_df = pd.DataFrame(results['points']['drift'])
plt.figure(figsize=(12, 6))
plt.plot(drift_df['date'], drift_df['rolling_mae'])
plt.axhline(y=drift_df['rolling_mae'].iloc[:100].mean(),
            color='r', linestyle='--', label='Baseline')
plt.fill_between(drift_df['date'],
                drift_df['rolling_mae'] * 0.8,
                drift_df['rolling_mae'] * 1.2,
                alpha=0.2, label='±20% drift threshold')
plt.legend()
plt.title('Model Drift Over Time')
plt.show()
```

---

## Final Score Projection

| Component | Current | After Phase 1 | After Phase 2 | After Phase 3 |
|-----------|---------|---------------|---------------|---------------|
| **Architecture** | 92 | 92 | 94 (+interactions) | 95 (+online learning) |
| **Feature Engineering** | 95 | 95 | 97 (+interactions) | 97 |
| **Prediction Analysis** | 78 | 88 (+live system) | 92 (+SHAP) | 95 (+dashboard) |
| **Data Pipeline** | 91 | 91 | 93 (+online) | 95 (+monitoring) |
| **Robustness** | 85 | 90 (+backtest) | 93 (+drift) | 95 (+alerts) |
| **Comprehensiveness** | 96 | 96 | 98 (+temporal) | 98 |
| **Scalability** | 82 | 85 (+tests) | 90 (+CI/CD) | 95 (+API) |
| **TOTAL** | **88** | **92** | **95** | **97** |

---

## Implementation Priority (Next 2 Weeks)

### Week 1: Get to 92/100
1. **Day 1-2**: Complete feature engineering in `predict_live.py`
2. **Day 3-4**: Add SHAP explainability
3. **Day 5-6**: Complete backtesting engine
4. **Day 7**: Test end-to-end on historical games

### Week 2: Get to 95/100
1. **Day 8-9**: Interaction features discovery
2. **Day 10-11**: Online learning pipeline
3. **Day 12-13**: Streamlit dashboard
4. **Day 14**: Model monitoring setup

---

## Testing Checklist

Before deploying:
- [ ] Validate features match training data exactly
- [ ] Backtest on last 30 days (RMSE < 3.5 for points)
- [ ] Calibration check: 80% intervals contain 78-82% of actuals
- [ ] SHAP explanations make sense (no unexpected features)
- [ ] Drift detection triggers on synthetic degradation
- [ ] Dashboard loads in <3 seconds
- [ ] API responds in <500ms
- [ ] Notebooks run end-to-end without errors

---

## Success Metrics

**Model Performance**:
- MAE (Points): < 2.3 (currently ~2.5)
- MAPE (Points): < 14% (currently ~15%)
- R² (Points): > 0.80 (currently ~0.78)
- Calibration Error: < 5% (80% intervals within 75-85%)

**System Performance**:
- Prediction latency: < 500ms per player
- Daily backtest runtime: < 10 minutes
- Dashboard load time: < 3 seconds
- Uptime: > 99.5%

**Research Quality**:
- Published on arXiv or similar (optional)
- Open-source documentation
- Reproducible results
- Peer review by community

---

## Resources & Dependencies

**Python Packages**:
```bash
pip install pandas numpy scikit-learn lightgbm
pip install pytorch-tabnet torch
pip install nba-api shap
pip install streamlit plotly
pip install fastapi uvicorn
pip install pytest black flake8
```

**Cloud Resources** (if needed):
- Google Colab Pro ($10/month) for training
- AWS EC2 t3.medium for API ($30/month)
- MongoDB Atlas Free tier for results storage

**Time Commitment**:
- Week 1-2: 20-30 hours
- Week 3-4: 15-20 hours
- Week 5-6: 10-15 hours
- **Total: 45-65 hours over 6 weeks**

---

## Questions to Address

1. **Do you want to focus on player props only, or also game-level predictions (moneyline/spread)?**
   - Current system handles both, but notebooks focus on players

2. **Should we integrate real betting lines from The Odds API?**
   - Cost: ~$50/month for live lines
   - Benefit: Real profit tracking instead of simulation

3. **Open-source or private?**
   - Open-source: Get community feedback, build reputation
   - Private: Keep edge for personal use

4. **Target deployment platform?**
   - Local (laptop/desktop)
   - Cloud (AWS/GCP)
   - Hybrid (train in Colab, predict locally)

---

## Next Steps

1. **Review this roadmap** - Any changes needed?
2. **Choose Phase 1 start date** - When to begin?
3. **Set up development environment** - All dependencies installed?
4. **Create GitHub repo** (optional) - Version control ready?

Let me know when you're ready to start, and we'll dive into completing `predict_live.py` with full feature engineering!
