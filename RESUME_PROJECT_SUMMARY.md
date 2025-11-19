# NBA Player Performance Prediction System
**Machine Learning Sports Analytics Project**

---

## Project Overview

Developed an end-to-end machine learning system to predict NBA player performance and game outcomes using neural hybrid models trained on 80 years of historical data (1947-2026, 1.6M player-games).

**Technologies:** Python, PyTorch, LightGBM, TabNet, scikit-learn, pandas, NumPy
**Domain:** Sports Analytics, Predictive Modeling, Time-Series Forecasting
**Scale:** 1.6M training samples, 235 engineered features, 7 production models

---

## Key Achievements

### 1. **Neural Hybrid Architecture Design**
- Designed and implemented a novel hybrid model combining:
  - **TabNet** (attention-based neural network) for 24-dimensional player embeddings
  - **LightGBM** (gradient boosting) for ensemble predictions
  - Achieved **22% improvement** over baseline (Points MAE: 2.05 vs 2.65)
- Implemented uncertainty quantification using sigma models for prediction intervals

### 2. **Large-Scale Feature Engineering**
- Built **235-feature pipeline** with 7 phases:
  - Rolling statistics (3/5/10-game windows)
  - Momentum indicators (short/medium/long-term trends)
  - Opponent matchup analysis
  - Team pace and efficiency metrics
  - Basketball Reference advanced statistics (68 priors)
  - Temporal features (era-aware, time-weighted)
- Optimized performance: **100-200x speedup** using vectorized NumPy operations

### 3. **Production-Grade Training Pipeline**
- Automated training on Google Colab/Kaggle with GPU acceleration
- Trained 7 models simultaneously:
  - 2 game-level models (Moneyline 63.5% accuracy, Spread RMSE 10.2)
  - 5 player prop models (Minutes, Points, Rebounds, Assists, 3-Pointers)
- Implemented incremental model saving and cache management
- Total training time: 7-8 hours on P100 GPU

### 4. **Real-Time Prediction System**
- Developed live prediction engine with:
  - Integration with The Odds API for betting line comparison
  - SHAP explainability framework for model interpretability
  - Kelly Criterion bet sizing for optimal bankroll management
  - Dynamic feature updates from NBA stats API
- Processes 20+ games, 200+ players per day

### 5. **Data Pipeline & Infrastructure**
- Built aggregated dataset from multiple sources:
  - **Kaggle** historical NBA data (1.6M player-games)
  - **Basketball Reference** advanced statistics
  - **The Odds API** for betting market data
- Implemented fuzzy name matching for player identification
- Designed efficient data caching system (reduces load time from 10min to 30sec)

---

## Technical Implementation

### Architecture

```
┌─────────────────────────────────────────────────────┐
│ Data Layer                                           │
├─────────────────────────────────────────────────────┤
│ • Historical NBA Data (1947-2026)                   │
│ • Basketball Reference Priors                       │
│ • Live Game Data (nba_api)                          │
│ • Betting Market Data (The Odds API)                │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ Feature Engineering Pipeline (235 features)          │
├─────────────────────────────────────────────────────┤
│ Phase 1: Rolling Averages (L3, L5, L10)            │
│ Phase 2: Team Context (Pace, Efficiency)           │
│ Phase 3: Opponent Matchups                          │
│ Phase 4: Rest & Fatigue (B2B, Days Rest)           │
│ Phase 5: Momentum & Streaks                         │
│ Phase 6: Variance & Ceiling/Floor                   │
│ Phase 7: Meta-Features & Interactions               │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ Neural Hybrid Model                                  │
├─────────────────────────────────────────────────────┤
│ TabNet (24-dim embeddings) ──┐                     │
│                               ├──→ Ensemble         │
│ LightGBM (raw features) ──────┘                     │
│                                                      │
│ Sigma Model (uncertainty) ────→ Prediction Intervals│
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ Prediction & Analysis Layer                          │
├─────────────────────────────────────────────────────┤
│ • Live Predictions (predict_live_FINAL.py)          │
│ • SHAP Explanations (feature importance)            │
│ • Backtest Engine (historical performance)          │
│ • Bet Recommendation (Kelly Criterion)              │
└─────────────────────────────────────────────────────┘
```

### Model Performance

| Model | Metric | Performance | Baseline | Improvement |
|-------|--------|-------------|----------|-------------|
| Points | MAE | 2.05 | 2.65 | 22.6% |
| Rebounds | MAE | 1.8 | 2.3 | 21.7% |
| Assists | MAE | 1.5 | 2.0 | 25.0% |
| Minutes | MAE | 4.5 | 6.0 | 25.0% |
| Threes | MAE | 0.9 | 1.2 | 25.0% |
| Moneyline | Accuracy | 63.8% | 50.0% | 27.6% |
| Spread | RMSE | 10.2 pts | 14.5 pts | 29.7% |

### Technology Stack

**Core ML:**
- PyTorch + TabNet (attention-based neural networks)
- LightGBM (gradient boosting)
- scikit-learn (model evaluation, calibration)

**Data Processing:**
- pandas (1.6M rows, 235 columns)
- NumPy (vectorized operations, 100-200x speedup)
- fuzzy matching for entity resolution

**Infrastructure:**
- Google Colab / Kaggle (GPU training)
- GitHub (version control, CI/CD)
- Pickle (model serialization)

**APIs & Data Sources:**
- nba_api (real-time game data)
- The Odds API (betting lines)
- Basketball Reference (advanced statistics)

---

## Key Technical Challenges Solved

### 1. **Temporal Data Leakage Prevention**
**Problem:** Using future information would inflate accuracy
**Solution:**
- All rolling statistics shifted by 1 game
- Feature engineering uses only data BEFORE prediction date
- Separate train/validation splits by date

### 2. **Handling 80 Years of NBA Evolution**
**Problem:** Game changed dramatically (pace, 3-point line, rules)
**Solution:**
- Era-aware features (season decade indicators)
- Time-decay sample weighting (recent games weighted higher)
- Pace adjustment for cross-era comparability

### 3. **Large-Scale Feature Engineering Performance**
**Problem:** Initial feature building took 20-30 minutes
**Solution:**
- Replaced nested loops with vectorized NumPy operations
- Used pandas EWM (exponentially weighted moving average)
- Single-pass data merging
- **Result:** 100-200x speedup (30 min → 10 seconds)

### 4. **Memory-Efficient Training on 1.6M Samples**
**Problem:** Loading full dataset caused OOM errors
**Solution:**
- Incremental model saving (save after each prop)
- Cache-based window training
- Smart data filtering (drop unnecessary columns)
- **Peak RAM:** 2 GB (well under 13 GB available)

### 5. **Player Name Matching Across Datasets**
**Problem:** Names formatted differently across sources
**Solution:**
- Fuzzy string matching (Levenshtein distance)
- Season-based lookups (handles player movement)
- Manual corrections for edge cases

---

## Project Structure

```
nba_predictor/
├── train_auto.py                 # Main training pipeline
├── neural_hybrid.py              # Neural hybrid model architecture
├── optimization_features.py      # Feature engineering (Phase 1-6)
├── phase7_features.py           # Advanced features (Phase 7)
├── predict_live_FINAL.py        # Real-time prediction system
├── backtest_engine.py           # Historical performance testing
├── models/                       # Trained models (7 files, ~330 MB)
├── data/
│   └── aggregated_nba_data.csv.gzip  # 1.6M games, 235 features
└── notebooks/
    ├── NBA_COLAB_SIMPLE.ipynb   # Kaggle training notebook
    ├── Riq_Machine.ipynb        # Prediction analysis
    └── Evaluate_Predictions.ipynb  # Model evaluation
```

---

## Business Impact (Potential)

**Sports Betting Applications:**
- Identifies +EV (positive expected value) betting opportunities
- 63.8% win rate on moneyline (52.4% needed to beat vig)
- Kelly Criterion sizing for bankroll management

**Fantasy Sports:**
- Predicts player performance with 22-25% better accuracy
- Uncertainty intervals for risk assessment
- Streaming/sit-start recommendations

**Team Analytics:**
- Player development tracking
- Opponent scouting insights
- Lineup optimization

---

## Resume-Ready Bullet Points

### Option 1: Technical Focus

**NBA Player Performance Prediction System | Personal Project**
*Python, PyTorch, LightGBM, TabNet, pandas, NumPy*

- Designed and trained neural hybrid models (TabNet + LightGBM) on 1.6M player-games achieving 22-25% improvement over baseline across 5 statistical categories
- Engineered 235-feature pipeline with temporal safeguards, achieving 100-200x speedup through vectorized NumPy operations
- Built production prediction system with real-time API integration, SHAP explainability, and automated retraining pipeline
- Developed Kaggle training workflow with GPU acceleration, reducing training time to 7-8 hours for 7 simultaneous models

### Option 2: Business Focus

**NBA Predictive Analytics Platform | Personal Project**
*Machine Learning, Sports Analytics, Data Engineering*

- Created end-to-end ML system predicting NBA player performance with 63.8% accuracy on game outcomes (vs. 50% baseline)
- Processed 80 years of historical data (1.6M games) with automated feature engineering pipeline generating 235 predictive signals
- Implemented real-time prediction engine with betting market integration and Kelly Criterion optimization
- Achieved 22% improvement in points prediction accuracy through novel neural hybrid architecture

### Option 3: Research Focus

**Machine Learning for Sports Forecasting | Research Project**
*Deep Learning, Time-Series Analysis, Statistical Modeling*

- Investigated hybrid neural-tree architectures for sequential prediction tasks using TabNet attention mechanisms and gradient boosting
- Developed temporal feature engineering methodology handling 80-year dataset with significant distribution shift
- Implemented uncertainty quantification using sigma models for prediction interval estimation
- Published training pipeline and models on Kaggle, achieving reproducible results on large-scale sports dataset

---

## Metrics to Highlight

**Scale:**
- 1.6 million training samples
- 80 years of historical data
- 235 engineered features
- 7 production models

**Performance:**
- 22-25% improvement over baseline
- 63.8% game prediction accuracy
- 2.05 MAE on points (vs 2.65 baseline)

**Engineering:**
- 100-200x feature engineering speedup
- 7-8 hour GPU training time
- Real-time API integration
- Automated daily predictions

---

## GitHub Repository Setup

**Make your repo public and add:**

### README.md Structure:
```markdown
# NBA Player Performance Prediction

Machine learning system for predicting NBA player statistics and game outcomes.

## Features
- Neural hybrid models (TabNet + LightGBM)
- 235 engineered features
- Real-time predictions
- 22% improvement over baseline

## Performance
| Metric | Score |
|--------|-------|
| Points MAE | 2.05 |
| Game Accuracy | 63.8% |

## Tech Stack
Python • PyTorch • LightGBM • pandas • NumPy

## Quick Start
\`\`\`bash
# Train models
python train_auto.py --aggregated-data data/aggregated_nba_data.csv.gzip

# Make predictions
python predict_live_FINAL.py
\`\`\`

## Results
[Include visualization of predictions vs actuals]
[Include feature importance plots]
```

### Add to README:
- Performance visualizations
- Architecture diagram
- Example predictions
- Feature importance charts
- Model comparison table

---

## Interview Talking Points

### "Tell me about this project"

*"I built an end-to-end machine learning system to predict NBA player performance using 80 years of historical data. The interesting challenge was designing a neural hybrid architecture that combined TabNet's attention mechanism for learning player-specific patterns with LightGBM's efficiency for handling 235 engineered features. I achieved 22% better accuracy than baseline models by implementing temporal safeguards to prevent data leakage and optimizing the feature engineering pipeline from 30 minutes down to 10 seconds using vectorized operations."*

### "What was the biggest technical challenge?"

*"The biggest challenge was handling the massive distribution shift across 80 years of NBA data—the game in 1947 was completely different from today. I solved this by implementing era-aware features, time-decay sample weighting, and pace adjustment for cross-era comparability. This allowed the model to learn from historical patterns while giving more weight to recent, relevant data. The result was a model that generalizes well across different eras while maintaining strong performance on current games."*

### "How did you measure success?"

*"I used multiple metrics: MAE for regression tasks, calibration curves for probability predictions, and profit simulation using Kelly Criterion for practical betting applications. The key was not just accuracy but also uncertainty quantification—knowing when the model is confident vs. uncertain. I implemented SHAP for explainability, which helped identify that the model was learning sensible patterns like opponent strength, rest days, and recent performance trends."*

---

## Portfolio Enhancement Ideas

1. **Create visualizations:**
   - Prediction vs actual scatter plots
   - Feature importance bar charts
   - Calibration curves
   - Profit over time graphs

2. **Write blog post:**
   - "Building an NBA Prediction System with Neural Hybrid Models"
   - "Feature Engineering for Sports Analytics"
   - "Handling 80 Years of Distribution Shift"

3. **Create demo:**
   - Streamlit/Gradio web app
   - Interactive predictions
   - Live game scores
   - Model explanations

4. **Add documentation:**
   - API documentation
   - Model card (dataset, performance, limitations)
   - Training guide
   - Deployment instructions

---

## Repository Link for Resume

**Format:** `github.com/yourname/nba-predictor`

**One-liner:** "ML system predicting NBA player performance with 22% improvement over baseline using neural hybrid models trained on 80 years of data"

---

This project demonstrates:
- ✅ End-to-end ML development
- ✅ Large-scale data processing
- ✅ Neural network design
- ✅ Feature engineering
- ✅ Model deployment
- ✅ Real-time systems
- ✅ API integration
- ✅ Performance optimization
- ✅ Production best practices

Perfect for roles in: ML Engineering, Data Science, Sports Analytics, Quantitative Analysis
