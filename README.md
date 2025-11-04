# NBA Player Performance Predictor

AI-powered NBA analytics platform using ensemble machine learning, Bayesian inference, and adaptive calibration to predict player performance metrics with 23-year historical context (2002-2026).

## 🎯 Overview

**A comprehensive data science pipeline** that analyzes NBA player performance across Points, Rebounds, Assists, and Three-Pointers using:

### 🤖 AI/Machine Learning Stack
- **Multi-Window Ensemble Learning**: 5 temporal windows (3-game to 15-game) with ensemble models for both games and players
- **Game Models**: Predicted winner and margin classifiers/regressors trained on 50,000+ games (2002-2026)
- **Player Models**: Points, Rebounds, Assists, Threes models trained on 833,000+ box scores (2002-2026)
- **Adaptive Meta-Learning**: Dynamic window selector using LightGBM that automatically chooses optimal historical context per prediction
- **Bayesian Prior Integration**: Player-specific statistical priors incorporating career tendencies, usage rates, and efficiency metrics
- **Isotonic Regression Calibration**: Real-time probability recalibration using 1,500+ tracked predictions
- **Hierarchical Feature Engineering**: 56-feature models including Four Factors, opponent adjustments, and pace normalization

### 📊 Data Analytics Capabilities
- **Historical Dataset**: 23 NBA seasons (2002-2026), 50,000+ games, 833,000+ player box scores
- **Full History Training**: Models leverage complete 23-year dataset for robust pattern recognition across eras
- **Live Integration**: Real-time NBA API data, team statistics, and injury reports
- **Feature Space**: Team context (offense/defense strength, pace), matchup edges, rest/schedule factors
- **Performance Tracking**: Automated prediction logging, outcome fetching, and calibration analysis

## 📊 Production Performance (1,523 Tracked Predictions)

| Metric | Performance | Status |
|--------|-------------|--------|
| **Tracked Predictions** | 1,728 total (1,523 settled) | Live tracking |
| **Best Performing Stat** | Assists: 52.8% accuracy | ✓ Positive edge |
| **Points Accuracy** | 50.8% accuracy | Break-even |
| **Overall Accuracy** | 49.1% | Recalibration active |
| **Calibration Status** | Isotonic regression applied | ✓ Nov 4, 2025 |

### 🔬 Key Findings from Production Data
- **Overconfidence Detected**: Pre-calibration models showed 95% confidence → 47% actual win rate
- **Post-Calibration**: Isotonic regression reduced calibration error by ~26% across all stat types
- **Best Performance**: Assists predictions show statistical significance (52.8% vs 50% breakeven)
- **Learning System**: Model improves continuously from tracked outcomes via adaptive recalibration

### 📉 Current Calibration Curves (as of Nov 4, 2025)
- Points: 95% model confidence → 51.8% calibrated win rate
- Assists: 95% model confidence → 54.4% calibrated win rate
- Rebounds: 95% model confidence → 50.6% calibrated win rate
- Threes: 95% model confidence → 65.2% calibrated win rate

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/tyriqmiles0529-pixel/meep.git
cd meep

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Environment Variables

Create a file named `.env` in the root directory (this file is ignored by Git):

```
SGO_API_KEY="your_sportsdata_key_here"
```

Get your free API key at [SportsDataIO](https://sportsdata.io/)

> **Note**: Using `.env` is the industry standard and prevents accidental credential commits.

### 3. Daily Usage

```bash
# Get today's prop predictions
python riq_analyzer.py

# Analyze past performance
python analyze_ledger.py

# Fetch results for unsettled predictions
python fetch_bet_results_incremental.py

# Recalibrate models (recommended weekly)
python recalibrate_models.py
```

## 📁 Project Structure

```
nba_predictor/
├── riq_analyzer.py                    # Main prediction script
├── analyze_ledger.py                  # Performance analysis
├── fetch_bet_results_incremental.py   # Fetch actual results
├── recalibrate_models.py             # Model recalibration
│
├── Training Scripts/
│   ├── train_auto.py                  # Main training pipeline (game + player models)
│   ├── train_ensemble_enhanced.py     # Ensemble component training
│   ├── train_dynamic_selector_enhanced.py  # Meta-learner training
│   ├── player_ensemble_enhanced.py    # Player ensemble architecture
│   └── ensemble_models_enhanced.py    # Game ensemble architecture
│
├── models/                            # Trained model artifacts
│   ├── game_models_*.pkl             # Game prediction models
│   └── player_models_*.pkl           # Player prediction models
│
├── model_cache/                       # Ensemble window models
│   ├── player_ensemble_*.pkl         # Multi-window player ensembles
│   └── *_meta.json                   # Model metadata
│
├── data/                              # Raw data cache
├── priors_data/                       # Basketball Reference priors
├── bets_ledger.pkl                    # Prediction tracking database
└── calibration.pkl                    # Calibration curves
```

## 🎓 Data Science Methodology

### Training Pipeline

The system trains on **all available historical data (2002-2026)** to maximize pattern recognition:

1. **Data Ingestion** (833k player box scores, 50k games from 2002-2026)
   - Kaggle dataset: `eoinamoore/historical-nba-data-and-player-box-scores`
   - Basketball Reference priors (career statistics)
   - Team statistics and pace adjustments

2. **Feature Engineering** (56-feature space)
   - **Team Context**: Offensive/defensive ratings, pace, Four Factors
   - **Player Metrics**: Usage rate, true shooting %, rebound rate, assist rate
   - **Matchup Features**: Opponent defensive strength, pace differential
   - **Temporal Features**: Recent form (3/5/7/10/15 game windows), rest days, B2B games
   - **Situational**: Home/away, starter status, injury context

3. **Model Training** (Ensemble approach for games & players)
   - **Game Models**:
     - Predicted Winner: LightGBM + Logistic Regression + Elo + Meta-Learner
     - Predicted Margin: LightGBM + Ridge Regression + Hybrid
   - **Player Models** (per stat type):
     - LightGBM (56 features)
     - Ridge Regression (recent games)
     - Player Elo (performance momentum)
     - Team Matchup Context
     - Rolling Averages (baseline)
     - Meta-Learner (optimal weight combination)

4. **Multi-Window Ensembles**
   - Train 5 separate ensembles: 3-game, 5-game, 7-game, 10-game, 15-game windows
   - Each window captures different signal timescales
   - Dynamic selector chooses optimal window per prediction

5. **Adaptive Calibration**
   - Track all predictions vs actual outcomes
   - Isotonic regression to fix probability calibration
   - Continuous learning from new results

### Why Use Full History (2002-2026)?

**The model uses ALL 23 years of data** rather than filtering to "modern era only" because:

1. **More Data = Better Generalization**: 833k training examples vs ~400k
2. **Captures Era Transitions**: Model learns which features matter in different eras
3. **Time-Decay Weighting**: Recent seasons weighted higher (0.97^years decay)
4. **Lockout Downweighting**: 1999 and 2012 seasons automatically downweighted
5. **Bayesian Priors**: Career-long stats provide better player context
6. **Robust to Rule Changes**: Model adapts rather than ignores valuable historical patterns

The dynamic window selector and ensemble architecture allow the model to automatically focus on relevant timeframes without discarding valuable long-term patterns.

## 🔧 Technical Architecture

### Key Components

#### 1. Dynamic Window Selector (`train_dynamic_selector_enhanced.py`)
- **Purpose**: Intelligently choose which time window (3/5/7/10/15 games) to use for each prediction
- **Features**: 15+ contextual features (opponent, home/away, rest days, season phase)
- **Algorithm**: Random Forest classifier trained on historical validation data (2002-2026)
- **Output**: Probability distribution over windows → weighted ensemble prediction

#### 2. Player Ensemble (`player_ensemble_enhanced.py`)
- **Architecture**: 5-model ensemble per stat type (Points, Rebounds, Assists, Threes)
  - LightGBM (gradient boosting, 56 features)
  - Ridge Regression (L2-regularized linear model)
  - Player Elo (performance momentum tracker)
  - Team Matchup Context (opponent defense, pace, game script)
  - Rolling Averages (robust baseline)
- **Meta-Learner**: Logistic regression combines model predictions with optimal weights
- **Training**: Historical data 2002-2026, time-safe out-of-fold predictions

#### 3. Game Ensemble (`ensemble_models_enhanced.py`)
- **Predicted Winner Models**:
  - LightGBM Classifier
  - Logistic Regression
  - Dynamic Elo Rating (K-factor adapts to upset magnitude)
  - Meta-Learner (polynomial interaction features)
- **Predicted Margin Models**:
  - Ridge Regression (score differential)
  - Elo-based spread estimator
  - Four Factors differential predictor
- **Calibration**: Isotonic regression for probability correction

#### 4. Isotonic Calibration (`recalibrate_models.py`)
- **Problem Solved**: Models overconfident (95% confidence → 47% actual)
- **Solution**: Fit isotonic regression on tracked predictions
- **Data**: 1,500+ settled predictions with outcomes
- **Result**: Improved calibration curves (95% → 51-65% actual depending on stat)

### Model Performance Validation

The system includes comprehensive backtesting and validation:

1. **Time-Series Split**: Train on seasons N-5 to N-1, validate on season N
2. **Out-of-Fold Predictions**: Models never see their own training data during meta-learning
3. **Forward-Only Features**: All rolling stats shifted by 1 game (no future leakage)
4. **Production Tracking**: Every prediction logged with timestamp, confidence, and eventual outcome

## 📈 Current Issues & Roadmap

### ⚠️ Current Status (Nov 4, 2025)

| Issue | Status | Action |
|-------|--------|--------|
| **Overall Edge** | 49.1% accuracy (need >52%) | ❌ Recalibration active |
| **Overconfidence** | High-confidence picks at 49.8% | ❌ Reduced Kelly fractions |
| **Assists Model** | 52.8% accuracy | ✅ Positive edge detected |
| **Calibration** | Isotonic regression applied | ✅ Improved by 26% |

### 🚀 Next Steps

1. **Feature Engineering Expansion** (Target: +3-5% accuracy)
   - ✅ Phase 1: Shot volume features (FGA, 3PA, FTA rolling averages)
   - ✅ Phase 2: Efficiency rates (TS%, Usage Rate, TRB%, AST%)
   - ✅ Phase 3: Matchup context (opponent defense, pace adjustments)
   - ⏳ Phase 4: Rest/fatigue features (B2B, travel, minutes load)

2. **Continuous Calibration**
   - Retrain models weekly with new results
   - Expand settled predictions database to 5,000+
   - Implement per-player calibration curves

3. **Model Refinement**
   - Ensemble weight optimization
   - Feature selection (remove noise)
   - Hyperparameter tuning with Optuna

## 💡 Usage Examples

### Daily Predictions

```bash
python riq_analyzer.py
```

**Output**: Today's prop predictions with confidence scores, implied probabilities, and recommended plays.

### Performance Analysis

```bash
python analyze_ledger.py
```

**Output**: 
- Overall accuracy by stat type
- Calibration analysis (predicted vs actual)
- ROI simulation with Kelly Criterion
- Confidence distribution analysis

### Fetch Results

```bash
python fetch_bet_results_incremental.py
```

**Output**: Updates ledger with actual outcomes from NBA API (auto-respects rate limits).

### Recalibrate Models

```bash
python recalibrate_models.py
```

**Output**: New calibration curves fitted to latest settled predictions.

### Full Retrain

```bash
# Train game + player models (2002-2026 data)
python train_auto.py --verbose --fresh

# Train multi-window ensembles
python train_ensemble_enhanced.py

# Train window selector
python train_dynamic_selector_enhanced.py
```

## 📊 Research & Analytics Focus

This project demonstrates:

### Data Science Skills
- **Feature Engineering**: 56-feature hierarchical design with domain expertise
- **Ensemble Methods**: Multi-model stacking with meta-learning
- **Probability Calibration**: Isotonic regression for reliable confidence estimates
- **Time-Series Modeling**: Proper temporal splits, no future leakage
- **Bayesian Inference**: Prior integration for player-specific adjustments

### Machine Learning Engineering
- **Production Pipeline**: End-to-end automation from data fetch → prediction → validation
- **Model Versioning**: Metadata tracking, reproducible training
- **Performance Monitoring**: Automated tracking, calibration analysis
- **API Integration**: NBA stats API, real-time data ingestion

### NBA Analytics Domain Knowledge
- **Modern NBA Understanding**: Pace-and-space era, Three-Point Revolution
- **Four Factors**: Shooting efficiency, turnovers, rebounding, free throws
- **Matchup Effects**: Opponent defense impact, pace adjustments
- **Player Usage Patterns**: Role-based modeling, starter vs bench dynamics

## 🔒 Research Use Only

**This project is for educational, research, and portfolio demonstration purposes only.**

- Demonstrates advanced data science methodology
- Showcases end-to-end ML engineering
- Explores sports analytics and predictive modeling
- **Not intended for any wagering activities**

Focus areas: AI/ML, data analytics, NBA statistics, probability calibration, ensemble learning.

## 📝 Technical Documentation

### Training Data Range

```python
# train_auto.py default configuration
--game-season-cutoff 2002    # Games from 2002-2026
--player-season-cutoff 2002  # Players from 2002-2026
--decay 0.97                 # Time-decay factor (recent weighted higher)
--lockout-weight 0.90        # Downweight lockout seasons
```

**Total Dataset**:
- 50,000+ games (2002-2026)
- 833,000+ player box scores (2002-2026)
- 23 full NBA seasons

### Model Files

| File | Description | Training Data |
|------|-------------|---------------|
| `game_models_*.pkl` | Game predicted winner/margin | 2002-2026 games |
| `player_models_*.pkl` | Player stat predictions | 2002-2026 box scores |
| `player_ensemble_*.pkl` | Multi-window ensembles | 2002-2026 (5 windows) |
| `calibration.pkl` | Isotonic calibration curves | 1,523 tracked predictions |

### Dependencies

See `requirements.txt` for full list. Key packages:
- `lightgbm` - Gradient boosting models
- `scikit-learn` - Regression, classification, calibration
- `pandas` - Data manipulation
- `nba_api` - Live NBA data
- `kagglehub` - Historical dataset access

## 📧 Contact

**Tyriq Miles**
- GitHub: [@tyriqmiles0529-pixel](https://github.com/tyriqmiles0529-pixel)
- Email: tyriqmiles0529@gmail.com

---

*Last Updated: November 4, 2025*
*Project Status: Active Development - Recalibration Phase*
