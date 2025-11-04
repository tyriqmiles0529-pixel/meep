# NBA Player Props Predictor

Advanced NBA player prop betting analyzer using ensemble machine learning models with dynamic window selection and enhanced calibration.

## 🎯 Overview

This system predicts NBA player prop outcomes (Points, Rebounds, Assists, Threes) using:
- **Ensemble models** with multiple time windows (3, 5, 7, 10, 15 games)
- **Enhanced dynamic selector** that picks the best window per prediction
- **Bayesian priors** for player tendencies
- **Team context** and matchup analysis
- **Real-time calibration** from past predictions

## 📊 Current Performance (1,523 Predictions)

| Metric | Performance |
|--------|-------------|
| **Overall Accuracy** | 49.1% (needs recalibration) |
| **Best Prop** | Assists: 52.8% ✓ |
| **Points** | 50.8% (break-even) |
| **Rebounds** | 46.8% ❌ |
| **Threes** | 43.3% ❌ |

⚠️ **Status**: Model needs recalibration - currently overconfident on high-probability predictions.

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

### 2. Setup API Keys

Create `keys.py`:
```python
SGO_API_KEY = "your_sportsdata_key_here"
```

Get your free API key at [SportsDataIO](https://sportsdata.io/)

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
├── ensemble_models_enhanced.py        # Ensemble model architecture
├── player_ensemble_enhanced.py        # Enhanced dynamic selector
│
├── models/                            # Trained models
│   ├── ensemble_[STAT]_[WINDOW]/    # Per-window models
│   └── enhanced_selector_[STAT].pkl   # Window selectors
│
├── priors_data/                       # Player Bayesian priors
│   └── player_priors_[STAT]_window_[N].csv
│
├── data/                              # Historical NBA data
├── bets_ledger.pkl                    # Prediction tracking
└── keys.py                            # API keys (not committed)
```

## 🔧 Key Scripts

### Production Scripts

| Script | Purpose |
|--------|---------|
| `riq_analyzer.py` | Generate daily predictions |
| `analyze_ledger.py` | Analyze prediction performance |
| `fetch_bet_results_incremental.py` | Fetch actual game results |
| `recalibrate_models.py` | Recalibrate with new data |

### Training Scripts

| Script | Purpose |
|--------|---------|
| `train_ensemble_enhanced.py` | Train ensemble models |
| `train_dynamic_selector_enhanced.py` | Train window selectors |
| `train_auto.py` | Full training pipeline |

### Analysis Scripts

| Script | Purpose |
|--------|---------|
| `backtest_enhanced_selector.py` | Historical backtest |
| `compare_ensemble_baseline.py` | Compare approaches |
| `test_enhanced_selector_live.py` | Test selector |

## 🎓 How It Works

### 1. Ensemble Architecture

For each stat (PTS, REB, AST, 3PM), we train **5 separate models** on different rolling windows:
- 3-game window (recent form)
- 5-game window (short-term trends)
- 7-game window (balanced)
- 10-game window (stable baseline)
- 15-game window (long-term patterns)

### 2. Enhanced Dynamic Selector

A meta-learner that:
- Analyzes player history and context
- Selects the best window for each prediction
- Considers volatility, usage, and team context
- Improves accuracy by ~0.5% over simple averaging

### 3. Bayesian Priors

Player-specific priors capture:
- Career tendencies (mean, variance)
- Shot volume patterns (FGA, 3PA, FTA)
- Usage rates and efficiency
- Helps with low-sample situations

### 4. Prediction Flow

```
1. Fetch today's props from SportsDataIO
2. For each prop:
   a. Calculate features from recent games
   b. Apply Bayesian priors
   c. Enhanced selector picks best window
   d. Generate prediction with confidence
3. Filter by edge threshold (>2% recommended)
4. Log predictions to ledger
5. Display ranked recommendations
```

## 📈 Model Training

### Full Retrain (Recommended Monthly)

```bash
# Train all ensemble models (5 windows × 4 stats = 20 models)
python train_ensemble_enhanced.py

# Train enhanced selectors (4 stat types)
python train_dynamic_selector_enhanced.py
```

Training time: ~2-3 hours on full historical data (2017-2024)

### Incremental Update (Weekly)

```bash
# Recalibrate with recent predictions only
python recalibrate_models.py
```

## 🎯 Prediction Output Example

```
TOP RECOMMENDATIONS (Positive Edge):

1. Jalen Brunson OVER 25.5 Points
   Line: 25.5 | Prediction: 29.68 | Edge: +15.9%
   Confidence: 87.3% | Window: 7-game | Odds: -110
   
2. Giannis Antetokounmpo OVER 11.5 Rebounds
   Line: 11.5 | Prediction: 12.84 | Edge: +11.6%
   Confidence: 82.1% | Window: 10-game | Odds: -115
```

## 📊 Performance Tracking

All predictions are automatically logged to `bets_ledger.pkl` with:
- Prediction details (player, stat, line, pick)
- Model metadata (confidence, window, odds)
- Actual results (fetched later)
- Win/loss tracking

### Analyze Performance

```bash
python analyze_ledger.py
```

Shows:
- Overall accuracy by stat type
- Calibration analysis (predicted prob vs actual win rate)
- Edge analysis (are you beating the closing line?)
- Recommendations for improvement

## 🔄 Data Updates

### Fetch Results for Past Predictions

```bash
# Incremental fetch (skips already-fetched)
python fetch_bet_results_incremental.py

# Run multiple times if rate-limited (50 players per run)
```

Uses fuzzy date matching (±1 day) to handle sportsbook vs NBA schedule differences.

## ⚠️ Current Issues & Solutions

### Issue 1: Overconfidence
**Problem**: High-confidence predictions (>90%) only winning ~47%  
**Solution**: Run `recalibrate_models.py` to adjust confidence scores

### Issue 2: Poor Rebounds/Threes Performance
**Problem**: Accuracy below breakeven  
**Solution**: 
- Add Phase 1 features (FGA, 3PA, efficiency rates)
- Retrain with expanded feature set
- See `FEATURE_ENGINEERING_ROADMAP.md`

### Issue 3: Small Edge
**Problem**: 49.1% accuracy (need 52.4% at -110 odds)  
**Solution**:
- Increase edge threshold (3%+ only)
- Focus on ASSISTS props (52.8% accuracy)
- Wait for recalibration before betting

## 🛠️ Troubleshooting

### API Rate Limits
```bash
# Fetch incrementally (50 players per run)
python fetch_bet_results_incremental.py

# Wait 60 seconds between runs if rate-limited
```

### Player Not Found
Some players have name mismatches between sportsbook and NBA database. Check `player_name_mapping.py` for fixes.

### Memory Issues
```bash
# Clear model cache
python clear_ensemble_cache.py

# Or manually delete model_cache/
```

## 📚 Documentation

- `FEATURE_ENGINEERING_ROADMAP.md` - Planned feature improvements
- `ENSEMBLE_INTEGRATION_GUIDE.md` - Technical architecture
- `ENHANCED_SELECTOR_INTEGRATION_STATUS.md` - Selector implementation
- `QUICK_START_UNIFIED.md` - Alternative quick start guide

## 🤝 Contributing

This is a personal project, but suggestions welcome via issues.

## 📄 License

Private repository - not for commercial use without permission.

## 🎲 Disclaimer

**This tool is for educational and research purposes only.** 

Sports betting involves risk. Past performance does not guarantee future results. The current model shows **negative edge** and requires recalibration before use. Never bet more than you can afford to lose.

## 📞 Support

For issues or questions, open a GitHub issue or contact the repository owner.

---

**Last Updated**: November 4, 2025  
**Model Version**: Enhanced Ensemble v2.0  
**Training Data**: NBA seasons 2017-2024  
**Predictions Tracked**: 1,728 (1,523 settled)
