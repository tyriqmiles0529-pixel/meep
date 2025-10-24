# RIQ MEEPING MACHINE 🚀

NBA prop betting analyzer with **Expected Log Growth (ELG)** optimization and optional **ML-powered projections**.

---

## Quick Start

### 1. Set up API credentials

#### Sports API (Required)
```bash
export API_SPORTS_KEY='your_apisports_io_key'
```

Get your key from: https://api-sports.io/

#### Kaggle API (Optional - for ML training)
```bash
python setup_kaggle.py
```

Follow the prompts to set up Kaggle credentials for accessing training data.

### 2. Install dependencies
```bash
pip install requests pandas numpy lightgbm scikit-learn
```

### 3. Run the analyzer
```bash
python nba_prop_analyzer_fixed.py
```

**Runtime:** ~30-50 seconds (Fast Mode)
**Output:** Top 5 props per category, ranked by ELG

---

## What's Included

### 📊 Core Analyzer
- **`nba_prop_analyzer_fixed.py`** - Main analyzer with ELG + dynamic Kelly
- Uses heuristic projections (EWMA, pace, defense adjustments)
- Fast mode optimized for <50s runtime

### 🤖 ML Training Pipeline (Optional)
- **`train_prop_model.py`** - Train LightGBM models on historical data
- **`explore_dataset.py`** - Explore Kaggle dataset structure
- **`setup_kaggle.py`** - Interactive Kaggle authentication setup

### 📖 Documentation
- **`QUICK_START.md`** - User setup guide
- **`ELG_OPTIMIZATION_NOTES.md`** - Technical deep dive (23KB)
- **`MODEL_INTEGRATION.md`** - ML integration guide
- **`KAGGLE_SETUP.md`** - Kaggle auth details

### 🔧 Reference Modules
- **`riq_scoring.py`** - ELG, Kelly, exposure caps utilities
- **`riq_prop_models.py`** - Prop-specific statistical models

---

## Key Features

### ✅ Expected Log Growth (ELG)
- Not arbitrary composite scores
- Directly optimizes for long-term compound returns
- Theoretically grounded in Kelly Criterion

### ✅ Dynamic Fractional Kelly
- Adapts from 25%-50% based on posterior uncertainty
- Conservative 30th percentile threshold
- Drawdown scaling (reduces bets during losing streaks)

### ✅ Beta Posterior Sampling
- Bayesian uncertainty quantification
- Prop-specific effective sample sizes
- Only bets if conservative estimate > break-even

### ✅ Prop-Specific Models
- **Normal:** PTS, AST, REB (continuous stats)
- **Negative Binomial:** 3PM (count data, overdispersion)

### ✅ Early-Season Blending
- Empirical Bayes: blends prior + current season
- Stable projections even with 1-2 games

### ✅ Fast Mode
- Runtime budget: 50s (avoids timeouts)
- On-demand player lookups (no roster build)
- Reduced API overhead

---

## Configuration

Edit `nba_prop_analyzer_fixed.py`:

```python
# Speed
FAST_MODE = True  # False for comprehensive analysis

# Bankroll
BANKROLL = 100.0

# Kelly fractions
q_conservative = 0.30  # Lower = more conservative (e.g., 0.20)
fk_high = 0.50         # Lower = smaller bets (e.g., 0.40)

# Early-season blending
PRIOR_GAMES_STRENGTH = 12.0  # Higher = more weight on prior season
TEAM_CONTINUITY_DEFAULT = 0.7  # 0.5-0.9
```

---

## ML Training (Optional)

### Setup
```bash
python setup_kaggle.py  # Authenticate with Kaggle
python explore_dataset.py  # Inspect data structure
```

### Train
```bash
python train_prop_model.py
```

This trains LightGBM models for:
- Points
- Assists
- Rebounds
- 3-Pointers

Models saved to `models/` directory.

### Integration

Once trained, models replace heuristic projections:
- **Before:** EWMA + pace/defense adjustments
- **After:** LightGBM predictions with 50+ features

The ELG/Kelly framework stays the same - just better inputs!

---

## Example Output

```
========================================================================
RIQ MEEPING MACHINE 🚀 — Unified Analyzer
========================================================================
Season: 2025-2026 | Stats: prior=2024-2025 | Bankroll: $100.00
Odds Range: -500 to 500 | Ranking: ELG + dynamic Kelly
FAST_MODE: ON | Time Budget: 50s
========================================================================

Points
------
🟢 # 1 | LeBron James         | Points   | ELG: 0.012345
     Game: Lakers vs Celtics
     Line: 25.5   | Proj: 28.50  | Δ: +3.00 | σ: 4.20
     🏀 Pace: 1.050x | 🛡️ Defense: 0.980x
     Pick: OVER   | Odds: -115
     Kelly: 3.50% | Stake: $3.50 | Profit: $3.04
     EV: +8.20% | Win Prob: 61.2%
```

---

## Files Generated

### Persistent Data (`.gitignore`d)
- `player_cache.pkl` - Player IDs, stats, team stats
- `prop_weights.pkl` - Learned confidence multipliers
- `prop_results.pkl` - Historical outcomes
- `equity_curve.pkl` - Bankroll history

### Output
- `prop_analysis_YYYYMMDD_HHMMSS.json` - Full bet details

### Models (after training)
- `models/points_model.pkl`
- `models/assists_model.pkl`
- `models/rebounds_model.pkl`
- `models/threepoint_goals_model.pkl`

---

## Workflow

### Daily Usage (Heuristic Mode)
```bash
export API_SPORTS_KEY='your_key'
python nba_prop_analyzer_fixed.py
# Review output, place bets
```

### One-Time: ML Training
```bash
python setup_kaggle.py
python explore_dataset.py
python train_prop_model.py
# Models saved to models/
```

### Daily Usage (ML Mode - future)
```bash
export API_SPORTS_KEY='your_key'
export USE_ML_MODELS='true'
python nba_prop_analyzer_fixed.py
# Uses trained models for projections
```

---

## Architecture

### Current: Heuristic Projections
```
Fetch odds/stats → EWMA projection → Prop-specific model (Normal/NB)
→ Beta posterior → Dynamic Kelly → ELG ranking → Top 5 per category
```

### Future: ML Projections
```
Fetch odds/stats → Feature engineering (50+ features)
→ LightGBM prediction → Prop-specific model → Beta posterior
→ Dynamic Kelly → ELG ranking → Top 5 per category
```

**Key insight:** ELG/Kelly framework is projection-agnostic. We can swap in better projections without changing the decision logic!

---

## Roadmap

- [x] ELG + dynamic Kelly framework
- [x] Prop-specific distributions (Normal, Negative Binomial)
- [x] Early-season Empirical Bayes blending
- [x] Fast mode (<50s runtime)
- [x] ML training pipeline (LightGBM)
- [ ] ML integration with live analyzer
- [ ] Backtesting framework
- [ ] Probability calibration (isotonic regression)
- [ ] MLflow experiment tracking
- [ ] Real-time feature engineering
- [ ] Production deployment (API)

---

## Warnings

⚠️ **Educational tool, not financial advice**
⚠️ **Sports betting involves risk of loss**
⚠️ **Validate with backtesting before real money**
⚠️ **Check local gambling laws**

---

## Support

- **Issues:** https://github.com/tyriqmiles0529-pixel/meep/issues
- **Docs:** See `ELG_OPTIMIZATION_NOTES.md` for technical details
- **Quick Start:** See `QUICK_START.md` for setup

---

**Happy Meeping! 🚀**
