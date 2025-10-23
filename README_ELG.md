# NBA Prop Analyzer - ELG Edition

## 🚀 Quick Start

```bash
# Install dependencies
pip install numpy pandas requests

# Run the analyzer
python nba_prop_analyzer_fixed.py
```

## 📋 What's New

This is a **unified analyzer** with Expected Log Growth (ELG) scoring:

✨ **Key Features:**
- 🎯 **ELG Scoring** - Maximize long-term compound growth
- 🧠 **Prop-Aware Models** - Normal for PTS/AST/REB, Poisson for 3PM
- 🛡️ **Conservative Gates** - p_c > p_be, ELG > 0 (no fixed MIN_CONFIDENCE)
- 📊 **Top 5 Per Category** - Points, Assists, Rebounds, 3PM, Moneyline, Spread
- 💰 **Exposure Caps** - Per game, player, team, and total
- 🔄 **Early Season Blending** - Smart combination of last/current season

## 📁 Project Structure

```
meep/
├── nba_prop_analyzer_fixed.py    # 🎯 Main entry point
├── riq_scoring.py                 # ELG and Kelly logic
├── riq_prop_models.py             # Prop-aware models
├── nba_prop_analyzer_optimized.py # Deprecated (use fixed)
├── test_unified_analyzer.py       # Test suite
├── USAGE_GUIDE.md                 # User guide
├── OPTIMIZATION_NOTES.md          # Technical details
└── IMPLEMENTATION_SUMMARY.md      # Project overview
```

## 🎓 Understanding ELG

**Traditional Kelly:** Uses point estimate → overbets when optimistic

**ELG Kelly:** Uses conservative quantile → naturally reduces risk

```python
# Sample from posterior uncertainty
p_samples = Beta(p_hat, n_eff)

# Use 25th percentile for sizing
p_conservative = p_25

# Only bet if conservative p beats break-even
if p_conservative > p_break_even and ELG > 0:
    place_bet()
```

## 📊 Sample Output

```
📊 POINTS
----------------------------------------------------------------------
🟢 #1 | LeBron James            | ELG: 0.0234
     Game: Lakers vs Warriors
     Line: 25.5   | Proj: 27.32  | Pick: OVER
     Win: 62.3% | Kelly: 3.45% | Stake: $3.45
     EV: +8.2% | ROI: 91.0%

📊 ASSISTS
----------------------------------------------------------------------
🟡 #1 | Chris Paul              | ELG: 0.0189
     Game: Suns vs Mavs
     Line: 8.5    | Proj: 9.21   | Pick: OVER
     Win: 58.7% | Kelly: 2.87% | Stake: $2.87
     EV: +5.4% | ROI: 73.0%
```

## 🔧 Configuration

Edit `nba_prop_analyzer_fixed.py`:

```python
# Bankroll
BANKROLL = 100.0

# Kelly sizing
KELLY_CONFIG = KellyConfig(
    min_kelly_stake=0.01,        # Min bet size
    max_kelly_fraction=0.25,      # Max 25% of bankroll
    conservative_quantile=0.25,   # Use p_25
    elg_samples=1000              # Monte Carlo samples
)

# Exposure caps
EXPOSURE_CAPS = ExposureCaps(
    max_per_game=0.15,    # 15% max per game
    max_per_player=0.10,  # 10% max per player
    max_per_team=0.20,    # 20% max per team
    max_total=0.50        # 50% total max
)
```

## 🧪 Testing

```bash
# Run test suite
python test_unified_analyzer.py

# Expected output:
# ✅ Prop Projection: Pass
# ✅ Win Probability: Pass
# ✅ ELG Scoring: Pass
# ✅ Portfolio Selection: Pass
# ✅ Season Blending: Pass
# ✅ Conservative Edge Gates: Pass
```

## 📖 Documentation

- **[USAGE_GUIDE.md](USAGE_GUIDE.md)** - How to use the analyzer
- **[OPTIMIZATION_NOTES.md](OPTIMIZATION_NOTES.md)** - ELG methodology
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Project details

## 🔄 Migration from Old System

If you were using the old analyzer:

```bash
# Old (deprecated)
python nba_prop_analyzer_optimized.py

# New (use this)
python nba_prop_analyzer_fixed.py
```

**What changed:**
- ❌ No more 40% MIN_CONFIDENCE gate
- ❌ No more 25%-90% probability caps
- ✅ ELG-based ranking
- ✅ Top 5 per category
- ✅ Conservative edge gates

**What stayed:**
- ✅ All output fields preserved
- ✅ JSON structure extended (not replaced)
- ✅ API calls unchanged

## 📈 Key Metrics

**Confidence Indicators:**
- 🟢 High (65%+): Strong confidence
- 🟡 Medium (55-65%): Moderate confidence  
- 🟠 Lower (50-55%): Positive edge but lower confidence

**Important Fields:**
- **ELG**: Expected Log Growth (primary ranking metric)
- **Win%**: Probability of winning
- **Kelly%**: Fraction of bankroll
- **Stake**: Dollar amount to bet
- **EV**: Expected value (%)
- **ROI**: Return on investment (%)

## 🛡️ Risk Management

**Built-in Safeguards:**
1. Conservative probability quantile (p_25)
2. Positive ELG requirement
3. Break-even edge requirement
4. Exposure caps on all levels
5. Minimum stake threshold

**Best Practices:**
- Start with low `max_kelly_fraction` (0.10)
- Update BANKROLL regularly
- Don't override exposure caps
- Track actual vs predicted ELG

## 🤝 Contributing

This is a research project. Suggestions welcome:
- Correlation modeling
- Historical backtesting
- Additional prop types
- UI improvements

## ⚠️ Disclaimer

**For educational purposes only.**

This tool provides analysis based on statistical models and does not guarantee profits. Sports betting involves risk. Always gamble responsibly and within your means.

## 📚 References

**Kelly Criterion & ELG:**
- Kelly, J. L. (1956). "A New Interpretation of Information Rate"
- Thorp, E. O. (2006). "The Kelly Criterion in Blackjack Sports Betting"
- MacLean et al. (2011). "The Kelly Capital Growth Investment Criterion"

**Statistical Methods:**
- Normal distribution for continuous stats
- Negative Binomial for count data
- Beta-Binomial conjugate priors
- Robust variance estimation (MAD)

## 📞 Support

For issues or questions:
1. Check [USAGE_GUIDE.md](USAGE_GUIDE.md)
2. Review [OPTIMIZATION_NOTES.md](OPTIMIZATION_NOTES.md)
3. Run test suite: `python test_unified_analyzer.py`
4. Enable DEBUG_MODE in config

---

**Version:** 2.0 (ELG Edition)  
**Last Updated:** October 2025  
**License:** Educational Use Only
