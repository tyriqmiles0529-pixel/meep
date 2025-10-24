# RIQ MEEPING MACHINE - Quick Start Guide

## Setup

### 1. Set your API key

```bash
export API_SPORTS_KEY='your_api_sports_io_key_here'
```

Or create a `.env` file:
```bash
echo "API_SPORTS_KEY=your_key_here" > .env
source .env
```

### 2. Install dependencies

```bash
pip install requests pandas numpy
```

## Run the Analyzer

```bash
python nba_prop_analyzer_fixed.py
```

**Expected runtime:** ~30-50 seconds (Fast Mode ON)

## Output

### Console
- Top 5 props per category (Points, Assists, Rebounds, 3PM, Moneyline, Spread)
- Ranked by **Expected Log Growth (ELG)**
- Shows projection, line, odds, Kelly stake, win probability

### JSON
- Saved to `prop_analysis_YYYYMMDD_HHMMSS.json`
- Full bet details for all top props

## Configuration

Edit `nba_prop_analyzer_fixed.py`:

```python
# Bankroll
BANKROLL = 100.0  # Set to your actual bankroll

# Speed vs Thoroughness
FAST_MODE = True   # False for comprehensive analysis (slower)

# Conservatism
q_conservative = 0.30  # Lower = more conservative (e.g., 0.20)
fk_high = 0.50         # Lower = smaller bets (e.g., 0.40)

# Early-season blending
PRIOR_GAMES_STRENGTH = 12.0  # Higher = more weight on prior season
TEAM_CONTINUITY_DEFAULT = 0.7  # 0.5-0.9 (lower = less continuity)
```

## Key Features

### ✅ Expected Log Growth (ELG)
- **Not** arbitrary composite scores
- Directly optimizes for long-term compound returns
- Theoretically grounded in Kelly Criterion

### ✅ Dynamic Fractional Kelly
- Adapts to model uncertainty
- 25%-50% of full Kelly based on posterior width
- Conservative 30th percentile threshold

### ✅ Beta Posterior Sampling
- Bayesian uncertainty quantification
- Prop-specific effective sample sizes
- Risk-adjusted bet sizing

### ✅ Prop-Specific Models
- **Normal:** PTS, AST, REB (continuous stats)
- **Negative Binomial:** 3PM (count data, overdispersion)

### ✅ Early-Season Blending
- Empirical Bayes: smoothly blends prior + current season
- Stable projections even with 1-2 games

### ✅ Drawdown Protection
- Scales bet sizes down 40% during 30% drawdowns
- Equity curve persistence across runs

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

## Files

- **`nba_prop_analyzer_fixed.py`** - Main analyzer (all-in-one, no external imports)
- **`ELG_OPTIMIZATION_NOTES.md`** - Comprehensive technical documentation
- **`riq_scoring.py`** - Modular ELG utilities (reference only)
- **`riq_prop_models.py`** - Modular prop models (reference only)

## Persistent Data (Created on First Run)

- `player_cache.pkl` - Player IDs, stats, team stats
- `prop_weights.pkl` - Learned confidence multipliers
- `prop_results.pkl` - Historical outcomes for adaptive learning
- `equity_curve.pkl` - Bankroll history for drawdown scaling

## Troubleshooting

### API unreachable or key invalid
- Verify your API key is set correctly
- Check API-Sports.io plan (need PRO for player stats)
- Test with: `curl -H "x-apisports-key: YOUR_KEY" https://v1.basketball.api-sports.io/status`

### No props passed ELG gates
- This is normal when no +EV props are available
- Try adjusting `MIN_ODDS` and `MAX_ODDS` if too restrictive
- Lower `q_conservative` for more aggressive bets (not recommended)

### Script runs too long / times out
- Ensure `FAST_MODE = True`
- Reduce `MAX_GAMES` or `MAX_PLAYER_PROPS_ANALYZE`
- Increase `RUN_TIME_BUDGET_SEC` if needed

## Next Steps

1. **Test Run:** Run once to verify setup
2. **Tune Config:** Adjust bankroll, conservatism, speed
3. **Daily Use:** Run daily before games for fresh odds
4. **Track Results:** Record outcomes in `prop_results.pkl` (manual for now)
5. **Backtest:** Collect historical results for validation

## Documentation

- **Technical Details:** See `ELG_OPTIMIZATION_NOTES.md`
- **API Docs:** https://api-sports.io/documentation/basketball/v1
- **Kelly Criterion:** https://en.wikipedia.org/wiki/Kelly_criterion

## Warnings

⚠️ **Educational tool, not financial advice**
⚠️ **Sports betting involves risk of loss**
⚠️ **Validate with backtesting before real money**
⚠️ **Check local gambling laws**

---

**Happy Meeping! 🚀**
