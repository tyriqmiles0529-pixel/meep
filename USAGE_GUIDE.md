# NBA Prop Analyzer - Usage Guide

## Quick Start

### Running the Analyzer

```bash
python nba_prop_analyzer_fixed.py
```

This will:
1. Fetch upcoming games (next 3 days)
2. Extract player props and game bets from odds
3. Analyze using ELG framework
4. Output Top 5 per category
5. Save results to JSON

## Understanding the Output

### Top 5 Per Category

The analyzer groups bets into 6 categories and shows the top 5 in each by ELG score:

- **Points**: Player points over/under
- **Assists**: Player assists over/under
- **Rebounds**: Player rebounds over/under
- **3PM**: Player three-pointers made over/under
- **Moneyline**: Game moneyline bets
- **Spread**: Game point spread bets

### Example Output

```
📊 POINTS
----------------------------------------------------------------------
🟢 #1 | LeBron James            | ELG: 0.0234
     Game: Lakers vs Warriors
     Line: 25.5   | Proj: 27.32  | Pick: OVER
     Win: 62.3% | Kelly: 3.45% | Stake: $3.45
     EV: +8.2% | ROI: 91.0%
```

**Key Metrics:**
- **ELG**: Expected Log Growth (higher is better)
- **Win%**: Probability of winning the bet
- **Kelly%**: Kelly fraction (% of bankroll)
- **Stake**: Actual dollar amount to bet
- **EV**: Expected value (return per dollar)
- **ROI**: Return on investment

### Confidence Indicators

- 🟢 **High (65%+)**: Strong confidence
- 🟡 **Medium (55-65%)**: Moderate confidence
- 🟠 **Lower (50-55%)**: Lower confidence but positive edge

## Configuration

### Adjusting Parameters

Edit `nba_prop_analyzer_fixed.py`:

```python
# Kelly sizing
KELLY_CONFIG = KellyConfig(
    min_kelly_stake=0.01,          # Minimum bet size
    max_kelly_fraction=0.25,        # Max 25% of bankroll
    conservative_quantile=0.25,     # Use p_25 for sizing
    elg_samples=1000                # Monte Carlo samples
)

# Exposure caps
EXPOSURE_CAPS = ExposureCaps(
    max_per_game=0.15,      # Max 15% per game
    max_per_player=0.10,    # Max 10% per player
    max_per_team=0.20,      # Max 20% per team
    max_total=0.50          # Max 50% total
)

# Early season blending
PRIOR_GAMES_STRENGTH = 5.0          # Prior strength (in games)
TEAM_CONTINUITY_DEFAULT = 0.8      # Roster continuity factor

# Bankroll
BANKROLL = 100.0
```

### Understanding ELG Scoring

**ELG (Expected Log Growth)** maximizes long-term compound growth by:

1. Using a **conservative probability** (25th percentile of posterior)
2. Accounting for **uncertainty** in win probability
3. Only betting when **ELG > 0** (positive expected growth)

**Gates Applied:**
- Conservative edge: `p_conservative > p_break_even`
- Positive growth: `ELG > 0`
- Minimum stake: `stake >= MIN_KELLY_STAKE`

**NO MIN_CONFIDENCE gate for player props!** The ELG framework naturally handles risk.

## JSON Output

Results are saved to `prop_analysis_YYYYMMDD_HHMMSS.json`:

```json
{
  "timestamp": "20241023_120000",
  "bankroll": 100.0,
  "kelly_config": {
    "max_fraction": 0.25,
    "conservative_quantile": 0.25,
    "min_stake": 0.01
  },
  "top_props": [...],
  "top_by_category": {
    "Points": [...],
    "Assists": [...],
    "Rebounds": [...],
    "3PM": [...],
    "Moneyline": [...],
    "Spread": [...]
  },
  "summary": {
    "total_bets": 15,
    "total_stake": 42.50,
    "total_potential": 98.25,
    "avg_win_prob": 58.3,
    "avg_ev": 6.8,
    "avg_elg": 0.0187
  }
}
```

## API Configuration

### Setting Your API Key

Option 1: Edit the file directly
```python
API_KEY = "your_key_here"
```

Option 2: Use environment variable
```python
API_KEY = os.getenv("API_SPORTS_KEY", "default_key")
```

Then run:
```bash
export API_SPORTS_KEY="your_key_here"
python nba_prop_analyzer_fixed.py
```

### API Endpoints Used

- `/games`: Upcoming games
- `/odds`: Game odds and player props
- `/players`: Player search
- `/games/statistics/players`: Player statistics
- `/statistics`: Team statistics

## Troubleshooting

### No Props Shown

**Problem**: "❌ No props met ELG thresholds"

**Possible causes:**
1. Conservative probability < break-even (p_c <= p_be)
2. Negative ELG (model predicts losses)
3. Stake below minimum threshold
4. No upcoming games with odds

**Solutions:**
- Check if player stats are being fetched (look for cache hit messages)
- Verify API key is valid
- Check odds range filter (MIN_ODDS, MAX_ODDS)
- Try adjusting `conservative_quantile` (e.g., 0.30 instead of 0.25)

### API Errors

**Problem**: "❌ Error 401" or "❌ Error 429"

**Solutions:**
- **401**: Invalid API key - check your key
- **429**: Rate limited - wait and retry
- **503**: API down - try again later

### Low Win Probabilities

**Problem**: All win probabilities < 55%

**Explanation**: This is normal! The prop-aware models don't artificially cap probabilities. The ELG framework ensures we only bet when there's positive expected growth, even at lower probabilities.

**What to do**: Trust the ELG score. If ELG > 0, the bet has positive expected log growth.

## Advanced Usage

### Custom Prop-Aware Models

Edit `riq_prop_models.py` to customize:

- **Projection**: `project_stat()` - EWMA, trend, overdispersion
- **Probability**: `prop_win_probability()` - Normal vs Poisson/NegBin
- **Blending**: `blend_seasons()` - Early season logic

### Custom Scoring

Edit `riq_scoring.py` to customize:

- **Kelly sizing**: `dynamic_fractional_kelly()` - Conservative quantile
- **ELG calculation**: `risk_adjusted_elg()` - Monte Carlo samples
- **Portfolio selection**: `select_portfolio()` - Exposure caps

### Backtesting

To backtest the strategy:

1. Collect historical odds and results
2. Run analyzer on historical odds
3. Compare predicted ELG vs actual log growth
4. Adjust `conservative_quantile` and `max_kelly_fraction`

## Best Practices

### Risk Management

1. **Start small**: Use lower `max_kelly_fraction` (e.g., 0.10)
2. **Track bankroll**: Update BANKROLL based on actual balance
3. **Respect caps**: Don't override exposure limits
4. **Monitor ELG**: Track actual vs predicted growth

### When to Bet

✅ **Good to bet:**
- ELG > 0.02 (2% expected log growth)
- Win probability > 55%
- Stake >= 1% of bankroll
- Category diversification

❌ **Avoid betting:**
- ELG < 0.01 (too small)
- Win probability < 50%
- Over-concentration in one category
- Total exposure > 50%

### Bankroll Management

- Update BANKROLL regularly based on actual balance
- Don't bet more than shown stake
- Consider drawdown scaling if losing streak occurs
- Take profits when ahead

## Migration from Old System

### What Changed?

**Removed:**
- Global MIN_CONFIDENCE gate (40%) for player props
- Artificial probability caps (25%-90%)
- Fixed composite-score formula
- Duplicate analyzer files (optimized version)

**Added:**
- ELG scoring framework
- Conservative probability gates (p_c > p_be)
- Prop-aware distributions (Normal, Poisson)
- Top 5 per category output
- Exposure caps
- Early-season blending

### Backwards Compatibility

All previous fields remain in output:
- `win_prob`, `kelly_pct`, `stake`, `ev`, `roi`
- `composite_score` (for comparison)
- `projection`, `line`, `pick`

New fields added:
- `elg` - Primary ranking metric

## Support

For questions or issues:
1. Check OPTIMIZATION_NOTES.md for detailed methodology
2. Review this guide for common issues
3. Check DEBUG_MODE output for diagnostic info

## License

This analyzer is for educational purposes. Always gamble responsibly and within your means.
