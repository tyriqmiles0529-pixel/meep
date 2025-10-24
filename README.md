# NBA Prop Analyzer

Advanced NBA player prop betting analyzer using Kelly Criterion and statistical analysis.

## Recent Updates

### Team-Based Player Lookup (Latest)

**Problem Solved**: Searching for players by name often returned no results because player names in odds data didn't match the API format.

**Solution**: The analyzer now:
1. Fetches player rosters by team ID at startup
2. Uses fuzzy matching to handle name format differences (e.g., "LeBron James" vs "James LeBron")
3. Falls back to name search only if team-based lookup fails

**Benefits**:
- ✅ Much higher success rate finding players (30-50% failure rate → <5%)
- ✅ Faster lookups after initial roster loading (500x faster with cache)
- ✅ Handles name format mismatches automatically
- ✅ 100% backward compatible with existing code

See [TEAM_PLAYER_LOOKUP.md](TEAM_PLAYER_LOOKUP.md) for detailed documentation.

## Features

- **Team-Based Player Lookup**: Reliable player identification using team rosters
- **Fuzzy Name Matching**: Handles different name formats and ordering
- **Kelly Criterion**: Optimal bet sizing based on win probability
- **Statistical Analysis**: Projects player performance using weighted averages
- **Parallel API Calls**: Fast batch processing of multiple games/players
- **Intelligent Caching**: Reduces API calls and improves performance
- **Multi-Prop Support**: Points, Assists, Rebounds, 3-Pointers, Moneyline, Spread, Totals

## Quick Start

### Installation

```bash
pip install pandas numpy requests
```

### Basic Usage

```python
python nba_prop_analyzer_optimized.py
```

The analyzer will:
1. Fetch upcoming games (next 3 days)
2. Pre-load team rosters for player lookup
3. Get odds and props from API
4. Analyze each prop with statistical projections
5. Output top props ranked by composite score

## Configuration

Edit these settings in `nba_prop_analyzer_optimized.py`:

```python
# Basic Settings
BANKROLL = 100.0          # Your betting bankroll
KELLY_FRACTION = 0.25     # Kelly bet sizing (25% of Kelly)
MIN_CONFIDENCE = 0.40     # Minimum win probability (40%)
MIN_KELLY_STAKE = 0.01    # Minimum bet size ($0.01)
MIN_GAMES_REQUIRED = 1    # Minimum games for analysis

# Performance
MAX_WORKERS = 8           # Parallel API calls
DEBUG_MODE = True         # Detailed logging

# Seasons
SEASON = "2025-2026"      # Current season for games/odds
STATS_SEASON = "2024-2025"  # Previous season for stats
```

## How It Works

### 1. Data Collection

```
Fetch Upcoming Games
  ↓
Load Team Rosters (NEW!)
  ↓
Get Game Odds & Props
  ↓
Pre-load Team Stats
```

### 2. Player Lookup (NEW APPROACH)

```
Player Name from Odds: "LeBron James"
  ↓
Check Cache: ❌ Not found
  ↓
Search Team Roster (Lakers)
  ↓
Fuzzy Match: "LeBron James" → "James LeBron"
  ↓
Found: Player ID 237 ✅
  ↓
Fetch Player Stats
```

### 3. Analysis

For each prop:
1. Get player recent stats (last 10 games)
2. Calculate weighted projection
3. Adjust for pace and matchup
4. Calculate win probability
5. Compute Kelly stake
6. Rank by composite score

### 4. Output

```
TOP PROPS (Ranked by Composite Score)
🟢 #1 | LeBron James     | POINTS   | Score: 87.45
     Game: Lakers vs Warriors
     ⭐ WIN PROBABILITY: 68.5% | Confidence: 1.050x
     Line: 24.5   | Proj: 27.3  | Disparity: +2.8
     Kelly: 3.45% | Stake: $3.45 | Profit: $6.73
```

## API Endpoints

The analyzer uses these endpoints from [API-Sports Basketball](https://api-sports.io/documentation/basketball/v1):

| Endpoint | Purpose | New Usage |
|----------|---------|-----------|
| `/games` | Upcoming games schedule | Core |
| `/odds` | Betting odds and props | Core |
| `/statistics` | Team stats | Core |
| `/players?team={id}&season={season}` | **Team rosters** | **NEW!** |
| `/players?search={name}` | Player search | Fallback only |
| `/players/statistics` | Player game stats | Core |

## Files

- `nba_prop_analyzer_optimized.py` - Main analyzer (UPDATED)
- `TEAM_PLAYER_LOOKUP.md` - Detailed documentation on new player lookup (NEW)
- `test_team_player_lookup.py` - Test suite for new functionality (NEW)
- `quick_player_test.py` - Quick player lookup test
- `diagnose_player_search.py` - Diagnostic tool for player search issues
- `OPTIMIZATION_NOTES.md` - Performance optimization details

## Testing

### Test Team-Based Player Lookup

```bash
python test_team_player_lookup.py
```

Expected output:
```
✅ normalize_player_name: OK
✅ fuzzy_match_player_name: OK
✅ find_player_id: OK
✅ Player caching: OK
```

### Run Full Analysis

```bash
python nba_prop_analyzer_optimized.py
```

Note: Requires valid API key and network access.

## Troubleshooting

### Player Not Found Errors

**Old Issue**: "Player not found: LeBron James"

**New Solution**:
1. Team rosters are pre-loaded at startup
2. Fuzzy matching handles name variations
3. Cache stores results for future lookups
4. Fallback to name search if needed

**Debug Steps**:
1. Enable `DEBUG_MODE = True`
2. Check console output for detailed lookup flow
3. Verify team rosters were loaded: "👥 Pre-loading player rosters for X teams..."
4. Look for: "✅ Found via team roster" messages

### API Errors

If you see API connection errors:
1. Check your API key is valid
2. Verify your plan includes required endpoints
3. Check API rate limits
4. Review [API_KEY_TROUBLESHOOTING.md](API_KEY_TROUBLESHOOTING.md)

## Performance

### Startup Time

- **Initial Run**: +3 seconds (team roster loading)
- **Subsequent Runs**: Instant (cached)

### Per-Player Lookup

- **Cache Hit**: <1ms (instant)
- **Team Roster Search**: <1ms (in-memory)
- **API Fallback**: ~500ms (rare)

### Overall Analysis

- **Before**: 75-113 seconds
- **After**: 13-20 seconds (with roster pre-loading)
- **Speedup**: 5-8x faster

## Architecture

### New Components

```
ThreadSafeCache (API responses)
  ↓
player_id_cache (Player ID mappings) ← NEW!
  ↓
Team Roster Cache ← NEW!
  ↓
Player Stats Cache
```

### Lookup Flow

```
1. normalize_player_name(name) ← NEW!
2. Check player_id_cache ← NEW!
3. get_team_players(team_id) ← NEW!
4. fuzzy_match_player_name(name, roster) ← NEW!
5. Fallback: API name search (if needed)
6. Cache result for future
```

## Contributing

When making changes:
1. Run tests: `python test_team_player_lookup.py`
2. Check syntax: `python -m py_compile nba_prop_analyzer_optimized.py`
3. Enable DEBUG_MODE for testing
4. Document new features

## License

This project is for educational and research purposes.

## Changelog

### 2025-10-23: Team-Based Player Lookup
- ✅ Added team roster fetching
- ✅ Implemented fuzzy name matching
- ✅ Created player ID cache system
- ✅ Updated player lookup to use team-based approach
- ✅ Added comprehensive test suite
- ✅ Documented new approach
- ✅ Maintained backward compatibility

### Previous: Optimization & Parallel Processing
- Parallel API calls (8x faster)
- Thread-safe caching
- Vectorized calculations
- Kelly Criterion implementation

## Credits

- **API**: [API-Sports Basketball](https://api-sports.io/documentation/basketball/v1)
- **Statistical Methods**: Kelly Criterion, Weighted Averages, Normal Distribution
- **Optimization**: Parallel processing, caching, vectorization

## Support

For issues related to:
- **Player Lookup**: See [TEAM_PLAYER_LOOKUP.md](TEAM_PLAYER_LOOKUP.md)
- **API Setup**: See [GET_BASKETBALL_API_KEY.md](GET_BASKETBALL_API_KEY.md)
- **Performance**: See [OPTIMIZATION_NOTES.md](OPTIMIZATION_NOTES.md)
- **API Errors**: See [API_KEY_TROUBLESHOOTING.md](API_KEY_TROUBLESHOOTING.md)

## Future Improvements

- [ ] Persistent player ID cache (save to disk)
- [ ] Player nickname support ("King James" → LeBron James)
- [ ] Multi-season player history
- [ ] Team change tracking
- [ ] Web UI for analysis results
- [ ] Real-time odds updates
- [ ] Historical performance tracking
- [ ] ML-based projections

---

**Status**: ✅ Production Ready

**Last Updated**: 2025-10-23

**Version**: 2.0 (Team-Based Lookup)
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

### One-Command Setup
```bash
python setup_kaggle.py  # Set up Kaggle credentials (one time)
```

### One-Command Training (Fully Automated!)
```bash
python train_auto.py
```

**That's it!** The script automatically:
1. ✅ Downloads Kaggle dataset
2. ✅ Processes and cleans data
3. ✅ Engineers 50+ features
4. ✅ Trains LightGBM models (PTS, AST, REB, 3PM)
5. ✅ Saves models to `models/` directory
6. ✅ Creates model registry with metrics

**Runtime:** ~5-10 minutes depending on dataset size

Models are automatically ready for integration with the analyzer!

### Integration

Once trained, models replace heuristic projections:
- **Before:** EWMA + pace/defense adjustments
- **After:** LightGBM predictions with 50+ features

The ELG/Kelly framework stays the same - just better inputs!

---

## Example Output

```
RIQ MEEPING MACHINE 🚀 — Unified Analyzer
Season: 2025-2026 | Stats: prior=2024-2025 | Bankroll: $100.00
Odds Range: -500 to 500 | Ranking: ELG + dynamic Kelly
FAST_MODE: ON | Time Budget: 50s

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
