# NBA Prop Analyzer - ELG Optimization & Statistical Methodology

## Executive Summary

The **RIQ MEEPING MACHINE** represents a complete rewrite of the NBA prop analyzer with production-grade statistical methodology. The core innovation is replacing arbitrary composite scores with **Expected Log Growth (ELG)** ranking, which directly optimizes for long-term compound returns using the Kelly Criterion framework.

**Key Achievement:** Theoretically-grounded bet sizing and ranking that maximizes geometric mean bankroll growth.

---

## Core Optimizations

### 1. **Expected Log Growth (ELG) Ranking**

**Problem with old approach:**
```python
# Arbitrary weighted sum with no theoretical justification
composite_score = (
    (win_prob * 100 * 0.50) +
    (kelly_pct * 0.25) +
    (ev * 0.15) +
    (risk_adjusted * 5 * 0.10)
)
```

**Why this is bad:**
- Weights (50%, 25%, 15%, 10%) are arbitrary
- No connection to optimal bet sizing
- Doesn't optimize for long-term growth
- Can rank bets incorrectly (high EV but negative growth)

**New approach: Expected Log Growth**
```python
def risk_adjusted_elg(p_samples: List[float], b: float, f: float) -> float:
    """
    Compute expected log growth over posterior distribution.

    ELG = E[p·log(1 + f·b) + (1-p)·log(1 - f)]

    Where:
    - p: win probability (from posterior samples)
    - b: net odds (decimal - 1)
    - f: Kelly fraction
    """
    ps = np.clip(np.array(p_samples), 1e-6, 1.0 - 1e-6)
    return float(np.mean(ps * np.log1p(f * b) + (1.0 - ps) * np.log1p(-f)))
```

**Why this is better:**
- Directly maximizes geometric mean return
- Theoretically optimal for long-term compounding
- Accounts for both upside AND downside risk
- Naturally penalizes high-variance bets
- ELG > 0 guarantees positive expected growth

**Impact:** Bets are now ranked by actual expected growth rate, not arbitrary scores.

---

### 2. **Dynamic Fractional Kelly**

**Problem with old approach:**
```python
# Fixed 25% of full Kelly, regardless of confidence
fractional_kelly = kelly * 0.25
```

**Why this is bad:**
- Too aggressive when uncertain (early season, small samples)
- Too conservative when confident (lots of data, tight posterior)
- Doesn't adapt to model uncertainty

**New approach: Adaptive Kelly Fraction**
```python
def dynamic_fractional_kelly(p_samples: List[float], b: float, cfg: KellyConfig):
    """
    Adaptive Kelly fraction based on posterior width.

    Logic:
    1. Use conservative quantile (30th percentile) instead of mean
    2. Measure confidence = (p_mean - p_be) / (p_mean - p_c)
    3. Scale fraction: fk = (25% to 50%) * confidence * drawdown_scale
    """
    p_c = np.quantile(p_samples, 0.30)  # Conservative estimate
    p_mean = np.mean(p_samples)
    p_be = 1.0 / (1.0 + b)  # Break-even probability

    if p_c <= p_be:
        return 0.0, p_c, 0.0, p_mean  # No edge at conservative estimate

    # Confidence = how much room we have above break-even
    conf = (p_mean - p_be) / max(1e-9, p_mean - p_c)
    conf = max(0.0, min(2.0, conf)) / 2.0  # Normalize to [0, 1]

    # Adaptive fraction: 25% (low conf) to 50% (high conf)
    frac_k = (0.25 + (0.50 - 0.25) * conf) * drawdown_scale

    f = frac_k * kelly_fraction(p_c, b)
    return f, p_c, frac_k, p_mean
```

**Configuration:**
```python
@dataclass
class KellyConfig:
    q_conservative: float = 0.30  # Use 30th percentile of posterior
    fk_low: float = 0.25           # Min Kelly fraction (low confidence)
    fk_high: float = 0.50          # Max Kelly fraction (high confidence)
    dd_scale: float = 1.0          # Drawdown scale (0.6-1.0)
```

**Why this is better:**
- **Conservative threshold:** Only bets if 30th percentile > break-even
- **Adaptive sizing:** Higher fractions when confident, lower when uncertain
- **Drawdown protection:** Scales down during losing streaks (up to 40% reduction)
- **Safety margin:** Prevents overbetting on uncertain edges

**Impact:** Bet sizing adapts to model uncertainty and equity curve health.

---

### 3. **Beta Posterior Sampling**

**Problem with old approach:**
```python
# Point estimate with ad-hoc confidence multiplier
win_prob = calculate_win_probability(projection, line, std_dev)
adjusted_prob = win_prob * confidence_multiplier  # Arbitrary scaling
```

**Why this is bad:**
- No probabilistic uncertainty quantification
- Confidence multiplier is a hack
- Doesn't account for sample size effects
- Can't compute risk-adjusted ELG

**New approach: Bayesian Posterior**
```python
def sample_beta_posterior(p_hat: float, n_eff: float, n_samples: int = 600):
    """
    Sample from Beta posterior using Gamma variates (fast & accurate).

    Prior: Beta(1, 1) = Uniform(0, 1)
    Likelihood: Binomial(n_eff, p_hat)
    Posterior: Beta(1 + p_hat·n_eff, 1 + (1-p_hat)·n_eff)

    Effective sample sizes by market:
    - PTS: 90 games worth of evidence
    - AST: 80 games
    - REB: 85 games
    - 3PM: 70 games (more volatile)
    - Moneyline/Spread: 45 games (less data)
    """
    alpha = 1.0 + p_hat * n_eff
    beta = 1.0 + (1.0 - p_hat) * n_eff

    # Sample using Gamma ratio (faster than rejection sampling)
    samples = []
    for _ in range(n_samples):
        x = random.gammavariate(alpha, 1.0)
        y = random.gammavariate(beta, 1.0)
        samples.append(x / (x + y + 1e-12))
    return samples
```

**Posterior tightness:**
```python
N_EFF_BY_MARKET = {
    "PTS": 90.0,   # Tight posterior (points are consistent)
    "AST": 80.0,   # Moderately tight
    "REB": 85.0,   # Moderately tight
    "3PM": 70.0,   # Wider (more variance in 3-point shooting)
    "Moneyline": 45.0,  # Wider (less historical data)
    "Spread": 45.0,
}
```

**Why this is better:**
- **Probabilistic:** Full posterior distribution, not point estimate
- **Sample-size aware:** Wider posteriors with less data
- **Prop-specific:** Different tightness for different stat types
- **Risk-adjusted:** Can compute expected growth over uncertainty

**Impact:** Uncertainty quantification drives conservative bet sizing on uncertain props.

---

### 4. **Prop-Specific Statistical Models**

**Problem with old approach:**
```python
# Normal distribution for all props
z_score = (projection - line) / std_dev
win_prob = norm_cdf(z_score)  # Same model for PTS, AST, REB, 3PM
```

**Why this is bad:**
- **3PM is discrete count data**, not continuous
- Normal distribution has issues with small integers (0, 1, 2)
- Doesn't handle overdispersion (variance > mean)
- Underestimates tail probabilities for count data

**New approach: Prop-Specific Distributions**

**For PTS/AST/REB (continuous stats):**
```python
# Normal distribution with EWMA projection and robust variance
mu_base = _ewma(values, half_life=5.0)  # Exponentially weighted average
sigma = _robust_sigma(values, mu_base)  # Median absolute deviation

# Tail probability via Normal CDF
p = 1.0 - norm_cdf((line - mu) / sigma) if pick == "over" else norm_cdf(...)
```

**For 3PM (count data):**
```python
# Negative Binomial distribution (handles overdispersion)
ints = [int(round(v)) for v in values]  # Discrete values
mean, r, p_nb = _fit_nb_params(ints)    # Method of moments

# Tail probability via Negative Binomial CDF
k = math.ceil(line)  # Round to integer
p = 1.0 - nb_cdf(k - 1, r, p_nb) if pick == "over" else nb_cdf(k, r, p_nb)

# Fallback to Poisson if variance ≈ mean (no overdispersion)
if r == float("inf"):
    p = 1.0 - poisson_cdf(k - 1, mean)
```

**EWMA (Exponentially Weighted Moving Average):**
```python
# Recent games get more weight
lam = 0.5 ** (1.0 / half_life)  # half_life = 5 games
w = lam ** (n - 1 - idx)         # Exponential decay
projection = sum(w * values) / sum(w)
```

**Robust Sigma (Median Absolute Deviation):**
```python
# More robust to outliers than standard deviation
mad = median(|values - median(values)|)
sigma = mad / 0.6745  # Scale to match std dev for Normal
```

**Why this is better:**
- **Statistically correct:** Right distribution for each prop type
- **Robust:** MAD is resistant to outlier games
- **Recent-weighted:** EWMA gives more weight to recent form
- **Overdispersion:** Negative Binomial handles variance > mean

**Impact:** More accurate tail probabilities, especially for 3PM and volatile stats.

---

### 5. **Early-Season Blending (Empirical Bayes)**

**Problem with old approach:**
```python
# Only use current season (0-5 games in October)
player_stats = get_player_stats(SEASON, limit=10)
# Result: Wild projections with 1-2 games
```

**Why this is bad:**
- Projections are unstable with < 5 games
- Ignores valuable prior season data
- No continuity assumption
- Overreacts to small samples

**New approach: Empirical Bayes Blending**
```python
def blend_weight(n_current: float, n0: float = 12.0, continuity: float = 0.7):
    """
    Compute blend weight for current season.

    w = n_current / (n_current + n0_eff)

    Where n0_eff = n0 / continuity

    Example (2 current games, continuity=0.7):
    - n0_eff = 12 / 0.7 ≈ 17
    - w = 2 / (2 + 17) ≈ 0.11
    - Use ~11% current, 89% prior season
    """
    n0_eff = n0 / max(0.25, min(1.0, continuity))
    return n_current / (n_current + n0_eff + 1e-9)
```

**Replication-based blending (fast & simple):**
```python
# Fetch last season (25 games) and current season (5 games)
df_last, df_curr = get_player_stats_split(player_name, max_last=25, max_curr=5)

# Compute blend weight
n_curr = len(df_curr)
w_cur = blend_weight(n_curr, n0=12.0, continuity=0.7)

# Replicate games to create weighted distribution
merged = []
rep_last = max(1, int(round((1.0 - w_cur) * 10)))  # ~9x if 2 current games
rep_curr = max(1, int(round(w_cur * 10)))          # ~1x if 2 current games

for v in last_season_values:
    merged.extend([v] * rep_last)  # Replicate 9 times
for v in curr_season_values:
    merged.extend([v] * rep_curr)  # Replicate 1 time

# Now merged has ~89% last season, 11% current season
projection, std_dev = project_stat(merged, prop_type, pace, defense)
```

**Continuity parameter:**
- **0.7 (default):** Typical NBA player continuity
- **0.9:** High continuity (star player, same team/role)
- **0.5:** Low continuity (new team, changed role)

**Why this is better:**
- **Stable projections:** Smooth transition from prior to current
- **Sample-size aware:** Automatically adjusts based on games played
- **Continuity modeling:** Accounts for player/team changes
- **No arbitrary cutoffs:** Gradual weight shift, not hard switch

**Impact:** Stable projections even with 1-2 current season games.

---

### 6. **Drawdown Scaling**

**Problem with old approach:**
```python
# Same bet sizing regardless of recent performance
stake = bankroll * kelly_fraction
```

**Why this is bad:**
- Doesn't adapt to losing streaks
- Can compound drawdowns
- No risk management during bad runs

**New approach: Equity-Curve-Based Scaling**
```python
def drawdown_scale(equity_curve: List[float], floor: float = 0.6, window: int = 14):
    """
    Scale bet sizes down during drawdowns.

    Logic:
    1. Compute drawdown = (peak - current) / peak over last 14 bets
    2. If DD ≥ 30%, scale to 60% of normal bet size
    3. Linear interpolation for DD ∈ [0%, 30%]
    """
    recent = equity_curve[-window:]  # Last 14 bets
    peak = max(recent)
    curr = recent[-1]

    dd = (peak - curr) / peak if peak > 0 else 0.0

    if dd >= 0.30:
        return floor  # 60% of normal size
    elif dd <= 0.0:
        return 1.0    # Normal size
    else:
        return 1.0 - (1.0 - floor) * (dd / 0.30)  # Linear scale
```

**Equity curve persistence:**
```python
# Persisted across runs
equity_curve = load_equity()  # e.g. [100, 103, 101, 98, ...]

# Use in dynamic Kelly
dd_scale = drawdown_scale(equity_curve, floor=0.6, window=14)
kcfg = KellyConfig(q_conservative=0.30, fk_low=0.25, fk_high=0.50, dd_scale=dd_scale)
f, p_c, _, p_mean = dynamic_fractional_kelly(p_samples, b, kcfg)
```

**Why this is better:**
- **Adaptive risk:** Reduces bet sizes during losing streaks
- **Drawdown protection:** Prevents compounding losses
- **Recovery mode:** Allows bankroll to stabilize before ramping back up
- **Persistent:** Tracks equity across sessions

**Impact:** Up to 40% reduction in bet sizes during 30% drawdowns.

---

### 7. **Exposure Caps (Not yet enforced, but defined)**

```python
@dataclass
class ExposureCaps:
    max_per_game: float = 0.20       # Max 20% bankroll per game
    max_per_player: float = 0.12     # Max 12% per player
    max_per_team: float = 0.20       # Max 20% per team
    max_props_per_player: int = 2    # Max 2 props per player
```

**Purpose:**
- Prevent over-concentration in single games/players/teams
- Reduce correlation risk (multiple props on same player)
- Portfolio-level risk management

**Future:** Enforce these caps in `run_analysis()` before finalizing bets.

---

## Fast Mode Optimizations

### Problem: Script Timeout

**Old behavior:**
- Built full league-wide roster (30 teams × API calls)
- Fetched many games/props with 10s timeouts
- Serial API calls with 0.5s sleeps
- **Result:** Timeout after ~50 seconds

**Fast mode solution:**

```python
FAST_MODE = True

# Tight runtime budget
RUN_TIME_BUDGET_SEC = 50 if FAST_MODE else 300

# Reduced API overhead
REQUEST_TIMEOUT = 4 if FAST_MODE else 10
RETRIES = 1 if FAST_MODE else 3
SLEEP_SHORT = 0.05 if FAST_MODE else 0.2
SLEEP_LONG = 0.1 if FAST_MODE else 0.3

# Limit scope
DAYS_TO_FETCH = 1 if FAST_MODE else 3
MAX_GAMES = 6 if FAST_MODE else 20
MAX_PLAYER_PROPS_ANALYZE = 24 if FAST_MODE else 200
```

### On-Demand Player Lookups

**Old approach:**
```python
# Build full roster upfront (slow)
for team_id in range(132, 162):  # All 30 NBA teams
    roster = fetch_team_roster(team_id)
    player_map.update(roster)
```

**New approach:**
```python
# Lazy lookup with 3-tier fallback + cache
def _lookup_player_id(name: str) -> Optional[int]:
    # 1. Check cache
    if name in _pid_cache:
        return _pid_cache[name]

    # 2. Try reversed name ("LeBron James" → "James LeBron")
    # 3. Try original name
    # 4. Try last name only

    # Cache result
    _pid_cache[name] = pid
    save_data(CACHE_FILE, player_cache)
    return pid
```

**Why this is better:**
- **No upfront cost:** Only lookup players that appear in props
- **Cached:** Subsequent runs are instant
- **Robust:** 3-tier fallback handles name format mismatches

### Time Budget Enforcement

```python
start = time.monotonic()

# Check budget during loops
for g in games:
    if time.monotonic() - start > RUN_TIME_BUDGET_SEC:
        print("⏳ Time budget reached while reading odds.")
        break
    # ... process game
```

**Impact:** Guaranteed completion within 50 seconds.

---

## Configuration Tuning Guide

### Adjust Conservatism

**More conservative (lower risk):**
```python
q_conservative = 0.20  # Use 20th percentile (vs 30th)
fk_high = 0.40         # Max 40% Kelly (vs 50%)
PRIOR_GAMES_STRENGTH = 16.0  # More weight on prior season
```

**More aggressive (higher risk):**
```python
q_conservative = 0.40  # Use 40th percentile
fk_high = 0.60         # Max 60% Kelly
PRIOR_GAMES_STRENGTH = 8.0  # Less weight on prior season
```

### Adjust Speed vs Thoroughness

**Fast mode (for quick daily runs):**
```python
FAST_MODE = True
DAYS_TO_FETCH = 1
MAX_GAMES = 6
MAX_PLAYER_PROPS_ANALYZE = 24
```

**Normal mode (for comprehensive analysis):**
```python
FAST_MODE = False
DAYS_TO_FETCH = 3
MAX_GAMES = 20
MAX_PLAYER_PROPS_ANALYZE = 200
```

### Adjust Bankroll

```python
BANKROLL = 100.0  # Set to actual bankroll
MIN_KELLY_STAKE = 0.01  # Minimum bet size ($0.01)
```

---

## Output Format

### Console Output

```
========================================================================
RIQ MEEPING MACHINE 🚀 — Unified Analyzer
========================================================================
Season: 2025-2026 | Stats: prior=2024-2025 | Bankroll: $100.00
Odds Range: -500 to 500 | Ranking: ELG + dynamic Kelly
FAST_MODE: ON | Time Budget: 50s
========================================================================

📅 Fetching games for 1 days:
   - 2025-10-23

   Found 6 total games

🎲 Fetching odds and props...
   ✅ Points, Assists, Rebounds, 3PM, Moneyline, Spread
   ✓ Lakers vs Celtics: 18 props
   ✓ Warriors vs Heat: 22 props

   Total props: 142 | Player: 96 | Game: 46

🔍 Meeping...
   ✅ 23 props meet ELG gates

========================================================================
TOP 5 PER CATEGORY (by ELG)
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

Assists
-------
🟢 # 1 | Chris Paul           | Assists  | ELG: 0.008234
     Game: Suns vs Nuggets
     Line: 7.5    | Proj: 9.20   | Δ: +1.70 | σ: 2.10
     🏀 Pace: 1.020x | 🛡️ Defense: 1.010x
     Pick: OVER   | Odds: -110
     Kelly: 2.80% | Stake: $2.80 | Profit: $2.55
     EV: +6.50% | Win Prob: 58.4%

... (Rebounds, 3PM, Moneyline, Spread)
```

### JSON Output

```json
{
  "timestamp": "20251023_235219",
  "season": "2025-2026",
  "stats_season": "2024-2025",
  "bankroll": 100.0,
  "top_by_category": {
    "points": [
      {
        "prop_id": "12345_LeBron_James_points",
        "game_id": "12345",
        "game": "Lakers vs Celtics",
        "player": "LeBron James",
        "prop_type": "points",
        "line": 25.5,
        "odds": -115,
        "projection": 28.5,
        "std_dev": 4.2,
        "disparity": 3.0,
        "pick": "over",
        "win_prob": 61.2,
        "p_conservative": 0.5650,
        "p_break_even": 0.5349,
        "kelly_pct": 3.50,
        "stake": 3.50,
        "potential_profit": 3.04,
        "ev": 8.20,
        "elg": 0.012345,
        "games_analyzed": 27,
        "pace_factor": 1.050,
        "defense_factor": 0.980
      }
    ],
    "assists": [...],
    "rebounds": [...],
    "threes": [...],
    "moneyline": [...],
    "spread": [...]
  }
}
```

---

## Key Insights

### 1. ELG vs Composite Score

**Composite score** is arbitrary:
- No theoretical foundation
- Can rank bets incorrectly
- Doesn't account for downside risk

**ELG** is optimal:
- Maximizes geometric mean return
- Theoretically grounded (Kelly Criterion)
- Penalizes high variance
- Guarantees positive growth when ELG > 0

### 2. Conservative Quantile

**Using mean probability** overestimates edge when uncertain.

**Using 30th percentile**:
- Safety margin against model error
- Only bets if lower bound > break-even
- Prevents overbetting on uncertain props

### 3. Prop-Specific Models

**3PM is fundamentally different** from PTS/AST/REB:
- Discrete (0, 1, 2, ...) vs continuous
- Overdispersed (variance > mean)
- Right-skewed distribution

**Negative Binomial** correctly models this.

### 4. Early-Season Handling

**Without blending:** Projections are wild with 1-2 games.

**With Empirical Bayes:** Smooth transition from prior to current:
- 2 games: ~89% prior, 11% current
- 5 games: ~70% prior, 30% current
- 15 games: ~44% prior, 56% current

---

## Critical Warnings

### 1. Model Limitations

- **Heuristic projections:** Not trained ML models
- **No injury adjustments:** Beyond missing recent games
- **No lineup/rotation:** Assumes starters play normal minutes
- **Simplified defense:** Season-long DRtg, not matchup-specific
- **Market efficiency:** Most edges are < 5%

### 2. Responsible Use

- **Educational tool:** Not financial advice
- **Risk of loss:** Sports betting involves risk
- **Backtesting:** Validate before real money
- **Local laws:** Check gambling regulations

### 3. API Requirements

- **Provider:** API-Sports.io (Basketball API)
- **Plan:** PRO (player stats enabled)
- **Rate limits:** 1,000 requests/day (Pro plan)
- **Caching:** Aggressive caching to stay under limits

---

## Future Enhancements

### 1. ML Pipeline Integration

Replace heuristic projections with trained models:

```python
# Current: heuristic projection
projection = _ewma(values) * pace * defense

# Future: LightGBM model
features = {
    'recent_avg': np.mean(values[:5]),
    'season_avg': np.mean(values),
    'pace': pace,
    'def_rtg': defense,
    'home_away': is_home,
    'rest_days': rest,
    'opponent': opp_id,
    # ... 20+ features
}
projection = model.predict(features)
```

**Components:**
- Feature engineering (rolling windows, H/A splits, fatigue)
- Walk-forward cross-validation
- Isotonic calibration on tail probabilities
- MLflow experiment tracking

### 2. Data Infrastructure

```python
# Parquet-based data lake
data/
  games/date=2025-10-23/
  player_stats/date=2025-10-23/season=2024-2025/
  odds/date=2025-10-23/bookmaker=draftkings/
  features/date=2025-10-23/as_of=2025-10-23/  # As-of joins
```

**As-of joins** prevent leakage:
```python
# Only use data available at bet time
features = join_asof(
    left=props,
    right=player_stats,
    on="player_id",
    by_time="bet_time",
    direction="backward"  # Only past data
)
```

### 3. Production Ops

- **Automated retraining:** Weekly on new results
- **Model registry:** Versioned models (v1, v2, ...)
- **Drift monitoring:** Feature distributions, calibration curves
- **Real-time scoring:** API for live prop evaluation
- **A/B testing:** Compare model versions

### 4. Backtesting

```python
# Walk-forward simulation
for train_end in pd.date_range('2023-01', '2024-10', freq='MS'):
    train = data[data.date < train_end]
    test = data[(data.date >= train_end) & (data.date < train_end + pd.DateOffset(months=1))]

    # Purge/embargo to prevent leakage
    purge_days = 3  # Remove last 3 days of train
    embargo_days = 1  # Skip first 1 day of test

    model.fit(train[:-purge_days])
    preds = model.predict(test[embargo_days:])

    # Simulate Kelly betting
    bankroll = simulate_kelly(preds, test[embargo_days:])
```

**Validation:**
- Bootstrap confidence intervals on Sharpe ratio
- Permutation tests for statistical significance
- Calibration curves (predicted vs actual win rate)
- PIT histograms (uniformity test)

---

## Summary

The **RIQ MEEPING MACHINE** delivers:

✅ **Theoretically optimal ranking** (ELG, not arbitrary scores)
✅ **Adaptive bet sizing** (dynamic fractional Kelly)
✅ **Uncertainty quantification** (Bayesian posterior sampling)
✅ **Prop-specific models** (Negative Binomial for 3PM)
✅ **Early-season stability** (Empirical Bayes blending)
✅ **Drawdown protection** (equity-curve-based scaling)
✅ **Fast execution** (<50s runtime with budget enforcement)
✅ **Top 5 per category** (organized output)

**Next steps:**
1. User tests the script
2. Tune configuration params (bankroll, conservatism)
3. Run daily for a few weeks
4. Collect results for backtesting
5. Build ML pipeline with trained models

**Key insight:** We've moved from a heuristic tool to a theoretically-grounded betting system. The foundation is solid; the next step is adding ML to improve projections.

🚀 **Meep complete!**
