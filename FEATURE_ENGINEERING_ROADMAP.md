# Feature Engineering Roadmap - NBA Predictor

## Current Status

Your models currently use **20 basic features**:
- Game context (home/away, season)
- Team stats (pace, offensive/defensive strength)
- Matchup features
- Player context (starter flag, points per minute, minutes)

**Missing**: Shot volume, efficiency rates, advanced metrics

## Phase 1: High-Impact Volume & Efficiency (Target: +2-4% RMSE improvement)

### 1.1 Shot Volume Features (CRITICAL)

**Why**: Volume = Opportunity. Players who shoot more score more.

Add to player features:
```python
# Rolling averages (last 5 games)
fga_avg_last5 = fieldGoalsAttempted.rolling(5).mean()
fga_per_minute = fga_avg_last5 / minutes_avg_last5

three_pa_avg_last5 = threePointersAttempted.rolling(5).mean()
three_pa_per_minute = three_pa_avg_last5 / minutes_avg_last5

fta_avg_last5 = freeThrowsAttempted.rolling(5).mean()
```

**Expected Impact**: +1.5-2.5% for PTS, +2-3% for 3PM

### 1.2 True Shooting % (CRITICAL)

**Why**: Best measure of scoring efficiency. Accounts for 3s and FTs.

```python
# True Shooting % (season and rolling)
ts_pct = points / (2 * (fieldGoalsAttempted + 0.44 * freeThrowsAttempted))

ts_pct_season = ts_pct.expanding().mean()  # Season average
ts_pct_last5 = ts_pct.rolling(5).mean()    # Recent form
ts_pct_last10 = ts_pct.rolling(10).mean()  # Medium term
```

**Expected Impact**: +0.5-1% for PTS

### 1.3 Usage Rate (CRITICAL)

**Why**: Captures how much of team's offense runs through a player.

```python
# Simplified usage rate (requires team totals)
# USG% = (Player FGA + 0.44*FTA + TOV) / Team equivalent * minutes adjustment
player_possessions = fieldGoalsAttempted + 0.44 * freeThrowsAttempted + turnovers
team_possessions = team_fga + 0.44 * team_fta + team_tov

usage_rate = (player_possessions / (numMinutes / 5)) / (team_possessions / 48)
usage_rate_avg_last5 = usage_rate.rolling(5).mean()
```

**Expected Impact**: +1-2% for PTS, AST

### 1.4 Rebound Rate (CRITICAL for REB)

**Why**: Pace-independent measure of rebounding ability.

```python
# Total Rebound %
available_rebounds = team_rebounds + opponent_rebounds
trb_pct = reboundsTotal / available_rebounds * (team_minutes / 5) / numMinutes

# Separate offensive and defensive
orb_pct = reboundsOffensive / opponent_drb * (team_minutes / 5) / numMinutes
drb_pct = reboundsDefensive / opponent_orb * (team_minutes / 5) / numMinutes

trb_pct_last5 = trb_pct.rolling(5).mean()
```

**Expected Impact**: +2-3% for REB

### 1.5 Assist Rate (IMPORTANT for AST)

**Why**: % of teammate FGs assisted while on court.

```python
# Assist Rate
teammate_fgm = team_fgm - fieldGoalsMade  # Teammate made FGs
ast_pct = assists / teammate_fgm * (team_minutes / 5) / numMinutes

ast_pct_last5 = ast_pct.rolling(5).mean()

# Assist-to-Turnover Ratio
ast_to_tov = assists / max(turnovers, 1)
ast_to_tov_last5 = ast_to_tov.rolling(5).mean()
```

**Expected Impact**: +1-2% for AST

### 1.6 Shot Selection Features

**Why**: Role identification (shooter vs. non-shooter).

```python
# 3-Point Attempt Rate (% of FGA that are 3PA)
three_par = threePointersAttempted / max(fieldGoalsAttempted, 1)
three_par_last5 = three_par.rolling(5).mean()

# Free Throw Rate (ability to get to the line)
ft_rate = freeThrowsAttempted / max(fieldGoalsAttempted, 1)
ft_rate_last5 = ft_rate.rolling(5).mean()
```

**Expected Impact**: +0.5-1% for PTS, 3PM

## Phase 2: Matchup & Context Features (Target: +1-2% improvement)

### 2.1 Opponent Defensive Metrics

From your team priors data:
```python
# Opponent strength
opp_def_rating        # Points allowed per 100 possessions
opp_pace              # Possessions per 48 minutes
opp_3p_allowed_per_game
opp_ast_allowed_per_game
```

### 2.2 Fatigue & Schedule

```python
# From game dates
is_b2b = (gameDate - prev_gameDate).days == 1
rest_days = (gameDate - prev_gameDate).days
games_in_last_7_days = count of games in rolling 7-day window
```

### 2.3 Injury Impact

```python
# Requires injury data (if available)
key_teammate_out_flag = 1 if teammate with usage_rate > 25% is out
usage_rate_missing = sum of usage rates for players out
```

## Phase 3: Advanced Features (Target: +0.5-1% improvement)

### 3.1 Hot/Cold Streaks

```python
# Performance vs. expectation
pts_vs_expected_last3 = recent_pts - season_avg_pts
ts_pct_momentum = (ts_pct_last3 - ts_pct_season)
```

### 3.2 Positional Matchups

```python
# If you have player positions
opp_def_vs_pg  # Points allowed to point guards
opp_def_vs_sg  # etc.
```

## Implementation Priority

### Week 1: Core Volume Features
- [ ] Add FGA rolling averages
- [ ] Add 3PA rolling averages
- [ ] Add FTA rolling averages
- [ ] Add per-minute versions
- [ ] Test on points model only

**Expected**: +1.5-2% for points

### Week 2: Efficiency Rates
- [ ] Calculate True Shooting %
- [ ] Calculate Usage Rate (requires team totals)
- [ ] Add season vs. rolling versions
- [ ] Test on all prop types

**Expected**: +1-2% additional improvement

### Week 3: Rebound & Assist Rates
- [ ] Calculate rebound rates (TRB%, ORB%, DRB%)
- [ ] Calculate assist rate
- [ ] Calculate assist-to-turnover ratio
- [ ] Test on REB and AST models

**Expected**: +2-3% for REB, +1-2% for AST

### Week 4: Shot Selection & Matchups
- [ ] Add 3PAr and FTr
- [ ] Add opponent defensive metrics
- [ ] Add rest/fatigue features
- [ ] Full backtest on all improvements

**Expected**: +0.5-1% additional improvement

## Total Expected Impact

If all phases successful:
- **Points**: +3-5% RMSE improvement
- **3PM**: +3-4% RMSE improvement
- **Rebounds**: +3-4% RMSE improvement
- **Assists**: +2-3% RMSE improvement

Combined with your current +0.5% from enhanced selector:
- **Overall improvement: +3.5-5.5% vs baseline**

This would be a **very strong** edge in player prop betting.

## Data Requirements

### Already Have ✅
- Player box scores with FGA, 3PA, FTA, REB, AST, TOV
- Game dates (for rest days)
- Team stats
- Historical data (2002-2025)

### Need to Calculate ⚙️
- Team totals per game (for usage rate)
- Opponent totals per game (for rebound rate)
- Rolling averages

### Nice to Have (Future) 🎯
- Player positions
- Injury reports
- Lineup data (for teammate impact)

## Technical Notes

### Feature Storage
Add new features to training data in `train_auto.py`:
1. Calculate rolling stats in preprocessing
2. Add to feature matrix before LightGBM training
3. Update `riq_analyzer.py` to calculate same features at inference

### Feature Count
Currently: 20 features
After Phase 1-2: 35-40 features
After Phase 3: 45-50 features

LightGBM handles this well, but watch for overfitting on smaller windows.

### Validation
Use same true backtest methodology:
- Train on historical windows
- Test on 2025 data
- Compare RMSE improvements
- Ensure gains are consistent across all stat types

## Next Steps

1. **Start with Phase 1.1** (shot volume) - easiest to implement, biggest impact
2. **Run quick test** on 2017-2021 window for points only
3. **If successful** (+1%+), proceed with full Phase 1
4. **Then move to Phase 2** once Phase 1 validated

Would you like me to start implementing Phase 1.1 (shot volume features)?
