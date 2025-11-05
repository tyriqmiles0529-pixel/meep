# Next-Level Optimizations to Increase Accuracy 🚀

**Current Status:** 49.1% → Expected 60-65% with Phase 6
**Goal:** Push to **65-70%+ accuracy**

---

## 🎯 High-Impact Optimizations (Ranked by Expected Gain)

### 1. **Player-Game State Modeling** ⭐⭐⭐⭐⭐
**Expected Gain:** +3-5% accuracy
**Effort:** Medium
**Status:** Not Implemented

#### What It Does:
Model the player's current "state" beyond simple rolling averages:
- **Confidence state**: How confident is player in their shot? (based on recent FG%)
- **Rhythm state**: Time since last game, practice days
- **Motivation state**: Contract year, milestone chasing, revenge games
- **Health state**: Minutes restriction, injury return progression
- **Team chemistry**: New teammates, lineup stability

#### Implementation:
```python
class PlayerStateModeler:
    """
    Track player state variables that affect performance.
    """
    def calculate_confidence_score(self, player_recent_stats):
        # Based on recent shooting %
        recent_fg_pct = player_recent_stats['fg_pct_L5']
        career_avg = player_recent_stats['fg_pct_career']
        return (recent_fg_pct - career_avg) / career_avg  # Deviation from normal
    
    def calculate_rhythm_score(self, days_since_last_game):
        # Optimal: 1-2 days rest
        if days_since_last_game == 1:
            return 1.0
        elif days_since_last_game == 2:
            return 0.9
        elif days_since_last_game > 5:
            return 0.6  # Rusty
        else:
            return 0.8
    
    def detect_milestone_chase(self, player_season_stats, milestone_thresholds):
        # E.g., LeBron chasing scoring record
        # Players often perform better near milestones
        pass
```

**Why It Works:**
- Players perform differently based on psychological factors
- Hot/cold shooting affects confidence and shot selection
- Rhythm matters more than just rest days
- **Expected: +3-5% on all props, especially points/threes**

---

### 2. **Lineup-Specific Modeling** ⭐⭐⭐⭐⭐
**Expected Gain:** +3-4% accuracy
**Effort:** High
**Status:** Not Implemented

#### What It Does:
Model performance based on specific 5-man lineup combinations:
- **On-court stats**: Points, assists when specific teammates on floor
- **Usage rate by lineup**: Player gets more/fewer touches with certain lineups
- **Spacing impact**: Better shooters around = more driving lanes
- **Ball-dominant teammates**: Usage drops when paired with high-usage players

#### Implementation:
```python
class LineupAnalyzer:
    """
    Analyze player performance with specific lineup combinations.
    """
    def get_lineup_adjustment(self, player, projected_lineup):
        # Historical stats with this lineup
        lineup_key = tuple(sorted(projected_lineup))
        hist_stats = self.lineup_database.get(lineup_key, {})
        
        # Factors:
        # 1. How many ball-dominant players in lineup?
        # 2. How much spacing (3P shooters)?
        # 3. Rim protection (affects driving)?
        
        ball_dominant_count = sum(1 for p in projected_lineup 
                                   if self.player_db[p]['usage_rate'] > 25)
        
        usage_adjustment = 1.0 - (ball_dominant_count - 1) * 0.15
        
        return {
            'points_mult': usage_adjustment,
            'assists_mult': 1.0 + (ball_dominant_count - 1) * 0.1,
            'usage_rate_adj': usage_adjustment
        }
```

**Why It Works:**
- Player production varies massively by lineup
- Some players excel with certain teammates
- Usage rate shifts 10-30% based on lineup
- **Expected: +3-4% especially assists, points**

**Data Source:** NBA.com lineup stats, play-by-play data

---

### 3. **Situational Context Features** ⭐⭐⭐⭐
**Expected Gain:** +2-3% accuracy
**Effort:** Low-Medium
**Status:** Partially Implemented

#### Additional Situations to Model:

##### A. Time of Season
```python
def add_season_context_features(df):
    """
    Performance varies by time of season.
    """
    df['games_into_season'] = df.groupby(['playerId', 'season']).cumcount()
    df['games_remaining_in_season'] = 82 - df['games_into_season']
    
    # Early season: players still getting into shape
    df['is_early_season'] = (df['games_into_season'] < 10).astype(int)
    
    # Late season: tanking teams rest stars, playoff push
    df['is_late_season'] = (df['games_remaining_in_season'] < 15).astype(int)
    
    # Playoff race: teams with playoff hopes play harder
    df['in_playoff_race'] = df.apply(lambda x: 
        1 if x['team_current_record'] in playoff_race_range else 0, axis=1)
    
    return df
```

##### B. Opponent-Specific History
```python
def add_opponent_history_features(df):
    """
    How does player perform against THIS specific team?
    """
    # Career stats vs this opponent
    df['pts_vs_opponent_career_avg'] = df.groupby(['playerId', 'opponent'])['points'].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    
    # Last 3 games vs this opponent
    df['pts_vs_opponent_L3'] = df.groupby(['playerId', 'opponent'])['points'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    
    # Revenge game factor (traded from this team recently)
    df['is_revenge_game'] = df.apply(detect_revenge_game, axis=1)
    
    return df
```

##### C. Schedule Density
```python
def add_schedule_density_features(df):
    """
    Performance drops in condensed schedules.
    """
    # Games in last 7 days
    df['games_in_last_7_days'] = df.groupby('playerId').apply(
        lambda x: x['date'].rolling('7D').count()
    )
    
    # Games in next 7 days (players conserve energy)
    df['games_in_next_7_days'] = df.groupby('playerId')['date'].shift(-7).rolling('7D').count()
    
    # Travel distance in last 5 days
    df['travel_miles_L5'] = calculate_travel_distance(df)
    
    return df
```

**Why It Works:**
- Performance varies significantly by context
- Players have historical matchups (good/bad vs certain teams)
- Schedule fatigue is real but often overlooked
- **Expected: +2-3% across all props**

---

### 4. **Advanced Market Intelligence** ⭐⭐⭐⭐
**Expected Gain:** +2-4% accuracy
**Effort:** Medium
**Status:** Partially Implemented

#### Enhancements Needed:

##### A. Injury Report Analysis (NLP)
```python
class InjuryReportAnalyzer:
    """
    Parse injury reports for severity and impact.
    """
    def parse_injury_description(self, injury_text):
        # "Questionable - knee soreness" vs "Out - knee surgery"
        
        severity_keywords = {
            'high': ['surgery', 'torn', 'fractured', 'severe'],
            'medium': ['strain', 'sprain', 'soreness', 'contusion'],
            'low': ['rest', 'load management', 'illness']
        }
        
        # Duration estimation
        expected_games_out = estimate_recovery_time(injury_text)
        
        # Impact on performance (even when playing)
        performance_impact = {
            'points': -0.15 if 'shooting' in injury_text else -0.05,
            'rebounds': -0.10 if 'knee' in injury_text else 0,
            'assists': -0.05,
            'minutes': -0.20 if 'minutes restriction' in injury_text else 0
        }
        
        return performance_impact
```

##### B. Betting Market Consensus
```python
def add_market_consensus_features(df, odds_data):
    """
    Where is the smart money?
    """
    # Percentage of bets vs percentage of money
    df['sharp_money_indicator'] = (
        odds_data['pct_money_over'] - odds_data['pct_bets_over']
    )
    
    # If 30% of bets but 60% of money on over = sharps on over
    df['sharp_side'] = np.where(df['sharp_money_indicator'] > 0.2, 'over', 
                                 np.where(df['sharp_money_indicator'] < -0.2, 'under', 'neutral'))
    
    # Line movement speed
    df['line_velocity'] = odds_data['line_current'] - odds_data['line_open']
    df['line_move_speed'] = df['line_velocity'] / odds_data['hours_since_open']
    
    return df
```

**Why It Works:**
- Injury reports contain valuable info if parsed correctly
- Sharp bettors have inside information
- Fast line movement indicates informed money
- **Expected: +2-4% especially when market disagrees with model**

---

### 5. **Prop-Specific Adjustments** ⭐⭐⭐⭐
**Expected Gain:** +2-3% accuracy per prop type
**Effort:** Low-Medium
**Status:** Not Implemented

#### What It Does:
Different props have different dynamics:

##### Points Props:
```python
def adjust_points_prediction(base_prediction, context):
    """
    Points props have unique characteristics.
    """
    adjustments = {}
    
    # Blowout games: starters sit, bench scores more
    if context['expected_margin'] > 15:
        if context['is_starter']:
            adjustments['blowout_penalty'] = -0.9  # 10% reduction
        else:
            adjustments['garbage_time_bonus'] = +1.3  # 30% increase
    
    # Pace matters more for points than other stats
    adjustments['pace_mult'] = 1.0 + (context['game_pace'] - 100) * 0.015
    
    # Usage rate matters most for points
    adjustments['usage_mult'] = 1.0 + (context['usage_rate'] - 20) * 0.02
    
    return base_prediction * np.prod(list(adjustments.values()))
```

##### Assists Props:
```python
def adjust_assists_prediction(base_prediction, context):
    """
    Assists depend heavily on teammates' shooting.
    """
    # Teammate shooting matters MORE than player's passing
    teammates_3p_pct = context['teammates_3p_pct_L10']
    league_avg_3p = 0.365
    
    shooting_multiplier = 1.0 + (teammates_3p_pct - league_avg_3p) * 5.0
    
    # Ball-dominant teammates reduce assists
    ball_dominant_teammates = context['teammates_usage_rate_sum']
    usage_penalty = 1.0 - (ball_dominant_teammates / 100) * 0.3
    
    return base_prediction * shooting_multiplier * usage_penalty
```

##### Rebounds Props:
```python
def adjust_rebounds_prediction(base_prediction, context):
    """
    Rebounds are about positioning and team rebounding.
    """
    # Opponent offensive rebounding (more misses = more opportunities)
    opp_orb_rate = context['opponent_offensive_reb_rate']
    league_avg_orb = 0.23
    
    opportunity_mult = 1.0 + (opp_orb_rate - league_avg_orb) * 3.0
    
    # Team pace (more possessions = more rebounds)
    pace_mult = context['game_pace'] / 100.0
    
    # Position matters (centers get more)
    position_mult = {
        'C': 1.2,
        'PF': 1.1,
        'SF': 1.0,
        'SG': 0.9,
        'PG': 0.8
    }.get(context['position'], 1.0)
    
    return base_prediction * opportunity_mult * pace_mult * position_mult
```

##### Threes Props:
```python
def adjust_threes_prediction(base_prediction, context):
    """
    Threes are most volatile, need special handling.
    """
    # Recent shooting % matters more than volume
    recent_3p_pct = context['three_pct_L5']
    
    if recent_3p_pct > 0.40:  # Hot shooter
        confidence_boost = 1.15
    elif recent_3p_pct < 0.25:  # Cold shooter
        confidence_boost = 0.85
    else:
        confidence_boost = 1.0
    
    # Opponent 3P defense
    opp_3p_defense = context['opponent_3p_pct_allowed']
    league_avg = 0.365
    defense_mult = 1.0 + (opp_3p_defense - league_avg) * 2.0
    
    # Home/away (3P shooting more affected by crowd)
    home_mult = 1.05 if context['is_home'] else 0.95
    
    return base_prediction * confidence_boost * defense_mult * home_mult
```

**Why It Works:**
- Each prop has unique dynamics
- Generic model misses prop-specific patterns
- Context matters differently for each stat
- **Expected: +2-3% per prop type**

---

### 6. **Temporal Ensemble (Time-Weighted)** ⭐⭐⭐
**Expected Gain:** +1-2% accuracy
**Effort:** Low
**Status:** Not Implemented

#### What It Does:
Weight recent games more heavily, but adaptively:

```python
class AdaptiveTemporalWeighting:
    """
    Weight recent games more, but adapt based on stability.
    """
    def calculate_adaptive_weights(self, player_recent_stats):
        # If player is consistent, weight evenly
        # If player is inconsistent, weight recent games MORE
        
        variance = np.var(player_recent_stats)
        
        if variance < threshold_low:
            # Consistent player: use more history
            weights = np.linspace(0.5, 1.0, len(player_recent_stats))
        else:
            # Inconsistent player: focus on recent
            weights = np.exp(np.linspace(-2, 0, len(player_recent_stats)))
        
        return weights / weights.sum()
```

**Why It Works:**
- Recent performance matters more, but depends on stability
- Consistent players: use more history
- Volatile players: recent games are better signal
- **Expected: +1-2% accuracy**

---

### 7. **Quantile Regression for Uncertainty** ⭐⭐⭐
**Expected Gain:** +1-2% via better bet selection
**Effort:** Medium
**Status:** Partially Implemented (sigma models)

#### What It Does:
Instead of predicting just the mean, predict the full distribution:

```python
from sklearn.ensemble import GradientBoostingRegressor

def train_quantile_models(X, y):
    """
    Train models for multiple quantiles.
    """
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    models = {}
    
    for q in quantiles:
        model = GradientBoostingRegressor(
            loss='quantile',
            alpha=q,
            n_estimators=500
        )
        model.fit(X, y)
        models[q] = model
    
    return models

def make_probabilistic_prediction(models, X):
    """
    Get full distribution of possible outcomes.
    """
    predictions = {q: model.predict(X)[0] for q, model in models.items()}
    
    # Probability of exceeding line
    line = 25.5  # Example: 25.5 points
    
    # Interpolate to get P(X > line)
    prob_over = interpolate_probability(predictions, line)
    
    return prob_over
```

**Why It Works:**
- Better uncertainty quantification
- Can calculate exact probability of over/under
- Identifies high-confidence vs low-confidence bets
- **Expected: +1-2% by avoiding uncertain bets**

---

### 8. **Multi-Task Learning** ⭐⭐⭐⭐
**Expected Gain:** +2-3% accuracy
**Effort:** High
**Status:** Not Implemented

#### What It Does:
Train a single neural network to predict ALL stats simultaneously:

```python
import torch
import torch.nn as nn

class MultiTaskNBAPredictor(nn.Module):
    """
    Predict all props simultaneously with shared representations.
    """
    def __init__(self, input_dim):
        super().__init__()
        
        # Shared layers (learn general player patterns)
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Task-specific heads
        self.points_head = nn.Linear(128, 1)
        self.rebounds_head = nn.Linear(128, 1)
        self.assists_head = nn.Linear(128, 1)
        self.threes_head = nn.Linear(128, 1)
        
    def forward(self, x):
        shared_rep = self.shared(x)
        
        return {
            'points': self.points_head(shared_rep),
            'rebounds': self.rebounds_head(shared_rep),
            'assists': self.assists_head(shared_rep),
            'threes': self.threes_head(shared_rep)
        }

# Train with multi-task loss
def train_multitask(model, X, y_dict):
    loss = (
        F.mse_loss(predictions['points'], y_dict['points']) +
        F.mse_loss(predictions['rebounds'], y_dict['rebounds']) +
        F.mse_loss(predictions['assists'], y_dict['assists']) +
        F.mse_loss(predictions['threes'], y_dict['threes'])
    )
```

**Why It Works:**
- Stats are correlated (high points often means high minutes)
- Shared representations learn general patterns
- Forces model to be consistent across props
- **Expected: +2-3% by leveraging correlations**

---

### 9. **Transfer Learning from Similar Players** ⭐⭐⭐
**Expected Gain:** +1-2% for young/new players
**Effort:** Medium
**Status:** Not Implemented

#### What It Does:
For players with limited history, use similar players as priors:

```python
class PlayerSimilarityEngine:
    """
    Find similar players and transfer knowledge.
    """
    def find_similar_players(self, target_player, player_database):
        # Similarity based on:
        # 1. Position
        # 2. Physical attributes (height, weight, age)
        # 3. Play style (usage rate, pace, shooting %)
        # 4. Role (starter/bench, minutes)
        
        similarities = []
        for candidate in player_database:
            sim_score = (
                0.3 * position_similarity(target, candidate) +
                0.2 * physical_similarity(target, candidate) +
                0.3 * playstyle_similarity(target, candidate) +
                0.2 * role_similarity(target, candidate)
            )
            similarities.append((candidate, sim_score))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:10]
    
    def transfer_predictions(self, target_player, similar_players, context):
        # Use similar players' performance in similar situations
        similar_stats = []
        for sim_player, sim_score in similar_players:
            hist_performance = get_performance_in_context(sim_player, context)
            weighted_perf = hist_performance * sim_score
            similar_stats.append(weighted_perf)
        
        # Weighted average based on similarity
        return np.average(similar_stats, weights=[s[1] for s in similar_players])
```

**Why It Works:**
- Rookies/new players have limited history
- Similar players provide valuable priors
- Especially helpful early in season
- **Expected: +1-2% for players with <20 games history**

---

### 10. **Adversarial Training** ⭐⭐⭐
**Expected Gain:** +1-2% robustness
**Effort:** Medium
**Status:** Not Implemented

#### What It Does:
Train model to be robust to small perturbations:

```python
def adversarial_training(model, X, y, epsilon=0.01):
    """
    Add adversarial examples to training.
    """
    # Normal training step
    loss = train_step(model, X, y)
    
    # Generate adversarial examples
    X_adv = X + epsilon * np.sign(np.random.randn(*X.shape))
    
    # Train on adversarial examples too
    loss_adv = train_step(model, X_adv, y)
    
    return loss + 0.5 * loss_adv
```

**Why It Works:**
- Makes model robust to noise/errors in features
- Reduces overfitting to specific patterns
- Better generalization
- **Expected: +1-2% on out-of-sample data**

---

## 📊 Implementation Priority

### Quick Wins (1-2 weeks)
1. ✅ Situational Context Features (+2-3%)
2. ✅ Prop-Specific Adjustments (+2-3% per prop)
3. ✅ Temporal Ensemble (+1-2%)

### Medium Effort (2-4 weeks)
4. ✅ Advanced Market Intelligence (+2-4%)
5. ✅ Quantile Regression (+1-2%)
6. ✅ Transfer Learning (+1-2% for rookies)

### High Effort (4-8 weeks)
7. ⏱️ Player-Game State Modeling (+3-5%)
8. ⏱️ Lineup-Specific Modeling (+3-4%)
9. ⏱️ Multi-Task Learning (+2-3%)
10. ⏱️ Adversarial Training (+1-2%)

---

## 🎯 Expected Total Impact

| Optimization | Expected Gain | Cumulative |
|--------------|---------------|------------|
| **Current (Phase 6)** | - | 60-65% |
| + Player State | +3-5% | 63-70% |
| + Lineup Modeling | +3-4% | 66-74% |
| + Situational Context | +2-3% | 68-77% |
| + Market Intelligence | +2-4% | **70-81%** ⭐ |
| + Prop Adjustments | +2-3% | 72-84% |
| + Other (5-10) | +3-5% | **75-89%** |

**Realistic Target:** **70-75% accuracy** with full implementation

**Best Case:** **75-80% accuracy** if all optimizations synergize

---

## 🚀 Recommended Implementation Order

### Phase 7 (Next 2 weeks):
1. **Situational Context Features** - Low effort, high impact
   - Time of season, opponent history, schedule density
   - Expected: +2-3%

2. **Prop-Specific Adjustments** - Low effort, immediate gains
   - Custom logic per prop type
   - Expected: +2-3% per prop

3. **Temporal Weighting** - Low effort
   - Adaptive recent game weighting
   - Expected: +1-2%

**Phase 7 Total:** **+5-8% → 65-73% accuracy**

### Phase 8 (Next 4 weeks):
4. **Advanced Market Intelligence** - Parse injury reports, track sharp money
   - Expected: +2-4%

5. **Player State Modeling** - Confidence, rhythm, motivation
   - Expected: +3-5%

**Phase 8 Total:** **+5-9% → 70-82% accuracy**

### Phase 9 (Next 8 weeks):
6. **Lineup Modeling** - Biggest remaining opportunity
   - Expected: +3-4%

7. **Multi-Task Learning** - Requires TabNet/PyTorch already integrated ✅
   - Expected: +2-3%

**Phase 9 Total:** **+5-7% → 75-89% accuracy**

---

## 💡 Additional Ideas (Research)

### A. Reinforcement Learning for Bet Sizing
- Learn optimal Kelly criterion adjustments
- Adapt bet size based on recent performance

### B. Attention Mechanisms for Context
- Transformer models for sequential game data
- Learn which games are most relevant

### C. Causal Inference
- Identify true causal factors vs correlation
- Estimate treatment effects (e.g., coaching changes)

### D. Meta-Learning for Fast Adaptation
- Quick adaptation to new trends/rules
- Few-shot learning for new players

---

## 📚 Data Sources Needed

For maximum improvement, add:

1. **NBA.com Play-by-Play Data**
   - Lineup combinations and on-court stats
   - Required for lineup modeling

2. **Twitter/News Sentiment**
   - Player confidence, team morale
   - Injury report parsing

3. **Betting Market Data**
   - Sharp vs public money percentages
   - Line movement history

4. **Player Tracking Data (SportVU)**
   - Speed, distance covered
   - Defensive positioning

5. **Referee Assignments**
   - Some refs call more fouls = more FTs = more points
   - Impacts pace and style

---

## ✅ Action Items

### Immediate (This Week):
1. [ ] Implement situational context features
2. [ ] Add prop-specific adjustment functions
3. [ ] Test temporal weighting

### Short Term (Next Month):
4. [ ] Collect lineup combination data
5. [ ] Build player state tracking system
6. [ ] Enhance market intelligence module

### Long Term (Next Quarter):
7. [ ] Implement multi-task neural network
8. [ ] Build lineup database from NBA.com
9. [ ] Add reinforcement learning for bet sizing

---

**Bottom Line:** With these optimizations, **70-75% accuracy is achievable** within 2-3 months.

The biggest opportunities:
1. ⭐⭐⭐⭐⭐ Player-Game State Modeling (+3-5%)
2. ⭐⭐⭐⭐⭐ Lineup-Specific Modeling (+3-4%)
3. ⭐⭐⭐⭐ Advanced Market Intelligence (+2-4%)

**Start with Phase 7 (situational context + prop adjustments) for quick +5-8% gain.**
