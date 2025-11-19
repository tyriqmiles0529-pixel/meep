# Backtest Engine - Implementation Plan

## Current State

**File:** `backtest_engine.py`
**Status:** 90% complete
**Missing:** Prediction generation (line 175)

---

## What's Missing (TODO at line 175)

The `generate_predictions()` method needs to:

1. **Iterate through dates** - Loop from start_date to end_date
2. **Get scheduled games** - Fetch games for each date
3. **Get player rosters** - For each game, get active players
4. **Engineer features** - Use logic from `predict_live_FINAL.py`
5. **Make predictions** - Call trained models
6. **Store results** - DataFrame with predictions

---

## Implementation Strategy

### Option 1: Reuse predict_live_FINAL.py Logic ⭐

**Advantages:**
- Already has all feature engineering
- Tested and working
- Loads aggregated data
- Builds dynamic features

**Implementation:**
```python
def generate_predictions(self, start_date, end_date):
    # Import from predict_live_FINAL
    from predict_live_FINAL import (
        load_aggregated_data,
        build_player_features,
        get_scheduled_games
    )

    # Load historical data
    df = load_aggregated_data()

    predictions = []

    # For each date in range
    for date in pd.date_range(start_date, end_date):
        # Get games scheduled for this date
        games = get_scheduled_games(date)

        # For each game
        for game in games:
            # Get rosters
            home_roster = get_roster(game['home_team'], date)
            away_roster = get_roster(game['away_team'], date)

            # For each player
            for player in home_roster + away_roster:
                # Build features using data BEFORE this date
                features = build_player_features(
                    player_id=player['id'],
                    game_date=date,
                    historical_data=df[df['date'] < date]
                )

                # Make predictions
                for prop in ['points', 'rebounds', 'assists', 'minutes', 'threes']:
                    if prop in self.models:
                        pred = self.models[prop].predict(features)
                        predictions.append({
                            'date': date,
                            'player': player['name'],
                            'prop': prop,
                            'predicted': pred
                        })

    return pd.DataFrame(predictions)
```

### Option 2: Use Cached Historical Predictions

**If you already ran predictions:**
```python
# Load from predict_live_FINAL.py output
predictions = pd.read_csv('predictions_history.csv')
predictions = predictions[
    (predictions['date'] >= start_date) &
    (predictions['date'] <= end_date)
]
```

---

## What We Need from predict_live_FINAL.py

Looking at `predict_live_FINAL.py`, we need these functions:

1. **load_aggregated_data()** - Load the aggregated CSV
2. **get_scheduled_games(date)** - Get games for a date
3. **build_player_features(player, date, data)** - Engineer features
4. **get_roster(team, date)** - Get active players

Let me check what exists...

---

## Next Steps

### Step 1: Review predict_live_FINAL.py
Check what functions are available to reuse.

### Step 2: Create Helper Functions
Extract reusable functions:
- `get_games_for_date(date)`
- `get_player_roster(team, date)`
- `engineer_features(player, date, historical_data)`

### Step 3: Implement generate_predictions()
Use the helpers to generate predictions for backtest period.

### Step 4: Add Actual Results Fetching
The `fetch_actual_results()` function also needs completion (line 88).

---

## Timeline

**While models train (~7 hours):**
1. Review predict_live_FINAL.py (15 min)
2. Extract helper functions (30 min)
3. Implement generate_predictions() (1 hour)
4. Implement fetch_actual_results() (30 min)
5. Test backtest on sample dates (30 min)

**Total: ~3 hours** (finishes before training)

---

## Questions to Answer

1. **Do you want to backtest on historical dates?**
   - E.g., October 2024 games
   - Requires fetching actual results from nba_api

2. **Or use cached predictions?**
   - If you have a predictions CSV from running predict_live_FINAL.py
   - Much faster, just load and analyze

3. **What date range to backtest?**
   - Last 30 days?
   - Entire 2024-25 season?
   - Specific period?

---

## Recommendation

**For now:** Complete the backtest engine implementation using predict_live_FINAL.py logic.

**After training finishes:** Use the trained models to backtest on recent games (Oct-Nov 2024).

This will show you:
- How accurate predictions are
- Where the model struggles
- Which props to bet on
- Optimal bet sizing

Want me to start implementing the prediction generation?
