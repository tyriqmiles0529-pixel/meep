# 📊 Complete NBA Data Sources - Actual Analysis

## Summary

You have **7 CSV files** with varying date ranges that need to be aggregated:

| File | Rows | Date Range | Granularity | Key Columns |
|------|------|------------|-------------|-------------|
| **PlayerStatistics.csv** | 1,632,909 | 1946-2025 | Game-level | gameDate, personId, points, rebounds, assists |
| **Advanced.csv** | 32,606 | 1947-2025 | Season-level | season, player_id, PER, TS%, Usage%, WS, BPM, VORP |
| **Per 100 Poss.csv** | 26,959 | **1974**-2025 | Season-level | season, player_id, per-100 stats |
| **Player Shooting.csv** | 17,521 | **1997**-2025 | Season-level | season, player_id, shooting splits |
| **Player Play By Play.csv** | 17,521 | **1997**-2025 | Season-level | season, player_id, on/off court stats |
| **Team Summaries.csv** | 1,876 | 1947-2025 | Season-level | season, team, W-L, SRS, Pace |
| **Team Abbrev.csv** | 1,788 | 1947-2025 | Season-level | season, team, abbreviation |

---

## Key Insights

### 1. **Different Start Years**
- **PlayerStatistics.csv**: 1946+ (79 years, FULL historical)
- **Advanced.csv**: 1947+ (PER, BPM, VORP, Win Shares)
- **Per 100 Poss.csv**: 1974+ (per-possession stats not available earlier)
- **Player Shooting.csv**: 1997+ (detailed shooting splits)
- **Player Play By Play.csv**: 1997+ (on/off court tracking)

### 2. **Granularity Mismatch**
- **PlayerStatistics.csv** = Game-by-game (1.6M rows)
- **All priors** = Season aggregates (17K-33K rows)

### 3. **Merge Strategy Required**
You need to merge season-level priors onto game-level data:

```python
# Example: Each game gets season context
Game on 2024-12-15: LeBron scores 28 points
+ Advanced.csv (2025 season): LeBron PER=24.5, Usage=31%
+ Per 100 Poss (2025 season): 30.2 PTS/100
+ Shooting (2025 season): 52.1% eFG%, 36.5% 3P%
→ Combined feature vector for that game
```

---

## Aggregation Requirements

### For Comprehensive Training (1946-2025):

**Full historical context:**
```
PlayerStatistics (1946-2025, 1.6M games)
├─ Merge Advanced stats (1947-2025) → PER, WS, BPM
├─ Merge Per 100 Poss (1974-2025) → Per-possession rates
├─ Merge Shooting splits (1997-2025) → eFG%, 3P zones
├─ Merge Play-by-Play (1997-2025) → On/off metrics
├─ Merge Team context (1947-2025) → Team SRS, Pace
└─ Result: Each game has full available context
```

**Missing data handling:**
- 1946-1973: No Per 100 Poss (fill with league averages)
- 1946-1996: No shooting splits (fill with basic FG%)
- Pre-1947: No Advanced stats (use basic box score stats)

---

## Monthly Update Strategy

### Current Situation:
- PlayerStatistics.csv: **Already has 1946-2025**
- Priors: **Already have 1947-2025**
- Both sources are COMPLETE

### For Monthly Updates:

**Option 1: Re-download from Kaggle**
```bash
# Downloads latest version with new games
kagglehub dataset download eoinamoore/historical-nba-data-and-player-box-scores
```

**Option 2: Incremental NBA API**
```python
# Add only new games since last update
from nba_api.stats.endpoints import LeagueGameLog
new_games = LeagueGameLog(season='2024-25',
                          date_from_nullable='2024-11-01')
# Append to PlayerStatistics.csv
```

**Option 3: Hybrid (RECOMMENDED)**
- **Monthly**: Re-download Kaggle dataset (gets everything)
- **Weekly**: Use NBA API for latest games (prediction only)
- **Training**: Use full Kaggle dataset monthly

---

## Aggregation Script Usage

### Analyze current data:
```bash
python aggregate_player_data.py --analyze-only
```

### Create aggregated dataset:
```bash
python aggregate_player_data.py \
  --player-csv PlayerStatistics.csv \
  --priors-dir priors_data \
  --output aggregated_player_data.csv
```

### Use in training:
```bash
python train_auto.py \
  --player-csv aggregated_player_data.csv \
  --priors-dataset priors_data \
  --player-season-cutoff 1974  # Or 1946 for full history
```

---

## Data Quality Notes

### ✅ What You Have:
- **Full historical game logs**: 1946-2025 (79 years!)
- **Advanced stats**: 1947-2025 (78 years)
- **Modern metrics**: 1997+ (shooting, on/off)
- **All 7 NBA eras** represented

### ⚠️ Limitations:
- **Per-100 possession**: Only 1974+ (ABA/NBA merger era)
- **Shooting zones**: Only 1997+ (tracking data era)
- **Play-by-play**: Only 1997+ (detailed tracking)

### 💡 Recommendation:
**Use 1974+ for training** to ensure all key stats available:
- Still have 51 years of data
- Include all modern eras (3-point line, pace-and-space)
- All advanced metrics available
- Fewer missing values to impute

**OR use 1946+ with intelligent imputation:**
- More historical context
- Fill missing advanced stats with league averages
- Use era-specific adjustments

---

## Next Steps

1. ✅ **Analyze**: Done - you have the complete picture above
2. ⏳ **Aggregate**: Merge all 7 sources into one comprehensive CSV
3. ⏳ **Update Training**: Modify train_auto.py to use aggregated data
4. ⏳ **Monthly Workflow**: Set up Kaggle re-download + incremental updates

Would you like me to:
- Complete the aggregation logic?
- Update the training notebooks?
- Create the monthly update workflow?
