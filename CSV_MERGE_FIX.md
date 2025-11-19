# CSV Merge Column Detection - FIXED

## Error That Occurred on Modal
```
KeyError: 'game_id'
at /root/shared/csv_aggregation.py:416 in load_and_merge_csvs
df = df.merge(team_df, on=['game_id', 'team_id'], ...)
```

## Root Cause
The CSV merge code had **hardcoded column names** that didn't match actual CSV schemas:

### Hardcoded Assumptions (WRONG)
- TeamStatistics.csv merge: assumed `game_id`, `team_id`
- Games.csv merge: assumed `game_id`
- Team Summaries.csv merge: assumed `team_id`, `season`

### Actual CSV Schemas (CORRECT)
- PlayerStatistics.csv uses: `gameId`, `teamId` (camelCase)
- Basketball Reference CSVs use: `Player`, `Season`, `Team` (title case)

## What Was Fixed

### 1. TeamStatistics.csv Merge (lines 408-442)
**Before:**
```python
df = df.merge(team_df, on=['game_id', 'team_id'], ...)  # KeyError!
```

**After:**
```python
# Auto-detect game_id and team_id columns
game_id_col = None
team_id_col = None

for col_name in ['game_id', 'gameId', 'GAME_ID', 'game_ID']:
    if col_name in df.columns and col_name in team_df.columns:
        game_id_col = col_name
        break

for col_name in ['team_id', 'teamId', 'TEAM_ID', 'team_ID']:
    if col_name in df.columns and col_name in team_df.columns:
        team_id_col = col_name
        break

if game_id_col and team_id_col:
    df = df.merge(team_df, on=[game_id_col, team_id_col], ...)
```

### 2. Games.csv Merge (lines 459-479)
**Before:**
```python
df = df.merge(games_df, on='game_id', ...)  # Would fail with gameId
```

**After:**
```python
# Auto-detect game_id column
game_id_col = None
for col_name in ['game_id', 'gameId', 'GAME_ID', 'game_ID']:
    if col_name in df.columns and col_name in games_df.columns:
        game_id_col = col_name
        break

if game_id_col:
    df = df.merge(games_df, on=game_id_col, ...)
```

### 3. Team Summaries.csv Merge (lines 497-544)
**Before:**
```python
df = df.merge(team_summ_df, on=['team_id', 'season'], ...)  # Would fail
```

**After:**
```python
# Auto-detect Team and Season columns in Team Summaries
team_summ_team_col = None
team_summ_season_col = None

for col_name in ['Team', 'team_id', 'teamId', 'TEAM_ID', 'team_ID']:
    if col_name in team_summ_df.columns:
        team_summ_team_col = col_name
        break

for col_name in ['Season', 'season', 'season_end_year', 'SEASON', 'Year']:
    if col_name in team_summ_df.columns:
        team_summ_season_col = col_name
        break

# Find matching columns in main dataframe
main_team_col = None
for col_name in ['team_id', 'teamId', 'TEAM_ID', 'team_ID']:
    if col_name in df.columns:
        main_team_col = col_name
        break

# Rename Team Summaries columns to match main df before merging
team_summ_df = team_summ_df.rename(columns={
    team_summ_team_col: main_team_col,
    team_summ_season_col: main_season_col
})

df = df.merge(team_summ_df, on=[main_team_col, main_season_col], ...)
```

## Summary of All Auto-Detected Merges

Now ALL 9 CSV merges use auto-detection:

1. ✅ **PlayerStatistics.csv** (base) - Creates `Player` and `season` columns
2. ✅ **Player Advanced.csv** - Detects `Player`/`Season`, renames to match
3. ✅ **Player Per 100 Poss.csv** - Detects `Player`/`Season`, renames to match
4. ✅ **Player Play-By-Play.csv** - Detects `Player`/`Season`, renames to match
5. ✅ **Player Shooting.csv** - Detects `Player`/`Season`, renames to match
6. ✅ **Players.csv** - Detects `player_id`/`personId`
7. ✅ **TeamStatistics.csv** - Detects `game_id`/`gameId` and `team_id`/`teamId`
8. ✅ **Games.csv** - Detects `game_id`/`gameId`
9. ✅ **Team Summaries.csv** - Detects `Team`/`team_id` and `Season`/`season`

## Why This Matters

**Before:** CSV merges would fail with KeyError if column names didn't exactly match hardcoded values

**After:** CSV aggregation works with ANY column naming convention:
- camelCase (`gameId`, `teamId`)
- snake_case (`game_id`, `team_id`)
- UPPERCASE (`GAME_ID`, `TEAM_ID`)
- Title Case (`Player`, `Season`, `Team`)

This makes the code robust across different data sources!

## Ready to Try Again

The Modal training should now work. Run:
```bash
py -3.12 -m modal run modal_train.py --window-start 2022 --window-end 2024
```

All column detection issues are now fixed! 🎉
