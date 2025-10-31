# Memory Optimization Opportunities - Season-Based Filtering

## Current Memory Usage Overview

Your training pipeline loads these datasets:

| Dataset | Source | Rows | Size | Season Range | Can Filter? |
|---------|--------|------|------|--------------|-------------|
| **TeamStatistics.csv** | Kaggle | 144k | ~40 MB | 1946-2026 | ✅ **YES** |
| **PlayerStatistics.csv** | Kaggle | 1.6M | ~300 MB | 1946-2026 | ✅ **YES** |
| **Games.csv** | Kaggle | 72k | ~15 MB | 1946-2026 | ✅ **YES** |
| **Team Priors** | Basketball Ref | 1.7k | ~300 KB | 1950-2025 | ✅ **DONE** (in code) |
| **Player Priors (4 CSVs)** | Basketball Ref | 185k | ~12 MB | 1950-2025 | ✅ **DONE** (now filtered!) |
| **Betting Odds** | Kaggle | varies | ~100 MB | 2007-2024 | ✅ **DONE** (disabled for historical) |

**Total memory with all data:** ~500-600 MB
**Total memory with 2002-2026 only:** ~200-300 MB (50% reduction!)

---

## Optimization Opportunities (Ranked by Impact)

### 🔥 HIGH IMPACT - Already Implemented

#### 1. ✅ Player Priors Season Filtering (Line 1851-1858)
**Status:** DONE (just implemented!)
**Impact:** 75% reduction (185k → 35k rows)
**Memory saved:** ~9 MB
**Code location:** train_auto.py lines 1851-1858

```python
# Filter priors to only seasons in player data (2002-2026)
ps_seasons = set(ps_join["season_end_year"].dropna().unique())
priors_players = priors_players[priors_players["season_for_game"].isin(ps_seasons)]
```

#### 2. ✅ Betting Odds Disabled for Historical Windows (Line 3077)
**Status:** DONE
**Impact:** Skips loading ~100 MB of odds data for 2002-2021 windows
**Memory saved:** ~100 MB per window
**Code location:** train_auto.py line 3077

#### 3. ✅ Games Season Filtering (Line 3616)
**Status:** DONE
**Impact:** Filters games from 72k (1946-2026) to ~32k (2002-2026)
**Memory saved:** ~55% of games data
**Code location:** train_auto.py line 3616

---

### 🚀 HIGH IMPACT - NOT YET IMPLEMENTED

#### 4. TeamStatistics.csv Season Filtering at Load Time
**Status:** ❌ NOT IMPLEMENTED
**Current:** Loads all 144k rows (1946-2026), then filters later
**Opportunity:** Filter during CSV read using pandas chunksize or date range
**Memory saved:** ~55% (144k → 65k rows, ~22 MB → ~10 MB)
**Complexity:** Medium

**How to implement:**

```python
# BEFORE (line 1104 - current)
ts = pd.read_csv(teams_path, usecols=usecols, dtype=dtype_spec, parse_dates=[date_c])

# AFTER (proposed optimization)
ts = pd.read_csv(teams_path, usecols=usecols, dtype=dtype_spec, parse_dates=[date_c])
# Filter immediately after load, before any processing
if "season" in ts.columns:
    ts = ts[ts["season"] >= 2002].copy()
elif date_c in ts.columns:
    # If no season column, filter by date
    ts = ts[ts[date_c] >= "2002-01-01"].copy()
```

**Lines to modify:** 1104-1130 in `build_games_from_teamstats()`

---

#### 5. PlayerStatistics.csv Season Filtering at Load Time
**Status:** ❌ NOT IMPLEMENTED
**Current:** Loads all 1.6M rows (1946-2026), filters at line 3937
**Opportunity:** Filter during CSV read
**Memory saved:** ~55% (1.6M → 833k rows, ~300 MB → ~135 MB)
**Complexity:** Easy (filter already exists, just move it earlier)

**Current filtering location:**
```python
# Line 3937 - CURRENT (filters AFTER loading all data)
- minutes: filtered by season >= 2002: 1,472,634 -> 692,792
- points: filtered by season >= 2002: 1,635,306 -> 833,839
```

**How to implement:**

```python
# BEFORE (line 1602 - current)
ps = pd.read_csv(player_path, low_memory=False, usecols=sorted(set(usecols)))

# AFTER (proposed optimization)
# Option 1: Filter after initial load (simplest)
ps = pd.read_csv(player_path, low_memory=False, usecols=sorted(set(usecols)))
if "season_end_year" in ps.columns:
    ps = ps[ps["season_end_year"] >= 2002].copy()

# Option 2: Use chunksize for even less memory (advanced)
chunks = []
for chunk in pd.read_csv(player_path, low_memory=False, usecols=sorted(set(usecols)), chunksize=50000):
    if "season_end_year" in chunk.columns:
        chunk = chunk[chunk["season_end_year"] >= 2002]
    chunks.append(chunk)
ps = pd.concat(chunks, ignore_index=True)
```

**Lines to modify:** 1602-1610 in `build_player_frames()`

---

#### 6. Team Priors Early Filtering
**Status:** ⚠️ PARTIALLY DONE (filters during merge, could filter earlier)
**Current:** Loads all 1.7k team-seasons, filters during merge
**Opportunity:** Filter immediately after load in `load_basketball_reference_priors()`
**Memory saved:** Small (~200 KB, but cleaner code)
**Complexity:** Easy

**How to implement:**

```python
# In load_basketball_reference_priors() function around line 2540
# AFTER loading Team Summaries.csv
priors_teams = pd.read_csv(team_summaries_path, low_memory=False)

# ADD THIS: Filter to 2002+ immediately
if "season" in priors_teams.columns:
    priors_teams = priors_teams[priors_teams["season"] >= 2002].copy()
```

**Lines to modify:** 2540-2560 in `load_basketball_reference_priors()`

---

### 💡 MEDIUM IMPACT

#### 7. Window-Specific Data Loading
**Status:** ❌ NOT IMPLEMENTED (architectural change)
**Current:** Load all 2002-2026 data, then train 5 windows sequentially
**Opportunity:** Load ONLY the 5 years needed for each window
**Memory saved:** 80% per window (only need 5 years at a time, not all 25)
**Complexity:** High (major architectural change)

**Example:**
```python
# CURRENT: Load all games 2002-2026 (32k games in memory)
games_df = load_all_games(2002, 2026)  # 32k games

# Train window 2002-2006 (uses all 32k for context)
# Train window 2007-2011 (uses all 32k for context)
# ...

# PROPOSED: Load only window's games + rolling context
for window in [(2002, 2006), (2007, 2011), ...]:
    # Load this window's games + 1 year prior for rolling stats
    window_games = load_games(window[0] - 1, window[1])  # ~7k games
    train_ensemble(window_games)
    del window_games  # Free memory before next window
```

**Why this is complex:**
- Rolling stats need prior season context
- Elo ratings need full history to be accurate
- Current caching system expects full dataset

**Recommendation:** Only implement if RAM becomes a bottleneck

---

### 🔍 LOW IMPACT (Already Optimized or Marginal Gains)

#### 8. ✅ Fuzzy Player Matching Disabled (Line 1949)
**Status:** DONE
**Impact:** Prevents 124 MiB memory spikes
**Memory saved:** Varies (prevents OOM errors)

#### 9. ✅ Historical Odds Skipped for Windows (Line 3077)
**Status:** DONE
**Impact:** Doesn't load odds CSV for 2002-2021 windows
**Memory saved:** ~100 MB per window

#### 10. Column Selection (usecols)
**Status:** ✅ ALREADY DONE throughout codebase
**Current:** Only loads needed columns via `usecols` parameter
**Example:** Line 1104, 1602, etc.
**Memory saved:** Already optimized

---

## Implementation Priority

### Phase 1: Quick Wins (Already Done! ✅)
1. ✅ Player priors season filtering (just implemented)
2. ✅ Games season filtering for ensemble windows
3. ✅ Betting odds disabled for historical windows
4. ✅ Fuzzy matching disabled

### Phase 2: High-Impact Optimizations (Recommended Next)
5. **TeamStatistics.csv early filtering** (lines 1104-1130)
   - Impact: Save ~22 MB, 55% of team data
   - Complexity: Low (2-3 lines of code)
   - Risk: Low (data already filtered later anyway)

6. **PlayerStatistics.csv early filtering** (lines 1602-1610)
   - Impact: Save ~165 MB, 55% of player data
   - Complexity: Low (2-3 lines of code)
   - Risk: Low (data already filtered at line 3937 anyway)

### Phase 3: Diminishing Returns (Optional)
7. Team priors early filtering (line 2540)
   - Impact: Save ~200 KB
   - Complexity: Low
   - Priority: Low (already pretty fast)

8. Window-specific data loading (major refactor)
   - Impact: Save 80% memory per window
   - Complexity: High (architectural change)
   - Priority: Only if RAM becomes bottleneck

---

## Recommended Implementation Order

**Immediate (5 minutes to implement):**

```python
# 1. Add to line 1110 (after loading TeamStatistics.csv)
if "season" in ts.columns:
    orig_len = len(ts)
    ts = ts[ts["season"] >= 2002].copy()
    if verbose:
        log(f"  Filtered TeamStatistics: {orig_len:,} → {len(ts):,} rows (2002+)", True)

# 2. Add to line 1608 (after loading PlayerStatistics.csv)
if "season_end_year" in ps.columns:
    orig_len = len(ps)
    ps = ps[ps["season_end_year"] >= 2002].copy()
    if verbose:
        log(f"  Filtered PlayerStatistics: {orig_len:,} → {len(ps):,} rows (2002+)", True)
```

**Expected memory savings:** ~200 MB total

---

## Current vs Optimized Memory Usage

### Before Any Optimizations (Baseline)
```
TeamStatistics:     144k rows × 40 MB
PlayerStatistics:   1.6M rows × 300 MB
Games:              72k rows × 15 MB
Team Priors:        1.7k rows × 300 KB
Player Priors:      185k rows × 12 MB
Betting Odds:       varies × 100 MB
─────────────────────────────────────
TOTAL:              ~470 MB in RAM
```

### After Phase 1 (Current State - Already Done!)
```
TeamStatistics:     144k rows × 40 MB (not filtered yet)
PlayerStatistics:   1.6M rows × 300 MB (not filtered yet)
Games:              32k rows × 7 MB ✅ (filtered)
Team Priors:        ~700 rows × 130 KB ✅ (filtered during merge)
Player Priors:      ~35k rows × 3 MB ✅ (NOW filtered!)
Betting Odds:       0 rows × 0 MB ✅ (skipped for historical)
─────────────────────────────────────
TOTAL:              ~350 MB in RAM (25% reduction!)
```

### After Phase 2 (Recommended - Easy to Implement)
```
TeamStatistics:     65k rows × 18 MB ✅ (filter at load)
PlayerStatistics:   833k rows × 135 MB ✅ (filter at load)
Games:              32k rows × 7 MB ✅ (already filtered)
Team Priors:        ~700 rows × 130 KB ✅ (already filtered)
Player Priors:      ~35k rows × 3 MB ✅ (already filtered)
Betting Odds:       0 rows × 0 MB ✅ (already skipped)
─────────────────────────────────────
TOTAL:              ~165 MB in RAM (65% reduction vs baseline!)
```

### After Phase 3 (Optional - Window-Specific Loading)
```
Per-Window Memory Usage:
  TeamStatistics:   ~13k rows × 3.5 MB (5 years only)
  PlayerStatistics: ~167k rows × 27 MB (5 years only)
  Games:            ~6.4k rows × 1.4 MB (5 years only)
  Priors:           ~7k rows × 600 KB (5 years only)
  Betting Odds:     0 MB (skipped)
─────────────────────────────────────
TOTAL per window:   ~33 MB in RAM (93% reduction vs baseline!)
```

---

## Summary

**Already Optimized (Phase 1):** ✅
- Games filtering (55% saved)
- Player priors filtering (75% saved)
- Betting odds disabled (100 MB saved per window)
- Fuzzy matching disabled (prevents OOM)

**Next Steps (Phase 2):** 🚀
- Add 2-3 lines to filter TeamStatistics at load → Save 22 MB
- Add 2-3 lines to filter PlayerStatistics at load → Save 165 MB
- **Total effort:** 5 minutes
- **Total savings:** ~200 MB (additional 40% reduction)

**Optional (Phase 3):** 💡
- Window-specific loading (architectural change)
- Only implement if RAM becomes a bottleneck
- Requires significant refactoring

**Bottom Line:**
- Current optimizations: ✅ 25% memory reduction (done!)
- Quick Phase 2 optimizations: 🚀 65% total reduction (5 minutes to implement)
- Full Phase 3 optimization: 💡 93% reduction (only if needed, high effort)
