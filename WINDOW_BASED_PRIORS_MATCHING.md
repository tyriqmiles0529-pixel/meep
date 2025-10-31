# Window-Based Player Priors Matching for 80%+ Match Rate

## Problem Analysis

Currently, player priors are loaded **ONCE** for ALL seasons (2002-2026):
- Line 2704-2796: Loads 185,226 player-seasons into memory
- Line 1898-2039: Merges all priors with all player data
- Line 1899-1907: Filters to relevant seasons AFTER loading (doesn't save memory)
- Line 2022: Skips fuzzy matching due to memory constraints

**Result:**
- High memory usage (all 24 seasons in memory at once)
- Fuzzy matching disabled (would cause OOM with 833k × 185k cross-join)
- Match rate limited to exact name + season matches only

---

## Solution: Per-Window Priors Loading

### Approach 1: Full Architectural Refactor (Complex, Best Long-Term)

**What it does:**
- Refactor `_build_player_datasets()` to accept a `season_filter` parameter
- Load priors CSVs with season filtering applied during CSV read
- Each 5-year window loads only its own priors (e.g., 2002-2006 loads only those 5 years)

**Benefits:**
- 80% memory reduction (5 years instead of 24 years)
- Can re-enable fuzzy matching (smaller dataset)
- 95-99% match rate for historical data

**Complexity:**
- Requires refactoring 6 functions
- Need to pass season range through multiple function calls
- Risk of breaking existing functionality
- Estimated effort: 4-6 hours

---

### Approach 2: Optimize Existing Fuzzy Matching (Simpler, Quick Win)

**What it does:**
- Keep current architecture (load all priors once)
- Optimize line 2022 fuzzy matching to work in chunks instead of full cross-join
- Process unmatched players in batches of 1000 instead of all at once

**Benefits:**
- Minimal code changes (20-30 lines)
- Can re-enable fuzzy matching
- +5-10% match rate improvement
- Low risk

**Complexity:**
- Simple loop optimization
- Estimated effort: 30 minutes

---

### Approach 3: Hybrid - Filter Priors Earlier (Medium Complexity)

**What it does:**
- Load priors CSVs with pre-filtering during CSV read (line 2719-2766)
- Add `usecols` and row filtering during `pd.read_csv()`
- Only load seasons that will actually be used

**Benefits:**
- 50-70% memory reduction (loads only 2002-2026 instead of 1974-2026)
- No architectural changes needed
- Can re-enable fuzzy matching

**Complexity:**
- Need to determine season range before loading priors
- Modify 4 CSV read statements
- Estimated effort: 1-2 hours

---

## RECOMMENDED: Approach 2 (Optimize Fuzzy Matching)

This gives you the **quickest path to 80%+ match rate** without major refactoring.

### Current Fuzzy Matching Code (DISABLED - Line 2022):

```python
if len(unmatched) > 0 and verbose:
    log(f"  Skipping fuzzy season match for {len(unmatched):,} unmatched rows (memory optimization)", True)
    # DISABLED: Fuzzy matching causes memory errors with large datasets
    # The +/- 1 season fallback creates Cartesian products that exhaust RAM
```

### Optimized Fuzzy Matching (CHUNKED):

```python
if len(unmatched) > 0:
    log(f"  Attempting fuzzy season match for {len(unmatched):,} unmatched rows (chunked processing)", verbose)

    # Process in chunks to avoid memory exhaustion
    CHUNK_SIZE = 1000  # Process 1000 unmatched players at a time
    num_chunks = (len(unmatched) + CHUNK_SIZE - 1) // CHUNK_SIZE

    fuzzy_matched_total = 0

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * CHUNK_SIZE
        end_idx = min((chunk_idx + 1) * CHUNK_SIZE, len(unmatched))
        chunk = unmatched.iloc[start_idx:end_idx]

        if verbose and chunk_idx % 10 == 0:
            log(f"    Processing chunk {chunk_idx+1}/{num_chunks} ({start_idx:,}-{end_idx:,})", True)

        # Try season +1
        chunk_plus = chunk.merge(
            priors_players[["__name_key__", "season_for_game"] + merge_cols],
            left_on=["__name_key__"],
            right_on=["__name_key__"],
            how="left",
            suffixes=("", "_fuzzy")
        )

        # Filter to rows where season_for_game is within +/- 1 of season_end_year
        valid_fuzzy = (
            (chunk_plus["season_for_game"] == chunk_plus["season_end_year"] + 1) |
            (chunk_plus["season_for_game"] == chunk_plus["season_end_year"] - 1)
        )
        chunk_plus = chunk_plus[valid_fuzzy]

        if len(chunk_plus) > 0:
            # Merge fuzzy matches back into ps_join
            fuzzy_cols = [c for c in merge_cols if f"{c}_fuzzy" in chunk_plus.columns]
            for col in fuzzy_cols:
                ps_join.loc[chunk_plus.index, col] = chunk_plus[f"{col}_fuzzy"]

            fuzzy_matched_total += len(chunk_plus)

    if verbose:
        log(f"    Fuzzy matching found {fuzzy_matched_total:,} additional matches (+{fuzzy_matched_total/len(ps_join)*100:.1f}%)", True)
```

---

## Implementation Plan

I'll implement **Approach 2** now, which will:

1. Re-enable fuzzy matching with chunked processing
2. Add +5-10% to match rate (from current 0% with bug, or 70-80% after fix → 80-90%)
3. Keep memory usage reasonable
4. Take ~30 minutes to implement and test

Then, if you want even more optimization later, we can do Approach 3 (filter priors earlier during CSV load).

---

## Expected Results

### Before Any Fixes:
- Match rate: 0.0% (bug in `_season_from_date`)
- Fuzzy matching: Disabled (memory constraints)

### After `_season_from_date` Fix Only:
- Match rate: 70-80% (exact name + season matches)
- Fuzzy matching: Still disabled

### After `_season_from_date` Fix + Chunked Fuzzy Matching:
- Match rate: **80-90%** (exact matches + fuzzy +/- 1 season matches)
- Fuzzy matching: Enabled (chunked to avoid OOM)
- Memory usage: Same as current (no increase)

---

## Alternative: If You Want Full Window-Based Loading

If you prefer Approach 1 (full refactor for per-window loading), I can implement that instead. It would:

- Require 4-6 hours of development
- Save 80% memory (5 years instead of 24 years per window)
- Allow even more aggressive fuzzy matching
- Achieve 95-99% match rate

But Approach 2 gets you to 80-90% in 30 minutes, so it's a better quick win.

---

## Which Approach Do You Want?

**Option A**: Chunked fuzzy matching (30 min, 80-90% match rate) ← RECOMMENDED
**Option B**: Filter priors during CSV load (1-2 hours, 80-90% match rate, 50% memory saved)
**Option C**: Full per-window priors loading (4-6 hours, 95-99% match rate, 80% memory saved)

Let me know which you prefer and I'll implement it!
