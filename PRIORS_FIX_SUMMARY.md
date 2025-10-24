# Fix for Priors Terminology Issue

## Problem Statement
The user asked: **"why are you calling them priors? do you have the appropriate columns for all csv being used?"**

This highlighted confusion about:
1. What "priors" means in the context of the code
2. Whether the necessary columns exist in all CSVs being used

## Root Cause
The code used the term "priors" ambiguously to refer to both:
- **Actual Basketball Reference prior-season statistics** (when available from Team Summaries.csv)
- **Baseline default values** (when priors aren't available or don't merge)

The diagnostic output showed "0 games with priors" but didn't clearly explain:
- That this meant actual Basketball Reference data wasn't being used
- Why the merge failed (missing teamTricode column)
- How to fix it

## Solution Implemented

### 1. Clarified Terminology in Comments

**GAME_FEATURES** (lines 184-188):
```python
# Basketball Reference prior-season stats (optional - requires teamTricode column + --priors-dataset)
# These columns hold actual prior-season stats when available, or baseline defaults when not
"home_o_rtg_prior", "home_d_rtg_prior", "home_pace_prior",
"away_o_rtg_prior", "away_d_rtg_prior", "away_pace_prior",
"home_srs_prior", "away_srs_prior",
```

**GAME_DEFAULTS** (lines 221-234):
```python
# Basketball Reference prior-season stats: baseline defaults used when actual priors unavailable
# These fields are populated with real prior-season stats when Basketball Reference data merges successfully
# When priors dataset is missing or merge fails (e.g., no team abbreviations), these baseline defaults are used
"home_o_rtg_prior": 110.0,  # league-average offensive rating baseline
"home_d_rtg_prior": 110.0,  # league-average defensive rating baseline
"home_pace_prior": 100.0,   # league-average pace baseline
...
```

### 2. Enhanced Diagnostic Output (lines 1707-1746)

**Before:**
```
🏀 TEAM PRIORS:
  home_o_rtg_prior: 0 non-default (0.0%)
⚠️ WARNING: NO games have real team priors - all using defaults!
```

**After:**
```
🏀 TEAM PRIORS (Basketball Reference prior-season stats):
   Note: These columns hold actual prior-season stats when available,
   or baseline defaults (110.0 o/d_rtg, 100.0 pace, 0.0 srs) when not.
   home_o_rtg_prior: 0 with actual priors (0.0%)

⚠️  WARNING: NO games have actual team priors - all using baseline defaults!
   This means Basketball Reference prior-season stats are NOT being used.
   Possible causes:
   • Missing 'teamTricode' column in TeamStatistics.csv
   • Season mismatch between games and priors dataset
   • No priors dataset provided (--priors-dataset)
   • Team abbreviations don't match between datasets

   🔍 Root cause: No team abbreviations available!
      Solution: Ensure TeamStatistics.csv has a 'teamTricode' column
```

### 3. Added Column Validation (lines 1293-1303, 1371-1379)

**Team priors validation (validates Team Summaries.csv):**
```python
# Validate required columns
required_cols = ["season", "abbreviation", "o_rtg", "d_rtg", "pace", "srs"]
missing_cols = [c for c in required_cols if c not in ts.columns]
if missing_cols:
    log(f"Warning: Team Summaries.csv is missing required columns: {missing_cols}", verbose)
    log(f"  Available columns: {list(ts.columns)}", verbose)
    log(f"  Team priors will not be loaded.", verbose)
```

**Player priors validation (validates Per 100 Poss.csv):**
```python
# Validate required columns
required_cols = ["season", "player_id"]
missing_cols = [c for c in required_cols if c not in per100.columns]
if missing_cols:
    log(f"Warning: Per 100 Poss.csv is missing required columns: {missing_cols}", verbose)
    log(f"  Player priors will not be loaded.", verbose)
```

### 4. Added User Guidance (lines 489-491, 1545-1548)

**When teamTricode is missing:**
```python
log("Warning: teamTricode column not found in TeamStatistics - team priors integration will not be possible", verbose)
log("  To enable priors: ensure your TeamStatistics.csv has a 'teamTricode' column with 3-letter team abbreviations", verbose)
```

**When team ID mapping fails:**
```python
log("- No team ID → abbreviation mapping available (teamTricode column not found in TeamStatistics)", verbose)
log("  Team priors from Basketball Reference will use baseline defaults instead of actual prior-season stats", verbose)
```

## Files Modified

### train_auto.py
- Updated GAME_FEATURES comments (lines 184-188)
- Updated GAME_DEFAULTS comments (lines 221-234)
- Added teamTricode validation (lines 489-491)
- Added team abbreviation mapping warnings (lines 1545-1548)
- Enhanced diagnostic output (lines 1707-1746)
- Added Team Summaries column validation (lines 1293-1303)
- Added Per 100 Poss column validation (lines 1371-1379)

### Files Added

1. **validate_priors_fix.py** - Automated validation script
   - Checks terminology in source code
   - Validates baseline default values
   - Verifies diagnostic output improvements

2. **test_priors_terminology.py** - Unit tests (requires numpy)
   - Tests GAME_DEFAULTS structure
   - Tests GAME_FEATURES completeness
   - Tests naming conventions

3. **MANUAL_VERIFICATION.py** - User guide
   - Explains what changed
   - Shows how to verify the fix
   - Provides example diagnostic output

## Validation Results

All automated checks pass:
```
✅ Terminology: 9/9 checks passed
   - Basketball Reference priors mentioned
   - Baseline defaults explained
   - Actual priors distinguished
   - League-average baseline noted
   - teamTricode requirement documented
   - Merge failure scenarios explained
   - Clear diagnostic for missing priors
   - Root cause identification
   - User-friendly guidance

✅ Baseline Values: 8/8 values correct
   - home/away_o_rtg_prior: 110.0
   - home/away_d_rtg_prior: 110.0
   - home/away_pace_prior: 100.0
   - home/away_srs_prior: 0.0

✅ Diagnostic Output: 8/8 improvements present
   - Emoji indicator for priors section
   - Explains what columns contain
   - Shows when defaults are used
   - Lists specific default values
   - Clear warning when no priors
   - Identifies root cause
   - Provides solution
   - Checks for abbreviations
```

## Impact

### Before the Fix
Users saw "priors" columns with 0% non-default values and didn't understand:
- What "priors" meant
- Why they were all defaults
- What CSV columns were needed
- How to fix it

### After the Fix
Users now understand:
1. **What "priors" means**: Basketball Reference prior-season statistics
2. **When actual priors are used**: When teamTricode column exists, priors dataset is provided, and seasons match
3. **When baseline defaults are used**: When the above conditions aren't met
4. **What CSV columns are required**: teamTricode, season, abbreviation, o_rtg, d_rtg, pace, srs, player_id
5. **How to fix missing priors**: Add teamTricode column to TeamStatistics.csv, ensure seasons match, provide --priors-dataset

## Summary

The fix achieves the following objectives:
1. ✅ Clarifying what "priors" means (Basketball Reference prior-season stats)
2. ✅ Documenting when priors are actual data vs baseline defaults
3. ✅ Validating that required CSV columns exist
4. ✅ Providing clear diagnostics when priors aren't available
5. ✅ Offering concrete solutions to fix missing priors

The terminology is now clear, the diagnostics are helpful, and users can understand and fix issues with priors integration.
