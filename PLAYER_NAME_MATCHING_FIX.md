# Player Name Matching Fix - CRITICAL

## Problem Discovered

**Symptom:** Only 0.7% of player-games matched with Basketball Reference priors (6,014 / 833k rows)

**Debug output showed:**
```
DEBUG - Raw Kaggle names: ['Shandon', 'Walt', 'Herb', 'Michaela', 'Michelle', ...]
DEBUG - Raw Priors names: ['Precious Achiuwa', 'Steven Adams', 'Bam Adebayo', ...]
```

Kaggle names were showing ONLY first names instead of full names!

---

## Root Cause Analysis

### The Bug (Line 1613-1614)

**BEFORE (buggy code):**
```python
# Line 1597-1601: Detect column names
name_full_col  = resolve_any([["playerName", "PLAYER_NAME", "player_name"]])
fname_col      = resolve_any([["firstName", "FIRST_NAME", "first_name"]])
lname_col      = resolve_any([["lastName", "LAST_NAME", "last_name"]])
name_col       = name_full_col or fname_col or lname_col  # ← Sets name_col = "firstName"

# Line 1613-1614: Build list of columns to load
want_cols = [gid_col, date_col, pid_col, name_col, tid_col, home_col, ...]  # ← Only loads firstName!
usecols = [c for c in want_cols if c is not None]
ps = pd.read_csv(player_path, low_memory=False, usecols=usecols)
```

**Problem:**
1. PlayerStatistics.csv has `firstName` and `lastName` columns (NOT a single `playerName` column)
2. Line 1597: `name_full_col = None` (no playerName column exists)
3. Line 1601: `name_col = fname_col` (which is "firstName")
4. Line 1613: Only `name_col` is included in `want_cols`, so only firstName is loaded
5. Line 1616: CSV is read with ONLY firstName column, lastName is never loaded!

**Result:**
- DataFrame has firstName column: `['Shandon', 'Walt', 'Herb', ...]`
- DataFrame has NO lastName column
- Line 1670 tries to combine: `fn + " " + ln`
- But `ln` is an empty Series (created as fallback at line 1669)
- `__name_join__` ends up with only first names: `['Shandon', 'Walt', ...]`

---

## The Fix

### Fix 1: Load BOTH firstName AND lastName (Line 1613-1614)

**AFTER (fixed code):**
```python
# FIXED: Include both fname_col and lname_col separately, not just name_col
want_cols = [gid_col, date_col, pid_col, name_full_col, fname_col, lname_col, tid_col, home_col,
             min_col, pts_col, reb_col, ast_col, tpm_col, starter_col]
usecols = [c for c in want_cols if c is not None]
```

**What changed:**
- OLD: `want_cols = [..., name_col, ...]` (only loaded firstName)
- NEW: `want_cols = [..., name_full_col, fname_col, lname_col, ...]` (loads ALL name columns)

**Result:**
- CSV now loads BOTH firstName and lastName columns
- Line 1670 can properly combine them: `"LeBron" + " " + "James"` → `"LeBron James"`
- `__name_join__` now has full names to match against Basketball Reference

---

### Fix 2: Enhanced Name Normalization (Lines 1924-1935)

**BEFORE:**
```python
def _name_key(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.normalize('NFKD')
         .str.encode('ascii', errors='ignore')
         .str.decode('ascii')
         .str.lower()
         .str.replace(r"[^a-z]+", " ", regex=True)
         .str.strip()
         .str.replace(r"\s+", " ", regex=True)
    )
```

**AFTER (with suffix removal):**
```python
def _name_key(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.normalize('NFKD')  # Normalize unicode (handles Dončić → Doncic)
         .str.encode('ascii', errors='ignore')
         .str.decode('ascii')
         .str.lower()
         .str.replace(r"[^a-z]+", " ", regex=True)  # Remove all non-letters
         .str.strip()
         .str.replace(r"\s+", " ", regex=True)  # Collapse multiple spaces
         .str.replace(r"\s+(jr|sr|ii|iii|iv|v)$", "", regex=True)  # Remove suffixes
    )
```

**What changed:**
- Added suffix removal for "Jr.", "Sr.", "II", "III", "IV", "V"
- Now "Gary Trent Jr." matches "Gary Trent" in Basketball Reference

---

## Expected Impact

### Before Fix
```
Raw Kaggle names: ['Shandon', 'Walt', 'Herb', ...]  ← Only first names!
Raw Priors names: ['Precious Achiuwa', 'Steven Adams', 'Bam Adebayo', ...]
Name overlap: 1 common names (0.7% match rate)  ← Almost nothing matched!
```

### After Fix
```
Raw Kaggle names: ['Shandon Anderson', 'Walt Williams', 'Herb Williams', ...]  ← Full names!
Raw Priors names: ['Precious Achiuwa', 'Steven Adams', 'Bam Adebayo', ...]
Name overlap: 500+ common names (50-80% match rate)  ← Much better!
```

---

## Why This Matters

### What Player Priors Provide

Basketball Reference priors add **68 additional features** for each player-game:
- **Efficiency**: PER (Player Efficiency Rating), TS% (True Shooting %), eFG%
- **Usage**: USG% (Usage Rate), how often player has the ball
- **Impact**: BPM (Box Plus/Minus), VORP (Value Over Replacement Player)
- **Shooting zones**: Corner 3%, Above the Break 3%, Mid-range %, Rim %
- **Defense**: Defensive Rating, Defensive Win Shares
- **Advanced**: Win Shares, Win Shares per 48, OBPM, DBPM

**With 0.7% match rate:**
- Only 6,014 player-games out of 833k get these features
- Models are missing critical context for 99.3% of player performances

**With 50-80% match rate:**
- 400k-650k player-games get priors
- Models can learn patterns like:
  - "High USG% player + low minutes = blowout garbage time"
  - "Corner 3% specialist + corner 3 attempt = higher make probability"
  - "High PER player on back-to-back = fatigue factor"

---

## Verification

### Test Script
```python
import pandas as pd

# Simulate PlayerStatistics structure
data = {
    'firstName': ['LeBron', 'Stephen', 'Kevin'],
    'lastName': ['James', 'Curry', 'Durant'],
}
ps = pd.DataFrame(data)

# Apply fix logic
fname_col = 'firstName'
lname_col = 'lastName'

fn = ps[fname_col].astype(str)
ln = ps[lname_col].astype(str)
ps['__name_join__'] = (fn.fillna('') + ' ' + ln.fillna('')).str.strip()

print(ps['__name_join__'].tolist())
# Output: ['LeBron James', 'Stephen Curry', 'Kevin Durant']  ✓
```

### On Next Training Run

Look for this output:
```
Detected player columns
- gid: gameId  date: gameDate  pid: personId  name_full: None  first: firstName  last: lastName  ← Both loaded!

DEBUG - Raw Kaggle names: ['LeBron James', 'Stephen Curry', 'Kevin Durant', ...]  ← Full names now!
DEBUG - Raw Priors names: ['Precious Achiuwa', 'Steven Adams', 'Bam Adebayo', ...]
Name overlap (sample up to 5k): [SHOULD BE 500-2000] common normalized names  ← Much higher!

Player priors matched by name for 450,000+ rows (50%+ match rate)  ← SUCCESS!
```

---

## Files Modified

### train_auto.py

**Line 1613-1616** - CSV column loading
```python
# OLD: want_cols = [..., name_col, ...]
# NEW: want_cols = [..., name_full_col, fname_col, lname_col, ...]
```

**Lines 1924-1935** - Name normalization
```python
# Added: .str.replace(r"\s+(jr|sr|ii|iii|iv|v)$", "", regex=True)
```

---

## Summary

**Root Cause:** Only firstName column was being loaded from CSV, lastName was ignored

**Fix Applied:** Load BOTH firstName AND lastName columns, plus enhanced suffix handling

**Expected Result:** Match rate increases from 0.7% to 50-80% (75x improvement!)

**Impact:** Models gain access to 68 Basketball Reference features for 400k-650k player-games instead of just 6k

**Next Step:** Run training with `--verbose` to verify debug output shows full names
