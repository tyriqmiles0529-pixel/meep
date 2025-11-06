# NBA Predictor - AI Handoff Document
**Date:** November 6, 2025  
**Session Focus:** Colab Training Setup & Debugging  
**Status:** 90% Complete - One Remaining Issue

---

## 🎯 CURRENT OBJECTIVE
Train NBA prediction models (game outcomes + player props) on Google Colab using GPU acceleration with full historical data (1974-2025).

---

## ✅ COMPLETED WORK

### 1. **Removed Windowed Training (Major Refactor)**
   - **Why:** Window-based training (5-year windows) was incompatible with GPU/TabNet neural hybrid
   - **Changes:**
     - Modified `train_auto.py` to skip windowed player training when `--no-window-ensemble` flag is used
     - Added non-windowed training path at lines 4934-4980
     - Made OOF predictions optional (lines 2546-2555) for non-windowed mode
   - **Result:** Training now processes ALL data at once (1974-2025) - optimal for neural networks
   - **Commits:** `5d2b8a1`, `ec60641`, `1a2e47b`

### 2. **Fixed Data Loading Issues**
   - **Problem:** PlayerStatistics.csv filtering was disabled (`if False` at line 1677)
   - **Problem:** Priors loading tried to download from Kaggle instead of checking local path first
   - **Solution:** Updated priors loading logic (lines 4327-4364) to check `os.path.exists()` FIRST
   - **Commits:** `56efd09`, `1fd3bc9`

### 3. **Fixed Colab Notebook Issues**
   - **Problem:** Files extracted to wrong directory, old code being cached
   - **Solutions:**
     - Added `os.chdir('/content')` to force extraction to correct path
     - Added `shutil.rmtree('meep')` to force fresh git clone
     - Added pre-flight verification showing file sizes and CSV counts
   - **Commits:** `74b9199`, `6464d95`, `95db1f9`

### 4. **Fixed Pandas Date Parsing**
   - **Problem:** `FutureWarning` and `NaT` errors from re-parsing datetime columns
   - **Solution:** Added check at line 167-170 to skip re-parsing if already datetime
   - **Commit:** `b63956a`

---

## ⚠️ CURRENT ISSUE: Basketball Reference Priors Not Merging

### Problem Description
The training script successfully:
- ✅ Loads priors from `/content/priors_data` (6 CSV files, 185,226 player-seasons)
- ✅ Merges team priors into games_df (51,586/62,085 games = 83.1% matched)
- ❌ **BUT all merged values are defaults (110.0) instead of real data**

### Evidence from Logs
```
✓ 51,308 games (82.6%) have Basketball Reference statistical priors
Fallback matched 51586 teams
Home priors matched: 51,586 / 62,085 (83.1%)

BUT THEN:

Value ranges:
  home_o_rtg_prior: 110.00 to 110.00  ← All defaults!
  Default value: 110.0

⚠️  WARNING: NO games have real Basketball Reference statistical priors
```

### Root Cause Analysis
1. **Merge is happening** (83% match rate)
2. **But values are NaN after merge** (then filled with defaults at line 4625)
3. **Likely cause:** Column name mismatch in `Team Summaries.csv`

### Code Location
- **Merge logic:** Lines 4408-4545 in `train_auto.py`
- **Column renaming:** Lines 4431-4451 (expects `o_rtg`, `d_rtg`, `pace`, `srs`, etc.)
- **Default filling:** Lines 4618-4625 (fills NaN with GAME_DEFAULTS)

### What to Check
**The `Team Summaries.csv` file needs these exact column names:**
- `abbreviation` (team abbrev like "LAL", "GSW")
- `season_for_game` (season year)
- `o_rtg` (offensive rating)
- `d_rtg` (defensive rating)
- `pace` (possessions per 48 min)
- `srs` (Simple Rating System)
- `e_fg_percent` (effective FG%)
- `tov_percent` (turnover %)
- `orb_percent` (offensive rebound %)
- `ft_fga` (free throw rate)
- `ts_percent` (true shooting %)
- `x3p_ar` (3-point attempt rate)
- `mov` (margin of victory)

**Diagnostic steps:**
1. Check actual column names in `Team Summaries.csv`
2. If names are different, update the rename mapping at lines 4431-4451
3. Alternative: Add debug logging to see what values are in the merged dataframe before fillna

---

## 📁 FILE STRUCTURE

### Key Files
- **`train_auto.py`** - Main training script (5,500+ lines)
- **`NBA_COLAB_SIMPLE.ipynb`** - Google Colab notebook (3 cells)
- **`riq_analyzer.py`** - Real-time prediction/betting analyzer
- **`optimization_features.py`** - Phase 6 advanced features
- **`phase7_features.py`** - Phase 7 situational features

### Data Files (Upload to Colab)
- **`PlayerStatistics.csv.zip`** (41 MB compressed → 317 MB extracted)
- **`priors_data.zip`** (4.8 MB compressed → 6 CSV files)
  - Team Summaries.csv
  - Team Abbrev.csv
  - Per 100 Poss.csv
  - Advanced.csv
  - Player Shooting.csv
  - Player Play By Play.csv

---

## 🚀 TRAINING CONFIGURATION

### Current Setup
```bash
python3 train_auto.py \
    --priors-dataset /content/priors_data \
    --player-csv /content/PlayerStatistics.csv \
    --verbose \
    --fresh \
    --neural-device gpu \
    --neural-epochs 50 \
    --no-window-ensemble \
    --game-season-cutoff 1974 \
    --player-season-cutoff 1974
```

### What This Does
- **Game models:** LightGBM for moneyline/spread (already working - 62.6% accuracy)
- **Player models:** TabNet + LightGBM hybrid for 5 props (minutes, points, rebounds, assists, threes)
- **Features:** 229 features including temporal (era/decade), Basketball Reference priors, advanced stats
- **Training time:** 25-35 minutes on GPU (L4/T4)

---

## 📊 FEATURES IMPLEMENTED

### Temporal Features (Lines 1235-1236, 229-230)
- ✅ `season_end_year` - NBA season identifier
- ✅ `season_decade` - 1940s-2020s categorization
- ✅ Time-decay sample weights (0.97 decay, recent games weighted higher)
- ✅ Lockout downweighting (1999, 2012 seasons)

### Basketball Reference Priors (68 features)
- **Team (19 features):** O/D ratings, pace, SRS, four factors, etc.
- **Player (68 features):** Per-100-poss stats, advanced metrics, shooting zones, play-by-play

### All Feature Phases
- ✅ Phase 1: Shot volume (FGA, 3PA, FTA rates + efficiency)
- ✅ Phase 2: Matchup context (pace, defensive difficulty)
- ✅ Phase 3: Advanced rates (usage%, rebound%, assist%)
- ✅ Phase 4: Rolling averages (3/5/10/20 game windows)
- ✅ Phase 5: Per-minute rates (pts/36, reb/36, etc.)
- ✅ Phase 6: Momentum, variance, ceiling/floor, workload
- ⚠️ Phase 7: Partially working (schedule density crashes on datetime comparison)

---

## 🐛 KNOWN ISSUES

### 1. Basketball Reference Priors Not Merging (CRITICAL - Current Issue)
- **Status:** Under investigation
- **Impact:** Models training without 68 advanced statistical features
- **Next step:** Check Team Summaries.csv column names

### 2. Phase 7 Features Crash
- **Error:** `Invalid comparison between dtype=datetime64[ns] and int`
- **Location:** Line 2872+ in `train_auto.py`
- **Impact:** Low (non-critical features)
- **Workaround:** Wrapped in try/except, training continues

### 3. PlayerStatistics.csv Filtering Disabled
- **Location:** Lines 1677, 1704 (`if False and...`)
- **Why:** "Temporarily disabled to debug Colab issue"
- **Impact:** Loads full 1946-2025 data (not a problem with current setup)
- **Note:** Can re-enable if memory issues arise

---

## 🔧 RECENT COMMITS

```
95db1f9 - Add pre-flight verification and force latest code pull
74b9199 - Extract files to /content/ with verification
56efd09 - Fix priors loading to check local path first before Kaggle download
1fd3bc9 - Make OOF predictions optional for non-windowed player training
ec60641 - Remove duplicate elif block and windowed training code
1a2e47b - Add non-windowed player training path for neural hybrid mode
5d2b8a1 - Disable window-based player training when --no-window-ensemble flag is used
6464d95 - Force fresh git clone in Colab to get latest code
b63956a - Fix verification to test with date range that exists in sample
```

---

## 📝 NEXT STEPS FOR NEW AI

### Immediate Priority
**Fix Basketball Reference priors merge:**

1. **Verify CSV column names:**
   ```python
   import pandas as pd
   team_summaries = pd.read_csv('/content/priors_data/Team Summaries.csv')
   print(team_summaries.columns.tolist())
   ```

2. **Check if columns match expected names:**
   - Expected: `o_rtg`, `d_rtg`, `pace`, `srs`, `e_fg_percent`, etc.
   - If different: Update rename dict at lines 4431-4451

3. **Add debug logging before fillna:**
   ```python
   # At line 4624, BEFORE fillna:
   sample_vals = games_df[["home_abbrev", "season_end_year", "home_o_rtg_prior"]].head(10)
   log(f"DEBUG - Values before fillna:\n{sample_vals}", verbose)
   ```

4. **Alternative fix:** If columns are correct, the issue might be:
   - Data type mismatch (season_end_year as int vs float)
   - NaN values in the priors CSV itself
   - Abbreviation standardization issue (e.g., "PHO" vs "PHX")

### After Fixing Priors
1. **Let training complete** (25-35 min)
2. **Download trained models** (Step 3 in notebook)
3. **Test predictions locally** with `riq_analyzer.py`
4. **Optional:** Fix Phase 7 datetime comparison issue

---

## 📚 REFERENCE DOCUMENTATION

### In Repo
- `START_HERE.md` - Original setup guide
- `COLAB_QUICKSTART.md` - Colab instructions
- `PHASE7_QUICKSTART.md` - Phase 7 features guide

### Key Functions
- `build_players_from_playerstats()` - Line 1607 (loads/processes player data)
- `load_basketball_reference_priors()` - Line 3990 (loads priors CSVs)
- `train_player_model_enhanced()` - Line 3640 (TabNet + LGB hybrid training)

---

## 💡 HELPFUL CONTEXT

### Why Non-Windowed Training?
- Neural networks (TabNet) learn better from more data at once
- Temporal features (era/decade) let the model distinguish different eras
- Window-based was originally for LightGBM which had concept drift issues
- GPU training on full dataset (1974-2025) takes same time as 1 window

### Why GPU Required?
- TabNet (neural network) component requires GPU for reasonable training time
- CPU training would take 2-3 hours per prop (25+ hours total)
- GPU training: ~30 min for all 5 props

### Dataset Info
- **Games:** 62,085 (1974-2025, includes current season)
- **Player-games:** 1.6M+ rows (full history)
- **Training uses:** Cutoff at 1974 = 50 years of data

---

## 🎯 SUCCESS CRITERIA

Training is successful when:
1. ✅ Game models: 62%+ accuracy (ACHIEVED)
2. ⏳ Player models: RMSE < 3.5 for points/rebounds/assists
3. ⏳ Basketball Reference priors: >80% of games using real data (NOT DEFAULTS)
4. ⏳ Neural hybrid: Completes 50 epochs without errors
5. ⏳ Models saved: 5 player prop models + game models in `/content/meep/models/`

---

## 📞 CONTACT INFO

**GitHub Repo:** https://github.com/tyriqmiles0529-pixel/meep  
**Main Branch:** `main` (latest commit: 95db1f9)

---

**Good luck! The issue should be a simple column name mismatch - check Team Summaries.csv first!**
