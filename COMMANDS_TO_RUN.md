# Commands to Run - NBA Predictor Cleanup & Update

## ✅ COMPLETED AUTOMATICALLY
- Updated README.md with correct information (2002-2026, not 2017-2026)
- Fixed timezone issue in analyze_ledger.py (was already fixed)
- Verified git status (only main branch exists - no cleanup needed)

## 📋 ANSWERS TO YOUR QUESTIONS

### Q: Why is full history not used for training?
**A: IT IS!** The defaults in train_auto.py are:
```python
--game-season-cutoff 2002    # Uses 2002-2026
--player-season-cutoff 2002  # Uses 2002-2026
```

The confusion came from old documentation that said "2017-2026". This has been corrected in the README.

### Q: Are phases integrated?
**A: NOT YET**. The phase features (shot volume, efficiency rates, etc.) are:
- ✅ Documented in your roadmap
- ❌ Not yet implemented in the training code
- 📝 Next priority for +3-5% accuracy improvement

### Q: Can negative odds have positive EV?
**A: YES!** Expected Value (EV) depends on BOTH:
1. **Your model's win probability** (e.g., 55%)
2. **The odds offered** (e.g., -110 = 52.4% implied)

Example:
- Model says: 55% chance player goes OVER
- Books offer: -110 (requires 52.4% to break even)
- **EV = (0.55 × profit) - (0.45 × stake) > 0** ✅ Positive!

Even heavy favorites (-200) can have +EV if your model is better calibrated than the market.

### Q: Is enhanced selector in production model?
**A: YES!** The enhanced selector (train_dynamic_selector_enhanced.py) IS being used by riq_analyzer.py when it loads ensemble windows.

### Q: Is it weighted to prevent overfitting?
**A: YES!** Multiple safeguards:
1. **Time-decay weighting**: 0.97^years (recent data weighted higher)
2. **Lockout downweighting**: 1999, 2012 seasons at 0.90x weight
3. **Time-series split**: Never train on future data
4. **Out-of-fold predictions**: Meta-learner never sees its training data
5. **Multi-window ensemble**: Different time horizons reduce overfitting

The model DOES use old data, but weights it appropriately to balance "more data" vs "relevance to modern NBA".

### Q: Are all necessary files for backtesting, validation, recalibration there?
**A: MOSTLY YES, but some gaps:**

✅ **Present:**
- Training: train_auto.py, train_ensemble_enhanced.py, train_dynamic_selector_enhanced.py
- Prediction: riq_analyzer.py (daily predictions)
- Tracking: analyze_ledger.py (performance analysis)
- Fetching: fetch_bet_results_incremental.py (get actuals)
- Recalibration: recalibrate_models.py (isotonic regression)

❌ **Missing:**
- backtest_full_history.py (referenced but not found)
- Automated validation pipeline
- Feature engineering phases (not yet implemented)

### Q: Should I delete outdated files?
**A: YES - SAFE TO DELETE:**

1. **player_prediction.py** - Outdated simple weighted average (not used in pipeline)
2. **keys.py** - Should use .env instead (security best practice)
3. **CLEAR_CACHES.bat** - Can recreate if needed
4. **NUL** - Windows artifact

## 🔧 COMMANDS TO RUN NOW

### 1. Delete outdated files
```powershell
# Remove outdated prediction script
Remove-Item player_prediction.py

# Remove keys.py (use .env instead - already noted in README)
# NOTE: First copy your API key!
# Get-Content keys.py  # Copy the SGO_API_KEY value
# Then create .env file with: SGO_API_KEY="your_key_here"
# Remove-Item keys.py

# Remove Windows artifacts
Remove-Item NUL -ErrorAction SilentlyContinue
```

### 2. Commit the updated README
```powershell
git add README.md
git commit -m "docs: Update README with correct training data range (2002-2026) and comprehensive technical details"
git push origin main
```

### 3. Fetch all available results (run multiple times until 50% done)
```powershell
# This will update your ledger with actual outcomes
python fetch_bet_results_incremental.py

# Check progress
python analyze_ledger.py
```

### 4. Recalibrate with maximum data
```powershell
# Once you have ~500+ settled predictions
python recalibrate_models.py
```

### 5. Check current model status
```powershell
python analyze_ledger.py
```

## 🚀 NEXT PRIORITIES (In Order)

### Priority 1: Get More Settled Predictions
**Goal**: 500-1000+ settled predictions for robust recalibration

```powershell
# Keep running until you've fetched all available results
python fetch_bet_results_incremental.py
```

### Priority 2: Implement Phase 1 Features (Shot Volume)
**Expected Impact**: +1.5-2% accuracy on Points predictions

You'll need to modify `train_auto.py` to add:
- FGA (field goal attempts) rolling averages
- 3PA (three-point attempts) rolling averages
- FTA (free throw attempts) rolling averages
- Per-minute versions of each

### Priority 3: Full Retrain with New Features
```powershell
python train_auto.py --verbose --fresh
python train_ensemble_enhanced.py
python train_dynamic_selector_enhanced.py
```

## 📊 CURRENT STATUS SUMMARY

✅ **Working Well:**
- Ensemble architecture (game + player models)
- Dynamic window selector
- Isotonic calibration
- Tracking system (1,728 predictions logged)
- Data pipeline (2002-2026 historical data)

⚠️ **Needs Improvement:**
- Overall accuracy (49.1% → target 52%+)
- Overconfidence (high-confidence picks underperforming)
- Feature engineering (phases not yet integrated)

🎯 **Best Performing:**
- Assists model: 52.8% ✅ (positive edge!)

## 🎤 MEETING PREP (NBA, Data Analytics, AI)

**Key Points to Emphasize:**

1. **Data Volume**: 23 years, 833k player box scores, 50k games
2. **AI/ML Stack**: Ensemble learning, Bayesian priors, adaptive calibration
3. **Production System**: Live tracking, continuous learning, automated recalibration
4. **NBA Domain Knowledge**: Four Factors, pace adjustments, modern era understanding
5. **Honest Performance**: 49.1% overall, but 52.8% on assists (positive edge detected)
6. **Next Steps**: Feature engineering phases for +3-5% improvement

Your README now clearly explains all of this!

## 📝 FINAL NOTES

- **Training data**: ALL files now consistent - 2002-2026 (NOT 2017-2026)
- **README**: Comprehensive, technically accurate, professional
- **Git**: Only main branch exists (no cleanup needed)
- **Pipeline**: train_auto.py → train_ensemble_enhanced.py → train_dynamic_selector_enhanced.py → riq_analyzer.py
- **Game models**: YES, for predicted winner and margin (used as features in player models)
- **Phases**: Documented but NOT YET IMPLEMENTED (your next coding task)
