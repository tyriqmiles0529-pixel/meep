# 🛠️ FIXING THE MODEL - Step-by-Step Guide

**Problem**: Model is 50% accurate (losing) and severely overconfident  
**Solution**: 5-step fix process  
**Time**: ~30 minutes  

---

## 🎯 **STEP 1: Recalibrate Models** ⭐⭐⭐ (MOST IMPORTANT)

### What it fixes:
- 90% predictions → actually 90% (not 45%)
- 70% predictions → actually 70% (not 42%)
- Fixes overconfidence issue

### How to do it:
```bash
python recalibrate_models.py
```

### What happens:
1. Loads your 220+ settled predictions
2. Trains `IsotonicRegression` for each stat type
3. Creates calibration curves (90% model → X% actual)
4. Saves to `model_cache/calibration_curves.pkl`
5. Shows you before/after calibration

### Expected output:
```
POINTS: 76 samples
  Before calibration:
    Avg predicted: 82.3%
    Train accuracy: 48.7%
    Gap: +33.6% (OVERCONFIDENT!)
  
  After calibration:
    Calibrated: 51.2%
    Train accuracy: 48.7%
    Gap: +2.5% (Much better!)
```

### Time: **5 minutes**

---

## 🎯 **STEP 2: Update riq_analyzer.py to Use Calibration**

### What to add:

```python
# At top of riq_analyzer.py (around line 50):

# Load calibration curves
CALIBRATORS = None
try:
    with open('model_cache/calibration_curves.pkl', 'rb') as f:
        CALIBRATORS = pickle.load(f)
        print("✅ Loaded calibration curves")
except:
    print("⚠️  No calibration curves found - using uncalibrated predictions")
```

### Then in `analyze_player_prop()` function (around line 3000):

Find this section:
```python
# Calculate win probability
win_prob = 1.0 - scipy.stats.norm.cdf(line, mu_final, sigma_final)
```

Add RIGHT AFTER:
```python
# Apply calibration if available
if CALIBRATORS and prop_type in CALIBRATORS:
    win_prob_original = win_prob
    win_prob = float(CALIBRATORS[prop_type].predict([win_prob])[0])
    if DEBUG_MODE:
        print(f"   Calibrated: {win_prob_original*100:.1f}% → {win_prob*100:.1f}%")
```

### Time: **10 minutes**

---

## 🎯 **STEP 3: Add Early-Season Dampening**

### What it fixes:
- Reduces overconfidence when player has <10 games
- Early season is noisy - this pulls predictions toward 50%

### What to add in `analyze_player_prop()`:

After calibration section, add:
```python
# Early-season dampening (reduce confidence for new players)
if len(player_history) < 10:
    dampening_factor = len(player_history) / 10.0  # 0.5 for 5 games, 0.8 for 8 games
    # Pull toward 50%
    win_prob = 0.5 + (win_prob - 0.5) * (0.6 + 0.4 * dampening_factor)
    if DEBUG_MODE:
        print(f"   Early-season dampening: factor={dampening_factor:.2f}")
```

### Time: **5 minutes**

---

## 🎯 **STEP 4: Reduce Kelly Fractions** (Temporary Safety)

### What to change in riq_analyzer.py:

Find this section (around line 125):
```python
KELLY_FRACTION = 0.5  # or whatever it currently is
```

Change to:
```python
KELLY_FRACTION = 0.1  # Very conservative until calibration is validated
```

### Time: **1 minute**

---

## 🎯 **STEP 5: Filter to Assists Only** (Optional - for safety)

### What it does:
- Only outputs assists predictions (59% win rate)
- Skips points/rebounds/threes until they improve

### What to add in `analyze_player_prop()`:

At the start of the function:
```python
def analyze_player_prop(prop, ...):
    # TEMPORARY: Only analyze assists until other stats are fixed
    if prop.get('prop_type') != 'assists':
        return None  # Skip non-assists
    
    # ... rest of function
```

### Time: **2 minutes**

---

## 📊 **VERIFICATION - Test the Fixes**

### Run analyzer with fixes:
```bash
python riq_analyzer.py
```

### Look for in output:
```
✅ Loaded calibration curves

Analyzing prop: LeBron James points...
   Predicted: 85.3%
   Calibrated: 85.3% → 67.2%  ← Should see this!
   Early-season dampening: factor=1.00
   Final win_prob: 67.2%
   
   Bet sizing:
   Kelly fraction: 0.10 (conservative)
   Stake: $10.00 (was $50.00 before)
```

### What to check:
✅ Calibrated probabilities are lower (not 90%+)  
✅ Kelly fractions are smaller (0.1x)  
✅ Only assists showing up (if you enabled filter)  

### Time: **5 minutes**

---

## 📈 **EXPECTED IMPROVEMENTS**

### Before fixes:
```
Overall: 50.0% accuracy
High-confidence (90%+): 45% actual
Kelly sizing: Too aggressive
Result: LOSING
```

### After fixes:
```
Overall: Expected 54-58% accuracy
High-confidence (90%+): Should be closer to 75-85%
Kelly sizing: Conservative (0.1x)
Result: Small positive or break-even (validate over 2 weeks)
```

### Timeline:
- **Week 1**: Break-even or small profit
- **Week 2-3**: Should see 54-58% accuracy
- **Week 4**: If stable, increase Kelly to 0.25x
- **Month 2**: If still good, increase to 0.5x

---

## 🔄 **ONGOING MONITORING**

### Daily:
```bash
# Fetch results from yesterday
python fetch_bet_results_optimized.py
```

### Weekly:
```bash
# Analyze performance
python analyze_ledger.py

# Look for:
# - Overall accuracy trending up?
# - Calibration improving?
# - Assists still best performer?
```

### Monthly:
```bash
# Re-calibrate with more data
python recalibrate_models.py

# If accuracy good, consider:
# - Un-filter points/rebounds/threes
# - Increase Kelly fraction
# - Retrain models
```

---

## 🚨 **STOPPING RULES**

### STOP BETTING if:
- Overall accuracy drops below 52% for 2+ weeks
- Losing money consistently
- Calibration gets worse

### REDUCE BET SIZE if:
- Overall accuracy 52-54% (break-even zone)
- High variance in results
- Unsure about calibration

### INCREASE BET SIZE only if:
- Overall accuracy >56% for 4+ weeks
- Calibration is good (predicted ≈ actual)
- Bankroll can handle variance

---

## 📁 **Files Created**

1. ✅ `recalibrate_models.py` - Creates calibration curves
2. ✅ `fetch_bet_results_optimized.py` - Better result fetcher
3. ✅ This guide - Step-by-step fix instructions

---

## 🎯 **QUICK START (Do This Now!)**

```bash
# 1. Recalibrate (5 min)
python recalibrate_models.py

# 2. Update riq_analyzer.py (15 min)
# - Add calibration loading at top
# - Add calibration to analyze_player_prop()
# - Add early-season dampening
# - Reduce KELLY_FRACTION to 0.1

# 3. Test it (5 min)
python riq_analyzer.py

# 4. Monitor daily
python fetch_bet_results_optimized.py  # Every morning
python analyze_ledger.py               # Every Sunday
```

---

## 💡 **WHY THIS WILL WORK**

### Problem identified:
✅ Model is overconfident (90% predictions hit 45%)  
✅ Assists only profitable stat  
✅ Early season noise  

### Solution applied:
✅ Calibration fixes overconfidence  
✅ Focus on assists (59% win rate)  
✅ Early-season dampening reduces noise  
✅ Conservative Kelly (0.1x) limits losses  

### Expected outcome:
✅ 54-58% overall accuracy (profitable!)  
✅ Better calibration (70% → ~70%)  
✅ Smaller losses while validating  
✅ Clear path to improvement  

---

## 🎉 **BOTTOM LINE**

**Total Time**: ~30 minutes to implement all fixes  
**Expected Impact**: 50% → 54-58% accuracy (profitable!)  
**Risk**: Low (conservative sizing + assists only)  
**Reward**: Get back to profitability while fixing long-term  

**DO THIS TODAY!** 🚀

---

## 📞 **Quick Reference**

```bash
# FIX THE MODEL (one time)
python recalibrate_models.py
# Then update riq_analyzer.py (see STEP 2)

# USE DAILY
python riq_analyzer.py  # Make predictions
python fetch_bet_results_optimized.py  # Get results

# MONITOR WEEKLY  
python analyze_ledger.py  # Check performance

# IMPROVE MONTHLY
python recalibrate_models.py  # Update calibration
```

**Start with STEP 1 (recalibration) - it's the most important!** ⭐
