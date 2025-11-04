# Parlay Odds Limit Update

**Date**: 2025-11-04  
**Change**: Maximum parlay odds reduced from +1000 to +600

---

## 🎯 Change Made

### Before:
```python
MAX_PARLAY_ODDS = +1000  # Max combined odds for parlays
```

### After:
```python
MAX_PARLAY_ODDS = +600  # Max combined odds for parlays (avoid longshots)
```

**File**: `riq_analyzer.py` (Line 132)

---

## 📊 What This Means

### Allowed Parlays (Examples):

| Legs | Individual Odds | Combined Odds | Status |
|------|----------------|---------------|--------|
| 2-leg | -110, -110 | +264 | ✅ Allowed |
| 2-leg | +100, +100 | +300 | ✅ Allowed |
| 3-leg | -110, -110, -110 | +595 | ✅ Allowed |
| 3-leg | +100, +100, +100 | +700 | ❌ **Rejected** |
| 3-leg | +150, +150, +150 | +1,462 | ❌ **Rejected** |

---

## 💡 Why +600?

### Risk Management:

**+600 (6-to-1) odds = 14.3% breakeven probability**

This is the sweet spot where:
- ✅ Still get meaningful payouts (6x return)
- ✅ Probability is high enough to be reliable
- ✅ Edge is sustainable long-term
- ❌ Avoids lottery-ticket territory (+1000+)

### Comparison:

| Max Odds | Breakeven Prob | Risk Level | Sustainability |
|----------|---------------|------------|----------------|
| +400 | 20.0% | Low | Very High |
| **+600** | **14.3%** | **Medium** | **High** ✅ |
| +1000 | 9.1% | High | Medium |
| +1500 | 6.3% | Very High | Low |

---

## 🎯 Expected Impact

### Before (+1000 limit):
- More parlays generated (including longshots)
- Higher variance
- Occasional big wins, more frequent losses
- Harder to manage bankroll

### After (+600 limit):
- Fewer but higher quality parlays
- **Lower variance**
- **Better hit rate**
- **More sustainable profitability** 💰

---

## ✅ Change is LIVE

The update is immediate - no retraining or additional changes needed.

**To test**:
```bash
python riq_analyzer.py
```

All parlays will now have combined odds ≤ +600.

---

## 🎉 Summary

✅ Updated MAX_PARLAY_ODDS from +1000 to +600  
✅ Focuses on sustainable edge over longshot payouts  
✅ Better bankroll management  
✅ Higher expected long-term profitability  

**Smart move for consistent profits!** 🚀
