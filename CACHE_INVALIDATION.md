# IMPORTANT: Cache Invalidation Notice

## 🚨 When to Clear Cache

The window ensemble caching system currently validates caches based on:
- ✅ Season coverage (which seasons are included)
- ❌ Feature set (NOT currently validated)

### When You Add New Features:

**YOU MUST** clear the cache to force retraining with new features:

```powershell
# Option 1: Delete cache manually
Remove-Item -Recurse -Force model_cache

# Option 2: Use --fresh flag (keeps cache but forces retrain)
python train_auto.py --verbose --fresh --enable-window-ensemble
```

### Current Feature Version: **5.0**

**Phase 1-3:** Shot volume, matchup, advanced rates  
**Phase 4:** Opponent defense, rest/B2B, role changes, game script  
**Phase 5:** Position classification, starter status, injury tracking  

---

## ⚠️ What Happens If You Don't Clear Cache

If you run training WITHOUT clearing cache after adding features:

1. **Old windows (2002-2021):** Uses cached models (**WITHOUT new features**)
2. **Current window (2022-2026):** Trains fresh (**WITH new features**)
3. **Result:** Inconsistent feature sets across windows!

This causes:
- ❌ Dynamic selector gets confused (windows have different features)
- ❌ Predictions less accurate (old windows missing context)
- ❌ Ensemble performance degraded

---

## ✅ Solution Applied

Added `FEATURE_VERSION` constant to train_auto.py (line 106):

```python
FEATURE_VERSION = "5.0"  # Phase 5 features
```

**Future improvement:** Add feature version check to cache validation logic.

**For now:** Always use `--fresh` or delete cache when adding features!

---

## 📋 When You Added Features Today:

**Phase 4 & 5 features added** - Cache is now INVALID

**Before training, run:**
```powershell
Remove-Item -Recurse -Force model_cache
```

**Then train:**
```powershell
python train_auto.py --verbose --fresh --enable-window-ensemble
```

This ensures ALL windows (2002-2026) get Phase 1-5 features! ✅
