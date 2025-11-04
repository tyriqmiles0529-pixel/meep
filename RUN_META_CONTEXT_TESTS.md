# Meta-Learned Context Testing - Progressive Validation

## 🎯 Testing Strategy

Progressive validation in 3 phases:

1. **Current Season** (2022-2026) - Quick test on ~112K samples
2. **5-Year Window** (2020-2024) - Validate on ~150K samples
3. **Full Data** (2002-2026) - Comprehensive test on ~1.3M samples

---

## Phase 1: Current Season Test (2022-2026)

**Purpose:** Quick validation that meta-learned context improves beyond baseline

**Command:**
```bash
python train_meta_context_test.py
```

**What to look for:**
- ✅ Meta-learner weights for each context feature
- ✅ RMSE/MAE metrics
- ✅ Which context features got highest weights

**Expected output:**
```
Meta-learner weights (learned from data):
  Ridge pred:      +0.XXX
  LightGBM pred:   +0.XXX
  Elo pred:        +0.XXX
  Rolling avg:     +0.XXX
  Baseline:        +0.XXX
  Combined pace:   +0.XXX  ← Context feature weight
  Team ORTG:       +0.XXX  ← Context feature weight
  ...
```

**Decision criteria:**
- If RMSE < 6.044 (points), proceed to Phase 2
- If RMSE < 2.513 (rebounds), proceed to Phase 2
- If weights make basketball sense, proceed to Phase 2

---

## Phase 2: 5-Year Backtest (2020-2024)

**Purpose:** Validate improvements hold on larger dataset

**Command:**
```bash
python train_meta_context_5year.py
```

**What to look for:**
- ✅ RMSE improvement vs baseline (target: >+1%)
- ✅ Consistency across seasons
- ✅ No overfitting (metrics should be similar to Phase 1)

**Expected improvement:**
- Points: RMSE < 6.15 (baseline: 6.15)
- Rebounds: RMSE < 2.57 (baseline: 2.57)
- Assists: RMSE < 1.77 (baseline: 1.77)
- Threes: RMSE < 1.17 (baseline: 1.17)

**Decision criteria:**
- If improvement >= +1% on 3+ stats, proceed to Phase 3
- If minutes still worse, skip ensemble for minutes

---

## Phase 3: Full Data Backtest (2002-2026)

**Purpose:** Comprehensive validation across all eras

**Command:**
```bash
python train_meta_context_full.py
```

**What to look for:**
- ✅ RMSE improvement across 24 seasons
- ✅ Per-window performance (check if all windows benefit)
- ✅ Era differences (2000s vs 2010s vs 2020s)

**Expected improvement:**
- Average RMSE: +1.5% to +3% (vs current +0.5%)
- Best stats: Rebounds, threes (expect +2-4%)

**Decision criteria:**
- If average improvement >= +1.5%, **deploy to production**
- If improvement < +1%, revert to original ensemble
- If minutes worse, use LightGBM-only for minutes

---

## 📊 Comparison Framework

After each phase, compare to baseline using:

```bash
python compare_meta_context_results.py --phase [1|2|3]
```

This will show:
- Side-by-side RMSE comparison
- Learned context weights
- Feature importance ranking
- Improvement percentages

---

## 🚦 Decision Tree

```
Phase 1 (Current Season)
├─ RMSE improved?
│  ├─ Yes → Phase 2
│  └─ No → Stop, investigate why
│
Phase 2 (5-Year Window)
├─ Improvement >= +1%?
│  ├─ Yes → Phase 3
│  └─ No → Stop, use original ensemble
│
Phase 3 (Full Data)
├─ Improvement >= +1.5%?
│  ├─ Yes → Deploy to production
│  └─ No → Use original ensemble
```

---

## 💡 What Meta-Learning Will Reveal

The Ridge meta-learner coefficients will empirically show:

### High-Weight Features (expect positive weights)
- **Pace** - More possessions = more stats
- **Team ORTG** - Better offense = more opportunities
- **Assist rate** - For assists prediction especially

### Low-Weight Features (may be near zero)
- **Opponent pace** - Less direct impact
- **Usage Gini** - Already captured in LightGBM

### Stat-Specific Patterns
- **Points:** Expect ORTG, pace to dominate
- **Assists:** Expect assist rate, ORTG high weight
- **Rebounds:** Expect 3PA rate (rebounding environment) high
- **Threes:** Expect 3PA rate, offensive scheme high
- **Minutes:** Expect usage, role features high (but may still struggle)

---

## 🎓 Learning Objectives

By the end of Phase 3, you'll know:

1. **Which context features actually matter** (empirically, not conceptually)
2. **Optimal weights for each stat type** (learned from 1.3M samples)
3. **Whether hand-crafted weights were close** (compare to learned weights)
4. **If meta-learning beats conceptual weights** (+1-3% more improvement?)

---

## ⚠️ Important Notes

1. **Phase 1 is fast** (~5-10 min) - Run this first to validate approach
2. **Phase 2 is medium** (~15-20 min) - Only run if Phase 1 succeeds
3. **Phase 3 is slow** (~1-2 hours) - Only run if Phase 2 succeeds
4. **Always compare to baseline** after each phase
5. **Watch for overfitting** - Metrics should be consistent across phases

---

## 📁 Output Files

Each phase generates:
- `model_cache/player_ensemble_meta_context_XXXX_YYYY.pkl` - Trained model
- `model_cache/player_ensemble_meta_context_XXXX_YYYY_meta.json` - Metadata with learned weights
- `backtest_meta_phaseX.json` - Backtest results

Keep all files for comparison!
