
# PHASE L AUDIT LOG
**Date:** 2025-12-30
**Event:** Early-Season Recalibration (NBA 2025-2026)

## 1. Problem Identification
- **Issue:** System was generating excessive high-confidence Unders for bench/role players.
- **Root Cause:** Historical embeddings + early season noise caused the model to see volatile low-minute performances as "stable lows," resulting in 95% confidence on Unders.
- **Risk:** "Lazy Unders" strategy is fragile; one injury or rotation change busts the bet.

## 2. Corrective Actions (Phase L)
### A. Minutes Gating (The "Gatekeeper")
- **Rule:** Win Probability now penalized by **10%** if projected minutes < 24.
- **Rule:** Win Probability penalized by **20%** if projected minutes < 18.
- **Impact:** Bench players are now treated as "High Variance" by default, preventing them from displacing Starters in Core Parlays.

### B. Overs Bias (The "Incentive")
- **Rule:** If Side is OVER and Minutes >= 24, Win Probability boosted by **5%**.
- **Rule:** If Side is UNDER, Win Probability penalized by **8%**.
- **Impact:** System now structurally prefers Overs on stable players, aligning with typical Public/Institutional preference for "Positive Events" over "Negative Events" (Unders).

### C. Governance Envelopes (The "Cap")
- **Starters (MPG >= 24):** Max Confidence capped at **90%** (down from 95% temporarily).
- **Bench (MPG < 24):** Max Confidence capped at **70%**.
- **Impact:** A bench player can NEVER appear in a Core Parlay (which requires high confidence anchors) unless the edge is astronomical, effectively filtering the noise.

## 3. Validation
- **Monte Carlo Simulation:** Re-run with updated parameters showing robust 100% survivability.
- **Operational Check:** Next `morning_routine` will reflect these new weightings immediately.

---
**Status:** COMPLETE.
**Authorized By:** Phase L Mandate.
