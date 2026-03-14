# Phase K Roadmap: The "Real Fund" Approach

**Goal:** Formalize a quantitative, audit-first operational model akin to an institutional fund.
**Core Philosophy:** Move from "Betting Strategy" to "Portfolio Management" via rigorous simulation and drift monitoring.

---

## 1. Formalize Phase K (Monitoring, CLV, Audit, Drift)
The "Black Box" must become transparent. We move from simply logging "bets" to logging the *lifecycle* of the decision.

### 1.1 CLV (Closing Line Value) Tracking
*   **Metric:** Log `Line_At_Bet_Time` vs `Line_At_Tipoff` (Closing Line).
*   **Requirement:** CLV is the truest predictor of long-term profitability, independent of game outcome.
*   **Action Plan:**
    *   Create `clv_tracker.py` to poll odds at tip-off time for previously logged bets.
    *   Update `betting_ledger.csv` schema to include `Closing_Line`, `Closing_Odds`, `CLV_%`.

### 1.2 Quantitative Audit Logs
*   **Metric:** Detailed "Why" for every decision.
*   **Requirement:** Every pass/fail decision in the filter chain (e.g., "Pass: Edge 1.2 < 1.5", "Pass: Correlation Violation") should be logged.
*   **Action Plan:**
    *   Implement structured logging in `betting_strategy.py` (JSON output per run).
    *   Log all candidate filters: `EdgeFilter`, `CorrelationFilter`, `KellyFilter`.

### 1.3 Drift Monitoring
*   **Metric:** Rolling error (RMSE) and Feature Drift (Covariate Shift).
*   **Requirement:** Detect if the model's accuracy is decaying or if the league environment (e.g., pace, scoring) has shifted fundamentally.
*   **Action Plan:**
    *   Build `monitor_drift.py` to compare recent (L7 days) prediction error vs. historical baseline.
    *   Flag "Concept Drift" if average Error increases by > 1 std dev.

---

## 2. Monte Carlo Sportsbook Simulation
The "Engine" for Strategy Validation. Stop guessing payout rules; derive them.

*   **Concept:** Build a "Simulated Sportsbook" environment.
*   **Inputs:**
    *   **Phase K Logs:** Real historical predictions and their outcomes.
    *   **Odds Distributions:** Historical odds snapshots (not just fixed -110).
*   **Variables to Stress Test:**
    *   **Variance:** Inject randomness into game outcomes (e.g., simulate the game 10,000 times based on prob distro).
    *   **Execution Friction:** Simulate slippage (odds moving before you bet).
    *   **Streakiness:** Simulate "Cold Streaks" to test bankroll resilience (Ruin Probability).

---

## 3. Lock Parlay Leg Rules via Simulation
Use the Monte Carlo engine to mathematically solve the "Parlay Structure" problem.

*   **Hypothesis Testing:**
    *   Is a **3-leg (+200)** parlay consistently better than a **2-leg (+100)** parlay?
    *   Does adding a 4th leg *actually* increase EV after accounting for increased variance and vigorish (vig)?
*   **Optimization Target:** Maximize **Sharpe Ratio** (Growth / Volatility), not just raw ROI.
*   **Output:** "Golden Rules" for leg selection (e.g., "NEVER exceed 4 legs", "ALWAYS disjoint legs").

---

## 4. Tune Payout vs. Hit Rate
Final Polish: Adjusting the "Knobs" of the strategy.

*   **Risk Appetite:**
    *   Once the "Golden Rules" (Step 3) are set, tune the **Kelly Multiplier** (risk fraction).
    *   *Fund Logic:* If Hit Rate is stable (low variance), leverage can increase. If Hit Rate is volatile, leverage must decrease.
*   **Targeting:**
    *   Aggressively target specific Payout tiers that the Simulation (Step 2) identified as "Mispriced" by the market.

---

## Execution Checklist

- [ ] **Phase K.1:** Update Ledger to Version 2.0 (CLV columns).
- [ ] **Phase K.1:** Build `clv_tracker.py` (Scheduled job).
- [ ] **Phase K.2:** Build `monte_carlo_engine.py`.
- [ ] **Phase K.3:** Run "Parlay Structure" Simulation (10k iterations).
- [ ] **Phase K.4:** Update `betting_strategy.py` with final "Golden Rules".
