
# Phase K.6 — Institutional Operations Prompt

**Objective:** Run the system in fully disciplined institutional mode with zero discretionary changes. Focus on monitoring, reporting, and governance compliance only.

## Instructions for Machine:

### 1. Capital & Allocation
- **Daily allocation is exactly 10 Units.**
- Maintain **80% Core / 15% Growth / 5% Moonshot** allocation.
- **Do not modify** allocations, stake sizes, or bankroll fractions based on outcomes.

### 2. Confidence & Probabilities
- Enforce **maximum 95% confidence cap**.
- Log confidence per leg and per parlay.
- If calculated probability exceeds 95%, **clip at 95%** before any betting logic.

### 3. Parlay Construction
- Generate parlays only, **no individual legs** for betting.
- Favor **overs** in all prop categories.
- Construct Core (low-to-mid odds), Growth (mid-to-high odds), and Moonshot (high odds) parlays.
- Only generate Round Robins if mathematically profitable hedged parlays exist.

### 4. Monitoring & Audit
- Execute `clv_tracker.py` nightly to log Closing Line Value (CLV).
- Execute `monte_carlo_engine.py` periodically to simulate exposure and Sharpe Ratio of Core/Growth portfolios.
- Run `monitor_systems.py` to track:
  - Confidence compliance
  - Allocation adherence
  - Portfolio drift

### 5. Governance Rules
- Any observed governance violation triggers **log only**, do not adjust strategy automatically.
- Maintain a full audit trail for all bets, calculations, and decisions.

### 6. Operational Mandate
- **Do not change strategy**, retrain models, or rebalance portfolios without explicit **Phase L** authorization.
- Output reports in **Units only**, omit bankroll/dollar values.
- Ensure all logs, reports, and audits are reproducible and timestamped.

### 7. Next Steps
- Continue monitoring in Phase K mode.
- Wait for CLV and audit trends before considering **Phase L — Adaptive Intelligence / Dynamic Adjustments**.

---
*End of Phase K.6 Instructions*
