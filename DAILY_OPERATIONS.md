# NBA Predictor: Daily Operations Guide (Phase J.7 Production Live)

This guide details how to run the production betting strategy for Riq's Picks.

---

## 🚦 PRODUCTION STATUS: ✅ ACTIVE (LIVE)
| Parameter | Value |
|:---|:---|
| **V4 Ensemble** | PTS, AST, REB, 3PM |
| **Current Mode** | Production Go-Live |
| **Starting Bankroll** | $20.00 |

---

## PHASE J.7 PRODUCTION CONTROLS

### ⚠️ Hard Constraints
| Constraint | Value | Notes |
|:---|:---|:---|
| **Primary Bookmaker** | FanDuel | All production bets target FD first |
| **Secondary Bookmaker** | DraftKings | Fallback |
| **Edge Threshold** | ±1.5 Delta | Minimum predicted value difference |
| **Kelly Fraction** | 0.10 (1/10 Kelly) | Conservative capital protection |
| **Daily Exposure Cap** | 15% of bankroll | Auto-scaled to protect bankroll |
| **Correlation Rules** | Exclusive Games & Players | No same game/player in same parlay |
| **3PM Market** | ✅ ENABLED (J.6) | sigma=0.9; same ±1.5 delta threshold |

---

## 1. Prerequisites & Setup

### Environment Variables
- `RAPIDAPI_KEY`: Your key for the NBA Player Props API.
- `ODDS_MODE`: Set to `live` (default).

### Directory Structure
- `predictions/live_ensemble_2025.csv`: Latest V4 ensemble predictions.
- `historical_data/`: Directory for simulated odds data.
- `betting_ledger.csv`: Appended automatically per run.

---

## 2. Daily Execution

Run the following command each day after the inference pipeline (`generate_final_projections.py`) has completed:

```powershell
# Run the strategy
python run_phase_i.py
```

### What happens during a run?
1. **Load Predictions**: Reads `predictions/live_ensemble_2025.csv`.
2. **Fetch Odds**: Fetches latest props from The Odds API (FanDuel PRIMARY).
3. **Generate Picks**: Applies ±1.5 delta filter and 0.10 Kelly sizing.
4. **Build Parlays**: 
   - Constructs top 5 **3/4 leg Round Robins**.
   - Constructs top 5 **3/4 leg Regular Parlays**.
   - Enforces "Player Name (TEAM)" display format.

### Outputs
- **Daily Picks File**: `MM.DD.YY - Riq's Picks.md`
- **Ledger Update**: `betting_ledger.csv`

---

## 3. Maintenance Note: "Minutes Leakage"
Audit J.7 identified temporal leakage in validation. While live inference is SAFE (uses proxies), treat validation hit rates as over-optimistic (~5-8%). Sizing is already adjusted (0.10 Kelly) to accommodate this delta.
