# NBA Predictor: Daily Operations Guide

This guide details how to run the production betting strategy (Phase I) for Riq's Picks.

## 1. Prerequisites & Setup

### Environment Variables
The following environment variables are required to be set in your terminal (or passed to the script):
- `RAPIDAPI_KEY`: Your RapidAPI key for the NBA Player Props API.
- `ODDS_MODE`: Set to `live` (default) or `simulated`.

### Directory Structure
Ensure the following folders and files exist:
- `predictions/live_ensemble_2025.csv`: Latest ensemble predictions from the inference pipeline.
- `historical_data/`: Directory for any simulated odds data if needed.
- `betting_ledger.csv`: Will be created/appended automatically.

---

## 2. Daily Execution

Run the following command each day after the `daily_inference_v4.py` script has completed:

```powershell
# Set API Key (one time per session)
$env:RAPIDAPI_KEY = "9ef7289093msh76adf5ee5bedb5fp15e0d6jsnc2a0d0ed9abe"

# Run the strategy
python run_phase_i.py
```

### What happens during a run?
1. **Load Predictions**: Reads `predictions/live_ensemble_2025.csv`.
2. **Fetch Odds**: 
   - In `live` mode, it fetches the latest game schedules and player props from RapidAPI.
   - In `simulated` mode, it reads from a historical CSV.
3. **Generate Picks**: Applies the V4 ensemble strategy, calculates EV using a Kelly Criterion multiplier, and filters for Top-5 picks per prop.
4. **Build Parlays**: Constructs non-correlated Round Robin parlays (2, 3, 4 leg) from distinct games.

### Outputs
- **Daily Picks File**: `MM.DD.YY - Riq's Picks.md`. This is your human-readable betting card.
- **Ledger Update**: `betting_ledger.csv`. A complete row-by-row accounting of every pick, including the odds source and implied probability for auditing.

---

## 3. Failure Modes & Troubleshooting

| Issue | Symptom | Fix |
|---|---|---|
| **API Error** | `[ERROR] Failed to fetch events` | Check internet connection and `RAPIDAPI_KEY` validity/limit. |
| **No Matches** | `matched 0 players` | This usually means player names in the API don't match the historical/prediction dataset. Check for naming discrepancies (e.g., "Nic Claxton" vs "Nicolas Claxton"). |
| **No Picks** | `[INFO] No qualifying bets found` | The model found no value today (EV <= 0 for all props). This is a valid safety result. |
| **Missing Projections** | `KeyError: 'pred_...'` | Ensure the `live_ensemble_2025.csv` contains the required prediction columns. |

### Verification
A successful run will end with:
`[SUCCESS] Phase I Complete. Final capital allocation saved.`
You should immediately see a new `.md` file in the current directory.
