---
description: Daily routine for NBA Paper Trading and Data Collection
---

# NBA Paper Trading Daily Workflow

Follow this routine every day to generate betting picks and accumulate historical data.

## 1. Morning: Generate Picks & Archive Odds
Run the paper trading script with today's date. This fetches live odds from RapidAPI, archives them to `historical_data/`, and outputs the best value bets.

```powershell
# Replace YYYY-MM-DD with today's date (e.g., 2025-12-14)
python paper_trading_live.py --date 2025-12-14
```

**Output Review:**
- Look for **"ALL QUALIFYING PICKS (Sorted by Value)"**.
- Focus on picks with positive **"Rec. Stake"** (Recommended Stake).
- Picks labeled **[ALT LINE PREDICTED]** suggest a massive edge—check your sportsbook for alternative lines (e.g., if line is 18.5 but we predict 25, look for Over 22.5 for better odds).

## 2. Verify Data Archival
Determine that the odds were saved correctly for future model training.

```powershell
ls historical_data
```
*Ensure `odds_2025-12-14.csv` was created.*

## 3. Evening: Check Results (Optional)
The system currently tracks results in `paper_ledger.json` automatically when you run the script, assuming we add a "Resolve" step later.
*Note: Currently `paper_trading_live.py` is set up for **Generating Picks**. Result resolution requires fetching box scores the next day.*

## 4. Weekly/Monthly: Retrain Models
As you accumulate more data (or every few weeks), retrain the core models to adapt to recent player form and new stats.

**Prerequisite**: Ensure `final_feature_matrix_with_per_min_1997_onward.csv` is in the `meep` folder.

```powershell
# Retrain for the current season (adjust start_season as needed)
python train_all_targets.py --epochs 20 --start_season 2025 --end_season 2025
```

## Troubleshooting
- **API Error**: If you see "0 props", check your RapidAPI key usage or try again in 5 minutes (rate limits).
- **Unicode Error**: If you see `charmap` errors, ensure your terminal supports UTF-8, though the script was patched to use `[OK]` instead of checkmarks.
