import pandas as pd
import os
from run_phase_i import fetch_live_odds_pivoted
from betting_strategy import BettingStrategy

# 1. Load context
preds = pd.read_csv("predictions/live_ensemble_2025.csv")
preds['player_name_key'] = preds['player'].str.lower().str.strip()

# 2. Get odds
odds_pivoted = fetch_live_odds_pivoted()
if 'player_name' in odds_pivoted.columns:
    odds_pivoted = odds_pivoted.drop(columns=['player_name'])
merged = pd.merge(preds, odds_pivoted, on='player_name_key', how='inner')

# 3. Inspect
print(f"Total matched players: {len(merged)}")
bs = BettingStrategy(load_models=False)
all_odds = []
for t in bs.targets:
    col = f'odds_{t}'
    if col in merged.columns:
        all_odds.extend(merged[col].dropna().tolist())

if all_odds:
    sorted_odds = sorted(all_odds)
    print("Top 10 most negative odds found:")
    print(sorted_odds[:10])
else:
    print("No odds found in merged data.")
