import pandas as pd
import numpy as np
from betting_strategy import BettingStrategy

# 1. Load Simulation Results (Bets)
try:
    bets = pd.read_csv('simulation_results.csv')
    bets['date'] = pd.to_datetime(bets['date'])
    print(f"Loaded {len(bets)} bets.")
except FileNotFoundError:
    print("simulation_results.csv not found.")
    exit()

# 2. Load Actual Stats (to get Minutes)
# We use BettingStrategy helper or just read csv directly
data_path = 'final_feature_matrix_with_per_min_1997_onward.csv'
df_actual = pd.read_csv(data_path, usecols=['date', 'player_name', 'minutes', 'season'])
df_actual['date'] = pd.to_datetime(df_actual['date'], utc=True).dt.tz_localize(None).dt.normalize()

# 3. Merge
bets = bets.rename(columns={'player': 'player_name'})
bets['player_name'] = bets['player_name'].astype(str)
df_actual['player_name'] = df_actual['player_name'].astype(str)
merged = pd.merge(bets, df_actual, on=['date', 'player_name'], how='left')

# 4. Analyze Failures
# Filter: Bets with high probability (>0.6) that LOST
high_prob_losses = merged[
    (merged['prob'] > 0.6) & 
    (merged['pnl'] < 0)
].copy()

print(f"\nHigh Prob (>0.6) Losses: {len(high_prob_losses)}")
print(f"Avg Actual Minutes in these losses: {high_prob_losses['minutes'].mean():.1f}")
print(f"Median Actual Minutes: {high_prob_losses['minutes'].median():.1f}")

# Distribution of minutes
print("\nMinutes Distribution for High Confidence Losers:")
print(high_prob_losses['minutes'].describe())

# How many were "DNFs" (e.g. < 15 mins)?
dnf_count = len(high_prob_losses[high_prob_losses['minutes'] < 15])
print(f"\nLosses where player played < 15 mins: {dnf_count} ({dnf_count/len(high_prob_losses):.1%})")

# Compare to Winners
high_prob_wins = merged[
    (merged['prob'] > 0.6) & 
    (merged['pnl'] > 0)
].copy()
print(f"\nAvg Actual Minutes in Winners: {high_prob_wins['minutes'].mean():.1f}")
