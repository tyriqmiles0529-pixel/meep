import pandas as pd
import numpy as np

try:
    df = pd.read_csv('simulation_results.csv')
    print(f"Loaded {len(df)} bets.")
except FileNotFoundError:
    print("simulation_results.csv not found.")
    exit()

# 1. Calibration Check
# Bucket probabilities
df['prob_bucket'] = (df['prob'] * 10).astype(int) / 10

# Win Flag
# Engine logic: 
#   Over: Win if Outcome=1 (Actual > Line)
#   Under: Win if Outcome=0 (Actual < Line)
# But wait, BacktestEngine Logic:
#   if bet['outcome'] == 1: pnl = ... (Over Logic?)
# Let's check BacktestEngine source code.
# The 'outcome' stored in results is 1 if Actual > Line.
# If side == 'Over', we win if outcome == 1.
# If side == 'Under', we win if outcome == 0.

df['won'] = np.where(
    ((df['side'] == 'Over') & (df['outcome'] == 1)) | 
    ((df['side'] == 'Under') & (df['outcome'] == 0)), 
    1, 0
)

calib = df.groupby('prob_bucket').agg({
    'won': ['mean', 'count'],
    'odds': 'mean'
})
print("\n--- Calibration Check (Win Rate vs Prob) ---")
print(calib)

# 2. EV Reality Check
# Expected Return vs Actual Return
df['expected_return'] = df['stake'] * df['ev'] # Roughly, ev is unitless? 
# calculated_ev function: (Prob * (Odds-1)) - ((1-Prob)*1) -> Profit per unit.
# So Expected PnL = Stake * EV.
total_expected_pnl = (df['stake'] * df['ev']).sum()
total_actual_pnl = df['pnl'].sum()

print("\n--- Profitability Check ---")
print(f"Total Expected PnL: {total_expected_pnl:.2f}")
print(f"Total Actual PnL:   {total_actual_pnl:.2f}")

# 3. Odds Analysis
print("\n--- Odds Stats ---")
print(df['odds'].describe())

# 4. Bankroll Check
print("\n--- Drawdown ---")
# Reconstruct running bankroll
df['cum_pnl'] = df['pnl'].cumsum()
print(f"Max Drawdown: {df['cum_pnl'].min()}")
