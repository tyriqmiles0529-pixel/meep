from simulation_engine import SportsbookSimulator, BacktestEngine
import pandas as pd
from datetime import timedelta

df = pd.read_csv("simulation_predictions_oelm_v2.csv")
df['date'] = pd.to_datetime(df['date'])

# 1 week simulation
start_date = df['date'].min()
end_date = start_date + timedelta(days=7)
df_1wk = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

print(f"Date range: {start_date.date()} to {end_date.date()}")

sim = SportsbookSimulator()
engine = BacktestEngine(df_1wk, sim)

# $1 per parlay = ~$9/day (9 parlays per day)
# base_stake = $9 (total daily), max_stake = $9 (flat)
results = engine.run(initial_bankroll=25.0, base_stake=9.0, max_stake=9.0)

final = engine.bankroll_history[-1]
roi = ((final - 25) / 25) * 100

print()
print("=== 1-WEEK SIMULATION ($1 per parlay) ===")
print(f"Starting: $25.00")
print(f"Final: ${final:.2f}")
print(f"Profit: ${final - 25:.2f}")
print(f"ROI: {roi:.2f}%")
print(f"Bets: {len(results)}")
print(f"Win Rate: {results['outcome'].mean()*100:.1f}%")
