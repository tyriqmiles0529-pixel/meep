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
results = engine.run(initial_bankroll=25.0, base_stake=2.5, max_stake=99999)

final = engine.bankroll_history[-1]
roi = ((final - 25) / 25) * 100

print()
print("=== 1-WEEK SIMULATION ===")
print(f"Starting: $25.00")
print(f"Final: ${final:.2f}")
print(f"ROI: {roi:.2f}%")
print(f"Bets: {len(results)}")
