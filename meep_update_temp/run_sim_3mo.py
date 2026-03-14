from simulation_engine import SportsbookSimulator, BacktestEngine
import pandas as pd
from datetime import timedelta

df = pd.read_csv("simulation_predictions_oelm_v2.csv")
df['date'] = pd.to_datetime(df['date'])

# Filter to only 3 months (90 days) from first date
start_date = df['date'].min()
end_date = start_date + timedelta(days=90)
df_3mo = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

print(f"Date range: {start_date.date()} to {end_date.date()}")
print(f"Total predictions: {len(df_3mo)}")

sim = SportsbookSimulator()
engine = BacktestEngine(df_3mo, sim)

# $25 starting bankroll, 3 month period
results = engine.run(initial_bankroll=25.0, base_stake=2.5, max_stake=99999)

final = engine.bankroll_history[-1]
roi = ((final - 25) / 25) * 100

print()
print("=== 3-MONTH SIMULATION RESULTS ===")
print(f"Starting Bankroll: $25.00")
print(f"Final Bankroll: ${final:.2f}")
print(f"ROI: {roi:.2f}%")
print(f"Total Bets: {len(results)}")
print(f"Win Rate: {results['outcome'].mean()*100:.1f}%")
