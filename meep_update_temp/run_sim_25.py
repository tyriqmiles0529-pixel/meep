from simulation_engine import SportsbookSimulator, BacktestEngine
import pandas as pd

df = pd.read_csv("simulation_predictions_2025.csv")
sim = SportsbookSimulator()
engine = BacktestEngine(df, sim)

# $25 starting bankroll
results = engine.run(initial_bankroll=25.0, base_stake=2.5, max_stake=99999)

final = engine.bankroll_history[-1]
roi = ((final - 25) / 25) * 100

print()
print("=== SIMULATION RESULTS ===")
print(f"Starting Bankroll: $25.00")
print(f"Final Bankroll: ${final:.2f}")
print(f"ROI: {roi:.2f}%")
print(f"Total Bets: {len(results)}")
print(f"Win Rate: {results['outcome'].mean()*100:.1f}%")
