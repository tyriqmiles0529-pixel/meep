import pandas as pd
from simulation_engine import SportsbookSimulator, BacktestEngine

df = pd.read_csv("simulation_predictions_oelm_v2.csv")
sim = SportsbookSimulator()
engine = BacktestEngine(df, sim)
results = engine.run(initial_bankroll=15.0, base_stake=2.5, max_stake=5.0)

# Separate win rates
parlays = results[results["side"] == "Parlay"]
straights = results[results["side"] != "Parlay"]

print()
print("=== WIN RATE BREAKDOWN ===")
straight_wr = straights["outcome"].mean()*100 if len(straights) > 0 else 0
straight_hits = int(straights["outcome"].sum()) if len(straights) > 0 else 0
print(f"Individual Pick Win Rate: {straight_wr:.1f}% ({straight_hits}/{len(straights)} hits)")

parlay_wr = parlays["outcome"].mean()*100 if len(parlays) > 0 else 0
parlay_hits = int(parlays["outcome"].sum()) if len(parlays) > 0 else 0
print(f"Parlay Success Rate: {parlay_wr:.1f}% ({parlay_hits}/{len(parlays)} hits)")

print()
print("Parlay breakdown by legs:")
for leg in ["2-leg", "3-leg", "4-leg"]:
    subset = parlays[parlays["target"] == leg]
    if len(subset) > 0:
        wr = subset["outcome"].mean()*100
        print(f"  {leg}: {wr:.1f}% win rate ({len(subset)} bets)")
