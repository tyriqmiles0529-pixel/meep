import pandas as pd
import numpy as np
from simulation_engine import SportsbookSimulator, BacktestEngine

df = pd.read_csv("simulation_predictions_oelm_v2.csv")

# Test different max stake percentages
STAKE_FRACTIONS = [
    ("1/10 (10%)", 0.10),
    ("1/8 (12.5%)", 0.125),
    ("1/6 (16.7%)", 0.167),
    ("1/5 (20%)", 0.20),
    ("1/4 (25%)", 0.25),
    ("1/3 (33%)", 0.33),
]

# Time periods
PERIODS = {
    "2 Weeks": 14,
    "1 Month": 30,
    "3 Months": 90,
}

print("=" * 80)
print("MAX STAKE OPTIMIZATION: Finding Optimal % of Bankroll")
print("Strategy: 2-3 Leg Focus | Starting: $15 | Base: $2.50")
print("=" * 80)

results_table = []

for name, fraction in STAKE_FRACTIONS:
    sim = SportsbookSimulator()
    engine = BacktestEngine(df, sim)
    
    # Modify run to use fraction-based max
    # max_stake = 0 disables cap, then we use fraction directly
    # We need to modify the engine temporarily
    
    # Run with max_stake=99999 so dynamic calc uses fraction
    results = engine.run(initial_bankroll=15.0, base_stake=2.5, max_stake=99999)
    
    row = {"Max Stake": name}
    
    if len(results) > 0:
        results['date'] = pd.to_datetime(results['date'])
        daily_bankroll = results.groupby('date')['bankroll_before'].last()
        start_date = daily_bankroll.index.min()
        
        for period_name, days in PERIODS.items():
            target_date = start_date + pd.Timedelta(days=days)
            mask = daily_bankroll.index <= target_date
            if mask.any():
                bankroll = daily_bankroll[mask].iloc[-1]
                row[period_name] = f"${bankroll:.2f}"
            else:
                row[period_name] = "N/A"
        
        row["Final"] = f"${engine.bankroll_history[-1]:.2f}"
        row["ROI"] = f"{((engine.bankroll_history[-1] - 15) / 15 * 100):.0f}%"
    
    results_table.append(row)
    print(f"\n{name}: Final ${engine.bankroll_history[-1]:.2f}")

print("\n" + "=" * 80)
print("COMPARISON TABLE")
print("=" * 80)
df_results = pd.DataFrame(results_table)
print(df_results.to_string(index=False))
