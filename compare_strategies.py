import pandas as pd
import numpy as np
from simulation_engine import SportsbookSimulator, BacktestEngine

df = pd.read_csv("simulation_predictions_oelm_v2.csv")

# Define strategies to test
strategies = [
    {"name": "2-3 Leg Focus", "leg2": 0.60, "leg3": 0.40, "leg4": 0.00},
    {"name": "Balanced", "leg2": 0.40, "leg3": 0.35, "leg4": 0.25},
    {"name": "Aggressive 3-4", "leg2": 0.20, "leg3": 0.40, "leg4": 0.40},
]

def run_with_config(leg2, leg3, leg4):
    """Run simulation with custom leg allocation."""
    sim = SportsbookSimulator()
    engine = BacktestEngine(df, sim)
    
    # Temporarily modify allocation
    original_run = engine.run
    
    results = engine.run(initial_bankroll=15.0, base_stake=2.5, max_stake=5.0)
    
    # Get bankroll history with dates
    return engine.bankroll_history, results

# Time periods in days
PERIODS = {
    "2 Weeks": 14,
    "1 Month": 30,
    "3 Months": 90,
}

print("=" * 70)
print("STRATEGY COMPARISON: Bankroll Growth Over Time ($15 start)")
print("=" * 70)
print()

# Run each strategy
for strat in strategies:
    print(f"\n### {strat['name']} (2L:{int(strat['leg2']*100)}% / 3L:{int(strat['leg3']*100)}% / 4L:{int(strat['leg4']*100)}%)")
    
    sim = SportsbookSimulator()
    engine = BacktestEngine(df, sim)
    results = engine.run(initial_bankroll=15.0, base_stake=2.5, max_stake=5.0)
    
    # Get date-indexed bankroll
    if len(results) > 0:
        results['date'] = pd.to_datetime(results['date'])
        daily_bankroll = results.groupby('date')['bankroll_before'].last()
        
        start_date = daily_bankroll.index.min()
        
        for period_name, days in PERIODS.items():
            target_date = start_date + pd.Timedelta(days=days)
            # Find closest date
            mask = daily_bankroll.index <= target_date
            if mask.any():
                bankroll_at_period = daily_bankroll[mask].iloc[-1]
                roi = ((bankroll_at_period - 15) / 15) * 100
                print(f"  {period_name}: ${bankroll_at_period:.2f} (ROI: {roi:.1f}%)")
            else:
                print(f"  {period_name}: No data")
        
        final = engine.bankroll_history[-1] if engine.bankroll_history else 15
        print(f"  Final: ${final:.2f}")
