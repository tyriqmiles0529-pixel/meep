
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
from multiprocessing import Pool, cpu_count

class MonteCarloEngine:
    def __init__(self, initial_bankroll=1000.0):
        self.initial_bankroll = initial_bankroll

    def simulate_bet_outcome(self, true_prob):
        """Simulate a single bet outcome based on true probability."""
        return np.random.random() < true_prob

    def run_strategy_simulation(self, bets_df, strategy_name="Flat Stake", n_simulations=1000, max_daily_risk=0.10):
        """
        Run Monte Carlo simulation for a given betting strategy.
        
        bets_df: DataFrame containing ['Model Prob', 'Odds', 'Edge']
        n_simulations: Number of authorized timelines (seasons) to simulate.
        """
        results = []
        
        print(f"--- Running {n_simulations} Simulations for Strategy: {strategy_name} ---")
        
        possible_outcomes = []
        
        # Pre-generate random outcomes for vectorization if possible, 
        # but for bankroll path dependency (Kelly), we often need sequential.
        # We will loop for clarity and realism first.
        
        sim_results = []
        
        for sim_id in range(n_simulations):
            bankroll = self.initial_bankroll
            history = [bankroll]
            peak_bankroll = bankroll
            max_drawdown = 0.0
            ruin = False
            
            # Group by "Day" to enforce daily limits? 
            # If ledger has dates, we group. If not, sequential.
            if 'Run Date' in bets_df.columns:
                days = bets_df.groupby('Run Date')
                day_keys = sorted(days.groups.keys())
            else:
                # Treat whole DF as one sequence if no dates
                days = [('All', bets_df)]
                day_keys = ['All']

            for day in day_keys:
                if bankroll <= 0:
                    ruin = True
                    break
                    
                day_bets = days.get_group(day) if 'Run Date' in bets_df.columns else bets_df
                
                # Available for today
                daily_budget = bankroll * max_daily_risk
                spent_today = 0
                
                daily_pnl = 0
                
                for _, bet in day_bets.iterrows():
                    # 1. Determine Stake
                    stake = 0
                    if strategy_name == "Flat 1U":
                        unit = bankroll / 50.0 # Conservative unit
                        stake = unit
                    elif strategy_name == "Kelly":
                        # f = (bp - q) / b
                        dec_odds = (1 + bet['Odds']/100) if bet['Odds'] > 0 else (1 + 100/abs(bet['Odds']))
                        b = dec_odds - 1
                        p = bet['Model Prob']
                        q = 1 - p
                        f = (b*p - q) / b
                        stake = bankroll * (f * 0.25) # Quarter Kelly
                        
                    # Cap stake at remaining budget
                    if spent_today + stake > daily_budget:
                        stake = daily_budget - spent_today
                    
                    if stake <= 0: continue
                    
                    spent_today += stake
                    
                    # 2. Simulate Outcome
                    # Assumption: Model Prob IS True Prob (Calibration assumption)
                    # To stress test: We can degrade Model Prob by a factor
                    true_prob = bet['Model Prob'] 
                    
                    won = self.simulate_bet_outcome(true_prob)
                    
                    # 3. PnL
                    if won:
                        dec_odds = (1 + bet['Odds']/100) if bet['Odds'] > 0 else (1 + 100/abs(bet['Odds']))
                        profit = stake * (dec_odds - 1)
                        daily_pnl += profit
                    else:
                        daily_pnl -= stake
                
                # End of Day Update
                bankroll += daily_pnl
                history.append(bankroll)
                
                peak_bankroll = max(peak_bankroll, bankroll)
                current_dd = (peak_bankroll - bankroll) / peak_bankroll
                max_drawdown = max(max_drawdown, current_dd)

            # End Simulation
            sim_results.append({
                'sim_id': sim_id,
                'final_bankroll': bankroll,
                'roi': (bankroll - self.initial_bankroll) / self.initial_bankroll,
                'max_drawdown': max_drawdown,
                'ruin': ruin
            })
            
        return pd.DataFrame(sim_results)

    def run_survival_simulation(self, bets_df, duration_days=30, n_simulations=1000, stop_loss_pct=0.08):
        """
        Phase S2 Simulation: Long-term Survival with Strict Daily Stop-Loss.
        Uses Bootstrapping of historical days to simulate future timelines.
        """
        print(f"--- Running Phase S2 Survival Sim ({duration_days} Days, {n_simulations} Sims) ---")
        
        # 1. Prepare Daily Batches
        if 'Run Date' in bets_df.columns:
            days_grouped = bets_df.groupby('Run Date')
            # Phase S2 Refinement: Exclude outlier backfill days (Volume > 100)
            # This ensures we simulate realistic daily volume (15-30 bets), not 900-bet anomalies.
            day_keys = [k for k in days_grouped.groups.keys() if len(days_grouped.get_group(k)) < 100]
            
            if not day_keys:
                 print("Warning: All days have high volume. Using all available days.")
                 day_keys = list(days_grouped.groups.keys())
            else:
                 print(f"Using {len(day_keys)} realistic days for bootstrapping (excluded outliers).")
        else:
            # Fallback for dummy: Chunk into 5-bet days
            n_chunks = max(1, len(bets_df) // 5)
            day_keys = list(range(n_chunks))
            # Mock grouping
            days_grouped = {k: bets_df.iloc[k*5:(k+1)*5] for k in day_keys}
            
        sim_results = []
        
        for sim_id in range(n_simulations):
            bankroll = self.initial_bankroll
            history = [bankroll]
            peak_bankroll = bankroll
            max_drawdown = 0.0
            ruin = False
            
            # Bootstrap Sequence of Days
            # We randomly select a "Day Pattern" from history for each day of the simulation
            chosen_day_indices = np.random.choice(len(day_keys), size=duration_days)
            
            for day_idx in chosen_day_indices:
                if bankroll <= 0:
                    ruin = True
                    break
                
                day_key = day_keys[day_idx]
                try:
                    day_bets = days_grouped.get_group(day_key)
                except AttributeError:
                    day_bets = days_grouped[day_key]
                
                daily_start_bankroll = bankroll
                daily_pnl = 0
                
                # S2 Rule: Flat Sizing (1.5% of Bankroll for Core Singles)
                # Survival Mode = Conservative.
                unit_size = daily_start_bankroll * 0.015 
                
                # Shuffle bets within the day to simulate random timing of outcomes
                day_bets_shuffled = day_bets.sample(frac=1)
                
                for _, bet in day_bets_shuffled.iterrows():
                    # S2 Rule: Daily Stop Loss
                    # If current daily loss exceeds 8%, HALT TRADING.
                    if daily_pnl < -(daily_start_bankroll * stop_loss_pct):
                        break # Stop trading for this day
                        
                    stake = unit_size
                    
                    # Simulate Outcome
                    true_prob = bet['Model Prob']
                    won = self.simulate_bet_outcome(true_prob)
                    
                    if won:
                        # Handle varied odds formats if needed
                        o = float(bet['Odds'])
                        dec_odds = (1 + o/100) if o > 0 else (1 + 100/abs(o))
                        profit = stake * (dec_odds - 1)
                        daily_pnl += profit
                    else:
                        daily_pnl -= stake
                
                bankroll += daily_pnl
                history.append(bankroll)
                
                peak_bankroll = max(peak_bankroll, bankroll)
                current_dd = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
                max_drawdown = max(max_drawdown, current_dd)
            
            sim_results.append({
                'sim_id': sim_id,
                'final_bankroll': bankroll,
                'roi': (bankroll - self.initial_bankroll) / self.initial_bankroll,
                'max_drawdown': max_drawdown,
                'ruin': ruin
            })
            
        return pd.DataFrame(sim_results)

    def analyze_results(self, results_df):
        """Generate Sharpe Ratio, Ruin Prob, and Value at Risk."""
        ruin_rate = results_df['ruin'].mean()
        avg_roi = results_df['roi'].mean()
        median_roi = results_df['roi'].median()
        avg_dd = results_df['max_drawdown'].mean()
        
        print(f"Survivability: {100*(1-ruin_rate):.1f}%")
        print(f"Avg ROI: {avg_roi*100:.1f}%")
        print(f"Median ROI: {median_roi*100:.1f}%")
        print(f"Avg Max Drawdown: {avg_dd*100:.1f}%")
        
        return results_df

# --- Comparison Runner ---
if __name__ == "__main__":
    # 1. Load Data
    try:
        # Load Ledger, filtered for valid probs
        if pd.io.common.file_exists("betting_ledger.csv"):
            all_bets = pd.read_csv("betting_ledger.csv")
        else:
            raise FileNotFoundError("betting_ledger.csv not found")

        # Ensure we have numeric probs
        all_bets = all_bets[pd.to_numeric(all_bets['Model Prob'], errors='coerce').notnull()]
        all_bets['Model Prob'] = all_bets['Model Prob'].astype(float)
        all_bets['Odds'] = all_bets['Odds'].astype(float)
        
        print(f"Loaded {len(all_bets)} historical bets for simulation.")
        
    except Exception as e:
        print(f"Warning: Using Dummy Data ({e})")
        # Create Dummy Data if Ledger fails
        data = {
            'Run Date': ['2025-01-01']*50 + ['2025-01-02']*50,
            'Model Prob': np.random.uniform(0.55, 0.75, 100), # S2 requires >56% filtering logic implicitly in data quality
            'Odds': np.random.choice([-110, -120, -150, -200], 100) # Conservative odds
        }
        all_bets = pd.DataFrame(data)

    # User Request: Starting Bankroll 20
    engine = MonteCarloEngine(initial_bankroll=20.0)
    
    # 2. Run Strategies
    # Phase S2: 1 Month, 3 Months, 1 Year
    periods = [30, 90, 365]
    
    for p in periods:
        res = engine.run_survival_simulation(all_bets, duration_days=p, n_simulations=500, stop_loss_pct=0.08)
        print(f"\n--- Results for {p} Days (Phase S2 Hardening) ---")
        engine.analyze_results(res)
    
    # Export One Year for detailed review
    res.to_csv("simulation_results_s2_1year.csv", index=False)
