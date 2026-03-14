
import pandas as pd
import numpy as np
import itertools
from monte_carlo_engine import MonteCarloEngine

def generate_structure_ledger(base_df, structure_type='2-Leg'):
    """
    Transforms a ledger of single bets into a ledger of parlay/structure bets.
    Assumes base_df has ['Run Date', 'Model Prob', 'Odds', 'EV']
    """
    hypothetical_bets = []
    
    # Filter for valid numeric data
    df = base_df.copy()
    df['Odds'] = pd.to_numeric(df['Odds'], errors='coerce')
    df['Model Prob'] = pd.to_numeric(df['Model Prob'], errors='coerce')
    df = df.dropna(subset=['Odds', 'Model Prob'])
    
    # Group by Date to form daily cards
    grouped = df.groupby('Run Date')
    
    print(f"Generating {structure_type} bets from {len(df)} singles across {len(grouped)} dates...")
    
    for date, group in grouped:
        # Sort by EV/Prob to pick "Best" legs for the structure
        # (Simulating we only parlay our best edges)
        best_legs = group.sort_values('Model Prob', ascending=False)
        
        legs_pool = best_legs.to_dict('records')
        
        if structure_type == '2-Leg':
            # Create pairs from top 6 legs (3 parlays)
            # Legs 0-1, 2-3, 4-5
            idx = 0
            while idx + 1 < len(legs_pool):
                if idx > 10: break # Cap daily volume
                l1 = legs_pool[idx]
                l2 = legs_pool[idx+1]
                
                # Combine
                comb_prob = l1['Model Prob'] * l2['Model Prob']
                
                d1 = (1 + l1['Odds']/100) if l1['Odds'] > 0 else (1 + 100/abs(l1['Odds']))
                d2 = (1 + l2['Odds']/100) if l2['Odds'] > 0 else (1 + 100/abs(l2['Odds']))
                comb_dec = d1 * d2
                
                us_odds = int((comb_dec - 1) * 100) if comb_dec >= 2.0 else int(-100 / (comb_dec - 1))
                
                hypothetical_bets.append({
                    'Run Date': date,
                    'Model Prob': comb_prob,
                    'Odds': us_odds,
                    'Type': '2-Leg Parlay'
                })
                idx += 2
                
        elif structure_type == '3-Leg':
            # Tripulets
            idx = 0
            while idx + 2 < len(legs_pool):
                if idx > 12: break
                l1 = legs_pool[idx]; l2 = legs_pool[idx+1]; l3 = legs_pool[idx+2]
                
                comb_prob = l1['Model Prob'] * l2['Model Prob'] * l3['Model Prob']
                
                d1 = (1 + l1['Odds']/100) if l1['Odds'] > 0 else (1 + 100/abs(l1['Odds']))
                d2 = (1 + l2['Odds']/100) if l2['Odds'] > 0 else (1 + 100/abs(l2['Odds']))
                d3 = (1 + l3['Odds']/100) if l3['Odds'] > 0 else (1 + 100/abs(l3['Odds']))
                comb_dec = d1 * d2 * d3
                
                us_odds = int((comb_dec - 1) * 100) if comb_dec >= 2.0 else int(-100 / (comb_dec - 1))
                
                hypothetical_bets.append({
                    'Run Date': date,
                    'Model Prob': comb_prob,
                    'Odds': us_odds,
                    'Type': '3-Leg Parlay'
                })
                idx += 3

        elif structure_type == '4-Leg':
            idx = 0
            while idx + 3 < len(legs_pool):
                if idx > 12: break
                legs = [legs_pool[idx+i] for i in range(4)]
                
                comb_prob = np.prod([l['Model Prob'] for l in legs])
                dec_odds = np.prod([(1 + l['Odds']/100) if l['Odds'] > 0 else (1 + 100/abs(l['Odds'])) for l in legs])
                
                us_odds = int((dec_odds - 1) * 100) if dec_odds >= 2.0 else int(-100 / (dec_odds - 1))
                
                hypothetical_bets.append({
                    'Run Date': date,
                    'Model Prob': comb_prob,
                    'Odds': us_odds,
                    'Type': '4-Leg Parlay'
                })
                idx += 4

        elif structure_type == 'Round Robin (3x2)':
            # Take top 3 legs, make 3x 2-leg parlays
            if len(legs_pool) >= 3:
                top_3 = legs_pool[:3]
                combos = itertools.combinations(top_3, 2)
                for pair in combos:
                    l1, l2 = pair
                    comb_prob = l1['Model Prob'] * l2['Model Prob']
                    d1 = (1 + l1['Odds']/100) if l1['Odds'] > 0 else (1 + 100/abs(l1['Odds']))
                    d2 = (1 + l2['Odds']/100) if l2['Odds'] > 0 else (1 + 100/abs(l2['Odds']))
                    comb_dec = d1 * d2
                    us_odds = int((comb_dec - 1) * 100) if comb_dec >= 2.0 else int(-100 / (comb_dec - 1))
                    
                    hypothetical_bets.append({
                        'Run Date': date,
                        'Model Prob': comb_prob,
                        'Odds': us_odds,
                        'Type': 'RR Sub-Parlay'
                    })

        elif structure_type == 'Lotto (10-Leg)':
            # Make one giant parlay from top 10
            if len(legs_pool) >= 8:
                legs = legs_pool[:10]
                comb_prob = np.prod([l['Model Prob'] for l in legs])
                dec_odds = np.prod([(1 + l['Odds']/100) if l['Odds'] > 0 else (1 + 100/abs(l['Odds'])) for l in legs])
                # Cap odds realistically? Books cap at +250000 or similar
                us_odds = int((dec_odds - 1) * 100) if dec_odds >= 2.0 else int(-100 / (dec_odds - 1))
                
                hypothetical_bets.append({
                    'Run Date': date,
                    'Model Prob': comb_prob,
                    'Odds': us_odds,
                    'Type': 'Lotto Slip'
                })

    return pd.DataFrame(hypothetical_bets)

if __name__ == "__main__":
    # Load Real Data
    try:
        raw_ledger = pd.read_csv("betting_ledger.csv")
    except:
        print("No ledger found.")
        exit()

    engine = MonteCarloEngine(initial_bankroll=20.0)
    summary = []

    strategies = ['2-Leg', '3-Leg', '4-Leg', 'Round Robin (3x2)', 'Lotto (10-Leg)']
    
    for stra in strategies:
        print(f"\nBuilding Ledger for: {stra}")
        h_df = generate_structure_ledger(raw_ledger, structure_type=stra)
        
        if h_df.empty:
            print("Not enough data to form structures.")
            continue
            
        print(f"Simulating {len(h_df)} {stra} bets...")
        # Use Flat Stake to isolate Structure Performance (removing Kelly sizing noise for now)
        sim_res = engine.run_strategy_simulation(h_df, strategy_name="Flat 1U", n_simulations=1000)
        
        # Stats
        ruin = sim_res['ruin'].mean()
        roi = sim_res['roi'].mean()
        dd = sim_res['max_drawdown'].mean()
        sharpe = sim_res['roi'].mean() / (sim_res['roi'].std() + 1e-9) # Simplified Sharpe
        
        summary.append({
            'Structure': stra,
            'ROI': f"{roi*100:.1f}%",
            'Survivability': f"{(1-ruin)*100:.1f}%",
            'Drawdown': f"{dd*100:.1f}%",
            'Sharpe': round(sharpe, 2)
        })

    print("\n\n=== PARLAY STRUCTURE STRESS TEST RESULTS (Flat 1U) ===")
    res_df = pd.DataFrame(summary)
    print(res_df.to_markdown(index=False))
