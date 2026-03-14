import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJS_FILE = "data/validation_projections_v4.csv"
ODDS_FILE = "historical_data/the_odds_api_historical.csv"

def kelly_stress_test():
    print("=== Phase J.4.5: Kelly Stress Test (Simulated Backtest) ===")
    
    # 1. Load Joined Data
    projs = pd.read_csv(PROJS_FILE)
    odds = pd.read_csv(ODDS_FILE)
    
    projs['player_key'] = projs['PLAYER_NAME'].str.upper().str.strip()
    odds['player_key'] = odds['player_name'].str.upper().str.strip()
    
    df = pd.merge(odds, projs, left_on=['player_key', 'game_date'], right_on=['player_key', 'GAME_DATE'])
    
    mapping = {
        'player_points': {'proj': 'proj_PTS', 'actual': 'PTS'},
        'player_assists': {'proj': 'proj_AST', 'actual': 'AST'},
        'player_rebounds': {'proj': 'proj_REB', 'actual': 'REB'}
    }
    
    def extract_metrics(row):
        m = row['market']
        if m in mapping:
            return pd.Series([row[mapping[m]['proj']], row[mapping[m]['actual']]])
        return pd.Series([np.nan, np.nan])
        
    df[['prediction', 'actual_outcome']] = df.apply(extract_metrics, axis=1)
    df = df.dropna(subset=['prediction', 'actual_outcome'])
    
    # Filter for PRODUCTION Threshold delta >= 1.5
    df['delta'] = df['prediction'] - df['line']
    df['abs_delta'] = df['delta'].abs()
    df = df[df['abs_delta'] >= 1.5].copy()
    
    # American to Decimal Odds
    def to_dec(o):
        if o > 0: return 1 + (o/100)
        return 1 + (100/abs(o))
    
    df['dec_odds'] = df['odds'].apply(to_dec)
    
    # Hit calculation
    def check_hit(row):
        if row['delta'] >= 0: return 1 if row['actual_outcome'] > row['line'] else 0
        return 1 if row['actual_outcome'] < row['line'] else 0
    df['hit'] = df.apply(check_hit, axis=1)
    
    # Sort by date for simulation
    df = df.sort_values('game_date')
    
    # Win Prob Estimation (Simplified for backtest: use empirical bucket rate as proxy or sigma-normal)
    # Let's use the sigma-normal logic from BettingStrategy to be realistic
    from scipy.stats import norm
    sigmas = {'player_points': 5.2, 'player_assists': 2.1, 'player_rebounds': 2.4}
    
    def get_prob(row):
        s = sigmas.get(row['market'], 4.0)
        z = row['abs_delta'] / s
        return norm.cdf(z) # Simplified over/under prob
    
    df['win_prob'] = df.apply(get_prob, axis=1)
    
    # Kelly Stress Test Simulations
    fractions = [0.25, 0.10, 0.05]
    starting_bankroll = 1000.0
    
    results = {}
    
    for f in fractions:
        bankroll = starting_bankroll
        history = [bankroll]
        
        for _, row in df.iterrows():
            # Full Kelly = (bp - q) / b
            b = row['dec_odds'] - 1
            p = row['win_prob']
            q = 1 - p
            kelly = (b * p - q) / b
            
            # Application of fraction
            stake = bankroll * max(0, kelly) * f
            
            if row['hit']:
                bankroll += stake * b
            else:
                bankroll -= stake
                
            history.append(bankroll)
            if bankroll <= 0: break
            
        results[f] = history
        print(f"Fraction {f}: Final Bankroll = ${bankroll:.2f} | Count = {len(history)} bets")

    # Plot results (Optional, but good for summary)
    # plt.figure(figsize=(10,6))
    # for f, h in results.items():
    #     plt.plot(h, label=f"Kelly {f}")
    # plt.yscale('log')
    # plt.legend()
    # plt.savefig('kelly_stress_test.png')

if __name__ == "__main__":
    kelly_stress_test()
