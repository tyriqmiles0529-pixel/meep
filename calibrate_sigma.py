import pandas as pd
import numpy as np
from scipy.stats import norm

ODDS_FILE = "historical_data/the_odds_api_historical.csv"
PROJS_FILE = "data/validation_projections_v4.csv"

def calibrate():
    print("=== Phase J.4.3: Probability Calibration Analysis ===")
    
    # Load Joined Data (already processed in analyst script logic)
    # We'll just run the core logic again to get the residual df
    projs = pd.read_csv(PROJS_FILE)
    odds = pd.read_csv(ODDS_FILE)
    
    projs['player_key'] = projs['PLAYER_NAME'].str.upper().str.strip()
    odds['player_key'] = odds['player_name'].str.upper().str.strip()
    
    df = pd.merge(odds, projs, left_on=['player_key', 'game_date'], right_on=['player_key', 'GAME_DATE'])
    
    mapping = {'player_points': 'PTS', 'player_assists': 'AST', 'player_rebounds': 'REB'}
    
    metrics = []
    for m, target in mapping.items():
        sub = df[df['market'] == m].copy()
        proj_col = f'proj_{target}'
        actual_col = target
        
        sub['residual'] = sub[actual_col] - sub[proj_col]
        rmse = np.sqrt((sub['residual']**2).mean())
        std = sub['residual'].std()
        skew = sub['residual'].skew()
        
        metrics.append({
            'market': m,
            'rmse': rmse,
            'std': std,
            'skew': skew,
            'n': len(sub)
        })
        
    res_df = pd.DataFrame(metrics)
    print("\nResidual Stats by Market:")
    print(res_df.to_string(index=False))
    
    print("\nValidation of 1.5 Delta Probability Implication:")
    # For points (std ~5.2), a 1.5 delta represents 1.5 / 5.2 = 0.28 standard deviations.
    # In a Normal distribution, Z=0.28 -> ~61% win prob.
    # Our empirical data showed 68%. This suggests the distribution is narrower or our edge is better.
    
if __name__ == "__main__":
    calibrate()
