import pandas as pd
import numpy as np

def calculate_directional_accuracy(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return

    # We need a 'season_average' or 'rolling_average' column to act as the Line
    # If not present, we can approximate it if we have 'actual_value' history
    
    # Ideally, prediction file has 'average' or similar.
    # Let's inspect columns first or assume we calc it.
    
    print(f"Analyzing {len(df)} predictions...")
    
    # Simple Proxy: Use the previous game's Season Average (if available) or calculate rolling locally
    # Group by Player and sort by date
    if 'player_name' in df.columns and 'date' in df.columns and 'actual_value' in df.columns:
        df = df.sort_values(['player_name', 'date'])
        
        # Calculate trailing average (proxy for Line)
        # Using expanding mean shifted by 1 (so we don't peek at today)
        df['proxy_line'] = df.groupby('player_name')['actual_value'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        
        # Drop first game (no average)
        df = df.dropna(subset=['proxy_line', 'pred_value', 'actual_value'])
        
        results = []
        
        for target in df['target'].unique():
            subset = df[df['target'] == target]
            if len(subset) < 100: continue
            
            # Prediction Direction relative to Proxy Line
            # Edge: How much do we differ from "The Line"?
            subset['edge'] = subset['pred_value'] - subset['proxy_line']
            
            # Bet Logic: 
            # If Model > Line + Threshold -> Over
            # If Model < Line - Threshold -> Under
            # Using Threshold = 0.5 (half a point edge)
            threshold = 0.5
            
            bets = subset[abs(subset['edge']) > threshold].copy()
            
            bets['bet_direction'] = np.where(bets['edge'] > 0, 'OVER', 'UNDER')
            
            # Outcome Logic
            # Result > Line -> Over Wins
            # Result < Line -> Under Wins
            bets['result_direction'] = np.where(bets['actual_value'] > bets['proxy_line'], 'OVER', 'UNDER')
            
            # Tie (Push) -> Result == Line (Exact Push)
            # In expanding mean, exact is rare, but we handle it.
            
            # Win Check
            bets['win'] = bets['bet_direction'] == bets['result_direction']
            
            win_rate = bets['win'].mean()
            count = len(bets)
            
            results.append({
                'Target': target,
                'Win Rate': f"{win_rate:.1%}",
                'Bets Placed': count,
                'Avg Edge': f"{bets['edge'].abs().mean():.2f}"
            })
            
        print("\n=== Directional Accuracy (vs Season Avg Proxy) ===")
        print(pd.DataFrame(results))
        
    else:
        print("Missing required columns (player_name, date, actual_value) for analysis.")

if __name__ == "__main__":
    # Check if we have prediction files
    # Try the one from check_rmse
    calculate_directional_accuracy('simulation_predictions_oelm_v2.csv')
