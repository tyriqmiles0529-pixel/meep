import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def main():
    print("Loading dataset...")
    df = pd.read_csv('final_feature_matrix_with_per_min_1997_onward.csv')
    
    # Ensure date is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['player_id', 'date'])
    
    targets = ['points', 'assists', 'rebounds', 'three_pointers']
    windows = [3, 5, 10]
    
    results = []
    
    print("\n=== Simple Rolling Average Baseline ===")
    
    for target in targets:
        if target not in df.columns:
            print(f"Skipping {target} (not found)")
            continue
            
        print(f"\nTarget: {target}")
        
        # Calculate rolling averages (shift 1 to avoid leakage)
        for w in windows:
            # Group by player, shift 1, then rolling mean
            # We can use the transform method for efficiency
            pred_col = f'pred_{target}_roll_{w}'
            df[pred_col] = df.groupby('player_id')[target].transform(lambda x: x.shift(1).rolling(w).mean())
            
            # Evaluate on non-NaN rows (where we have enough history)
            valid_mask = df[pred_col].notna() & df[target].notna()
            
            y_true = df.loc[valid_mask, target]
            y_pred = df.loc[valid_mask, pred_col]
            
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            
            print(f"  Window {w}: RMSE={rmse:.4f}, MAE={mae:.4f}")
            results.append({'target': target, 'window': w, 'rmse': rmse, 'mae': mae})

    # Save results summary
    print("\nSummary:")
    res_df = pd.DataFrame(results)
    print(res_df)

if __name__ == "__main__":
    main()
