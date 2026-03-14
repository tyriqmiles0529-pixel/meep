import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

try:
    df = pd.read_csv('simulation_predictions_oelm_v2.csv')
    print(f"Loaded {len(df)} predictions.")
except FileNotFoundError:
    print("simulation_predictions_oelm_v2.csv not found.")
    exit()

# 1. Verify Player Names
print("\n--- Player Name Sample ---")
print(df['player_name'].head(10))
is_decoded = df['player_name'].dtype == 'O' # Object/String
print(f"Player Names Decoded? {is_decoded}")

# 2. Check RMSE per Target
print("\n--- Actual RMSE vs Assumed ---")
assumed_rmses = {'points': 4.5, 'rebounds': 2.0, 'assists': 1.8, 'three_pointers': 0.8}

for target in assumed_rmses.keys():
    subset = df[df['target'] == target]
    if subset.empty:
        continue
        
    mse = mean_squared_error(subset['actual_value'], subset['pred_value'])
    rmse = np.sqrt(mse)
    
    print(f"Target: {target}")
    print(f"  Assumed RMSE: {assumed_rmses[target]}")
    print(f"  Actual RMSE:  {rmse:.4f}")
    print(f"  Diff:         {rmse - assumed_rmses[target]:.4f}")
