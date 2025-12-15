import pandas as pd
import itertools
import json
import os
from betting_strategy import BettingStrategy

def run_optimization():
    print("Starting Optimization Loop...")
    
    # Define Parameter Grid
    confidence_thresholds = [0, 10, 20]
    kelly_fractions = [0.125, 0.25, 0.5]
    min_evs = [0.0, 0.02, 0.05]
    
    # Combinations
    combinations = list(itertools.product(confidence_thresholds, kelly_fractions, min_evs))
    
    results = []
    
    # Initialize Strategy
    # Note: We assume models are available in 'models' dir
    bs = BettingStrategy(models_dir='models', data_path='final_feature_matrix_with_per_min_1997_onward.csv')
    bs.load_resources() # Load initial resources (data processor)
    
    best_sharpe = -100
    best_params = None
    
    for conf, kelly, ev in combinations:
        print(f"\nTesting: Conf={conf}, Kelly={kelly}, MinEV={ev}")
        
        # Run Backtest (2021-2026)
        # We use a shorter range for speed if needed, but 2020-2026 is robust.
        metrics = bs.backtest(start_season=2021, end_season=2026, 
                              confidence_threshold=conf, 
                              kelly_fraction=kelly, 
                              min_ev=ev)
                              
        metrics['params'] = {
            'confidence_threshold': conf,
            'kelly_fraction': kelly,
            'min_ev': ev
        }
        
        # Remove history DF to save space in JSON
        if 'history' in metrics:
            del metrics['history']
            
        results.append(metrics)
        
        print(f"Result: ROI={metrics['roi']:.2%}, Sharpe={metrics['sharpe']:.2f}, DD={metrics['drawdown']:.2%}")
        
        if metrics['sharpe'] > best_sharpe:
            best_sharpe = metrics['sharpe']
            best_params = metrics['params']
            
    # Save Results
    with open('OPTIMIZATION_RESULTS.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\nOptimization Complete.")
    print(f"Best Params (Sharpe): {best_params}")
    print(f"Best Sharpe: {best_sharpe:.2f}")

if __name__ == "__main__":
    run_optimization()
