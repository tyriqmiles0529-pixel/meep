import pandas as pd
import numpy as np
from simulation_engine import SportsbookSimulator, BacktestEngine
from betting_strategy import BettingStrategy
import logging
import os
print(f"CWD: {os.getcwd()}")
print(f"Files in CWD: {os.listdir()}")



def process_date_batch(args):
    """
    Worker function for parallel processing of dates.
    args: (date_list, models_dir, data_path, season_map)
    """
    dates, models_dir, data_path, season_map = args
    
    # Each worker needs its own BettingStrategy instance to avoid shared state issues
    # But loading strategy is heavy (data loading). 
    # Optimization: Pass the relevant data slices directly or use shared memory.
    # For simplicity v1: Workers load strategy once.
    # Actually, data loading is fast enough if we shard by season.
    
    # Better approach: Iterate seasons in main process, load model once, 
    # then parallelize the daily loop within that season using a shared dataframe (via global or copy).
    # Since Python MP pickles args, passing large DF is bad.
    
    # HYBRID APPROACH:
    # 1. Main process iterates Seasons.
    # 2. Main process loads Models for Season S.
    # 3. Main process spawns pool to process Dates in Season S.
    return []

# Optimized Generator
def generate_predictions_optimized(start_season=2021, end_season=2026):
    from multiprocessing import Pool, cpu_count
    import os
    
    print("Initializing BettingStrategy...")
    bs = BettingStrategy(models_dir='models', data_path='final_feature_matrix_with_per_min_1997_onward.csv')
    bs.load_resources() # Loads Dataframe
    
    # Verify Encoders
    if 'player_name' in bs.processor.label_encoders:
        print("Player Name Encoder found.")
    else:
        print("WARNING: Player Name Encoder NOT found.")
    
    df = bs.processor.df
    print(f"Total Data Rows: {len(df)}")
    df_test = df[(df['season'] >= start_season) & (df['season'] <= end_season)].copy()
    print(f"Test Data Rows ({start_season}-{end_season}): {len(df_test)}")
    
    all_predictions = []
    
    # Iterate by Season to minimize model loading overhead
    unique_seasons = sorted(df_test['season'].unique())
    
    for season in unique_seasons:
        print(f"Processing Season {season}...")
        bs.load_season_models(season)
        
        season_df = df_test[df_test['season'] == season].copy()
        
        # Generate Embeddings (Vectorized)
        if 'global' in bs.ft_extractors:
            print(f"  Generating Embeddings for {len(season_df)} rows...")
            cat_cols = bs.processor.get_cat_cols()
            # Ensure order matches what FT expects? 
            # FTTransformerFeatureExtractor.transform expects numpy array of encoded ints
            try:
                # Ensure numeric
                for col in cat_cols:
                    season_df[col] = pd.to_numeric(season_df[col], errors='coerce').fillna(0).astype(int)
                
                X_cat = season_df[cat_cols].values
                
                # Transform
                embeddings = bs.ft_extractors['global'].transform(X_cat)
                
                # Add to DF
                emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
                df_emb = pd.DataFrame(embeddings, columns=emb_cols, index=season_df.index)
                season_df = pd.concat([season_df, df_emb], axis=1)
                print("  Embeddings generated.")
            except Exception as e:
                import traceback
                print(f"  Embedding Generation Failed: {e}")
                print(traceback.format_exc())
        
        # Features
        features = bs.processor.feature_columns.copy()
        if 'global' in bs.ft_extractors:
             features += [c for c in season_df.columns if c.startswith('emb_')]
        
        valid_features = [f for f in features if f in season_df.columns]
        X_season = season_df[valid_features]
        
        season_preds_df = season_df[['date', 'player_name', 'season']].copy()
        
        for target in bs.targets:
            if target in bs.predictors:
                # Batch predict!
                preds = bs.predictors[target].predict(X_season, use_stacking=True)
                season_preds_df[f'pred_{target}'] = preds
        
        # Merge priros
        for target in bs.targets:
             abbr_map = {
                 'points': 'prior_pts', 
                 'rebounds': 'prior_reb', 
                 'assists': 'prior_ast',
                 'three_pointers': 'threePointersMade_last_game' # Proxy
             }
             prior_col = abbr_map.get(target)
             if prior_col in season_df.columns:
                 season_preds_df[f'prior_{target}'] = season_df[prior_col].values
             else:
                 season_preds_df[f'prior_{target}'] = np.nan
                 
        # Ensure pred cols exist
        for target in bs.targets:
            if f'pred_{target}' not in season_preds_df.columns:
                season_preds_df[f'pred_{target}'] = np.nan
                 
        # Merge Actuals
        for target in bs.targets:
            if target in season_df.columns:
                season_preds_df[f'actual_{target}'] = season_df[target].values
            else:
                 season_preds_df[f'actual_{target}'] = np.nan
            
        # Merge Opponent Def
        if 'opp_pts_allowed_last_20' in season_df.columns:
            season_preds_df['opp_def'] = season_df['opp_pts_allowed_last_20'].values
        else:
            season_preds_df['opp_def'] = np.nan
            
        all_predictions.append(season_preds_df)
        
    full_preds = pd.concat(all_predictions)
    print(f"Generated {len(full_preds)} predictions (Encoded).")
    print(f"Sample Encoded Names: {full_preds['player_name'].head(5).tolist()}")
    
    # Decode Player Names
    if 'player_name' in bs.processor.label_encoders:
        le = bs.processor.label_encoders['player_name']
        try:
            full_preds['player_name'] = full_preds['player_name'].astype(int)
            full_preds['player_name'] = le.inverse_transform(full_preds['player_name'])
            print(f"Decoded Names successfully. Sample: {full_preds['player_name'].head(5).tolist()}")
        except Exception as e:
            print(f"Decoding Failed: {e}")
    
    # ===== BIAS CORRECTION (Walk-Forward) =====
    print("\nCalculating Walk-Forward Bias Correction...")
    bias_dict = {}
    for target in bs.targets:
        bias_dict[target] = {}
        pred_col = f'pred_{target}'
        actual_col = f'actual_{target}'
        
        if pred_col not in full_preds.columns or actual_col not in full_preds.columns:
            continue
            
        for season in sorted(full_preds['season'].unique()):
            # Use ONLY data from PRIOR seasons to calculate bias (walk-forward)
            prior_mask = (full_preds['season'] < season) & full_preds[pred_col].notna() & full_preds[actual_col].notna()
            if prior_mask.sum() > 100: # Need enough samples
                errors = full_preds.loc[prior_mask, pred_col] - full_preds.loc[prior_mask, actual_col]
                bias_dict[target][season] = errors.mean()
            else:
                bias_dict[target][season] = 0.0 # No bias correction for first season or sparse data
        print(f"Bias for {target}: {bias_dict[target]}")
    
    # Apply OELM and Prob Calc Vectorized
    # Iterate targets
    from scipy.stats import norm, skewnorm
    rmses = {'points': 4.5, 'rebounds': 2.0, 'assists': 1.8, 'three_pointers': 0.8}
    skews = {'points': 2.0, 'rebounds': 2.5, 'assists': 2.2, 'three_pointers': 1.5}
    LEAGUE_AVG_PTS_ALLOWED = 112.0
    
    final_output = []
    
    for target in bs.targets:
        # Check if columns exist before filtering
        req_cols = [f'pred_{target}', f'prior_{target}', f'actual_{target}']
        if not all(c in full_preds.columns for c in req_cols):
             print(f"Skipping {target}: Missing columns in predictions.")
             continue
             
        # Filter rows where we have pred, prior, actual
        mask = (
            full_preds[f'pred_{target}'].notna() & 
            full_preds[f'prior_{target}'].notna() & 
            full_preds[f'actual_{target}'].notna()
        )
        subset = full_preds[mask].copy()
        print(f"Target {target}: {len(subset)} valid rows for simulation.")
        
        if subset.empty: continue
        
        # OELM Logic (Vectorized)
        base_line = subset[f'prior_{target}']
        smart_line = base_line.copy()
        
        if target == 'points' and 'opp_def' in subset.columns:
            adj_factor = subset['opp_def'] / LEAGUE_AVG_PTS_ALLOWED
            adj_factor = 1 + (adj_factor - 1) * 0.7
            smart_line = base_line * adj_factor
            
        line_val = (smart_line * 2).round() / 2
        pred_val = subset[f'pred_{target}']
        
        # ===== APPLY BIAS CORRECTION =====
        # Map season to its bias value
        season_bias = subset['season'].map(lambda s: bias_dict.get(target, {}).get(s, 0.0))
        corrected_pred = pred_val - season_bias
        
        # Prob Calc (using CORRECTED predictions)
        rmse = rmses.get(target, 4.5)
        skew_a = skews.get(target, 0)
        
        # True Probability (Model's view, AFTER bias correction)
        true_prob = 1 - skewnorm.cdf(line_val, skew_a, loc=corrected_pred, scale=rmse)
        
        # House Probability (Inject Noise to mimic inefficiencies/market noise)
        # Noise level 5%
        noise = np.random.normal(0, 0.05, size=len(true_prob))
        prob_over = np.clip(true_prob + noise, 0.01, 0.99)
        
        subset_out = pd.DataFrame({
            'date': subset['date'],
            'player_name': subset['player_name'],
            'target': target,
            'pred_value': pred_val,
            'corrected_pred': corrected_pred, # For debugging
            'bias': season_bias, # For debugging
            'line_value': line_val,
            'actual_value': subset[f'actual_{target}'],
            'prob_over': prob_over, # Used for Odds Generation (House)
            'true_prob': true_prob, # Saved for analysis
            'season': subset['season']
        })
        final_output.append(subset_out)
    
    if not final_output:
        return pd.DataFrame()
        
    return pd.concat(final_output)

if __name__ == "__main__":
    print("Step 1: Generating Historical Predictions (Optimized Batch Inference)...")
    # Generate or Load
    cache_file = 'simulation_predictions_oelm_v2.csv'
    try:
        preds_df = pd.read_csv(cache_file)
        preds_df['date'] = pd.to_datetime(preds_df['date'])
        print("Loaded cached predictions.")
    except FileNotFoundError:
        # Use Optimized Generator
        preds_df = generate_predictions_optimized()
        preds_df.to_csv(cache_file, index=False)
        print("Generated and saved predictions.")
        
    print(f"Total Predictions: {len(preds_df)}")
    
    # --- Walk-Forward Calibration ---
    print("\nStep 1.5: Applying Walk-Forward Calibration...")
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import brier_score_loss
    
    # We need to know the season for each date
    # Re-fetch season info (it's not in preds_df, but we can infer or merge)
    # Actually, generate_predictions_for_simulation uses 'date', we can map date->season loosely
    # Or better, we can assume consecutive blocks.
    # Let's just re-merge season from the processor if possible, or simple year Logic
    # Simple logic: Oct-Dec is Part of Season Y+1? Nba seasons span years. 
    # Best way: Pass season in predictions_list.
    
    # Only calibrate if we have 'season' column
    # We will patch generate_predictions to include 'season' first? 
    # Or just use moving window of 1000 bets? 
    # Let's use Expanding Window calibration.
    
    preds_df = preds_df.sort_values('date')
    calibrated_probs = []
    
    # Initial window size
    window_size = 500
    
    ir = IsotonicRegression(out_of_bounds='clip')
    
    # We calibrate 'prob_over' vs 'outcome_over'
    # outcome_over = 1 if actual > line else 0
    preds_df['outcome_over'] = (preds_df['actual_value'] > preds_df['line_value']).astype(int)
    
    # Vectorized approch involves looping? 
    # For speed, let's recalibrate every month or season.
    # We will use "Prior Season" for calibration.
    
    dates = pd.to_datetime(preds_df['date'])
    preds_df['year'] = dates.dt.year
    preds_df['month'] = dates.dt.month
    
    # Approximate seasons (start Aug)
    preds_df['season_id'] = np.where(preds_df['month'] >= 10, preds_df['year'] + 1, preds_df['year'])
    
    seasons = sorted(preds_df['season_id'].unique())
    print(f"Seasons found: {seasons}")
    
    final_preds = []
    
    for i, season in enumerate(seasons):
        season_data = preds_df[preds_df['season_id'] == season].copy()
        
        if i == 0:
            # First season: No prior data to calibrate on. Use Raw Probs.
            # Or skip betting? Let's use Raw but warn.
            print(f"Season {season}: Using Raw Probabilities (Start of History)")
            season_data['calibrated_prob'] = season_data['prob_over']
        else:
            # Train on ALL prior history (Expanding Window)
            history = preds_df[preds_df['season_id'] < season]
            
            # Fit IR
            X_train = history['prob_over'].values
            y_train = history['outcome_over'].values
            
            ir.fit(X_train, y_train)
            
            # Predict
            X_test = season_data['prob_over'].values
            season_data['calibrated_prob'] = ir.predict(X_test)
            print(f"Season {season}: Calibrated using {len(history)} prior samples.")
            
        final_preds.append(season_data)
        
    preds_df = pd.concat(final_preds)
    
    # Replace 'prob_over' for the simulation engine with 'calibrated_prob'
    # But keep raw for debug
    preds_df['raw_prob_over'] = preds_df['prob_over']
    preds_df['prob_over'] = preds_df['calibrated_prob']

    print("\nStep 2: Running Sportsbook Simulation...")
    
    # Configure Simulator
    # Scenario: Standard Book (4.5% vig, skewed noise)
    # We updated simulation_engine to support skew? No, run_simulation handles skew logic in prob gen.
    # The Simulator just adds noise.
    sim = SportsbookSimulator(vig=0.045, noise_sigma=0.03, favorite_bias=0.01)
    
    engine = BacktestEngine(preds_df, sim)
    
    # Run Backtest
    results = engine.run(initial_bankroll=1000, min_ev=0.03, kelly_fraction=0.125)
    
    # Save Results
    results.to_csv('simulation_results.csv', index=False)
    print("\nResults saved to simulation_results.csv")
    
    # Quick Stats
    if not results.empty:
        win_rate = results['pnl'].gt(0).mean()
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Total Bets: {len(results)}")
        print(f"Total PnL: {results['pnl'].sum():.2f}")
    else:
        print("No bets placed.")
