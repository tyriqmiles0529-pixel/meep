import pandas as pd
import numpy as np
import os
import argparse
from data_processor import BasketballDataProcessor
from ensemble_model_vm import EnsemblePredictor
from ft_transformer import FTTransformerFeatureExtractor

TARGETS = ['points', 'rebounds', 'assists']

def export_predictions(season=2025):
    print(f"Exporting predictions for Season {season}...")
    
    # 1. Load Data
    processor = BasketballDataProcessor('final_feature_matrix_with_per_min_1997_onward.csv')
    processor.load_data()
    processor.preprocess(target='points')
    
    # Filter for Test Season
    test_mask = (processor.df['season'] == season)
    X_test_base = processor.df.loc[test_mask, processor.feature_columns].copy()
    
    # 2. Add Embeddings (Must load the trained FT Transformer logic or skip if not essential for SIMULATION)
    # For simulation export, we want the MODEL's best guess.
    # We need to recreate the exact pipeline used in training (embedding + ensemble).
    
    # Re-extract embeddings (requires saved tokenizer or re-fitting on same vocab)
    # Assuming 'models/global_ft_{season}/ft_transformer.pt' exists or we can just skip embeddings for quick proxy?
    # NO, the ensemble relies on them. We must load.
    
    cat_cols = processor.get_cat_cols()
    cardinalities = [len(processor.label_encoders[col].classes_) for col in cat_cols]
    
    # Load FT
    ft_path = f"models/global_ft_{season}/ft_transformer.pt"
    if os.path.exists(ft_path):
        import torch
        ft_model = FTTransformerFeatureExtractor(cardinalities, embed_dim=16)
        ft_model.load_state_dict(torch.load(ft_path))
        
        X_test_cat = X_test_base[cat_cols].values
        emb_test = ft_model.transform(X_test_cat)
        
        emb_cols = [f"emb_{i}" for i in range(emb_test.shape[1])]
        X_test_emb = pd.DataFrame(emb_test, columns=emb_cols, index=X_test_base.index)
        X_test_final = pd.concat([X_test_base, X_test_emb], axis=1)
    else:
        print("Warning: FT Transformer model not found. Using base features only (Predictions might differ from training).")
        X_test_final = X_test_base
        
    all_preds = []
    
    # 3. Predict for each Target
    for target in TARGETS:
        print(f"Predicting {target}...")
        
        # Load Ensemble
        model_dir = f"models/{target}"
        if not os.path.exists(model_dir):
            print(f"Skipping {target}: No model found.")
            continue
            
        predictor = EnsemblePredictor(model_dir=model_dir)
        try:
            predictor.load_models(suffix=f"_{season}")
        except:
            print(f"Could not load models for {target}. Skipping.")
            continue
            
        # Get subset with valid target (to measure accuracy)
        # For simulation, we want preds for ALL games, even if actual is missing? No, we need actual to verify.
        valid_mask = processor.df.loc[test_mask, target].notna()
        indices = valid_mask[valid_mask].index
        
        X_curr = X_test_final.loc[indices]
        y_actual = processor.df.loc[indices, target]
        
        # Predict
        # Note: CatBoost needs to know cat features if dataframe passed, or handled in predictor
        # In train_all_targets, we passed cat_features.
        # Ensure consistency.
        preds = predictor.predict(X_curr, use_stacking=True)
        
        # Define Standard Deviations (Approximated from Training RMSE)
        std_dev_map = {'points': 5.2, 'rebounds': 3.3, 'assists': 2.5}
        std_dev = std_dev_map.get(target, 4.0)
        
        from scipy.stats import norm

        # Store
        for i, idx in enumerate(indices):
            row = processor.df.loc[idx]
            
            p_val = preds[i]
            
            # Simulate a "Vegas Line"
            # We assume Vegas is generally efficient but can differ from us.
            # Line = ModelPred + Noise (Market Disagreement)
            # This allows us to find edges.
            simulated_line = p_val + np.random.normal(0, std_dev * 0.3)
            
            # Round line to 0.5 to look real
            simulated_line = round(simulated_line * 2) / 2
            
            # Calculate Probability of OVER
            # Z = (Pred - Line) / StdDev
            z_score = (p_val - simulated_line) / std_dev
            prob_over = norm.cdf(z_score)
            
            # Clip prob to realistic bounds
            prob_over = max(0.05, min(0.95, prob_over))
            
            all_preds.append({
                'date': row['date'],
                'player_name': row['player_name'],
                'team': row['team'],
                'opponent': row['opponentteamname'], # Adjust col name if needed
                'target': target,
                'pred_value': p_val,
                'actual_value': y_actual.iloc[i],
                'line_value': simulated_line,
                'minutes': row.get('minutes', 0),
                'pred_minutes': row.get('minutes', 30), # Assuming actual minutes as perfect projection for now
                'prob_over': prob_over
            })
            
    # 4. Save CSV
    out_df = pd.DataFrame(all_preds)
    out_path = f"simulation_predictions_{season}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved {len(out_df)} predictions to {out_path}")

if __name__ == "__main__":
    export_predictions(2025)
