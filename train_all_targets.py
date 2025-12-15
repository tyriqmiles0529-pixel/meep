import argparse
import pandas as pd
import numpy as np
import os
import json
import torch
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from data_processor import BasketballDataProcessor
from ensemble_model_vm import EnsemblePredictor
from ft_transformer import FTTransformerFeatureExtractor

TARGETS = ['points', 'rebounds', 'assists']

def main():
    parser = argparse.ArgumentParser(description="Train NBA Hybrid Ensemble Model for ALL Targets")
    parser.add_argument('--epochs', type=int, default=50, help='Number of FT-Transformer epochs')
    parser.add_argument('--start_season', type=int, default=2020, help='Start validation season')
    parser.add_argument('--end_season', type=int, default=2026, help='End validation season')
    args = parser.parse_args()

    # 1. Load and Preprocess (Global)
    print("Loading and Preprocessing Data...")
    processor = BasketballDataProcessor('final_feature_matrix_with_per_min_1997_onward.csv')
    processor.load_data()
    # Preprocess with 'points' to set up features and lags. 
    # This assumes nulls are consistent across targets (if you didn't play, you have no stats).
    processor.preprocess(target='points') 
    
    # Ensure all targets are in the dataframe
    for t in TARGETS:
        if t not in processor.df.columns:
            raise ValueError(f"Target {t} not found in dataset!")

    metrics_report = {t: [] for t in TARGETS}
    
    # 2. Walk-Forward Loop
    for season in range(args.start_season, args.end_season + 1):
        if season == 2026: # Skip partial 2026 for training if needed, but user asked for it.
            # Actually, 2026 is "current" season, so we test on it?
            # train_pipeline.py skipped it. I will follow suit or include it if data exists.
            # User said "at least 2020-2024". I'll do 2020-2025.
            pass

        print(f"\n\n==================================================")
        print(f"=== Walk-Forward Validation: Testing on Season {season} ===")
        print(f"==================================================")
        
        # Manual Split to control X and y separately
        # Expanding Window: Train on all available history up to validation season
        val_season = season - 1
        test_season = season
        
        # train_mask = (processor.df['season'] < val_season)
        # To be safe and explicit about 1997 start:
        train_mask = (processor.df['season'] >= 1997) & (processor.df['season'] < val_season)
        
        val_mask = (processor.df['season'] == val_season)
        test_mask = (processor.df['season'] == test_season) # Strict test season
        
        # Features
        feature_cols = processor.feature_columns
        cat_cols = processor.get_cat_cols()
        
        X_train_base = processor.df.loc[train_mask, feature_cols].copy()
        X_val_base = processor.df.loc[val_mask, feature_cols].copy()
        X_test_base = processor.df.loc[test_mask, feature_cols].copy()
        
        print(f"Train: 1997-{val_season-1} ({len(X_train_base)}) | Val: {val_season} ({len(X_val_base)}) | Test: {test_season} ({len(X_test_base)})")
        
        # --- GLOBAL EMBEDDING PHASE ---
        # Train FT-Transformer ONCE for this fold using 'points' as proxy
        print(f"\n[Global] Training FT-Transformer on 'points' for embeddings...")
        
        # Calculate cardinalities
        cardinalities = []
        for col in cat_cols:
            cardinalities.append(len(processor.label_encoders[col].classes_))
            
        ft_extractor = FTTransformerFeatureExtractor(cardinalities, embed_dim=16, device='cpu')
        
        # Prepare Data
        X_train_cat = X_train_base[cat_cols].values
        y_train_pts = processor.df.loc[train_mask, 'points'].values
        
        # Fit
        ft_extractor.fit(X_train_cat, y_train_pts, epochs=args.epochs, batch_size=512)
        
        # Transform
        print("[Global] Extracting embeddings...")
        emb_train = ft_extractor.transform(X_train_cat)
        emb_val = ft_extractor.transform(X_val_base[cat_cols].values)
        emb_test = ft_extractor.transform(X_test_base[cat_cols].values)
        
        # Helper to merge embeddings
        emb_cols = [f"emb_{i}" for i in range(emb_train.shape[1])]
        def add_emb(df, emb):
            df_emb = pd.DataFrame(emb, columns=emb_cols, index=df.index)
            return pd.concat([df, df_emb], axis=1)
            
        X_train_emb = add_emb(X_train_base, emb_train)
        X_val_emb = add_emb(X_val_base, emb_val)
        X_test_emb = add_emb(X_test_base, emb_test)
        
        # Save FT Model (Global for this season)
        ft_dir = f"models/global_ft_{season}"
        os.makedirs(ft_dir, exist_ok=True)
        ft_extractor.save(os.path.join(ft_dir, "ft_transformer.pt"))
        
        # --- TARGET LOOP ---
        for target in TARGETS:
            print(f"\n--- Training Target: {target.upper()} ---")
            
            # Get Y
            # We must align X and y after dropping NaNs for THIS target
            # The embeddings and X_base are already aligned to the full df (which might have NaNs for this target)
            
            # Create a mask for valid target values
            valid_mask_train = processor.df.loc[train_mask, target].notna()
            valid_mask_val = processor.df.loc[val_mask, target].notna()
            valid_mask_test = processor.df.loc[test_mask, target].notna()
            
            # Apply mask to X and y
            y_train = processor.df.loc[train_mask, target][valid_mask_train]
            X_train_curr = X_train_emb[valid_mask_train]
            
            y_val = processor.df.loc[val_mask, target][valid_mask_val]
            X_val_curr = X_val_emb[valid_mask_val]
            
            y_test = processor.df.loc[test_mask, target][valid_mask_test]
            X_test_curr = X_test_emb[valid_mask_test]
            
            print(f"[{target.upper()}] Train size: {len(y_train)} | Val size: {len(y_val)}")

            # Initialize Ensemble
            predictor = EnsemblePredictor()
            predictor.model_dir = f"models/{target}" # Separate dir per target
            os.makedirs(predictor.model_dir, exist_ok=True)
            
            # Load Best Params (if exists)
            # We assume best_params.json is generic or we tune. 
            # For now, using defaults/hardcoded in EnsemblePredictor or generic best_params
            # TODO: Ideally tune per target, but for speed we use robust defaults/previous best
            
            # Train
            # Note: We pass cat_features so CatBoost knows them
            predictor.train(X_train_curr, y_train, X_val_curr, y_val, cat_features=cat_cols)
            
            # Stacking
            predictor.train_stacking(X_val_curr, y_val)
            
            # Evaluate
            preds = predictor.predict(X_test_curr, use_stacking=True)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            
            print(f"[{target.upper()}] Season {season} RMSE: {rmse:.4f} | R2: {r2:.4f}")
            
            # Store
            metrics_report[target].append({
                'season': season,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'weights': predictor.weights.tolist() if hasattr(predictor, 'weights') else []
            })
            
            # Save Model
            predictor.save_models(suffix=f"_{season}")

    # 3. Final Report
    print("\n\n=== FINAL REPORT ===")
    with open('ALL_TARGETS_REPORT.json', 'w') as f:
        json.dump(metrics_report, f, indent=4)
        
    for target, res in metrics_report.items():
        avg_rmse = np.mean([r['rmse'] for r in res])
        print(f"Target: {target} | Avg RMSE: {avg_rmse:.4f}")

if __name__ == "__main__":
    main()
