import argparse
from data_processor import BasketballDataProcessor
from ensemble_model import EnsemblePredictor
from ft_transformer import FTTransformerFeatureExtractor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pandas as pd
import json

def main():
    parser = argparse.ArgumentParser(description="Train NBA Hybrid Ensemble Model")
    parser.add_argument('--target', type=str, default='points', help='Target variable to predict')
    parser.add_argument('--tune', action='store_true', help='Run Optuna hyperparameter tuning')
    parser.add_argument('--trials', type=int, default=10, help='Number of Optuna trials')
    parser.add_argument('--sample', type=int, default=None, help='Number of rows to load for quick testing')
    parser.add_argument('--use_ft', action='store_true', help='Use FT-Transformer embeddings')
    parser.add_argument('--stacking', action='store_true', help='Use Stacking Meta-Learner instead of Weighted Blending')
    parser.add_argument('--epochs', type=int, default=5, help='Number of FT-Transformer epochs')
    args = parser.parse_args()

    # 1. Load and Preprocess
    processor = BasketballDataProcessor('final_feature_matrix_with_per_min_1997_onward.csv')
    processor.load_data(nrows=args.sample)
    processor.preprocess(target=args.target)
    
    # 2. Walk-Forward Validation
    # Instead of a single split, we loop through seasons to simulate real-world performance
    start_val_season = 2020
    end_test_season = 2026
    
    metrics = []
    
    # Initialize ft_extractor outside loop to save later
    ft_extractor = None
    
    for season in range(start_val_season, end_test_season + 1):
        print(f"\n=== Walk-Forward Validation: Testing on Season {season} ===")
        
        # Dynamic Split
        # Using a 3-year rolling window to adapt to recent trends (e.g., scoring inflation)
        # This prevents the model from being biased by 1990s/2000s defensive eras
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = processor.get_time_series_splits(
            val_season=season-1, 
            test_season=season,
            window_size=3 
        )
        
        # Initialize & Train
        predictor = EnsemblePredictor()
        
        # FT-Transformer Embeddings (Optional)
        if args.use_ft:
            cat_cols = processor.get_cat_cols()
            if cat_cols:
                print(f"Training FT-Transformer on {len(cat_cols)} categorical features...")
                # Calculate cardinalities
                # We need to ensure we use the global max from the processor's label encoders if possible,
                # or just max(X_train) + 1. Global is safer.
                cardinalities = []
                for col in cat_cols:
                    # processor.label_encoders[col].classes_ gives us the full list
                    cardinalities.append(len(processor.label_encoders[col].classes_))
                
                ft_extractor = FTTransformerFeatureExtractor(cardinalities, embed_dim=16, device='cpu')
                
                # Prepare data for FT (numpy arrays of ints)
                X_train_cat = X_train[cat_cols].values
                X_val_cat = X_val[cat_cols].values
                X_test_cat = X_test[cat_cols].values
                
                # Fit
                ft_extractor.fit(X_train_cat, y_train.values, epochs=args.epochs, batch_size=512)
                
                # Transform
                print("Extracting embeddings...")
                emb_train = ft_extractor.transform(X_train_cat)
                emb_val = ft_extractor.transform(X_val_cat)
                emb_test = ft_extractor.transform(X_test_cat)
                
                # Append to DataFrames
                # We need to name them
                emb_cols = [f"emb_{i}" for i in range(emb_train.shape[1])]
                
                # Helper to append
                def append_emb(df, emb):
                    df_emb = pd.DataFrame(emb, columns=emb_cols, index=df.index)
                    return pd.concat([df, df_emb], axis=1)
                
                X_train = append_emb(X_train, emb_train)
                X_val = append_emb(X_val, emb_val)
                X_test = append_emb(X_test, emb_test)
                
                print(f"Added {len(emb_cols)} embedding features.")
            else:
                print("No categorical columns found for FT-Transformer.")
        
        # Load Best Params
        best_params = {}
        try:
            with open('best_params.json', 'r') as f:
                best_params = json.load(f)
            print("Loaded best parameters from best_params.json")
        except FileNotFoundError:
            print("best_params.json not found. Using defaults.")
        
        if args.tune and season == start_val_season: # Only tune once for efficiency in this demo
            print("Running hyperparameter tuning (first fold only)...")
            xgb_params = predictor.tune_xgboost(X_train, y_train, X_val, y_val, n_trials=args.trials)
            lgb_params = predictor.tune_lightgbm(X_train, y_train, X_val, y_val, n_trials=args.trials)
        elif not args.tune:
            xgb_params = best_params.get('xgboost')
            lgb_params = best_params.get('lightgbm')
            
        # Get categorical features
        cat_features = processor.get_cat_cols()
            
        predictor.train(X_train, y_train, X_val, y_val, xgb_params, lgb_params, cat_features=cat_features)
        
        if args.stacking:
            predictor.train_stacking(X_val, y_val)
        
        # Evaluate on Test Season
        preds = predictor.predict(X_test, use_stacking=args.stacking)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        print(f"Season {season} Results - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
        metrics.append({'season': season, 'rmse': rmse, 'mae': mae, 'r2': r2})
        
    # Aggregate Results
    avg_rmse = np.mean([m['rmse'] for m in metrics])
    print(f"\nAverage Test RMSE over {len(metrics)} seasons: {avg_rmse:.4f}")
    
    # Feature Importance (from XGBoost as proxy)
    print("\nTop 10 Important Features (XGBoost):")
    importances = predictor.xgb_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    for i in range(10):
        print(f"{X_train.columns[indices[i]]}: {importances[indices[i]]:.4f}")

    # Save
    predictor.save_models(suffix=f"_{args.target}")
    
    # Explicitly save FT-Transformer
    if ft_extractor is not None:
        ft_path = os.path.join(predictor.model_dir, f'ft_transformer_{args.target}.pt')
        torch.save(ft_extractor.state_dict(), ft_path)
        print(f"FT-Transformer model saved to {ft_path}")

if __name__ == "__main__":
    main()
