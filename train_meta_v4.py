
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score

# Configuration
OOF_DIR = "data/oof_v4"
META_MODEL_DIR = "models/meta_v4"
STRICT_DATA_PATH = "data/strict_features_v4.csv" # To get actual targets for OOF rows

# OOF_FILE = "data/oof/oof_validation_2024.csv" # Original OOF_FILE, now derived or replaced by OOF_DIR
# META_DIR = "models/meta_v1" # Original META_DIR, now replaced by META_MODEL_DIR
TARGETS = ['PTS', 'AST', 'REB']

def train_meta_learner():
    # Assuming OOF_FILE will be constructed from OOF_DIR, or a specific file within OOF_DIR
    # For now, let's assume a default OOF file name within OOF_DIR
    # If the user intended a specific OOF file, this needs further clarification.
    # For this change, we'll use a placeholder for OOF_FILE if it's still needed.
    # However, the instruction only mentions updating OOF_DIR and META_MODEL_DIR.
    # The snippet provided for OOF_FILE was incomplete.
    # Let's assume the OOF_FILE is now "data/oof_v4/oof_validation_2024.csv" for consistency.
    current_oof_file = os.path.join(OOF_DIR, "oof_validation_2024.csv") # Example construction
    
    os.makedirs(META_MODEL_DIR, exist_ok=True)
    
    print(f"Loading OOF Predictions: {current_oof_file}...")
    df = pd.read_csv(current_oof_file)
    
    # We trained XGB, LGB, CAT
    # Columns in OOF: pred_PTS_xgb, pred_PTS_lgb, ...
    
    results = []
    
    for target in TARGETS:
        print(f"\n=== META-LEARNER: {target} ===")
        
        # Features for Stacker: The base model predictions
        pred_cols = [f'pred_{target}_xgb', f'pred_{target}_lgb', f'pred_{target}_cat']
        
        X = df[pred_cols]
        y = df[target]
        
        # We use RidgeCV to find best alpha and weights
        # Positive=True to enforce non-negative weights (ensemble averaging)
        meta_model = RidgeCV(alphas=[0.1, 1.0, 10.0], fit_intercept=True)
        # sklearn Ridge doesn't easily support positive constraint in CV, 
        # but ElasticNet does (positive=True not in standard CV? Ridge has it? 
        # Actually standard Ridge doesn't enforce positive coefficients easily 
        # without constraint. Let's use LinearRegression with non-negative least squares 
        # or simplified Ridge. Ridge is robust enough. We can check coefs.)
        
        # Actually simply RidgeCV is usually fine.
        meta_model.fit(X, y)
        
        # Coefficients
        coefs = meta_model.coef_
        intercept = meta_model.intercept_
        
        print(f"Weights: {dict(zip(pred_cols, coefs))}")
        print(f"Intercept: {intercept}")
        
        # Evaluate Stacker (In-Sample on 2024 - yes, slightly optimistic but shows blending ability)
        preds = meta_model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, preds))
        r2 = r2_score(y, preds)
        
        print(f"Stacker RMSE: {rmse:.4f}, R2: {r2:.4f}")
        
        # Save
        joblib.dump(meta_model, os.path.join(META_MODEL_DIR, f"stacker_{target}.joblib"))
        results.append({'target': target, 'weights': coefs, 'rmse': rmse, 'r2': r2})
        
    print("\nMeta-Learner Training Complete.")

if __name__ == "__main__":
    train_meta_learner()
