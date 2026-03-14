import optuna
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import json
import argparse

from sklearn.preprocessing import OrdinalEncoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=20, help='Number of trials per model')
    parser.add_argument('--target', type=str, default='points')
    args = parser.parse_args()

    print(f"Loading data for target: {args.target}...")
    df = pd.read_csv('final_feature_matrix_with_per_min_1997_onward.csv')
    
    # Drop rows with missing target
    print(f"Original shape: {df.shape}")
    df = df.dropna(subset=[args.target])
    print(f"Shape after dropping missing targets: {df.shape}")
    
    # Simple time-split for tuning (e.g. Train on 2015-2020, Val on 2021)
    df['date'] = pd.to_datetime(df['date'])
    df['season'] = df['season'].astype(int)
    
    train_mask = (df['season'] >= 2015) & (df['season'] <= 2020)
    val_mask = (df['season'] == 2021)
    
    # Features: Strict Allowlist to prevent leakage
    # We only want:
    # 1. Lags (*_last_game, *_avg)
    # 2. Priors (prior_*)
    # 3. Opponent Metrics (opp_*)
    # 4. Context (home, gameType, playerteamName, opponentteamName)
    
    all_cols = df.columns.tolist()
    features = [c for c in all_cols if 
                c.endswith('_last_game') or 
                c.endswith('_avg') or 
                c.startswith('prior_') or 
                c.startswith('opp_') or 
                c in ['home', 'gameType', 'playerteamName', 'opponentteamName']]
                
    print(f"Tuning with {len(features)} safe features: {features}")
    
    X_train = df.loc[train_mask, features].copy()
    y_train = df.loc[train_mask, args.target]
    X_val = df.loc[val_mask, features].copy()
    y_val = df.loc[val_mask, args.target]
    
    # Handle Categoricals
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    print(f"Encoding categorical columns: {cat_cols}")
    if cat_cols:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
        X_val[cat_cols] = encoder.transform(X_val[cat_cols])
        
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")
    
    best_params = {}
    
    # --- XGBoost Tuning ---
    print("\nTuning XGBoost...")
    def objective_xgb(trial):
        param = {
            'verbosity': 0,
            'objective': 'reg:squarederror',
            'tree_method': 'hist', # CPU optimized
            'booster': 'gbtree',
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'n_estimators': 1000,
            'early_stopping_rounds': 50
        }
        model = xgb.XGBRegressor(**param)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))

    study_xgb = optuna.create_study(direction='minimize')
    study_xgb.optimize(objective_xgb, n_trials=args.trials)
    best_params['xgboost'] = study_xgb.best_params
    print(f"Best XGB RMSE: {study_xgb.best_value:.4f}")

    # --- LightGBM Tuning ---
    print("\nTuning LightGBM...")
    def objective_lgb(trial):
        param = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'n_estimators': 1000
        }
        model = lgb.LGBMRegressor(**param)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        preds = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))

    study_lgb = optuna.create_study(direction='minimize')
    study_lgb.optimize(objective_lgb, n_trials=args.trials)
    best_params['lightgbm'] = study_lgb.best_params
    print(f"Best LGB RMSE: {study_lgb.best_value:.4f}")
    
    # --- CatBoost Tuning (Simplified) ---
    print("\nTuning CatBoost...")
    def objective_cat(trial):
        param = {
            'iterations': 1000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            'random_seed': 42,
            'verbose': False,
            'allow_writing_files': False
        }
        model = cb.CatBoostRegressor(**param)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
        preds = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))
        
    study_cat = optuna.create_study(direction='minimize')
    study_cat.optimize(objective_cat, n_trials=10) # Fewer trials for CatBoost as it's slower
    best_params['catboost'] = study_cat.best_params
    print(f"Best CatBoost RMSE: {study_cat.best_value:.4f}")

    # Save
    with open('best_params_honest.json', 'w') as f:
        json.dump(best_params, f, indent=4)
    print("Saved best parameters to best_params_honest.json")

if __name__ == "__main__":
    main()
