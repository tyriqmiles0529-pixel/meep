import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import numpy as np
import optuna
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

from simple_baselines import BaselineModels

class EnsemblePredictor:
    def __init__(self, model_dir='models'):
        self.xgb_model = None
        self.lgb_model = None
        self.cat_model = None
        self.baselines = BaselineModels()
        self.model_dir = model_dir
        
        # Weights for [XGB, LGB, Cat, Ridge]
        self.weights = [0.3, 0.3, 0.3, 0.1] 
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    # ... (tune_xgboost and tune_lightgbm remain the same) ...

    def train(self, X_train, y_train, X_val, y_val, xgb_params=None, lgb_params=None, cat_features=None):
        """Trains XGBoost, LightGBM, and Baseline Ridge."""
        
        # 1. Train Tree Models
        print("Training XGBoost...")
        if xgb_params is None:
            xgb_params = {'n_estimators': 1000, 'learning_rate': 0.05, 'max_depth': 6, 'tree_method': 'hist', 'early_stopping_rounds': 50}
        else:
            xgb_params['n_estimators'] = 1000
            xgb_params['early_stopping_rounds'] = 50
            xgb_params['tree_method'] = 'hist'
            
        self.xgb_model = xgb.XGBRegressor(**xgb_params)
        self.xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
        
        print("Training LightGBM...")
        if lgb_params is None:
            lgb_params = {'n_estimators': 1000, 'learning_rate': 0.05, 'num_leaves': 31}
        else:
            lgb_params['n_estimators'] = 1000
            
        self.lgb_model = lgb.LGBMRegressor(**lgb_params)
        callbacks = [lgb.early_stopping(stopping_rounds=50)]
        self.lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='rmse', callbacks=callbacks)

        # 2. Train CatBoost
        print("Training CatBoost...")
        # Identify categorical features indices for CatBoost
        # We assume X_train is a DataFrame. If numpy, we need indices.
        cat_features = []
        if hasattr(X_train, 'columns'):
            # We need to find columns that were identified as categorical in DataProcessor
            # Since we don't have the processor here, we rely on the caller passing cat_features indices or names
            # But for now, we'll try to infer or accept a param.
            # Actually, let's update the train signature to accept cat_features names
            pass
            
        # Default CatBoost Params
        cb_params = {
            'iterations': 2000,
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 3,
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            'random_seed': 42,
            'verbose': 100,
            'allow_writing_files': False
        }
        
        self.cat_model = cb.CatBoostRegressor(**cb_params)
        self.cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, cat_features=cat_features)

        # 3. Train Baseline (Ridge)
        print("Training Ridge Baseline...")
        self.baselines.train_ridge(X_train, y_train)
        
        # 3. Optimize Weights
        self.optimize_weights(X_val, y_val)
        
    def optimize_weights(self, X_val, y_val):
        """Finds optimal blending weights on validation set using constrained optimization."""
        print("Optimizing ensemble weights...")
        p_xgb = self.xgb_model.predict(X_val)
        p_lgb = self.lgb_model.predict(X_val)
        p_cat = self.cat_model.predict(X_val)
        p_ridge = self.baselines.predict(X_val)['ridge']
        
        predictions = np.column_stack([p_xgb, p_lgb, p_cat, p_ridge])
        
        def loss_func(weights):
            # weights must sum to 1
            final_pred = np.dot(predictions, weights)
            return np.sqrt(mean_squared_error(y_val, final_pred))
            
        # Constraints: sum(weights) = 1
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        # Bounds: 0 <= w <= 1
        bounds = [(0, 1)] * 4
        
        # Initial guess: equal weights
        init_weights = [0.25, 0.25, 0.25, 0.25]
        
        from scipy.optimize import minimize
        result = minimize(loss_func, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        
        self.weights = result.x
        print(f"Optimal Weights -> XGB: {self.weights[0]:.2f}, LGB: {self.weights[1]:.2f}, Cat: {self.weights[2]:.2f}, Ridge: {self.weights[3]:.2f} (RMSE: {result.fun:.4f})")

    def train_stacking(self, X_val, y_val):
        """Trains a Meta-Learner (Ridge) on the out-of-fold predictions of base models."""
        print("Training Stacking Meta-Learner...")
        p_xgb = self.xgb_model.predict(X_val)
        p_lgb = self.lgb_model.predict(X_val)
        p_cat = self.cat_model.predict(X_val)
        p_ridge = self.baselines.predict(X_val)['ridge']
        
        # Meta-features: Predictions from base models
        X_meta = np.column_stack([p_xgb, p_lgb, p_cat, p_ridge])
        
        # Meta-learner: Ridge Regression (simple, robust, interpretable)
        from sklearn.linear_model import Ridge
        self.meta_model = Ridge(alpha=1.0)
        self.meta_model.fit(X_meta, y_val)
        
        print(f"Stacking Coeffs: {self.meta_model.coef_}")
        print(f"Stacking Intercept: {self.meta_model.intercept_}")

    def predict(self, X, use_stacking=False):
        p_xgb = self.xgb_model.predict(X)
        p_lgb = self.lgb_model.predict(X)
        p_cat = self.cat_model.predict(X)
        p_ridge = self.baselines.predict(X)['ridge']
        
        if use_stacking and hasattr(self, 'meta_model'):
            X_meta = np.column_stack([p_xgb, p_lgb, p_cat, p_ridge])
            return self.meta_model.predict(X_meta)
        else:
            return (self.weights[0] * p_xgb + 
                    self.weights[1] * p_lgb + 
                    self.weights[2] * p_cat +
                    self.weights[3] * p_ridge)

    def save_models(self, suffix=''):
        joblib.dump(self.xgb_model, os.path.join(self.model_dir, f'xgb_model{suffix}.pkl'))
        joblib.dump(self.lgb_model, os.path.join(self.model_dir, f'lgb_model{suffix}.pkl'))
        self.cat_model.save_model(os.path.join(self.model_dir, f'cat_model{suffix}.cbm'))
        joblib.dump(self.baselines, os.path.join(self.model_dir, f'baselines_model{suffix}.pkl'))
        print("Models saved.")

    def load_models(self, suffix=''):
        """Loads models with the given suffix."""
        try:
            self.xgb_model = joblib.load(os.path.join(self.model_dir, f'xgb_model{suffix}.pkl'))
            self.lgb_model = joblib.load(os.path.join(self.model_dir, f'lgb_model{suffix}.pkl'))
            
            self.cat_model = cb.CatBoostRegressor()
            self.cat_model.load_model(os.path.join(self.model_dir, f'cat_model{suffix}.cbm'))
            
            self.baselines = joblib.load(os.path.join(self.model_dir, f'baselines_model{suffix}.pkl'))
            print(f"Models loaded with suffix '{suffix}'.")
        except Exception as e:
            print(f"Error loading models with suffix '{suffix}': {e}")
            raise
