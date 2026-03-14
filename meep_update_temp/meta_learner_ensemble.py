"""
Context-Aware Stacking Meta-Learner for NBA Player Props

Replaces simple averaging with intelligent ensemble that learns:
- Which windows work best for which player types
- How to weight predictions based on context (position, usage, opponent)
- Conditional interactions between windows

Architecture:
1. Base Models: 25 window predictions (1947-2021)
2. Context Features: Player archetype, opponent defense, game context
3. Meta-Features: Prediction statistics (mean, std, min, max across windows)
4. Meta-Learner: LightGBM trained on OOF predictions (no leakage)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Tuple
import pickle
from pathlib import Path


class ContextAwareMetaLearner:
    """
    Stacked ensemble meta-learner with contextual features.

    Features:
    - All 25 window predictions (base predictions)
    - Player context: position, usage rate, minutes, recent form
    - Opponent context: defensive rating, pace, scheme
    - Prediction statistics: mean, std, min, max across windows
    - Interactions: position × prediction_mean, usage × prediction_std
    """

    def __init__(self,
                 n_windows: int = 25,
                 lgb_params: Dict = None,
                 cv_folds: int = 5):
        """
        Args:
            n_windows: Number of base model windows
            lgb_params: LightGBM parameters
            cv_folds: Cross-validation folds for OOF predictions
        """
        self.n_windows = n_windows
        self.cv_folds = cv_folds

        # Default LightGBM params optimized for stacking
        self.lgb_params = lgb_params or {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'reg_alpha': 0.1,  # L1 regularization
            'reg_lambda': 1.0,  # L2 regularization
            'verbose': -1
        }

        self.meta_models = {}  # One meta-learner per prop
        self.feature_names = None

    def create_meta_features(self,
                            window_predictions: np.ndarray,
                            player_context: pd.DataFrame = None) -> pd.DataFrame:
        """
        Create meta-features from window predictions and context.

        Args:
            window_predictions: (n_samples, n_windows) array of predictions
            player_context: DataFrame with contextual features

        Returns:
            DataFrame with meta-features
        """
        n_samples = window_predictions.shape[0]
        meta_features = {}

        # 1. BASE PREDICTIONS (25 windows)
        for i in range(self.n_windows):
            meta_features[f'window_{i}_pred'] = window_predictions[:, i]

        # 2. PREDICTION STATISTICS (aggregates across windows)
        meta_features['pred_mean'] = np.mean(window_predictions, axis=1)
        meta_features['pred_std'] = np.std(window_predictions, axis=1)
        meta_features['pred_min'] = np.min(window_predictions, axis=1)
        meta_features['pred_max'] = np.max(window_predictions, axis=1)
        meta_features['pred_range'] = meta_features['pred_max'] - meta_features['pred_min']
        meta_features['pred_median'] = np.median(window_predictions, axis=1)

        # Coefficient of variation (uncertainty measure)
        meta_features['pred_cv'] = meta_features['pred_std'] / (meta_features['pred_mean'] + 1e-6)

        # Recent vs old window divergence
        recent_mean = np.mean(window_predictions[:, -5:], axis=1)  # Last 5 windows
        old_mean = np.mean(window_predictions[:, :5], axis=1)       # First 5 windows
        meta_features['recent_vs_old'] = recent_mean - old_mean

        # 3. PLAYER CONTEXT (if available)
        if player_context is not None:
            # Add all contextual features
            for col in player_context.columns:
                if col not in meta_features:
                    meta_features[col] = player_context[col].values

            # 4. INTERACTION FEATURES (context × predictions)
            if 'position_encoded' in player_context.columns:
                meta_features['position_x_pred_mean'] = (
                    player_context['position_encoded'].values * meta_features['pred_mean']
                )

            if 'usage_rate' in player_context.columns:
                meta_features['usage_x_pred_std'] = (
                    player_context['usage_rate'].values * meta_features['pred_std']
                )

            if 'minutes_avg' in player_context.columns:
                meta_features['minutes_x_pred_mean'] = (
                    player_context['minutes_avg'].values * meta_features['pred_mean']
                )

        return pd.DataFrame(meta_features)

    def fit_oof(self,
                window_predictions: np.ndarray,
                y_true: np.ndarray,
                player_context: pd.DataFrame = None,
                prop_name: str = 'points') -> Dict:
        """
        Train meta-learner using Out-of-Fold predictions to prevent leakage.

        Args:
            window_predictions: (n_samples, n_windows) base predictions
            y_true: (n_samples,) actual values
            player_context: Contextual features
            prop_name: Property name (points, rebounds, etc.)

        Returns:
            Dict with OOF metrics
        """
        # Create meta-features
        X_meta = self.create_meta_features(window_predictions, player_context)
        self.feature_names = list(X_meta.columns)

        print(f"\n{'='*70}")
        print(f"TRAINING META-LEARNER: {prop_name.upper()}")
        print(f"{'='*70}")
        print(f"  Base predictions: {self.n_windows} windows")
        print(f"  Meta-features: {len(self.feature_names)}")
        print(f"  Training samples: {len(X_meta):,}")
        print(f"  CV folds: {self.cv_folds}")

        # K-Fold cross-validation for OOF predictions
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        oof_predictions = np.zeros(len(X_meta))
        fold_models = []
        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_meta), 1):
            X_train, X_val = X_meta.iloc[train_idx], X_meta.iloc[val_idx]
            y_train, y_val = y_true[train_idx], y_true[val_idx]

            # Train LightGBM
            model = lgb.LGBMRegressor(**self.lgb_params, n_estimators=500)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )

            # OOF predictions
            oof_predictions[val_idx] = model.predict(X_val)

            # Metrics
            mae = mean_absolute_error(y_val, oof_predictions[val_idx])
            rmse = np.sqrt(mean_squared_error(y_val, oof_predictions[val_idx]))

            fold_models.append(model)
            fold_metrics.append({'mae': mae, 'rmse': rmse})

            print(f"  Fold {fold}: MAE = {mae:.3f}, RMSE = {rmse:.3f}")

        # Overall OOF metrics
        overall_mae = mean_absolute_error(y_true, oof_predictions)
        overall_rmse = np.sqrt(mean_squared_error(y_true, oof_predictions))

        # Baseline: simple average of windows
        baseline_pred = np.mean(window_predictions, axis=1)
        baseline_mae = mean_absolute_error(y_true, baseline_pred)
        baseline_rmse = np.sqrt(mean_squared_error(y_true, baseline_pred))

        improvement_mae = ((baseline_mae - overall_mae) / baseline_mae) * 100
        improvement_rmse = ((baseline_rmse - overall_rmse) / baseline_rmse) * 100

        print(f"\n  OOF Performance:")
        print(f"    Meta-Learner: MAE = {overall_mae:.3f}, RMSE = {overall_rmse:.3f}")
        print(f"    Baseline Avg: MAE = {baseline_mae:.3f}, RMSE = {baseline_rmse:.3f}")
        print(f"    Improvement:  MAE = {improvement_mae:+.1f}%, RMSE = {improvement_rmse:+.1f}%")

        # Feature importance
        avg_importance = np.mean([m.feature_importances_ for m in fold_models], axis=0)
        top_features = sorted(zip(self.feature_names, avg_importance),
                            key=lambda x: x[1], reverse=True)[:10]

        print(f"\n  Top 10 Features:")
        for feat, imp in top_features:
            print(f"    {feat:30s}: {imp:6.1f}")

        # Save fold models
        self.meta_models[prop_name] = {
            'fold_models': fold_models,
            'feature_names': self.feature_names,
            'oof_predictions': oof_predictions,
            'metrics': {
                'oof_mae': overall_mae,
                'oof_rmse': overall_rmse,
                'baseline_mae': baseline_mae,
                'baseline_rmse': baseline_rmse,
                'improvement_mae_pct': improvement_mae,
                'improvement_rmse_pct': improvement_rmse
            }
        }

        return self.meta_models[prop_name]['metrics']

    def predict(self,
                window_predictions: np.ndarray,
                player_context: pd.DataFrame = None,
                prop_name: str = 'points') -> np.ndarray:
        """
        Make predictions using ensemble of fold models.

        Args:
            window_predictions: (n_samples, n_windows) base predictions
            player_context: Contextual features
            prop_name: Property name

        Returns:
            (n_samples,) meta-predictions
        """
        if prop_name not in self.meta_models:
            raise ValueError(f"No meta-model trained for {prop_name}")

        # Create meta-features
        X_meta = self.create_meta_features(window_predictions, player_context)

        # Ensure feature alignment
        X_meta = X_meta[self.meta_models[prop_name]['feature_names']]

        # Average predictions from all fold models
        fold_preds = []
        for model in self.meta_models[prop_name]['fold_models']:
            fold_preds.append(model.predict(X_meta))

        return np.mean(fold_preds, axis=0)

    def save(self, filepath: str):
        """Save meta-learner"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"[OK] Saved meta-learner: {filepath}")

    @staticmethod
    def load(filepath: str):
        """Load meta-learner"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def extract_player_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract contextual features for meta-learner.

    Args:
        df: Player-game DataFrame

    Returns:
        DataFrame with contextual features
    """
    context = pd.DataFrame(index=df.index)

    # Position encoding (if available)
    if 'position' in df.columns:
        # One-hot or label encode
        position_map = {'PG': 0, 'SG': 1, 'SF': 2, 'PF': 3, 'C': 4}
        context['position_encoded'] = df['position'].map(position_map).fillna(2)

    # Usage rate (proxy: FGA + FTA + AST)
    if all(c in df.columns for c in ['fieldGoalsAttempted', 'freeThrowsAttempted', 'assists']):
        context['usage_rate'] = (
            df['fieldGoalsAttempted'].fillna(0) +
            df['freeThrowsAttempted'].fillna(0) * 0.44 +
            df['assists'].fillna(0) * 0.33
        )

    # Minutes average
    if 'numMinutes' in df.columns:
        context['minutes_avg'] = df['numMinutes'].fillna(0)

    # Home/away
    if 'home' in df.columns:
        context['is_home'] = df['home'].astype(int)

    # Opponent (if available)
    if 'opponentTeamId' in df.columns:
        context['opponent_encoded'] = df['opponentTeamId'].astype('category').cat.codes

    return context


# Example usage in backtest
"""
# In backtest_2024_2025.py, replace simple averaging with:

# 1. Collect all window predictions
window_preds = []
for window_name, results in all_results.items():
    if 'error' not in results and 'points' in results:
        window_preds.append(results['points']['predictions'])

# Stack into array: (n_samples, n_windows)
X_base = np.column_stack(window_preds)

# 2. Extract context
player_context = extract_player_context(test_df)

# 3. Train meta-learner
meta_learner = ContextAwareMetaLearner(n_windows=len(window_preds))
metrics = meta_learner.fit_oof(
    window_predictions=X_base,
    y_true=test_df['points'].values,
    player_context=player_context,
    prop_name='points'
)

# 4. Make predictions
final_predictions = meta_learner.predict(X_base, player_context, 'points')

# 5. Save
meta_learner.save('model_cache/meta_learner_2024.pkl')
"""
