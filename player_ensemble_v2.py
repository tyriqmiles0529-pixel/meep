"""
Player Ensemble V2 - With Weighted Team Context

Improvements over V1:
- Weighted team context factors (based on correlation analysis)
- Stat-specific impact weights
- More conservative context adjustments
- Better handling of missing data
"""

import numpy as np
import pickle
from typing import Dict, Optional
from collections import defaultdict
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Import weighted context model
from team_context_weighted import WeightedTeamContext


class RidgePlayerStatModel:
    """L2-regularized linear regression on player recent stats."""

    def __init__(self, alpha: float = 1.0, lookback_games: int = 10):
        self.alpha = alpha
        self.lookback_games = lookback_games
        self.model = Ridge(alpha=alpha)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, recent_stats: np.ndarray, target: float):
        """Fit on player's recent game stats."""
        if len(recent_stats) < 3:
            return

        X = recent_stats[-self.lookback_games:].reshape(-1, 1)
        y = np.array([target])

        if len(X) >= 3:
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            self.model.fit(X_scaled, y)
            self.is_fitted = True

    def predict(self, recent_stats: np.ndarray) -> float:
        """Predict next game stat value."""
        if not self.is_fitted or len(recent_stats) == 0:
            return np.mean(recent_stats) if len(recent_stats) > 0 else 0.0

        X = recent_stats[-self.lookback_games:].reshape(-1, 1)
        X_scaled = self.scaler.transform(X)
        return float(self.model.predict(X_scaled)[0])


class PlayerEloRating:
    """Elo rating system for player performance momentum."""

    def __init__(self, stat_name: str, base_k: float = 20.0):
        self.stat_name = stat_name
        self.base_k = base_k
        self.player_elos = defaultdict(lambda: 1500.0)

    def _dynamic_k_factor(self, expected: float, actual: float) -> float:
        """Adjust K-factor based on surprise magnitude."""
        surprise = abs(actual - expected) / max(expected, 1.0)
        k_multiplier = 0.5 + 2.0 * min(surprise, 1.0)
        return self.base_k * k_multiplier

    def get_prediction(self, player_id: str, baseline: float) -> float:
        """Get Elo-adjusted prediction."""
        elo = self.player_elos[player_id]
        # Convert Elo to multiplier (1500 = 1.0x, 1600 = 1.1x, 1400 = 0.9x)
        multiplier = 1.0 + (elo - 1500) / 1000
        return baseline * multiplier

    def update(self, player_id: str, actual: float, expected: float):
        """Update player Elo based on performance."""
        current_elo = self.player_elos[player_id]
        k = self._dynamic_k_factor(expected, actual)

        # Performance score (0 to 1, where 1 = exceeded expectation by 50%+)
        if expected > 0:
            performance = min(actual / expected, 1.5) / 1.5
        else:
            performance = 0.5

        # Elo update
        delta = k * (performance - 0.5)
        self.player_elos[player_id] = current_elo + delta


class PlayerStatEnsembleV2:
    """
    Enhanced ensemble with weighted team context.

    Components:
    1. Ridge regression (recent stats)
    2. LightGBM (68+ features)
    3. Player Elo (momentum)
    4. Rolling average
    5. Weighted team context adjustment
    """

    def __init__(self, stat_name: str):
        self.stat_name = stat_name

        # Components
        self.ridge_model = RidgePlayerStatModel(alpha=1.0)
        self.lgbm_model = None  # Loaded externally
        self.player_elo = PlayerEloRating(stat_name=stat_name)
        self.team_context = WeightedTeamContext()

        # Meta-learner (combines base predictions)
        self.meta_learner = Ridge(alpha=0.1)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def load_lgbm_model(self, model_path: str):
        """Load pre-trained LightGBM model."""
        try:
            with open(model_path, 'rb') as f:
                self.lgbm_model = pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load LightGBM model: {e}")

    def get_base_predictions(self,
                            player_id: str,
                            recent_stats: np.ndarray,
                            baseline: float,
                            player_team: Optional[str] = None,
                            opponent_team: Optional[str] = None,
                            lgbm_features: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get predictions from all 5 components.

        Returns: [ridge_pred, lgbm_pred, elo_pred, rolling_avg, context_pred]
        """
        # 1. Ridge prediction
        ridge_pred = self.ridge_model.predict(recent_stats) if self.ridge_model.is_fitted else baseline

        # 2. LightGBM prediction
        if self.lgbm_model is not None and lgbm_features is not None:
            try:
                lgbm_pred = float(self.lgbm_model.predict(lgbm_features.reshape(1, -1))[0])
            except:
                lgbm_pred = baseline
        else:
            lgbm_pred = baseline

        # 3. Player Elo prediction
        elo_pred = self.player_elo.get_prediction(str(player_id), baseline)

        # 4. Rolling average (last 10 games)
        rolling_avg = np.mean(recent_stats[-10:]) if len(recent_stats) > 0 else baseline

        # 5. Team context adjustment
        if player_team and opponent_team:
            context_pred = self.team_context.get_context_adjustment(
                player_team, opponent_team, self.stat_name, baseline
            )
        else:
            context_pred = baseline

        return np.array([ridge_pred, lgbm_pred, elo_pred, rolling_avg, context_pred])

    def fit_meta_learner(self, X_meta: np.ndarray, y: np.ndarray):
        """
        Fit meta-learner to combine base predictions.

        X_meta: (n_samples, 5) - predictions from 5 components
        y: (n_samples,) - actual stat values
        """
        # Handle NaN values - replace with column mean
        X_clean = X_meta.copy()
        for col_idx in range(X_clean.shape[1]):
            col = X_clean[:, col_idx]
            nan_mask = np.isnan(col)
            if np.any(nan_mask):
                col_mean = np.nanmean(col)
                if np.isnan(col_mean):
                    col_mean = 0.0
                X_clean[nan_mask, col_idx] = col_mean

        # Remove rows with any remaining NaN
        valid_rows = ~np.isnan(X_clean).any(axis=1)
        X_clean = X_clean[valid_rows]
        y_clean = y[valid_rows]

        if len(X_clean) < 10:
            print(f"Warning: Only {len(X_clean)} valid samples for meta-learner")
            return

        # Fit
        self.scaler.fit(X_clean)
        X_scaled = self.scaler.transform(X_clean)
        self.meta_learner.fit(X_scaled, y_clean)
        self.is_fitted = True

        # Print meta-learner weights
        print(f"    Meta-learner weights: {self.meta_learner.coef_}")
        print(f"    Intercept: {self.meta_learner.intercept_:.3f}")

    def predict(self,
                player_id: str,
                recent_stats: np.ndarray,
                baseline: float,
                player_team: Optional[str] = None,
                opponent_team: Optional[str] = None,
                lgbm_features: Optional[np.ndarray] = None) -> float:
        """
        Make ensemble prediction.

        If meta-learner is fitted, uses weighted combination.
        Otherwise, returns simple average of components.
        """
        base_preds = self.get_base_predictions(
            player_id, recent_stats, baseline,
            player_team, opponent_team, lgbm_features
        )

        # Handle NaN in predictions
        base_preds = np.nan_to_num(base_preds, nan=baseline)

        if self.is_fitted:
            # Use meta-learner
            X = base_preds.reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            return float(self.meta_learner.predict(X_scaled)[0])
        else:
            # Simple average
            return float(np.mean(base_preds))

    def update_elo(self, player_id: str, actual: float, expected: float):
        """Update player Elo after seeing result."""
        self.player_elo.update(str(player_id), actual, expected)


def save_ensemble(ensemble: PlayerStatEnsembleV2, path: str):
    """Save ensemble model to disk."""
    with open(path, 'wb') as f:
        pickle.dump(ensemble, f)


def load_ensemble(path: str) -> PlayerStatEnsembleV2:
    """Load ensemble model from disk."""
    with open(path, 'rb') as f:
        return pickle.load(f)
