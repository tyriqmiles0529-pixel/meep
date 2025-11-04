"""
Player Ensemble with Meta-Learned Team Context

Instead of hand-crafted weights, this version feeds raw team context features
directly to the Ridge meta-learner, which learns optimal weights from data.

Meta-learner input (15 features total):
- 5 base predictions (Ridge, LightGBM, Elo, Rolling, Team Context)
- 10 raw team context features (pace, efficiency, usage, etc.)

The Ridge regression coefficients become the empirically optimized weights.
"""

import numpy as np
import pickle
from typing import Dict, Optional, Tuple
from collections import defaultdict
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class RidgePlayerStatModel:
    """L2-regularized linear regression on player recent stats."""

    def __init__(self, alpha: float = 1.0, lookback_games: int = 10):
        self.alpha = alpha
        self.lookback_games = lookback_games
        self.model = Ridge(alpha=alpha)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def predict(self, recent_stats: np.ndarray) -> float:
        """Predict next game stat value."""
        if not self.is_fitted or len(recent_stats) == 0:
            return np.mean(recent_stats) if len(recent_stats) > 0 else 0.0

        X = recent_stats[-self.lookback_games:].reshape(-1, 1)
        if len(X) < 3:
            return np.mean(recent_stats)

        try:
            X_scaled = self.scaler.transform(X)
            return float(self.model.predict(X_scaled)[0])
        except:
            return np.mean(recent_stats)


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

        # Performance score (0 to 1)
        if expected > 0:
            performance = min(actual / expected, 1.5) / 1.5
        else:
            performance = 0.5

        # Elo update
        delta = k * (performance - 0.5)
        self.player_elos[player_id] = current_elo + delta


class TeamContextFeatures:
    """Extract raw team context features (no hand-crafted weights)."""

    def __init__(self):
        # Rolling team stats storage
        self.team_stats = defaultdict(lambda: {
            'pace': [],
            'ortg': [],
            'drtg': [],
            'ast_rate': [],
            '3pa_rate': [],
            'usage_gini': [],
            'transition_freq': []
        })

    def update_team_stats(self, team: str, game_stats: Dict):
        """Update rolling team statistics."""
        stats = self.team_stats[team]

        stats['pace'].append(game_stats.get('pace', 100.0))
        stats['ortg'].append(game_stats.get('ortg', 110.0))
        stats['drtg'].append(game_stats.get('drtg', 110.0))
        stats['ast_rate'].append(game_stats.get('ast_rate', 0.6))
        stats['3pa_rate'].append(game_stats.get('3pa_rate', 0.35))
        stats['usage_gini'].append(game_stats.get('usage_gini', 0.3))
        stats['transition_freq'].append(game_stats.get('transition_pct', 0.15))

        # Keep last 20 games
        for key in stats:
            if len(stats[key]) > 20:
                stats[key] = stats[key][-20:]

    def get_raw_features(self, player_team: str, opponent_team: str) -> np.ndarray:
        """
        Get raw team context features (normalized).

        Returns 10 features:
        1. Combined pace (normalized)
        2. Team offensive rating (normalized)
        3. Team defensive rating (normalized)
        4. Opponent defensive rating (normalized)
        5. Team assist rate
        6. Team 3PA rate
        7. Team usage concentration (Gini)
        8. Team transition frequency
        9. Opponent pace
        10. Opponent offensive rating
        """
        team_stats = self.team_stats[player_team]
        opp_stats = self.team_stats[opponent_team]

        features = []

        # 1. Combined pace (avg of both teams)
        team_pace = np.mean(team_stats['pace'][-10:]) if team_stats['pace'] else 100.0
        opp_pace = np.mean(opp_stats['pace'][-10:]) if opp_stats['pace'] else 100.0
        combined_pace = (team_pace + opp_pace) / 2
        features.append((combined_pace - 95) / 10)  # Normalize around 95-105

        # 2. Team offensive rating
        team_ortg = np.mean(team_stats['ortg'][-10:]) if team_stats['ortg'] else 110.0
        features.append((team_ortg - 105) / 10)  # Normalize around 105-115

        # 3. Team defensive rating
        team_drtg = np.mean(team_stats['drtg'][-10:]) if team_stats['drtg'] else 110.0
        features.append((115 - team_drtg) / 10)  # Inverse (lower is better)

        # 4. Opponent defensive rating
        opp_drtg = np.mean(opp_stats['drtg'][-10:]) if opp_stats['drtg'] else 110.0
        features.append((115 - opp_drtg) / 10)  # Inverse

        # 5. Team assist rate
        ast_rate = np.mean(team_stats['ast_rate'][-10:]) if team_stats['ast_rate'] else 0.6
        features.append((ast_rate - 0.5) / 0.2)  # Normalize around 0.5-0.7

        # 6. Team 3PA rate
        three_rate = np.mean(team_stats['3pa_rate'][-10:]) if team_stats['3pa_rate'] else 0.35
        features.append((three_rate - 0.3) / 0.15)  # Normalize around 0.3-0.45

        # 7. Usage concentration (Gini coefficient)
        usage_gini = np.mean(team_stats['usage_gini'][-10:]) if team_stats['usage_gini'] else 0.3
        features.append((usage_gini - 0.25) / 0.15)  # Normalize around 0.25-0.4

        # 8. Transition frequency
        trans_freq = np.mean(team_stats['transition_freq'][-10:]) if team_stats['transition_freq'] else 0.15
        features.append((trans_freq - 0.1) / 0.1)  # Normalize around 0.1-0.2

        # 9. Opponent pace
        features.append((opp_pace - 95) / 10)

        # 10. Opponent offensive rating
        opp_ortg = np.mean(opp_stats['ortg'][-10:]) if opp_stats['ortg'] else 110.0
        features.append((opp_ortg - 105) / 10)

        return np.array(features)


class PlayerStatEnsembleMetaContext:
    """
    Enhanced ensemble with meta-learned team context.

    Meta-learner sees:
    - 5 base predictions
    - 10 raw team context features

    Total: 15 features for Ridge to learn optimal weights from data.
    """

    def __init__(self, stat_name: str):
        self.stat_name = stat_name

        # Components
        self.ridge_model = RidgePlayerStatModel(alpha=1.0)
        self.lgbm_model = None
        self.player_elo = PlayerEloRating(stat_name=stat_name)
        self.team_context = TeamContextFeatures()

        # Meta-learner with expanded input
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

    def get_expanded_features(self,
                              player_id: str,
                              recent_stats: np.ndarray,
                              baseline: float,
                              player_team: Optional[str] = None,
                              opponent_team: Optional[str] = None) -> np.ndarray:
        """
        Get expanded feature vector for meta-learner.

        Returns 15 features:
        - [0:5] = Base predictions (Ridge, LightGBM, Elo, Rolling, Baseline)
        - [5:15] = Raw team context features
        """
        # Base predictions (5 features)
        ridge_pred = self.ridge_model.predict(recent_stats) if self.ridge_model.is_fitted else baseline
        lgbm_pred = baseline  # Placeholder (would use actual LightGBM in production)
        elo_pred = self.player_elo.get_prediction(str(player_id), baseline)
        rolling_avg = np.mean(recent_stats[-10:]) if len(recent_stats) > 0 else baseline

        base_features = np.array([ridge_pred, lgbm_pred, elo_pred, rolling_avg, baseline])

        # Team context features (10 features)
        if player_team and opponent_team:
            context_features = self.team_context.get_raw_features(player_team, opponent_team)
        else:
            context_features = np.zeros(10)

        # Combine: 5 base + 10 context = 15 total
        return np.concatenate([base_features, context_features])

    def fit_meta_learner(self, X_meta: np.ndarray, y: np.ndarray):
        """
        Fit meta-learner on expanded features.

        X_meta: (n_samples, 15) - 5 base predictions + 10 context features
        y: (n_samples,) - actual stat values
        """
        # Handle NaN values
        X_clean = X_meta.copy()
        for col_idx in range(X_clean.shape[1]):
            col = X_clean[:, col_idx]
            nan_mask = np.isnan(col)
            if np.any(nan_mask):
                col_mean = np.nanmean(col)
                if np.isnan(col_mean):
                    col_mean = 0.0
                X_clean[nan_mask, col_idx] = col_mean

        # Remove rows with remaining NaN
        valid_rows = ~np.isnan(X_clean).any(axis=1) & ~np.isnan(y)
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

        # Print learned weights
        coef = self.meta_learner.coef_
        print(f"\n    Meta-learner weights (learned from data):")
        print(f"      Ridge pred:      {coef[0]:+.3f}")
        print(f"      LightGBM pred:   {coef[1]:+.3f}")
        print(f"      Elo pred:        {coef[2]:+.3f}")
        print(f"      Rolling avg:     {coef[3]:+.3f}")
        print(f"      Baseline:        {coef[4]:+.3f}")
        print(f"      Combined pace:   {coef[5]:+.3f}")
        print(f"      Team ORTG:       {coef[6]:+.3f}")
        print(f"      Team DRTG:       {coef[7]:+.3f}")
        print(f"      Opp DRTG:        {coef[8]:+.3f}")
        print(f"      Assist rate:     {coef[9]:+.3f}")
        print(f"      3PA rate:        {coef[10]:+.3f}")
        print(f"      Usage Gini:      {coef[11]:+.3f}")
        print(f"      Transition freq: {coef[12]:+.3f}")
        print(f"      Opp pace:        {coef[13]:+.3f}")
        print(f"      Opp ORTG:        {coef[14]:+.3f}")
        print(f"    Intercept: {self.meta_learner.intercept_:.3f}")

    def predict(self,
                player_id: str,
                recent_stats: np.ndarray,
                baseline: float,
                player_team: Optional[str] = None,
                opponent_team: Optional[str] = None) -> float:
        """Make ensemble prediction with meta-learned context."""
        expanded_features = self.get_expanded_features(
            player_id, recent_stats, baseline, player_team, opponent_team
        )

        # Handle NaN
        expanded_features = np.nan_to_num(expanded_features, nan=0.0)

        if self.is_fitted:
            X = expanded_features.reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            return float(self.meta_learner.predict(X_scaled)[0])
        else:
            # Fallback: simple average of base predictions
            return float(np.mean(expanded_features[:5]))

    def update_elo(self, player_id: str, actual: float, expected: float):
        """Update player Elo after seeing result."""
        self.player_elo.update(str(player_id), actual, expected)


def save_ensemble(ensemble: PlayerStatEnsembleMetaContext, path: str):
    """Save ensemble model to disk."""
    with open(path, 'wb') as f:
        pickle.dump(ensemble, f)


def load_ensemble(path: str) -> PlayerStatEnsembleMetaContext:
    """Load ensemble model from disk."""
    with open(path, 'rb') as f:
        return pickle.load(f)
