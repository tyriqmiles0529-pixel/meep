"""
Enhanced Player Prediction Ensemble with Team Matchup Context

Architecture:
1. Player Elo Rating System (performance momentum)
2. LightGBM with 68+ features (current model)
3. Ridge Regression (L2-regularized linear model on recent stats)
4. Team Matchup Context (opponent defensive ratings, pace, etc.)
5. Rolling Averages (baseline)
6. Meta-Learner (combines all with optimal weights)

Key Innovation: Team context for assists, threes, and minutes
- Assists depend on team pace, offensive style, ball movement
- Threes depend on opponent 3P defense, team spacing
- Minutes depend on coach rotation, game script, blowouts
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 1. RIDGE REGRESSION MODEL
# ============================================================================

class RidgePlayerStatModel:
    """
    L2-regularized linear regression on player recent stats.

    Uses simpler features than LightGBM:
    - Last N games stats (points, rebounds, assists, etc.)
    - Opponent defensive rating
    - Home/away
    - Rest days
    - Minutes played

    Benefits:
    - Fast to train and retrain
    - Interpretable coefficients
    - Good baseline that complements complex LightGBM
    """

    def __init__(self, alpha=1.0, lookback_games=10):
        self.alpha = alpha
        self.lookback_games = lookback_games
        self.model = Ridge(alpha=alpha)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _build_features(self, player_recent_stats: pd.DataFrame,
                       opp_def_rating: float, is_home: bool,
                       rest_days: int, recent_minutes: float) -> np.ndarray:
        """
        Build feature vector from recent player stats.

        Features (example for points):
        - Avg points last N games
        - Std points last N games
        - Trend (linear slope)
        - Opponent defensive rating
        - Home (0/1)
        - Rest days
        - Recent minutes avg
        """
        features = []

        # Recent stats
        if len(player_recent_stats) > 0:
            features.append(player_recent_stats.mean())  # Average
            features.append(player_recent_stats.std())   # Volatility

            # Trend (simple linear regression slope)
            if len(player_recent_stats) >= 3:
                x = np.arange(len(player_recent_stats))
                slope = np.polyfit(x, player_recent_stats, 1)[0]
                features.append(slope)
            else:
                features.append(0.0)
        else:
            features.extend([0.0, 0.0, 0.0])

        # Context features
        features.append(opp_def_rating)
        features.append(1.0 if is_home else 0.0)
        features.append(rest_days)
        features.append(recent_minutes)

        return np.array(features).reshape(1, -1)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit Ridge model."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> float:
        """Predict stat value."""
        if not self.is_fitted:
            return 0.0

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)[0]


# ============================================================================
# 2. PLAYER ELO RATING SYSTEM
# ============================================================================

class PlayerEloRating:
    """
    Elo rating for player performance momentum.
    Tracks recent performance trends beyond season averages.
    """

    def __init__(self, base_k=15.0, stat_name='points'):
        """
        Args:
            base_k: Base K-factor for rating updates
            stat_name: Stat being tracked (points, assists, etc.)
        """
        self.base_k = base_k
        self.stat_name = stat_name
        self.player_ratings = {}  # {player_id: rating}
        self.rating_history = {}  # {player_id: [ratings]}
        self.stat_baselines = {}  # {player_id: career_avg}

    def _expected_performance(self, rating: float, baseline: float) -> float:
        """Expected stat value based on rating vs baseline."""
        # Rating of 1500 = baseline, +100 = ~10% above baseline
        performance_multiplier = 1.0 + (rating - 1500) / 1000.0
        return baseline * max(0.5, min(1.5, performance_multiplier))

    def _dynamic_k_factor(self, expected: float, actual: float) -> float:
        """
        Adjust K-factor based on surprise magnitude.
        Big deviation = faster adaptation.
        """
        if expected == 0:
            return self.base_k

        surprise = abs(actual - expected) / max(expected, 1.0)
        # K ranges from 0.5x to 2.5x base K
        k_multiplier = 0.5 + 2.0 * min(surprise, 1.0)
        return self.base_k * k_multiplier

    def update(self, player_id: str, actual_stat: float, baseline: float) -> float:
        """
        Update player rating after observing actual performance.

        Args:
            player_id: Player identifier
            actual_stat: Actual stat value (e.g., 28 points)
            baseline: Player's baseline expectation (e.g., 22 ppg)

        Returns:
            Updated rating
        """
        # Initialize if new player
        if player_id not in self.player_ratings:
            self.player_ratings[player_id] = 1500.0
            self.rating_history[player_id] = [1500.0]
            self.stat_baselines[player_id] = baseline

        rating = self.player_ratings[player_id]
        expected = self._expected_performance(rating, baseline)

        # Dynamic K-factor
        k = self._dynamic_k_factor(expected, actual_stat)

        # Rating update
        performance_ratio = actual_stat / max(expected, 0.1)
        rating_change = k * (performance_ratio - 1.0) * 100

        new_rating = np.clip(rating + rating_change, 1000, 2000)

        self.player_ratings[player_id] = new_rating
        self.rating_history[player_id].append(new_rating)

        return new_rating

    def get_prediction(self, player_id: str, baseline: float) -> float:
        """Get predicted stat value based on current rating."""
        if player_id not in self.player_ratings:
            return baseline

        rating = self.player_ratings[player_id]
        return self._expected_performance(rating, baseline)


# ============================================================================
# 2. TEAM MATCHUP CONTEXT FEATURES
# ============================================================================

class TeamMatchupContext:
    """
    Extract team-level context features that affect player performance.

    Critical for:
    - Assists: Team pace, ball movement, offensive style
    - Threes: Opponent 3P defense, team spacing strategy
    - Minutes: Blowout potential, rotation patterns
    - Rebounds: Team rebounding philosophy, opponent size
    """

    def __init__(self):
        self.team_stats = {}  # Rolling team statistics

    def update_team_stats(self, team_id: str, game_stats: Dict):
        """Update rolling team statistics."""
        if team_id not in self.team_stats:
            self.team_stats[team_id] = {
                'pace': [],
                'assist_rate': [],
                '3pa_rate': [],
                'def_rating': [],
                'recent_margins': []
            }

        stats = self.team_stats[team_id]
        stats['pace'].append(game_stats.get('pace', 100))
        stats['assist_rate'].append(game_stats.get('ast_pct', 0.6))
        stats['3pa_rate'].append(game_stats.get('3pa_rate', 0.35))
        stats['def_rating'].append(game_stats.get('def_rtg', 110))
        stats['recent_margins'].append(game_stats.get('margin', 0))

        # Keep last 10 games
        for key in stats:
            stats[key] = stats[key][-10:]

    def get_matchup_features(self, player_team: str, opp_team: str,
                            stat_name: str) -> Dict[str, float]:
        """
        Extract team context features for a specific stat.

        Returns dict of features like:
        - team_pace_10g
        - opp_def_rating_10g
        - team_ast_rate_10g (for assists)
        - opp_3p_def_10g (for threes)
        - blowout_risk (for minutes)
        """
        features = {}

        # Get team stats (default if not available)
        team_stats = self.team_stats.get(player_team, {})
        opp_stats = self.team_stats.get(opp_team, {})

        # Common features for all stats
        features['team_pace'] = np.mean(team_stats.get('pace', [100]))
        features['opp_pace'] = np.mean(opp_stats.get('pace', [100]))
        features['pace_delta'] = features['team_pace'] - features['opp_pace']
        features['opp_def_rating'] = np.mean(opp_stats.get('def_rating', [110]))

        # Stat-specific features
        if stat_name == 'assists':
            features['team_assist_rate'] = np.mean(team_stats.get('assist_rate', [0.6]))
            features['opp_assist_rate_allowed'] = np.mean(opp_stats.get('assist_rate', [0.6]))

        elif stat_name == 'threes':
            features['team_3pa_rate'] = np.mean(team_stats.get('3pa_rate', [0.35]))
            features['opp_3p_def'] = np.mean(opp_stats.get('def_rating', [110]))  # Proxy

        elif stat_name == 'minutes':
            # Blowout risk affects rotation
            team_margins = team_stats.get('recent_margins', [0])
            opp_margins = opp_stats.get('recent_margins', [0])
            features['team_dominance'] = np.mean(team_margins)
            features['opp_dominance'] = np.mean(opp_margins)
            features['blowout_risk'] = abs(features['team_dominance'] - features['opp_dominance']) / 10.0

        elif stat_name == 'rebounds':
            # Team rebounding philosophy
            features['team_def_rating'] = np.mean(team_stats.get('def_rating', [110]))
            features['competitive_game'] = 1.0 if abs(features.get('team_dominance', 0)) < 5 else 0.0

        return features


# ============================================================================
# 3. ENSEMBLE META-LEARNER
# ============================================================================

class PlayerStatEnsemble:
    """
    Meta-learner that combines multiple prediction sources with optimal weights.

    Components:
    1. Ridge regression (simple linear model on recent stats)
    2. LightGBM model (68+ features)
    3. Player Elo rating (performance momentum)
    4. Rolling average (baseline)
    5. Team matchup adjustment (context)

    Meta-learner learns optimal weights for each component.
    """

    def __init__(self, stat_name: str):
        self.stat_name = stat_name
        self.ridge_model = RidgePlayerStatModel(alpha=1.0)
        self.lgbm_model = None
        self.player_elo = PlayerEloRating(stat_name=stat_name)
        self.team_context = TeamMatchupContext()
        self.meta_learner = Ridge(alpha=0.1)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def load_lgbm_model(self, model_path: str):
        """Load pre-trained LightGBM model."""
        with open(model_path, 'rb') as f:
            self.lgbm_model = pickle.load(f)

    def _get_base_predictions(self, features: Dict, player_id: str,
                             rolling_avg: float, baseline: float,
                             team_id: str, opp_team_id: str) -> np.ndarray:
        """
        Get predictions from each base model.

        Returns array: [ridge_pred, lgbm_pred, elo_pred, rolling_avg, matchup_adj]
        """
        predictions = []

        # 1. Ridge prediction (simple linear model)
        if self.ridge_model.is_fitted:
            try:
                # TODO: Build proper Ridge features from recent stats
                X_ridge = np.zeros((1, 7))  # Placeholder
                ridge_pred = self.ridge_model.predict(X_ridge)
            except:
                ridge_pred = baseline
        else:
            ridge_pred = baseline
        predictions.append(ridge_pred)

        # 2. LightGBM prediction (complex model with 68+ features)
        if self.lgbm_model is not None:
            try:
                # Convert features dict to array (match training features)
                X = self._features_to_array(features)
                lgbm_pred = self.lgbm_model.predict(X)[0]
            except:
                lgbm_pred = baseline
        else:
            lgbm_pred = baseline
        predictions.append(lgbm_pred)

        # 3. Player Elo prediction (performance momentum)
        elo_pred = self.player_elo.get_prediction(player_id, baseline)
        predictions.append(elo_pred)

        # 4. Rolling average (baseline)
        predictions.append(rolling_avg)

        # 5. Matchup adjustment (team context)
        matchup_features = self.team_context.get_matchup_features(
            team_id, opp_team_id, self.stat_name
        )
        matchup_multiplier = self._compute_matchup_multiplier(matchup_features)
        matchup_pred = baseline * matchup_multiplier
        predictions.append(matchup_pred)

        return np.array(predictions)

    def _compute_matchup_multiplier(self, matchup_features: Dict) -> float:
        """
        Compute matchup adjustment multiplier.

        Examples:
        - High pace game → more assists, points
        - Strong opponent defense → fewer points
        - Blowout risk → fewer minutes for starters
        """
        multiplier = 1.0

        # Pace adjustment (affects most stats)
        pace = matchup_features.get('team_pace', 100)
        multiplier *= (pace / 100.0) ** 0.3  # Diminishing returns

        # Opponent defense
        opp_def = matchup_features.get('opp_def_rating', 110)
        if opp_def < 108:  # Elite defense
            multiplier *= 0.95
        elif opp_def > 112:  # Weak defense
            multiplier *= 1.05

        # Stat-specific adjustments
        if self.stat_name == 'assists':
            ast_rate = matchup_features.get('team_assist_rate', 0.6)
            multiplier *= (ast_rate / 0.6) ** 0.5

        elif self.stat_name == 'threes':
            three_rate = matchup_features.get('team_3pa_rate', 0.35)
            multiplier *= (three_rate / 0.35) ** 0.4

        elif self.stat_name == 'minutes':
            blowout_risk = matchup_features.get('blowout_risk', 0)
            multiplier *= (1.0 - 0.1 * blowout_risk)  # Reduce mins if blowout likely

        return np.clip(multiplier, 0.7, 1.3)

    def _features_to_array(self, features: Dict) -> np.ndarray:
        """Convert feature dict to array (placeholder - needs actual feature mapping)."""
        # This would match the exact feature order from training
        # For now, return dummy array
        return np.zeros((1, 10))

    def fit_meta_learner(self, X_meta: np.ndarray, y: np.ndarray):
        """
        Fit meta-learner to learn optimal weights for base predictions.

        Args:
            X_meta: Array of base predictions (n_samples, 5)
                    Columns: [ridge, lgbm, elo, rolling_avg, matchup]
            y: True stat values
        """
        # Handle NaN values - replace with column mean
        X_clean = X_meta.copy()

        # Check for NaN and replace with column mean
        for col_idx in range(X_clean.shape[1]):
            col = X_clean[:, col_idx]
            nan_mask = np.isnan(col)
            if np.any(nan_mask):
                col_mean = np.nanmean(col)
                if np.isnan(col_mean):
                    # All values are NaN, use 0
                    col_mean = 0.0
                X_clean[nan_mask, col_idx] = col_mean
                print(f"    Replaced {np.sum(nan_mask)} NaN values in column {col_idx} with mean {col_mean:.2f}")

        # Remove any rows that still have NaN (shouldn't happen, but be safe)
        valid_rows = ~np.isnan(X_clean).any(axis=1)
        if not np.all(valid_rows):
            print(f"    Removing {np.sum(~valid_rows)} rows with remaining NaN values")
            X_clean = X_clean[valid_rows]
            y = y[valid_rows]

        if len(X_clean) == 0:
            print(f"    ERROR: No valid training data after NaN removal")
            return

        X_scaled = self.scaler.fit_transform(X_clean)
        self.meta_learner.fit(X_scaled, y)
        self.is_fitted = True

        # Print learned weights
        weights = self.meta_learner.coef_
        print(f"\n{self.stat_name.upper()} Ensemble Weights:")
        print(f"  Ridge: {weights[0]:.3f}")
        print(f"  LightGBM: {weights[1]:.3f}")
        print(f"  Player Elo: {weights[2]:.3f}")
        print(f"  Rolling Avg: {weights[3]:.3f}")
        print(f"  Matchup Adj: {weights[4]:.3f}")

    def predict(self, features: Dict, player_id: str, rolling_avg: float,
               baseline: float, team_id: str, opp_team_id: str) -> Tuple[float, Dict]:
        """
        Make ensemble prediction.

        Returns:
            prediction: Final ensemble prediction
            breakdown: Dict with individual component predictions
        """
        # Get base predictions
        base_preds = self._get_base_predictions(
            features, player_id, rolling_avg, baseline, team_id, opp_team_id
        )

        # Meta-learner ensemble (if fitted)
        if self.is_fitted:
            X_meta = base_preds.reshape(1, -1)
            X_scaled = self.scaler.transform(X_meta)
            ensemble_pred = self.meta_learner.predict(X_scaled)[0]
        else:
            # Simple average if meta-learner not fitted
            ensemble_pred = np.mean(base_preds)

        breakdown = {
            'ridge': base_preds[0],
            'lgbm': base_preds[1],
            'elo': base_preds[2],
            'rolling_avg': base_preds[3],
            'matchup': base_preds[4],
            'ensemble': ensemble_pred
        }

        return ensemble_pred, breakdown

    def update_elo(self, player_id: str, actual_stat: float, baseline: float):
        """Update player Elo after observing result."""
        self.player_elo.update(player_id, actual_stat, baseline)

    def update_team_stats(self, team_id: str, game_stats: Dict):
        """Update team context statistics."""
        self.team_context.update_team_stats(team_id, game_stats)


# ============================================================================
# 4. TRAINING PIPELINE INTEGRATION
# ============================================================================

def train_ensemble_models(player_stats_df: pd.DataFrame,
                         lgbm_model_paths: Dict[str, str],
                         output_dir: str = "models") -> Dict[str, PlayerStatEnsemble]:
    """
    Train ensemble models for all player stats.

    Args:
        player_stats_df: Historical player game stats
        lgbm_model_paths: Paths to pre-trained LightGBM models
        output_dir: Where to save ensemble models

    Returns:
        Dict of trained ensemble models
    """
    ensembles = {}

    for stat_name in ['points', 'rebounds', 'assists', 'threes', 'minutes']:
        print(f"\n{'='*70}")
        print(f"Training {stat_name.upper()} Ensemble")
        print('='*70)

        # Create ensemble
        ensemble = PlayerStatEnsemble(stat_name=stat_name)

        # Load LightGBM model
        if stat_name in lgbm_model_paths:
            ensemble.load_lgbm_model(lgbm_model_paths[stat_name])

        # TODO: Collect base predictions from historical data
        # This requires replaying history to build Elo, rolling avgs, etc.

        # For now, placeholder
        print(f"⚠ Full training pipeline not yet implemented")
        print(f"Ensemble structure created for {stat_name}")

        ensembles[stat_name] = ensemble

    return ensembles


# ============================================================================
# 5. USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Player Ensemble Architecture Demo")
    print("="*70)

    # Create ensemble for points
    points_ensemble = PlayerStatEnsemble(stat_name='points')

    # Example prediction
    features = {}  # Would contain 68+ features
    prediction, breakdown = points_ensemble.predict(
        features=features,
        player_id="player_123",
        rolling_avg=22.5,
        baseline=21.0,
        team_id="LAL",
        opp_team_id="GSW"
    )

    print(f"\nPrediction: {prediction:.1f} points")
    print(f"Breakdown:")
    for component, value in breakdown.items():
        print(f"  {component}: {value:.1f}")

    print("\n" + "="*70)
    print("Next Steps:")
    print("="*70)
    print("1. Integrate with train_auto.py to replay history")
    print("2. Build meta-learner training data")
    print("3. Fit ensemble weights")
    print("4. Backtest ensemble vs current LightGBM-only models")
    print("5. Expected improvement: 10-20% RMSE reduction")
