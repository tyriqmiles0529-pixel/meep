"""
Ensemble sub-models for NBA game prediction.

Includes:
- L2 Ridge Regression (score differentials)
- Nate Silver's Elo Rating model
- Dean Oliver's Four Factors
- Logistic Regression meta-learner
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pickle
import math
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import Ridge, LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, brier_score_loss


# ============================================================================
# 1. L2 RIDGE REGRESSION (Score Differentials)
# ============================================================================

class RidgeScoreDiffModel:
    """L2 Ridge regression on historical score differences (home - away)."""
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=alpha, random_state=42))
        ])
        self.residual_std = None
    
    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray) -> Dict[str, float]:
        """Fit Ridge model and compute residual std."""
        self.pipeline.fit(X_train, y_train)
        y_pred_train = self.pipeline.predict(X_train)
        self.residual_std = float(np.std(y_train - y_pred_train, ddof=1))
        
        rmse = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))
        mae = float(mean_absolute_error(y_train, y_pred_train))
        
        return {'rmse': rmse, 'mae': mae, 'residual_std': self.residual_std}
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict margin (home - away)."""
        return self.pipeline.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Convert margin predictions to win probability."""
        margins = self.predict(X)
        std = self.residual_std if self.residual_std else 15.0
        # Assume normal distribution: P(home wins) = P(margin > 0)
        from scipy.stats import norm
        probs = norm.sf(-margins / std)  # 1 - CDF(-margin/std) = CDF(margin/std)
        return np.clip(probs, 0.001, 0.999)


# ============================================================================
# 2. ELO RATING MODEL (Nate Silver's variant)
# ============================================================================

class EloRating:
    """Nate Silver's Elo model for NBA teams."""
    
    def __init__(self, k_factor: float = 20.0, home_advantage: float = 70.0):
        """
        Args:
            k_factor: Rating adjustment magnitude per game
            home_advantage: Rating boost for home team
        """
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.ratings: Dict[str, float] = {}  # team_id -> rating
    
    def get_rating(self, team_id: str, default: float = 1500.0) -> float:
        """Get current rating for team."""
        return self.ratings.get(str(team_id), default)
    
    def set_rating(self, team_id: str, rating: float) -> None:
        """Set rating for team."""
        self.ratings[str(team_id)] = rating
    
    def expected_win_prob(self, home_rating: float, away_rating: float) -> float:
        """Probability home team wins given both teams' ratings."""
        diff = home_rating - away_rating
        return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
    
    def update_after_game(self, home_id: str, away_id: str, home_won: bool) -> Tuple[float, float]:
        """Update ratings after a game. Returns (new_home_rating, new_away_rating)."""
        home_id = str(home_id)
        away_id = str(away_id)
        
        home_r = self.get_rating(home_id)
        away_r = self.get_rating(away_id)
        
        # Expected: home team win probability with home advantage
        exp_home = self.expected_win_prob(home_r + self.home_advantage, away_r)
        
        # Actual: 1 if home won, 0 if away won
        actual = 1.0 if home_won else 0.0
        
        # Update
        home_r_new = home_r + self.k_factor * (actual - exp_home)
        away_r_new = away_r + self.k_factor * ((1.0 - actual) - (1.0 - exp_home))
        
        self.ratings[home_id] = home_r_new
        self.ratings[away_id] = away_r_new
        
        return home_r_new, away_r_new
    
    def add_elo_features(self, games_df: pd.DataFrame) -> Tuple[pd.DataFrame, EloRating]:
        """
        Add elo_home, elo_away, elo_diff columns to games_df.
        Simulates Elo progression through all historical games.
        """
        elo_features = []
        elo_sim = EloRating(self.k_factor, self.home_advantage)
        
        for _, row in games_df.iterrows():
            home_id = str(row['home_tid'])
            away_id = str(row['away_tid'])
            
            home_elo = elo_sim.get_rating(home_id)
            away_elo = elo_sim.get_rating(away_id)
            
            elo_features.append({
                'gid': row['gid'],
                'elo_home': home_elo,
                'elo_away': away_elo,
                'elo_diff': home_elo - away_elo
            })
            
            # Update after game
            home_won = row['home_score'] > row['away_score']
            elo_sim.update_after_game(home_id, away_id, home_won)
        
        elo_df = pd.DataFrame(elo_features)
        games_df = games_df.merge(elo_df, on='gid', how='left')
        
        return games_df, elo_sim


# ============================================================================
# 3. FOUR FACTORS MODEL (Dean Oliver)
# ============================================================================

class FourFactorsModel:
    """
    Dean Oliver's Four Factors:
    1. eFG% (Effective FG%) = (FG + 0.5*3P) / FGA
    2. TOV% (Turnover Rate) = TOV / (FGA + 0.44*FTA + TOV)
    3. ORB% (Offensive Rebound %) = ORB / (ORB + Opp DRB)
    4. FTR (Free Throw Rate) = FTA / FGA
    """
    
    def __init__(self):
        self.model = LinearRegression()
        self.feature_names = []
    
    def compute_team_four_factors(self, team_stats: Dict[str, float]) -> Dict[str, float]:
        """
        Compute four factors from team stats.
        Expected keys: fg, three_pm, fga, fta, tov, orb, drb
        """
        fg = team_stats.get('fg', 0)
        three_pm = team_stats.get('three_pm', 0)
        fga = team_stats.get('fga', 1)
        fta = team_stats.get('fta', 0)
        tov = team_stats.get('tov', 0)
        orb = team_stats.get('orb', 0)
        opp_drb = team_stats.get('opp_drb', 1)
        
        efg_pct = (fg + 0.5 * three_pm) / max(fga, 1)
        tov_pct = tov / max(fga + 0.44 * fta + tov, 1)
        orb_pct = orb / max(orb + opp_drb, 1)
        ftr = fta / max(fga, 1)
        
        return {
            'efg_pct': efg_pct,
            'tov_pct': tov_pct,
            'orb_pct': orb_pct,
            'ftr': ftr
        }
    
    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray) -> Dict[str, float]:
        """Fit four factors model on (home - away) margin."""
        self.feature_names = X_train.columns.tolist()
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_train)
        rmse = float(np.sqrt(mean_squared_error(y_train, y_pred)))
        mae = float(mean_absolute_error(y_train, y_pred))
        
        return {'rmse': rmse, 'mae': mae}
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict margin."""
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame, residual_std: float = 15.0) -> np.ndarray:
        """Convert margin predictions to win probability."""
        margins = self.predict(X)
        from scipy.stats import norm
        probs = norm.sf(-margins / residual_std)
        return np.clip(probs, 0.001, 0.999)


# ============================================================================
# 4. LOGISTIC REGRESSION META-LEARNER (Ensemble)
# ============================================================================

class LogisticEnsembler:
    """
    Meta-learner that blends sub-model predictions via Logistic Regression.
    Refits every N games to adapt to recent form (as per user specification).
    
    This learns a weighted blend of Ridge, Elo, Four Factors, and LGB predictions,
    continuously adapting the weights based on recent performance.
    """
    
    def __init__(self, refit_frequency: int = 20, random_state: int = 42):
        """
        Args:
            refit_frequency: Refit meta-learner every N games (user spec: every 20 games)
            random_state: Random seed for reproducibility
        """
        self.refit_frequency = refit_frequency
        self.random_state = random_state
        self.meta_model = LogisticRegression(
            fit_intercept=True,
            random_state=random_state,
            max_iter=1000,
            solver='lbfgs',
            C=1.0  # Regularization strength (default: good balance)
        )
        self.sub_model_weights = None
        self.coef_history = []
        self.latest_coefs = None  # Store latest weights for inference
    
    def fit(self, 
            X_meta: np.ndarray,  # Shape: (n_games, 4) [ridge, elo, ff, lgb]
            y: np.ndarray,        # Shape: (n_games,) - 1 if home wins, 0 if away
            gids: Optional[np.ndarray] = None
    ) -> Dict[str, any]:
        """
        Fit meta-learner with time-based continuous refitting every N games.
        
        This trains on all past games, then refits on rolling windows to capture
        how model quality changes over time. Perfect for early-season adaptation.
        
        Args:
            X_meta: Sub-model predictions stacked column-wise (ridge, elo, ff, lgb probs)
            y: Target (1 if home wins, 0 if away wins)
            gids: Game IDs (optional, for logging)
        
        Returns:
            Training history with logloss, weights, and refitting info
        """
        n_games = len(X_meta)
        oof_probs = np.full(n_games, np.nan, dtype=float)
        refit_intervals = list(range(0, n_games, self.refit_frequency))
        
        history = {
            'refits': 0,
            'oof_logloss': np.nan,
            'oof_brier': np.nan,
            'coefs': [],
            'refit_frequency': self.refit_frequency
        }
        
        # Time-based refitting: train on [:refit_i], validate on [refit_i:refit_{i+1})
        # This allows the meta-learner to continuously adapt as new games arrive
        for i in range(1, len(refit_intervals)):
            tr_start = 0
            tr_end = refit_intervals[i]
            val_start = refit_intervals[i]
            val_end = refit_intervals[i + 1] if i + 1 < len(refit_intervals) else n_games
            
            X_tr = X_meta[tr_start:tr_end]
            y_tr = y[tr_start:tr_end]
            X_val = X_meta[val_start:val_end]
            y_val = y[val_start:val_end] if val_end > val_start else np.array([])
            
            if len(X_tr) > 0 and len(X_val) > 0:
                self.meta_model.fit(X_tr, y_tr)
                oof_probs[val_start:val_end] = self.meta_model.predict_proba(X_val)[:, 1]
                
                # Log coefficients at each refit (shows how weights evolve)
                coef = self.meta_model.coef_[0].tolist()
                self.coef_history.append({
                    'refit_iteration': i,
                    'at_game_index': val_start,
                    'train_games': tr_end,
                    'coef': coef,  # [ridge_weight, elo_weight, ff_weight, lgb_weight]
                    'intercept': float(self.meta_model.intercept_[0])
                })
                history['refits'] += 1
                self.latest_coefs = coef  # Update for inference
        
        # Final metrics
        valid_mask = ~np.isnan(oof_probs)
        if valid_mask.any():
            history['oof_logloss'] = float(log_loss(y[valid_mask], oof_probs[valid_mask]))
            history['oof_brier'] = float(brier_score_loss(y[valid_mask], oof_probs[valid_mask]))
        
        history['coefs'] = self.coef_history
        self.sub_model_weights = self.meta_model.coef_[0]
        
        return history
    
    def predict_proba(self, X_meta: np.ndarray) -> np.ndarray:
        """Predict win probability for home team."""
        return self.meta_model.predict_proba(X_meta)[:, 1]
    
    def get_weights(self) -> Dict[str, float]:
        """Get sub-model weights."""
        return {
            'ridge': float(self.sub_model_weights[0]) if len(self.sub_model_weights) > 0 else 0,
            'elo': float(self.sub_model_weights[1]) if len(self.sub_model_weights) > 1 else 0,
            'four_factors': float(self.sub_model_weights[2]) if len(self.sub_model_weights) > 2 else 0,
            'lgb': float(self.sub_model_weights[3]) if len(self.sub_model_weights) > 3 else 0,
        }


# ============================================================================
# 5. EXHAUSTION FEATURES
# ============================================================================

def add_exhaustion_features(ps_df: pd.DataFrame) -> pd.DataFrame:
    """Add fatigue and usage-based features."""
    ps_df = ps_df.copy()
    
    # Season game count (how many games into the season)
    ps_df['season_games'] = ps_df.groupby(['playerId', 'season_end_year']).cumcount()
    ps_df['season_fatigue'] = np.clip(ps_df['season_games'] / 82.0, 0, 1)  # Normalized
    
    # Heavy usage indicator
    ps_df['heavy_usage'] = (ps_df['min_prev_mean10'] > 30).astype(float)
    
    # Back-to-back consecutive games
    ps_df['consecutive_b2b'] = ps_df.groupby('playerId')['player_b2b'].cumsum()
    ps_df['consecutive_b2b'] = ps_df['consecutive_b2b'] * ps_df['player_b2b']  # Zero out if not B2B
    
    # Rest accumulation (inverse: more rest = less fatigue)
    ps_df['rest_accumulated'] = ps_df.groupby('playerId')['days_rest'].cumsum()
    
    return ps_df
