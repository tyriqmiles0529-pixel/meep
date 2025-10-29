"""
Enhanced Ensemble Models with:
- Optimal refit frequency (20 games default, tunable)
- Dynamic K-factor for Elo based on upset magnitude
- Logistic meta-learner with polynomial interaction features
- Per-team/conference calibration
- Rolling Four Factors priors (10-game window)
- Game-level exhaustion features
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 1. RIDGE REGRESSION (Score Differential)
# ============================================================================

class RidgeScoreDiffModel:
    """L2-regularized ridge on past score differentials."""
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Convert margin prediction to win probability via sigmoid."""
        margin = self.predict(X)
        prob_home = 1.0 / (1.0 + np.exp(-margin / 10.0))  # Scale margin to probability
        return np.column_stack([1 - prob_home, prob_home])


# ============================================================================
# 2. DYNAMIC ELO RATING
# ============================================================================

class DynamicEloRating:
    """Elo with dynamic K-factor based on upset magnitude."""
    
    def __init__(self, base_k=20.0, home_advantage=70):
        self.base_k = base_k
        self.home_advantage = home_advantage
        self.team_ratings = {}
        self.rating_history = {}
        
    def _expected_win_prob(self, rating_diff):
        """Expected win probability from rating difference."""
        return 1.0 / (1.0 + 10.0 ** (-rating_diff / 400.0))
    
    def _dynamic_k_factor(self, expected_prob, actual_outcome):
        """
        Adjust K-factor based on surprise magnitude.
        High surprise (upset) → higher K → faster adaptation
        Low surprise (chalk) → lower K → stable rating
        """
        surprise = abs(actual_outcome - expected_prob)  # 0-1 scale
        # K ranges from 0.5x to 2x base K
        k_multiplier = 0.5 + 1.5 * surprise
        return self.base_k * k_multiplier
    
    def update(self, home_team, away_team, home_score, away_score):
        """Update ratings after a game."""
        # Initialize if needed
        if home_team not in self.team_ratings:
            self.team_ratings[home_team] = 1500.0
            self.rating_history[home_team] = [1500.0]
        if away_team not in self.team_ratings:
            self.team_ratings[away_team] = 1500.0
            self.rating_history[away_team] = [1500.0]
        
        home_rating = self.team_ratings[home_team]
        away_rating = self.team_ratings[away_team]
        
        # Rating difference with home advantage
        rating_diff = home_rating - away_rating + self.home_advantage
        
        # Expected win probability for home team
        expected_prob_home = self._expected_win_prob(rating_diff)
        
        # Actual outcome
        actual_home = 1.0 if home_score > away_score else 0.0
        
        # Dynamic K-factor based on upset
        k = self._dynamic_k_factor(expected_prob_home, actual_home)
        
        # Update ratings
        rating_change_home = k * (actual_home - expected_prob_home)
        rating_change_away = -rating_change_home  # Zero-sum
        
        self.team_ratings[home_team] = home_rating + rating_change_home
        self.team_ratings[away_team] = away_rating + rating_change_away
        
        # Store history
        self.rating_history[home_team].append(self.team_ratings[home_team])
        self.rating_history[away_team].append(self.team_ratings[away_team])
    
    def expected_win_prob(self, home_team, away_team):
        """Get expected win probability for home team."""
        home_rating = self.team_ratings.get(home_team, 1500.0)
        away_rating = self.team_ratings.get(away_team, 1500.0)
        rating_diff = home_rating - away_rating + self.home_advantage
        return self._expected_win_prob(rating_diff)


# ============================================================================
# 3. FOUR FACTORS WITH ROLLING PRIORS
# ============================================================================

class FourFactorsModelDynamic:
    """Dean Oliver's Four Factors with rolling 10-game priors."""
    
    def __init__(self, rolling_window=10):
        self.rolling_window = rolling_window
        self.model = Ridge(alpha=1.0)
        self.scaler = StandardScaler()
        self.team_game_history = {}  # Track per-team stats
        
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self
    
    def update_team_stats(self, team_id, stats_dict):
        """Update rolling stats for a team after each game."""
        if team_id not in self.team_game_history:
            self.team_game_history[team_id] = []
        self.team_game_history[team_id].append(stats_dict)
        # Keep only last rolling_window games
        if len(self.team_game_history[team_id]) > self.rolling_window:
            self.team_game_history[team_id] = self.team_game_history[team_id][-self.rolling_window:]
    
    def get_rolling_priors(self, team_id):
        """Get rolling average priors for a team."""
        if team_id not in self.team_game_history or len(self.team_game_history[team_id]) == 0:
            return None
        
        games = self.team_game_history[team_id]
        stats = ['efg_pct', 'tov_pct', 'orb_pct', 'ftr']
        
        rolling_avg = {}
        for stat in stats:
            values = [g.get(stat, 0.5) for g in games]
            rolling_avg[stat] = np.mean(values)
        return rolling_avg
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        margin = self.predict(X)
        prob_home = 1.0 / (1.0 + np.exp(-margin / 10.0))
        return np.column_stack([1 - prob_home, prob_home])


# ============================================================================
# 4. ENHANCED META-LEARNER WITH INTERACTIONS & CALIBRATION
# ============================================================================

class EnhancedLogisticEnsembler:
    """
    Logistic meta-learner with:
    - Polynomial interaction features
    - Per-team/conference calibration
    - Continuous refitting with optimal frequency
    - Coefficient logging
    """
    
    def __init__(self, refit_frequency=20, calibration_mode='global'):
        """
        Args:
            refit_frequency: Games between refits (10, 20, or 30)
            calibration_mode: 'global', 'home_away', or 'conference'
        """
        self.refit_frequency = refit_frequency
        self.calibration_mode = calibration_mode
        self.models = {}  # For per-team/conference models
        self.coefficients_history = []  # Track weight evolution
        self.game_counter = 0
        
    def _add_interaction_features(self, probs_array):
        """
        Convert [ridge_p, elo_p, ff_p, lgb_p] to interaction features.
        
        Features:
        - Original 4 probabilities
        - Squared terms (confidence)
        - Pairwise products (agreement)
        - Sum and product (ensemble consensus)
        """
        ridge_p, elo_p, ff_p, lgb_p = probs_array
        
        features = [
            ridge_p, elo_p, ff_p, lgb_p,  # Base
            ridge_p**2, elo_p**2, ff_p**2, lgb_p**2,  # Squared (confidence)
            ridge_p * elo_p, ridge_p * ff_p, ridge_p * lgb_p,  # Ridge interactions
            elo_p * ff_p, elo_p * lgb_p,  # Elo interactions
            ff_p * lgb_p,  # Four Factors × LGB
            np.mean([ridge_p, elo_p, ff_p, lgb_p]),  # Average (simple ensemble)
            np.max([ridge_p, elo_p, ff_p, lgb_p]),  # Max (most confident)
        ]
        return np.array(features)
    
    def _get_model_key(self, game_row):
        """Get calibration key based on mode."""
        if self.calibration_mode == 'global':
            return 'global'
        elif self.calibration_mode == 'home_away':
            return 'home' if game_row.get('home_team') else 'away'
        elif self.calibration_mode == 'conference':
            conference = game_row.get('home_conference', 'UNKNOWN')
            return conference
        return 'global'
    
    def fit(self, X_meta, y, game_rows=None):
        """
        Train meta-learner on stacked predictions.
        
        Args:
            X_meta: Array of shape (n_games, 4) with [ridge_p, elo_p, ff_p, lgb_p]
            y: Binary outcomes (0/1)
            game_rows: List of game dicts for calibration key lookup
        """
        if game_rows is None:
            game_rows = [{'home_team': 0}] * len(X_meta)
        
        # Add interaction features
        X_enhanced = np.array([self._add_interaction_features(row) for row in X_meta])
        
        if self.calibration_mode == 'global':
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_enhanced, y)
            self.models['global'] = model
        else:
            # Train separate models per key
            for key in set(self._get_model_key(row) for row in game_rows):
                mask = np.array([self._get_model_key(row) == key for row in game_rows])
                if mask.sum() > 10:  # Need minimum samples
                    model = LogisticRegression(max_iter=1000, random_state=42)
                    model.fit(X_enhanced[mask], y[mask])
                    self.models[key] = model
        
        # Log coefficients
        self._log_coefficients()
        return self
    
    def _log_coefficients(self):
        """Store model coefficients for analysis."""
        for key, model in self.models.items():
            self.coefficients_history.append({
                'key': key,
                'game_counter': self.game_counter,
                'intercept': model.intercept_[0],
                'coefficients': model.coef_[0].tolist(),  # [ridge, elo, ff, lgb, ridge^2, elo^2, ...]
            })
    
    def predict_proba(self, X_meta, game_row=None):
        """
        Predict probability with interaction features.
        
        Args:
            X_meta: Array of shape (1, 4) or (n, 4)
            game_row: Single game dict or None
        
        Returns:
            Predicted probabilities [P(away wins), P(home wins)]
        """
        X_meta = np.atleast_2d(X_meta)
        X_enhanced = np.array([self._add_interaction_features(row) for row in X_meta])
        
        if self.calibration_mode == 'global' or game_row is None:
            model = self.models.get('global', None)
        else:
            key = self._get_model_key(game_row)
            model = self.models.get(key, self.models.get('global', None))
        
        if model is None:
            raise ValueError("No trained model available")
        
        return model.predict_proba(X_enhanced)
    
    def should_refit(self):
        """Check if we should refit based on game counter."""
        self.game_counter += 1
        return (self.game_counter % self.refit_frequency) == 0
    
    def get_coefficients_history_df(self):
        """Return coefficients as DataFrame for analysis."""
        return pd.DataFrame(self.coefficients_history)


# ============================================================================
# 5. GAME-LEVEL EXHAUSTION FEATURES
# ============================================================================

def add_game_exhaustion_features(games_df):
    """
    Add exhaustion features at game level.
    
    Features:
    - season_fatigue: Progress through 82-game season (0-1)
    - b2b_indicator: 1 if team playing back-to-back
    - consecutive_b2b_count: Number of consecutive B2B games
    - days_since_last_game: Rest available
    - team_usage_rate: Avg usage% of team's active players
    """
    games_df = games_df.copy()
    
    # Sort by date
    games_df = games_df.sort_values('date').reset_index(drop=True)
    
    # Season fatigue: game number / 82 (use season_end_year if 'season' doesn't exist)
    season_col = 'season_end_year' if 'season_end_year' in games_df.columns else 'season'
    games_df['home_season_fatigue'] = games_df.groupby(season_col).cumcount() / 82.0
    games_df['away_season_fatigue'] = games_df.groupby(season_col).cumcount() / 82.0
    
    # Back-to-back tracking per team
    for team_col in ['home_team', 'away_team']:
        team_grouped = games_df.groupby(team_col)
        days_since = team_grouped['date'].diff().dt.days.fillna(999)
        games_df[f'{team_col.replace("_team", "")}_b2b'] = (days_since == 1).astype(int)
        
        # Consecutive B2B count (reset on 2+ day rest)
        consecutive_b2b = []
        current_count = 0
        for days in days_since:
            if days == 1:
                current_count += 1
            else:
                current_count = 0
            consecutive_b2b.append(current_count)
        games_df[f'{team_col.replace("_team", "")}_consecutive_b2b'] = consecutive_b2b
    
    # Days rest
    games_df['home_days_rest'] = games_df.groupby('home_team')['date'].diff().dt.days.fillna(3).clip(0, 7)
    games_df['away_days_rest'] = games_df.groupby('away_team')['date'].diff().dt.days.fillna(3).clip(0, 7)
    
    return games_df


# ============================================================================
# 6. UTILITY FUNCTIONS
# ============================================================================

def create_ensemble_training_data(ridge_model, elo_model, ff_model, lgb_model, 
                                  games_df, game_features, game_defaults):
    """
    Generate stacked meta-learner training data.
    
    Returns:
        X_meta: (n_games, 4) array with [ridge_p, elo_p, ff_p, lgb_p]
        y: (n_games,) binary outcomes
    """
    X_game = games_df[game_features].fillna(pd.Series(game_defaults, index=game_features)).astype(float)
    
    ridge_probs = ridge_model.predict_proba(X_game)[:, 1]
    ff_probs = ff_model.predict_proba(X_game)[:, 1]
    lgb_probs = lgb_model.predict_proba(X_game)[:, 1]
    
    # Elo probs from ratings
    elo_probs = []
    for _, row in games_df.iterrows():
        prob = elo_model.expected_win_prob(row['home_team'], row['away_team'])
        elo_probs.append(prob)
    elo_probs = np.array(elo_probs)
    
    X_meta = np.column_stack([ridge_probs, elo_probs, ff_probs, lgb_probs])
    y = (games_df['home_score'] > games_df['away_score']).astype(int).values
    
    return X_meta, y, games_df


def plot_coefficient_evolution(ensembler, output_path='coefficient_evolution.csv'):
    """Save coefficient history for analysis."""
    coef_df = ensembler.get_coefficients_history_df()
    coef_df.to_csv(output_path, index=False)
    print(f"Coefficient evolution saved to {output_path}")
    
    # Summary stats
    if len(coef_df) > 0:
        print("\n=== Meta-Learner Coefficient Evolution ===")
        print(f"Refits performed: {len(coef_df)}")
        print(f"Average coefficients (first 4 features: ridge, elo, ff, lgb):")
        avg_coefs = np.array([c['coefficients'][:4] for c in coef_df[-5:]])  # Last 5 refits
        print(f"  Ridge:  {avg_coefs[:, 0].mean():.4f}")
        print(f"  Elo:    {avg_coefs[:, 1].mean():.4f}")
        print(f"  FF:     {avg_coefs[:, 2].mean():.4f}")
        print(f"  LGB:    {avg_coefs[:, 3].mean():.4f}")


if __name__ == '__main__':
    print("Enhanced Ensemble Models module loaded.")
    print("Features:")
    print("  ✓ Dynamic Elo with upset-based K-factor")
    print("  ✓ Rolling Four Factors priors (10-game window)")
    print("  ✓ Logistic meta-learner with polynomial interactions")
    print("  ✓ Per-team/conference calibration")
    print("  ✓ Game-level exhaustion features")
    print("  ✓ Coefficient tracking and logging")
