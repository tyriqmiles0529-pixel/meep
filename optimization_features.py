"""
Advanced Optimization Features for NBA Prediction Pipeline

Implements:
1. Momentum Features (trend detection across multiple timeframes)
2. Better Window Selection (meta-learning for optimal window choice)
3. Market Signal Analysis (line movement, sharp vs public money)
4. Ensemble Stacking (weighted combination of all window predictions)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 1. MOMENTUM FEATURES
# ============================================================================

class MomentumAnalyzer:
    """
    Detect trends and momentum across multiple timeframes.
    
    Features generated:
    - Short-term momentum (last 3 games)
    - Medium-term momentum (last 7 games)
    - Long-term momentum (last 15 games)
    - Acceleration (change in momentum)
    - Hot/cold streaks
    - Performance vs expectation trend
    """
    
    def __init__(self, short_window=3, med_window=7, long_window=15):
        self.short_window = short_window
        self.med_window = med_window
        self.long_window = long_window
    
    def calculate_momentum(self, series: pd.Series, window: int) -> pd.Series:
        """
        Calculate momentum as linear regression slope over window.
        Positive = upward trend, Negative = downward trend
        """
        def slope(x):
            if len(x) < 2:
                return 0.0
            indices = np.arange(len(x))
            try:
                return np.polyfit(indices, x, 1)[0]
            except:
                return 0.0
        
        return series.rolling(window=window, min_periods=2).apply(slope, raw=False)
    
    def calculate_acceleration(self, momentum_short: pd.Series, momentum_med: pd.Series) -> pd.Series:
        """
        Acceleration = change in momentum (short vs medium term).
        Positive = accelerating improvement, Negative = decelerating
        """
        return momentum_short - momentum_med
    
    def detect_streak(self, series: pd.Series, threshold: float, window: int = 5) -> Tuple[pd.Series, pd.Series]:
        """
        Detect hot/cold streaks.
        
        Returns:
        - hot_streak: consecutive games above threshold
        - cold_streak: consecutive games below threshold (negative)
        """
        def count_streak(x):
            if len(x) == 0:
                return 0
            streak = 0
            for val in reversed(x):
                if val > threshold:
                    streak += 1
                else:
                    break
            return streak
        
        def count_cold_streak(x):
            if len(x) == 0:
                return 0
            streak = 0
            for val in reversed(x):
                if val < -threshold:
                    streak -= 1
                else:
                    break
            return streak
        
        hot_streak = series.rolling(window=window, min_periods=1).apply(count_streak, raw=False)
        cold_streak = series.rolling(window=window, min_periods=1).apply(count_cold_streak, raw=False)
        
        return hot_streak, cold_streak
    
    def add_momentum_features(self, df: pd.DataFrame, stat_col: str, 
                            group_by: str = 'playerId') -> pd.DataFrame:
        """
        Add all momentum features for a given stat column.
        
        Args:
            df: DataFrame with player/team stats
            stat_col: Column name to analyze (e.g., 'points', 'minutes')
            group_by: Column to group by (e.g., 'playerId', 'teamId')
        
        Returns:
            DataFrame with added momentum columns
        """
        result = df.copy()
        
        # Group by player/team
        grouped = result.groupby(group_by)[stat_col]
        
        # Calculate momentum at different timeframes
        result[f'{stat_col}_momentum_short'] = grouped.transform(
            lambda x: self.calculate_momentum(x, self.short_window)
        ).fillna(0.0)
        
        result[f'{stat_col}_momentum_med'] = grouped.transform(
            lambda x: self.calculate_momentum(x, self.med_window)
        ).fillna(0.0)
        
        result[f'{stat_col}_momentum_long'] = grouped.transform(
            lambda x: self.calculate_momentum(x, self.long_window)
        ).fillna(0.0)
        
        # Calculate acceleration (change in momentum)
        result[f'{stat_col}_acceleration'] = self.calculate_acceleration(
            result[f'{stat_col}_momentum_short'],
            result[f'{stat_col}_momentum_med']
        ).fillna(0.0)
        
        # Detect hot/cold streaks (using std as threshold)
        stat_std = result[stat_col].std()
        hot, cold = self.detect_streak(result[stat_col], threshold=stat_std * 0.5)
        result[f'{stat_col}_hot_streak'] = hot.fillna(0.0)
        result[f'{stat_col}_cold_streak'] = cold.fillna(0.0)
        
        # Normalize momentum features to [-1, 1] range for stability
        for col in [f'{stat_col}_momentum_short', f'{stat_col}_momentum_med', 
                   f'{stat_col}_momentum_long', f'{stat_col}_acceleration']:
            if result[col].std() > 0:
                result[col] = result[col].clip(-3 * result[col].std(), 3 * result[col].std())
                result[col] = result[col] / (3 * result[col].std() + 1e-6)
        
        return result


# ============================================================================
# 2. META-LEARNING WINDOW SELECTOR
# ============================================================================

class MetaWindowSelector:
    """
    Learn optimal window selection based on game/player context.
    
    Uses logistic regression to predict which window will be most accurate
    based on features like:
    - Recency (how recent is the data)
    - Sample size (number of games in window)
    - Player consistency (variance in stats)
    - League era (rule changes, pace changes)
    - Performance correlation (how well recent matches current context)
    """
    
    def __init__(self):
        self.model = LogisticRegression(multi_class='multinomial', max_iter=1000)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.window_labels = []
    
    def build_meta_features(self, window_data: Dict[str, pd.DataFrame],
                          target_date: pd.Timestamp,
                          player_id: str,
                          stat_type: str) -> np.ndarray:
        """
        Build meta-features for window selection.
        
        Args:
            window_data: Dict of {window_name: DataFrame} for each training window
            target_date: Date of prediction
            player_id: Player to predict for
            stat_type: Stat to predict ('points', 'rebounds', etc.)
        
        Returns:
            Feature vector for meta-learner
        """
        features = []
        
        for window_name, df in window_data.items():
            # Filter to player
            player_df = df[df['playerId'] == player_id] if 'playerId' in df.columns else df
            
            if len(player_df) == 0:
                # No data for this player in this window
                features.extend([0, 0, 0, 0, 0])
                continue
            
            # Recency: days between target and most recent game in window
            if 'date' in player_df.columns:
                max_date = pd.to_datetime(player_df['date']).max()
                recency_days = (target_date - max_date).days
            else:
                recency_days = 365  # Default if no date
            
            # Sample size
            sample_size = len(player_df)
            
            # Consistency: inverse of coefficient of variation
            if stat_type in player_df.columns:
                stat_mean = player_df[stat_type].mean()
                stat_std = player_df[stat_type].std()
                consistency = 1.0 / (1.0 + stat_std / (stat_mean + 1e-6))
            else:
                consistency = 0.0
            
            # Era similarity: absolute difference in average pace/scoring
            if 'season' in player_df.columns:
                window_avg_season = player_df['season'].mean()
                target_season = target_date.year
                era_diff = abs(target_season - window_avg_season)
            else:
                era_diff = 0
            
            # Trend alignment: is recent trend pointing in same direction as window avg
            if stat_type in player_df.columns and len(player_df) >= 5:
                recent_avg = player_df.tail(5)[stat_type].mean()
                overall_avg = player_df[stat_type].mean()
                trend_alignment = 1.0 if recent_avg > overall_avg else -1.0
            else:
                trend_alignment = 0.0
            
            features.extend([
                recency_days,
                sample_size,
                consistency,
                era_diff,
                trend_alignment
            ])
        
        return np.array(features).reshape(1, -1)
    
    def fit(self, X: np.ndarray, y: np.ndarray, window_labels: List[str]):
        """
        Fit meta-learner.
        
        Args:
            X: Meta-features (n_samples, n_features)
            y: Best window for each sample (window index)
            window_labels: Names of windows
        """
        self.window_labels = window_labels
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
    
    def predict_best_window(self, X: np.ndarray) -> str:
        """Predict which window to use."""
        if not self.is_fitted:
            return self.window_labels[0] if self.window_labels else "default"
        
        X_scaled = self.scaler.transform(X)
        window_idx = self.model.predict(X_scaled)[0]
        return self.window_labels[window_idx]
    
    def predict_window_weights(self, X: np.ndarray) -> np.ndarray:
        """Get probability distribution over windows (for ensemble weighting)."""
        if not self.is_fitted:
            n_windows = len(self.window_labels)
            return np.ones(n_windows) / n_windows
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[0]


# ============================================================================
# 3. MARKET SIGNAL ANALYZER
# ============================================================================

class MarketSignalAnalyzer:
    """
    Analyze betting market signals for predictive edge.
    
    Features:
    - Line movement (opening vs closing line)
    - Steam moves (sharp money indicators)
    - Reverse line movement (line moves against public)
    - Consensus fade opportunities
    - Market efficiency signals
    """
    
    def __init__(self):
        self.sharp_threshold = 0.03  # 3% move indicates sharp action
        self.rlm_threshold = 0.60    # 60% public betting threshold
    
    def calculate_line_movement(self, opening_line: float, closing_line: float) -> Dict[str, float]:
        """
        Calculate line movement features.
        
        Returns:
            - movement: absolute line change
            - movement_pct: percentage change
            - is_steam: whether this qualifies as a steam move
        """
        movement = closing_line - opening_line
        movement_pct = movement / (abs(opening_line) + 1e-6)
        is_steam = abs(movement_pct) >= self.sharp_threshold
        
        return {
            'line_movement': movement,
            'line_movement_pct': movement_pct,
            'is_steam_move': float(is_steam)
        }
    
    def detect_reverse_line_movement(self, opening_line: float, closing_line: float,
                                    public_bet_pct: float) -> Dict[str, float]:
        """
        Detect reverse line movement (RLM).
        
        RLM occurs when:
        - Public is heavily on one side (>60%)
        - Line moves in opposite direction (sharp money on other side)
        
        This is a strong indicator of sharp/professional money.
        """
        movement = closing_line - opening_line
        public_side = 1 if public_bet_pct > 0.5 else -1
        movement_direction = 1 if movement > 0 else -1
        
        # RLM detected if movement against public
        is_rlm = (public_side != movement_direction) and (abs(public_bet_pct - 0.5) > 0.1)
        rlm_strength = abs(public_bet_pct - 0.5) * abs(movement) if is_rlm else 0.0
        
        return {
            'is_reverse_line_movement': float(is_rlm),
            'rlm_strength': rlm_strength,
            'fade_public_signal': float(is_rlm and abs(public_bet_pct - 0.5) > 0.2)
        }
    
    def calculate_market_efficiency(self, implied_prob: float, true_prob: float,
                                   num_books: int = 1) -> Dict[str, float]:
        """
        Measure market efficiency.
        
        More books = more efficient (harder to beat)
        Large discrepancy between implied and true prob = opportunity
        """
        prob_diff = abs(implied_prob - true_prob)
        
        # Adjust for number of books (more books = tighter market)
        efficiency_score = min(num_books / 10.0, 1.0)  # Normalize to [0, 1]
        
        # Edge opportunity (larger diff + fewer books = more edge)
        edge_opportunity = prob_diff * (1.0 - efficiency_score)
        
        return {
            'market_efficiency': efficiency_score,
            'prob_discrepancy': prob_diff,
            'edge_opportunity': edge_opportunity
        }
    
    def add_market_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all market signal features to DataFrame.
        
        Expected columns:
        - opening_line, closing_line (for line movement)
        - public_bet_pct (optional, for RLM detection)
        - market_implied_prob, model_prob (for efficiency)
        - num_books (optional, defaults to 1)
        """
        result = df.copy()
        
        # Line movement features
        if 'opening_line' in result.columns and 'closing_line' in result.columns:
            movement_features = result.apply(
                lambda row: self.calculate_line_movement(row['opening_line'], row['closing_line']),
                axis=1
            )
            for key in ['line_movement', 'line_movement_pct', 'is_steam_move']:
                result[key] = [d[key] for d in movement_features]
        
        # Reverse line movement (if public data available)
        if all(col in result.columns for col in ['opening_line', 'closing_line', 'public_bet_pct']):
            rlm_features = result.apply(
                lambda row: self.detect_reverse_line_movement(
                    row['opening_line'], row['closing_line'], row['public_bet_pct']
                ),
                axis=1
            )
            for key in ['is_reverse_line_movement', 'rlm_strength', 'fade_public_signal']:
                result[key] = [d[key] for d in rlm_features]
        
        # Market efficiency (if both implied and model probs available)
        if 'market_implied_prob' in result.columns and 'model_prob' in result.columns:
            num_books = result['num_books'] if 'num_books' in result.columns else 1
            efficiency_features = result.apply(
                lambda row: self.calculate_market_efficiency(
                    row['market_implied_prob'],
                    row['model_prob'],
                    row.get('num_books', 1)
                ),
                axis=1
            )
            for key in ['market_efficiency', 'prob_discrepancy', 'edge_opportunity']:
                result[key] = [d[key] for d in efficiency_features]
        
        return result


# ============================================================================
# 4. ENSEMBLE STACKER
# ============================================================================

class EnsembleStacker:
    """
    Stack predictions from multiple windows using learned optimal weights.
    
    Methods:
    - Simple averaging
    - Weighted by recency
    - Learned weights (Ridge regression on validation set)
    - Dynamic weights (meta-learning per prediction)
    """
    
    def __init__(self, method='learned', alpha=1.0):
        """
        Args:
            method: 'simple', 'recency', 'learned', or 'dynamic'
            alpha: Ridge regularization parameter (for learned method)
        """
        self.method = method
        self.alpha = alpha
        self.weights = None
        self.model = None
        
        if method == 'learned':
            self.model = Ridge(alpha=alpha)
            self.scaler = StandardScaler()
    
    def fit(self, window_predictions: np.ndarray, y_true: np.ndarray,
           window_names: Optional[List[str]] = None):
        """
        Learn optimal ensemble weights.
        
        Args:
            window_predictions: (n_samples, n_windows) array of predictions
            y_true: (n_samples,) array of true values
            window_names: Names of windows (for recency weighting)
        """
        n_windows = window_predictions.shape[1]
        
        if self.method == 'simple':
            self.weights = np.ones(n_windows) / n_windows
        
        elif self.method == 'recency':
            # More recent windows get higher weight
            if window_names:
                # Extract years from window names (assumes format like "2020_2024")
                years = []
                for name in window_names:
                    try:
                        end_year = int(name.split('_')[1])
                        years.append(end_year)
                    except:
                        years.append(2020)  # Default
                
                # Linear weighting by recency
                max_year = max(years)
                weights = np.array([(year - min(years) + 1) for year in years])
                self.weights = weights / weights.sum()
            else:
                # Fallback to simple averaging
                self.weights = np.ones(n_windows) / n_windows
        
        elif self.method == 'learned':
            # Learn weights via Ridge regression
            X_scaled = self.scaler.fit_transform(window_predictions)
            self.model.fit(X_scaled, y_true)
            # Weights are the coefficients
            self.weights = self.model.coef_
            # Normalize to sum to 1
            self.weights = np.abs(self.weights) / np.abs(self.weights).sum()
    
    def predict(self, window_predictions: np.ndarray) -> np.ndarray:
        """
        Generate ensemble prediction.
        
        Args:
            window_predictions: (n_samples, n_windows) array of predictions
        
        Returns:
            (n_samples,) array of ensemble predictions
        """
        if self.method == 'learned' and self.model is not None:
            X_scaled = self.scaler.transform(window_predictions)
            return self.model.predict(X_scaled)
        else:
            # Weighted average
            if self.weights is None:
                # Fallback to simple average
                return window_predictions.mean(axis=1)
            return window_predictions @ self.weights
    
    def get_weights(self) -> np.ndarray:
        """Return current ensemble weights."""
        return self.weights if self.weights is not None else np.array([])


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def add_all_optimization_features(
    df: pd.DataFrame,
    stat_cols: List[str] = ['points', 'rebounds', 'assists', 'minutes'],
    group_by: str = 'playerId',
    add_momentum: bool = True,
    add_market: bool = True
) -> pd.DataFrame:
    """
    Convenience function to add all optimization features.
    
    Args:
        df: Input DataFrame
        stat_cols: Stat columns to add momentum features for
        group_by: Grouping column (playerId, teamId, etc.)
        add_momentum: Whether to add momentum features
        add_market: Whether to add market signal features
    
    Returns:
        DataFrame with added optimization features
    """
    result = df.copy()
    
    # Add momentum features
    if add_momentum:
        momentum = MomentumAnalyzer()
        for stat in stat_cols:
            if stat in result.columns:
                result = momentum.add_momentum_features(result, stat, group_by)
    
    # Add market signals
    if add_market:
        market = MarketSignalAnalyzer()
        result = market.add_market_signals(result)
    
    return result
