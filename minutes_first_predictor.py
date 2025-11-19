"""
Minutes-First Prediction Pipeline

Instead of predicting raw stats (points, rebounds, assists, threes),
this approach:
1. Predicts minutes (most stable/predictable)
2. Predicts rate stats (PPM, APM, RPM, 3PM - per minute rates)
3. Multiplies: final_stat = minutes * rate

Why this works:
- Minutes are more stable than raw stats (less variance)
- Rate stats are less noisy than totals
- Reduces compounding error
- Expected: +5-10% accuracy improvement

Usage:
    predictor = MinutesFirstEnsemblePredictor(model_cache_dir="model_cache")
    predictions = predictor.predict(X)
    # Returns: {'points': 25.3, 'rebounds': 8.2, 'assists': 5.1, 'threes': 2.8}
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from pathlib import Path
from ensemble_predictor import load_all_window_models, predict_with_window


class MinutesFirstEnsemblePredictor:
    """
    Minutes-first ensemble predictor with meta-learner support.

    Architecture:
    1. Ensemble predicts minutes (27 windows + meta-learner)
    2. Ensemble predicts rate stats (PPM, APM, RPM, 3PM)
    3. Final predictions = minutes * rates
    """

    def __init__(self, model_cache_dir: str = "model_cache", use_meta_learner: bool = True):
        """
        Args:
            model_cache_dir: Directory with window models
            use_meta_learner: Use meta-learner for weighting (if available)
        """
        self.model_cache_dir = model_cache_dir
        self.use_meta_learner = use_meta_learner

        # Load all window models
        print("[*] Loading 27 window models for minutes-first prediction...")
        self.window_models = load_all_window_models(model_cache_dir)
        print(f"[OK] Loaded {len(self.window_models)} windows")

        # Load meta-learner if available
        self.meta_learner = None
        if use_meta_learner:
            meta_paths = [
                Path("meta_learner_2025_2026.pkl"),
                Path(model_cache_dir) / "meta_learner_2025_2026.pkl",
                Path("meta_learner_2024_2025.pkl"),
            ]

            for meta_path in meta_paths:
                if meta_path.exists():
                    from meta_learner_ensemble import ContextAwareMetaLearner
                    self.meta_learner = ContextAwareMetaLearner.load(str(meta_path))
                    print(f"[OK] Loaded meta-learner from {meta_path}")
                    break

            if self.meta_learner is None:
                print(f"[!] Meta-learner not found, using simple averaging")

    def _predict_with_ensemble(self, X: pd.DataFrame, prop: str,
                               player_context: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Get ensemble predictions for a prop (minutes or rate stat)

        Args:
            X: Features (n_samples, n_features)
            prop: Property to predict ('minutes', 'ppm', 'apm', 'rpm', 'threepm')
            player_context: Optional player context for meta-learner

        Returns:
            predictions: (n_samples,)
        """
        # Get predictions from all windows
        window_preds = []
        for window_name, window_models in self.window_models.items():
            try:
                preds = predict_with_window(window_models, X, prop)
                window_preds.append(preds)
            except Exception as e:
                # Window may not have this prop, fallback to zero
                window_preds.append(np.zeros(len(X)))

        if not window_preds:
            raise ValueError(f"No window predictions available for {prop}")

        # Stack predictions: (n_samples, n_windows)
        X_base = np.column_stack(window_preds)

        # Use meta-learner if available
        if self.meta_learner and prop in self.meta_learner.meta_models:
            if player_context is None:
                # Extract basic context from features
                player_context = self._extract_context_from_features(X)

            return self.meta_learner.predict(
                window_predictions=X_base,
                player_context=player_context,
                prop_name=prop
            )

        # Fallback: simple average
        return np.mean(X_base, axis=1)

    def _extract_context_from_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract player context from features (for meta-learner)"""
        context = pd.DataFrame(index=X.index)

        # Position encoding
        if 'position' in X.columns:
            position_map = {'PG': 0, 'SG': 1, 'SF': 2, 'PF': 3, 'C': 4}
            context['position_encoded'] = X['position'].map(position_map).fillna(2)

        # Usage rate proxy
        if all(c in X.columns for c in ['fieldGoalsAttempted', 'freeThrowsAttempted', 'assists']):
            context['usage_rate'] = (
                X['fieldGoalsAttempted'].fillna(0) +
                X['freeThrowsAttempted'].fillna(0) * 0.44 +
                X['assists'].fillna(0) * 0.33
            )

        # Minutes
        if 'numMinutes' in X.columns:
            context['minutes_avg'] = X['numMinutes'].fillna(0)
        elif 'minutes' in X.columns:
            context['minutes_avg'] = X['minutes'].fillna(0)

        # Home/away
        if 'home' in X.columns:
            context['is_home'] = X['home'].astype(int)

        return context

    def _compute_rate_stats(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Compute per-minute rate stats from features.

        If features contain recent averages (e.g., points_L5_avg, numMinutes_L5_avg),
        we can derive rate stats. Otherwise, return zeros (will be predicted).

        Args:
            X: Features DataFrame

        Returns:
            DataFrame with columns: ['ppm', 'apm', 'rpm', 'threepm']
        """
        rate_stats = pd.DataFrame(index=X.index)

        # Try to derive from recent averages
        if 'points_L5_avg' in X.columns and 'numMinutes_L5_avg' in X.columns:
            rate_stats['ppm'] = X['points_L5_avg'] / X['numMinutes_L5_avg'].replace(0, 1)
            rate_stats['apm'] = X['assists_L5_avg'] / X['numMinutes_L5_avg'].replace(0, 1)
            rate_stats['rpm'] = X['reboundsTotal_L5_avg'] / X['numMinutes_L5_avg'].replace(0, 1)
            rate_stats['threepm'] = X['threes_L5_avg'] / X['numMinutes_L5_avg'].replace(0, 1)
        else:
            # Fallback: zeros (will be overridden by ensemble prediction)
            rate_stats['ppm'] = 0.0
            rate_stats['apm'] = 0.0
            rate_stats['rpm'] = 0.0
            rate_stats['threepm'] = 0.0

        return rate_stats

    def predict(self, X: pd.DataFrame, player_context: Optional[pd.DataFrame] = None) -> Dict[str, np.ndarray]:
        """
        Predict using minutes-first approach.

        Process:
        1. Predict minutes (ensemble of 27 windows + meta-learner)
        2. Predict rate stats (PPM, APM, RPM, 3PM)
        3. Multiply: final_stat = minutes * rate

        Args:
            X: Features (n_samples, n_features)
            player_context: Optional player context for meta-learner

        Returns:
            predictions: {
                'minutes': np.ndarray,
                'points': np.ndarray,
                'rebounds': np.ndarray,
                'assists': np.ndarray,
                'threes': np.ndarray,
                'ppm': np.ndarray,  # Rate stats (for inspection)
                'apm': np.ndarray,
                'rpm': np.ndarray,
                'threepm': np.ndarray
            }
        """
        # Step 1: Predict minutes (most stable)
        minutes = self._predict_with_ensemble(X, 'minutes', player_context)

        # Step 2: Predict rate stats
        # Check if models support rate stats directly
        try:
            ppm = self._predict_with_ensemble(X, 'ppm', player_context)
            apm = self._predict_with_ensemble(X, 'apm', player_context)
            rpm = self._predict_with_ensemble(X, 'rpm', player_context)
            threepm = self._predict_with_ensemble(X, 'threepm', player_context)
        except (ValueError, KeyError):
            # Fallback: Compute from features or predict totals and derive rates
            print("[!] Rate stats not in models, deriving from totals...")

            # Predict totals
            points = self._predict_with_ensemble(X, 'points', player_context)
            assists = self._predict_with_ensemble(X, 'assists', player_context)
            rebounds = self._predict_with_ensemble(X, 'rebounds', player_context)
            threes = self._predict_with_ensemble(X, 'threes', player_context)

            # Derive rates (avoid division by zero)
            ppm = points / np.maximum(minutes, 1.0)
            apm = assists / np.maximum(minutes, 1.0)
            rpm = rebounds / np.maximum(minutes, 1.0)
            threepm = threes / np.maximum(minutes, 1.0)

        # Step 3: Multiply minutes * rates = final predictions
        predictions = {
            'minutes': minutes,
            'points': minutes * ppm,
            'rebounds': minutes * rpm,
            'assists': minutes * apm,
            'threes': minutes * threepm,
            # Include rate stats for inspection
            'ppm': ppm,
            'apm': apm,
            'rpm': rpm,
            'threepm': threepm
        }

        return predictions

    def predict_all_props(self, X: pd.DataFrame, player_context: Optional[pd.DataFrame] = None) -> Dict[str, np.ndarray]:
        """Alias for predict() for compatibility with EnsemblePredictor API"""
        return self.predict(X, player_context)


# Example usage
if __name__ == "__main__":
    import pandas as pd

    # Load predictor
    predictor = MinutesFirstEnsemblePredictor(
        model_cache_dir="model_cache",
        use_meta_learner=True
    )

    # Create sample features (normally from riq_analyzer)
    X = pd.DataFrame({
        'points_L5_avg': [15.2],
        'assists_L5_avg': [5.3],
        'reboundsTotal_L5_avg': [7.1],
        'numMinutes_L5_avg': [28.5],
        'fieldGoalsAttempted': [12.3],
        'freeThrowsAttempted': [3.2],
        # ... more features
    })

    # Predict
    predictions = predictor.predict(X)

    print("Minutes-First Predictions:")
    print(f"  Minutes: {predictions['minutes'][0]:.1f}")
    print(f"  Points:  {predictions['points'][0]:.1f} (PPM: {predictions['ppm'][0]:.3f})")
    print(f"  Rebounds: {predictions['rebounds'][0]:.1f} (RPM: {predictions['rpm'][0]:.3f})")
    print(f"  Assists: {predictions['assists'][0]:.1f} (APM: {predictions['apm'][0]:.3f})")
    print(f"  Threes:  {predictions['threes'][0]:.1f} (3PM: {predictions['threepm'][0]:.3f})")
