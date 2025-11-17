"""
SHAP Explainability Module for NBA Predictions

Provides feature attribution and explanation for model predictions.
Shows which features drove each prediction (e.g., "High usage rate +2.3 pts").

Usage:
    from explainability import SHAPExplainer

    explainer = SHAPExplainer(model)
    explanation = explainer.explain_prediction(X_single, prop_type='points')
    print(explanation['top_features'])
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import pickle
import os

# Try to import SHAP (optional dependency)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not installed. Run: pip install shap")


class SHAPExplainer:
    """
    SHAP-based explainability for NBA prediction models.

    Provides:
    - Feature importance rankings
    - Individual prediction explanations
    - "Top N reasons" for each prediction
    """

    def __init__(self, model=None, model_path: Optional[str] = None, background_data: Optional[pd.DataFrame] = None):
        """
        Initialize SHAP explainer.

        Args:
            model: Trained model (LightGBM, NeuralHybrid, or sklearn)
            model_path: Path to pickled model file
            background_data: Sample of training data for SHAP (100-500 rows recommended)
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP library required. Install with: pip install shap")

        self.model = model
        if model_path and not model:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)

        self.background_data = background_data
        self.explainer = None
        self.feature_names = None

        # Cache for SHAP values
        self._shap_cache = {}

    def _get_base_model(self):
        """Extract the base model for SHAP (handles NeuralHybrid wrapper)."""
        if hasattr(self.model, 'lgbm'):
            # NeuralHybridPredictor - use LightGBM component
            return self.model.lgbm
        elif hasattr(self.model, 'hybrid_model'):
            # HybridWrapper - use the underlying model
            return self.model.hybrid_model
        else:
            return self.model

    def _init_explainer(self, X: pd.DataFrame):
        """Initialize SHAP explainer with background data."""
        if self.explainer is not None:
            return

        base_model = self._get_base_model()

        # Store feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
        else:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        # Choose appropriate explainer based on model type
        model_type = type(base_model).__name__

        if 'LGBMRegressor' in model_type or 'LGBMClassifier' in model_type:
            # TreeExplainer is fastest for tree models
            self.explainer = shap.TreeExplainer(base_model)
        elif 'GradientBoosting' in model_type or 'RandomForest' in model_type:
            self.explainer = shap.TreeExplainer(base_model)
        else:
            # Fallback to KernelExplainer (slower but universal)
            if self.background_data is not None:
                background = self.background_data
            else:
                # Use subset of X as background
                background = shap.sample(X, min(100, len(X)))

            self.explainer = shap.KernelExplainer(
                base_model.predict if hasattr(base_model, 'predict') else base_model,
                background
            )

    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute SHAP values for input data.

        Args:
            X: Features DataFrame

        Returns:
            SHAP values array (n_samples, n_features)
        """
        self._init_explainer(X)

        # Convert to numpy if needed (some explainers prefer this)
        X_arr = X.values if isinstance(X, pd.DataFrame) else X

        shap_values = self.explainer.shap_values(X_arr)

        # Handle multi-output (classification returns list)
        if isinstance(shap_values, list):
            # For binary classification, use positive class
            shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]

        return shap_values

    def explain_prediction(
        self,
        X_single: pd.DataFrame,
        prop_type: str = 'points',
        top_n: int = 5,
        include_plot: bool = False
    ) -> Dict[str, Any]:
        """
        Explain a single prediction.

        Args:
            X_single: Single row of features (1, n_features)
            prop_type: Type of prop (points, assists, rebounds, threes, minutes)
            top_n: Number of top features to return
            include_plot: Whether to generate SHAP plot

        Returns:
            Dictionary with:
            - prediction: Model prediction
            - top_features: List of (feature_name, shap_value, direction)
            - explanation_text: Human-readable explanation
            - shap_values: Raw SHAP values
        """
        if len(X_single) != 1:
            X_single = X_single.iloc[[0]] if isinstance(X_single, pd.DataFrame) else X_single[:1]

        # Get prediction
        prediction = self._predict(X_single)

        # Compute SHAP values
        shap_values = self.compute_shap_values(X_single)[0]  # First (only) row

        # Get feature names
        if self.feature_names is None:
            if isinstance(X_single, pd.DataFrame):
                self.feature_names = list(X_single.columns)
            else:
                self.feature_names = [f'feature_{i}' for i in range(len(shap_values))]

        # Sort by absolute SHAP value
        feature_importance = sorted(
            zip(self.feature_names, shap_values, X_single.values[0]),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        # Top N features
        top_features = []
        for feat_name, shap_val, feat_val in feature_importance[:top_n]:
            direction = "+" if shap_val > 0 else ""
            top_features.append({
                'feature': feat_name,
                'shap_value': float(shap_val),
                'feature_value': float(feat_val),
                'direction': 'positive' if shap_val > 0 else 'negative',
                'formatted': f"{feat_name}: {direction}{shap_val:.3f}"
            })

        # Generate human-readable explanation
        explanation_text = self._generate_explanation(prop_type, prediction, top_features)

        result = {
            'prediction': float(prediction),
            'prop_type': prop_type,
            'top_features': top_features,
            'explanation_text': explanation_text,
            'shap_values': shap_values.tolist(),
            'feature_names': self.feature_names,
            'base_value': float(self.explainer.expected_value) if hasattr(self.explainer, 'expected_value') else None
        }

        if include_plot:
            result['plot'] = self._generate_plot(X_single, shap_values)

        return result

    def _predict(self, X):
        """Get model prediction."""
        if hasattr(self.model, 'predict'):
            pred = self.model.predict(X)
            if isinstance(pred, np.ndarray):
                return pred[0] if len(pred) == 1 else pred
            return pred
        else:
            # Callable model
            return self.model(X)[0]

    def _generate_explanation(self, prop_type: str, prediction: float, top_features: List[Dict]) -> str:
        """Generate human-readable explanation."""
        lines = [f"Predicted {prop_type}: {prediction:.2f}"]
        lines.append("\nTop factors driving this prediction:")

        for i, feat in enumerate(top_features, 1):
            feat_name = feat['feature']
            shap_val = feat['shap_value']
            feat_val = feat['feature_value']

            # Human-readable feature names
            readable_name = self._make_readable(feat_name)

            if shap_val > 0:
                impact = f"increases prediction by {shap_val:.2f}"
            else:
                impact = f"decreases prediction by {abs(shap_val):.2f}"

            lines.append(f"  {i}. {readable_name} = {feat_val:.2f} → {impact}")

        return "\n".join(lines)

    def _make_readable(self, feature_name: str) -> str:
        """Convert feature name to readable format."""
        # Common mappings
        mappings = {
            'adv_usg_percent': 'Usage Rate',
            'adv_ts_percent': 'True Shooting %',
            'adv_per': 'Player Efficiency Rating',
            'per100_pts_per_100_poss': 'Points per 100 Poss',
            'per100_ast_per_100_poss': 'Assists per 100 Poss',
            'per100_trb_per_100_poss': 'Rebounds per 100 Poss',
            'shoot_percent_fga_from_x3p_range': '3P Shot Frequency',
            'shoot_corner_3_point_percent': 'Corner 3P %',
            'pbp_pg_percent': 'PG Position %',
            'home_off_strength': 'Home Team Offensive Strength',
            'away_def_strength': 'Away Team Defensive Strength',
            'match_pace_sum': 'Combined Pace',
            'numMinutes': 'Minutes Played',
            'threePointersMade': '3-Pointers Made',
            'reboundsTotal': 'Total Rebounds',
        }

        if feature_name in mappings:
            return mappings[feature_name]

        # Clean up common patterns
        name = feature_name
        name = name.replace('_', ' ').replace('percent', '%')
        name = name.replace('adv ', 'Advanced: ')
        name = name.replace('per100 ', 'Per100: ')
        name = name.replace('shoot ', 'Shooting: ')
        name = name.replace('pbp ', 'Play-by-Play: ')

        return name.title()

    def _generate_plot(self, X_single, shap_values):
        """Generate SHAP waterfall plot (returns matplotlib figure)."""
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6))

            # Create SHAP Explanation object
            explanation = shap.Explanation(
                values=shap_values,
                base_values=self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0,
                data=X_single.values[0] if isinstance(X_single, pd.DataFrame) else X_single[0],
                feature_names=self.feature_names
            )

            shap.plots.waterfall(explanation, max_display=10, show=False)

            return fig
        except Exception as e:
            print(f"Warning: Could not generate plot: {e}")
            return None

    def global_feature_importance(self, X: pd.DataFrame, max_samples: int = 1000) -> pd.DataFrame:
        """
        Compute global feature importance using SHAP.

        Args:
            X: Training/validation data
            max_samples: Maximum samples to use (for speed)

        Returns:
            DataFrame with feature importance rankings
        """
        # Sample if too large
        if len(X) > max_samples:
            X_sample = X.sample(max_samples, random_state=42)
        else:
            X_sample = X

        # Compute SHAP values
        shap_values = self.compute_shap_values(X_sample)

        # Mean absolute SHAP value per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'mean_abs_shap': mean_abs_shap,
            'std_shap': shap_values.std(axis=0),
        }).sort_values('mean_abs_shap', ascending=False)

        importance_df['rank'] = range(1, len(importance_df) + 1)
        importance_df['importance_pct'] = 100 * importance_df['mean_abs_shap'] / importance_df['mean_abs_shap'].sum()

        return importance_df


def add_shap_to_prediction(
    prop_result: Dict[str, Any],
    model,
    features: pd.DataFrame,
    prop_type: str = 'points',
    top_n: int = 3
) -> Dict[str, Any]:
    """
    Add SHAP explanation to an existing prediction result.

    Designed to integrate with riq_analyzer.py's analyze_player_prop output.

    Args:
        prop_result: Dictionary from analyze_player_prop
        model: Trained model
        features: Feature DataFrame for this prediction
        prop_type: Type of prop
        top_n: Number of top features to show

    Returns:
        prop_result with added 'explanation' field
    """
    if not SHAP_AVAILABLE:
        prop_result['explanation'] = "SHAP not available. Install with: pip install shap"
        return prop_result

    try:
        explainer = SHAPExplainer(model)
        explanation = explainer.explain_prediction(features, prop_type, top_n)

        # Add to result
        prop_result['explanation'] = {
            'top_factors': explanation['top_features'],
            'text': explanation['explanation_text'],
            'base_value': explanation['base_value']
        }

        # Add human-readable summary
        top_3_summary = []
        for feat in explanation['top_features'][:3]:
            name = explainer._make_readable(feat['feature'])
            impact = f"+{feat['shap_value']:.2f}" if feat['shap_value'] > 0 else f"{feat['shap_value']:.2f}"
            top_3_summary.append(f"{name} ({impact})")

        prop_result['why'] = " | ".join(top_3_summary)

    except Exception as e:
        prop_result['explanation'] = f"SHAP error: {e}"
        prop_result['why'] = "Explanation unavailable"

    return prop_result


# Convenience function for quick explanations
def explain_bet(
    player_name: str,
    prop_type: str,
    model_path: str,
    features: pd.DataFrame,
    line: float = None
) -> str:
    """
    Quick explanation for a single bet.

    Usage:
        explanation = explain_bet(
            "LeBron James", "points",
            "models/points_model.pkl",
            features_df
        )
        print(explanation)
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    explainer = SHAPExplainer(model)
    result = explainer.explain_prediction(features, prop_type)

    text = f"\n{'='*50}\n"
    text += f"SHAP EXPLANATION: {player_name} - {prop_type.upper()}\n"
    text += f"{'='*50}\n"
    text += f"Prediction: {result['prediction']:.2f}"

    if line:
        diff = result['prediction'] - line
        text += f" (Line: {line:.1f}, Diff: {diff:+.2f})\n"
    else:
        text += "\n"

    text += "\nKey Factors:\n"
    for i, feat in enumerate(result['top_features'], 1):
        name = explainer._make_readable(feat['feature'])
        val = feat['feature_value']
        impact = feat['shap_value']
        text += f"  {i}. {name} = {val:.2f} → {impact:+.3f}\n"

    text += f"{'='*50}\n"

    return text


if __name__ == "__main__":
    # Example usage
    print("SHAP Explainability Module for NBA Predictions")
    print("=" * 50)

    if not SHAP_AVAILABLE:
        print("Install SHAP: pip install shap")
        exit(1)

    # Demo with dummy model
    print("To use:")
    print("  1. Load your trained model")
    print("  2. Create SHAPExplainer(model)")
    print("  3. Call explainer.explain_prediction(X_single)")
    print("\nExample output:")
    print("""
    Predicted points: 24.7

    Top factors driving this prediction:
      1. Usage Rate = 28.3 → increases prediction by 3.42
      2. Minutes Played = 36.2 → increases prediction by 2.18
      3. Away Team Defensive Strength = 0.92 → decreases prediction by 1.54
      4. True Shooting % = 0.612 → increases prediction by 1.21
      5. Combined Pace = 102.4 → increases prediction by 0.89
    """)
