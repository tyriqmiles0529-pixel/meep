"""
Ensemble Predictor - Loads All 25 Windows + Meta-Learner

Use this instead of single-model predictions for maximum accuracy.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import torch


def load_all_window_models(model_cache_dir: str = "model_cache", max_year: Optional[int] = None) -> Dict:
    """Load all 25 window models from cache, optionally filtering by max training year"""
    import os

    # Force CPU mode BEFORE loading any models
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

    cache_path = Path(model_cache_dir)

    if not cache_path.exists():
        raise FileNotFoundError(f"Model cache not found: {model_cache_dir}")

    # Find all player model files
    model_files = sorted(cache_path.glob("player_models_*.pkl"))

    if not model_files:
        raise FileNotFoundError(f"No player models found in {model_cache_dir}")

    print(f"Loading models (Max Year: {max_year})...")

    all_models = {}
    for model_file in model_files:
        # Extract window years from filename
        stem = model_file.stem  # "player_models_2022_2024"
        parts = stem.split('_')
        if len(parts) >= 4:
            start_year = int(parts[2])
            end_year = int(parts[3])
            
            # Filter by max_year (prevent lookahead)
            if max_year is not None and end_year > max_year:
                continue

            # Load models with CPU mapping
            class CPU_Unpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if module == 'torch.storage' and name == '_load_from_bytes':
                        import io
                        return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
                    return super().find_class(module, name)

            with open(model_file, 'rb') as f:
                models = CPU_Unpickler(f).load()

            # Force all to CPU (critical for Modal)
            device = torch.device('cpu')
            if isinstance(models, dict):
                for key, model in models.items():
                    if hasattr(model, 'use_gpu'):
                        model.use_gpu = False
                    if hasattr(model, 'device_name'):
                        model.device_name = 'cpu'
                    if hasattr(model, 'device'):
                        model.device = 'cpu'

                    # Force TabNet to CPU
                    if hasattr(model, 'network') and hasattr(model.network, 'to'):
                        model.network.to(device)

                    # Force TabNet regressor device
                    if hasattr(model, 'tabnet') and hasattr(model.tabnet, 'device'):
                        model.tabnet.device = 'cpu'
                        if hasattr(model.tabnet, 'network'):
                            model.tabnet.network.to(device)

                    # Disable GPU flag in TabNet
                    if hasattr(model, 'tabnet') and hasattr(model.tabnet, 'device_name'):
                        model.tabnet.device_name = 'cpu'

            # Extract feature names
            feature_names = None
            if isinstance(models, dict):
                if 'multi_task_model' in models and hasattr(models['multi_task_model'], 'feature_names'):
                    feature_names = models['multi_task_model'].feature_names
                elif 'points' in models and hasattr(models['points'], 'feature_names'):
                    feature_names = models['points'].feature_names

            all_models[f"{start_year}-{end_year}"] = {
                'models': models,
                'start_year': start_year,
                'end_year': end_year,
                'feature_names': feature_names
            }
            print(f"  Loaded {start_year}-{end_year} ({len(feature_names) if feature_names else '?'} features)")

    return all_models


def predict_with_window(window_models: Dict, X: pd.DataFrame, prop: str) -> np.ndarray:
    """Get predictions from one window for one prop"""

    # Align features with model's training features
    if 'feature_names' in window_models and window_models['feature_names']:
        model_features = window_models['feature_names']

        # Only use features that model was trained on
        available_features = [f for f in model_features if f in X.columns]
        X_aligned = X[available_features].copy()

        # Add missing features as zeros (use concat to avoid fragmentation)
        missing_features = [f for f in model_features if f not in X_aligned.columns]
        if missing_features:
            missing_df = pd.DataFrame(0, index=X_aligned.index, columns=missing_features)
            X_aligned = pd.concat([X_aligned, missing_df], axis=1)

        # Ensure column order matches training
        X_aligned = X_aligned[model_features]
    else:
        X_aligned = X

    # Convert to numeric
    numeric_cols = []
    for col in X_aligned.columns:
        if X_aligned[col].dtype.name == 'category':
            try:
                X_aligned[col] = X_aligned[col].astype(float)
                numeric_cols.append(col)
            except:
                pass
        elif X_aligned[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            numeric_cols.append(col)

    X_aligned = X_aligned[numeric_cols].fillna(0)

    # Get predictions
    models = window_models['models']

    # Map prop names
    prop_map = {
        'points': 'points',
        'rebounds': 'rebounds',
        'assists': 'assists',
        'threes': 'threes',
        'minutes': 'minutes'
    }

    model_prop = prop_map.get(prop, prop)

    # Check if hybrid multi-task model
    if 'multi_task_model' in models:
        with torch.no_grad():
            preds = models['multi_task_model'].predict(X_aligned)
        if isinstance(preds, dict) and model_prop in preds:
            return preds[model_prop]

    # Old single-task models
    if model_prop in models and models[model_prop] is not None:
        with torch.no_grad():
            return models[model_prop].predict(X_aligned)

    # Fallback: return zeros
    return np.zeros(len(X_aligned))


class EnsemblePredictor:
    """
    Ensemble predictor using all 27 windows + meta-learner

    Supports two prediction modes:
    1. Direct: Predict stats directly (points, rebounds, assists, threes)
    2. Minutes-first: Predict minutes, then rates (PPM, APM, RPM), then multiply
    """

    def __init__(self, model_cache_dir: str = "model_cache", use_meta_learner: bool = True,
                 use_minutes_first: bool = False, max_year: Optional[int] = None):
        """
        Args:
            model_cache_dir: Directory with window models
            use_meta_learner: Use meta-learner for weighting (if available)
            use_minutes_first: Use minutes-first prediction pipeline (default: False)
            max_year: Maximum training year to include (for backtesting)
        """
        self.model_cache_dir = model_cache_dir
        self.use_meta_learner = use_meta_learner
        self.use_minutes_first = use_minutes_first
        self.max_year = max_year

        # Load all window models
        self.window_models = load_all_window_models(model_cache_dir, max_year=max_year)

        # Load meta-learner if available (try current season first, then previous)
        self.meta_learner = None
        if use_meta_learner:
            # Try 2025-2026 (current season)
            meta_paths = [
                Path("meta_learner_2025_2026.pkl"),
                Path(model_cache_dir) / "meta_learner_2025_2026.pkl",
                Path("meta_learner_2024_2025.pkl"),  # Fallback to previous season
            ]

            for meta_path in meta_paths:
                if meta_path.exists():
                    from meta_learner_ensemble import ContextAwareMetaLearner
                    self.meta_learner = ContextAwareMetaLearner.load(str(meta_path))
                    print(f"[OK] Loaded meta-learner from {meta_path}")
                    break

            if self.meta_learner is None:
                print(f"Meta-learner not found, using simple averaging")

    def predict(self, X: pd.DataFrame, prop: str, player_context: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Predict using ensemble of all windows

        Args:
            X: Features (n_samples, n_features)
            prop: Property to predict ('points', 'rebounds', 'assists', 'threes', 'minutes')
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
                print(f"Warning: Failed to get predictions from {window_name}: {e}")
                continue

        if not window_preds:
            raise ValueError("No window predictions available")

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
        """Extract player context from features"""
        context = pd.DataFrame(index=X.index)

        # Position encoding (if available)
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

    def predict_all_props(self, X: pd.DataFrame, player_context: Optional[pd.DataFrame] = None) -> Dict[str, np.ndarray]:
        """
        Predict all props at once.

        If use_minutes_first=True, uses minutes-first pipeline:
        1. Predict minutes
        2. Predict rate stats (PPM, APM, RPM, 3PM)
        3. Multiply: stat = minutes * rate

        Otherwise, predicts stats directly.
        """
        if self.use_minutes_first:
            # Minutes-first pipeline
            try:
                # Step 1: Predict minutes
                minutes = self.predict(X, 'minutes', player_context)

                # Step 2: Derive rate stats from totals (fallback method)
                # Get total predictions first
                points_total = self.predict(X, 'points', player_context)
                assists_total = self.predict(X, 'assists', player_context)
                rebounds_total = self.predict(X, 'rebounds', player_context)
                threes_total = self.predict(X, 'threes', player_context)

                # Derive rates (avoid division by zero)
                minutes_safe = np.maximum(minutes, 1.0)
                ppm = points_total / minutes_safe
                apm = assists_total / minutes_safe
                rpm = rebounds_total / minutes_safe
                threepm = threes_total / minutes_safe

                # Step 3: Multiply minutes * rates
                return {
                    'minutes': minutes,
                    'points': minutes * ppm,
                    'rebounds': minutes * rpm,
                    'assists': minutes * apm,
                    'threes': minutes * threepm,
                    # Include rates for inspection
                    'ppm': ppm,
                    'apm': apm,
                    'rpm': rpm,
                    'threepm': threepm
                }
            except Exception as e:
                print(f"[!] Minutes-first failed: {e}, falling back to direct prediction")
                # Fallback to direct if minutes-first fails

        # Direct prediction (original behavior)
        props = ['points', 'rebounds', 'assists', 'threes', 'minutes']

        predictions = {}
        for prop in props:
            try:
                predictions[prop] = self.predict(X, prop, player_context)
            except Exception as e:
                print(f"Warning: Failed to predict {prop}: {e}")
                predictions[prop] = np.zeros(len(X))

        return predictions


# Example usage
if __name__ == "__main__":
    # Load ensemble predictor
    ensemble = EnsemblePredictor(
        model_cache_dir="model_cache",
        use_meta_learner=True  # Use meta-learner if available
    )

    # Create sample features (normally from riq_analyzer)
    import pandas as pd
    X = pd.DataFrame({
        'points_L5_avg': [15.2],
        'assists_L5_avg': [5.3],
        'reboundsTotal_L5_avg': [7.1],
        # ... more features
    })

    # Predict all props
    predictions = ensemble.predict_all_props(X)

    print("Ensemble Predictions:")
    for prop, pred in predictions.items():
        print(f"  {prop}: {pred[0]:.1f}")
