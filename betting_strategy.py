
import torch
import torch.nn as nn

class FTTransformer(nn.Module):
    def __init__(self, num_players, num_continuous, embed_dim=16, depth=2, heads=4):
        super().__init__()

        # Categorical Embedding (Player ID)
        self.player_embedding = nn.Embedding(num_players, embed_dim)

        # Continuous Feature Encoding (Linear project to embed_dim)
        self.cont_encoder = nn.Linear(num_continuous, embed_dim)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=heads, dim_feedforward=embed_dim*2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Output Heads (Weak Supervision)
        # Predict Next Game PTS Bin (Low, Med, High, Explosion)
        self.head = nn.Linear(embed_dim, 4)

    def forward(self, x_cat, x_cont):
        # x_cat: [batch, 1] (PlayerID)
        # x_cont: [batch, num_cont]

        # 1. Embed Categorical
        # [batch, 1, dim]
        # x_cat might be [batch] or [batch, 1]. Ensure [batch].
        if x_cat.dim() == 2:
            x_cat = x_cat.squeeze(1)

        emb_cat = self.player_embedding(x_cat).unsqueeze(1)

        # 2. Embed Continuous
        # [batch, 1, dim]
        emb_cont = self.cont_encoder(x_cont).unsqueeze(1)

        # 3. Stack as Sequence: [Player, Stats]
        # x: [batch, 2, dim]
        x = torch.cat([emb_cat, emb_cont], dim=1)

        # 4. Transform
        x = self.transformer(x)

        # 5. Pooling (Mean of sequence) -> [batch, dim]
        representation = x.mean(dim=1)

        # 6. Prediction
        logits = self.head(representation)

        return logits, representation
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
import pandas as pd
import numpy as np
import os
import json
import joblib
import torch
import datetime
import sys
from data_processor import BasketballDataProcessor
# Add Odds Ingestion
try:
    from odds_ingestion.mock import MockOddsProvider
    from odds_ingestion.the_odds_api import TheOddsAPIProvider
    from odds_ingestion.rapid_api import RapidAPIProvider
except ImportError:
    MockOddsProvider = None
    TheOddsAPIProvider = None
    RapidAPIProvider = None

class BettingStrategy:
    def __init__(self, models_dir='models', data_path='final_feature_matrix_with_per_min_1997_onward.csv', provider='mock'):
        self.models_dir = models_dir
        self.data_path = data_path
        self.targets = ['points', 'rebounds', 'assists', 'three_pointers']
        self.predictors = {}
        self.ft_extractors = {}
        self.processor = None
        
        self.odds_provider = None
        if provider == 'the-odds-api' and TheOddsAPIProvider:
            self.odds_provider = TheOddsAPIProvider()
        elif provider == 'rapid-api' and RapidAPIProvider:
            self.odds_provider = RapidAPIProvider()
        elif MockOddsProvider:
            self.odds_provider = MockOddsProvider()
            if provider != 'mock':
                print(f"Warning: Provider '{provider}' not found or unavailable. Using Mock.")
        
    def load_resources(self):
        print("Loading Data Processor...")
        self.processor = BasketballDataProcessor(self.data_path)
        self.processor.load_data()
        # Preprocess for features (using 'points' as dummy target to generate features)
        self.processor.preprocess(target='points')
        
        print("Loading Models...")
        for target in self.targets:
            print(f"Loading {target} model...")
            # Load Ensemble
            pred = EnsemblePredictor()
            # Set model_dir to the target subdirectory (e.g. models/points)
            pred.model_dir = os.path.join(self.models_dir, target)
            # Load latest models (e.g. xgb_model_points.pkl)
            # Wait, ls showed xgb_model_points.pkl in models/points?
            # ls meep/nba_predictor/models/points showed xgb_model_2022.pkl etc.
            # It also showed xgb_model_points.pkl?
            # Step 3319 showed xgb_model_points.pkl in meep/nba_predictor/models (root models dir).
            # Step 3352 showed xgb_model_2022.pkl in meep/nba_predictor/models/points.
            
            # So for LATEST (production), we might use the ones in root models dir?
            # Or maybe we should use 2022 (latest available)?
            # Let's use 2022 as latest for now.
            
            # If using root models dir:
            # pred.model_dir = self.models_dir
            # pred.load_models(suffix=f"_{target}")
            
            # If using models/points/xgb_model_2022.pkl:
            # pred.model_dir = os.path.join(self.models_dir, target)
            # pred.load_models(suffix="_2022")
            
            # Let's assume we want to load the "latest" available season model.
            # For 2025 season (current), we might not have a model yet?
            # Or we use 2022 model for everything?
            # Let's try to load "_points" from root first (if it exists).
            # If not, try "_2022" from subdir.
            
            # Based on ls output:
            # models/xgb_model_points.pkl exists.
            # models/points/xgb_model_2022.pkl exists.
            
            # Let's use the root one for "default" loading.
            pred.model_dir = self.models_dir
            try:
                pred.load_models(suffix=f"_{target}")
                self.predictors[target] = pred
            except Exception as e:
                print(f"Failed to load default {target} model: {e}")
            
            # Load FT-Transformer (Global or per target? Train script saved global per season)
            # We need a strategy to pick the right FT model. 
            # For inference on new data, we should use the LATEST available model.
            # Let's assume we use the one from the latest trained season (e.g., 2025).
            ft_path = os.path.join(self.models_dir, f"global_ft_2025", "ft_transformer.pt")
            if os.path.exists(ft_path):
                # We need to know cardinalities to init the model structure first
                cat_cols = self.processor.get_cat_cols()
                cardinalities = [len(self.processor.label_encoders[col].classes_) for col in cat_cols]
                
                ft = FTTransformerFeatureExtractor(cardinalities, embed_dim=16, device='cpu')
                ft.load(ft_path)
                self.ft_extractors['global'] = ft
            else:
                print(f"Warning: FT-Transformer not found at {ft_path}")

    def generate_predictions(self, date_str):
        # Filter data for the specific date
        # This assumes we have data for that date in the csv (historical backtest)
        # For live, we'd need to fetch new data.
        
        df_day = self.processor.df[self.processor.df['date'] == date_str].copy()
        
        if df_day.empty:
            print(f"No games found for {date_str}")
            return None
            
        # Generate Embeddings
        if 'global' in self.ft_extractors:
            cat_cols = self.processor.get_cat_cols()
            X_cat = df_day[cat_cols].values
            embeddings = self.ft_extractors['global'].transform(X_cat)
            
            # Add to DF
            emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
            df_emb = pd.DataFrame(embeddings, columns=emb_cols, index=df_day.index)
            df_day = pd.concat([df_day, df_emb], axis=1)
            
        predictions = {}
        for target in self.targets:
            # Predict
            # We need to ensure columns match what the model expects
            # The EnsemblePredictor.predict method handles DMatrix conversion
            # But we need to pass the right columns.
            # We can use the processor's feature_columns + embedding columns
            
            # Filter for features only
            # Combine processor features + any embedding columns we added
            features_to_use = self.processor.feature_columns.copy()
            if 'global' in self.ft_extractors:
                 features_to_use += [c for c in df_day.columns if c.startswith('emb_')]
            
            # Ensure all features exist
            valid_features = [f for f in features_to_use if f in df_day.columns]
            X_pred = df_day[valid_features]
            
            if target in self.predictors:
                preds = self.predictors[target].predict(X_pred, use_stacking=True)
                predictions[target] = preds
            else:
                predictions[target] = np.zeros(len(df_day))
            
        # Combine into a results DF
        results = df_day[['player_name', 'playerteamName', 'opponentteamName', 'minutes']].copy()
        for t, p in predictions.items():
            results[f'pred_{t}'] = p
            
        # Add Confidence (Dummy for now, can be variance of ensemble members)
        # results['confidence'] = ...
        
        return results

    def calculate_ev(self, row, target, line, odds):
        # Simple EV calculation
        # Prob = Probability of winning bet
        # We need a distribution to get Prob(pred > line).
        # For now, assume normal distribution with std dev from validation RMSE.
        
        # Load RMSE from report
        rmses = {'points': 4.5, 'rebounds': 2.0, 'assists': 1.8, 'three_pointers': 0.8}
        skews = {'points': 2.0, 'rebounds': 2.5, 'assists': 2.2, 'three_pointers': 1.5}
        
        rmse = rmses.get(target, 4.5)
        skew_a = skews.get(target, 0)
        
        from scipy.stats import norm, skewnorm
        
        pred = row[f'pred_{target}']
        
        # Probability of Over
        prob_over = 1 - skewnorm.cdf(line, skew_a, loc=pred, scale=rmse)
        
        # Probability of Under
        prob_under = skewnorm.cdf(line, skew_a, loc=pred, scale=rmse)
        
        # Decimal Odds
        dec_odds = odds
        if odds < 0:
            dec_odds = 1 + (100 / abs(odds))
        else:
            dec_odds = 1 + (odds / 100)
            
        # EV
        # Assuming we bet OVER if pred > line
        if pred > line:
            win_prob = prob_over
        else:
            win_prob = prob_under
            
        ev = (win_prob * (dec_odds - 1)) - (1 - win_prob)
        
        return ev, win_prob

    def calculate_confidence(self, prediction, line, rmse):
        """
        Calculate confidence score (0-100) based on Z-score.
        """
        if rmse <= 0: return 0
        z_score = abs(prediction - line) / rmse
        # Map Z-score to 0-100. 
        # Z=1 (1 sigma) -> ~68% conf interval? 
        # Let's scale: Z=2 -> 100% confidence (arbitrary but practical)
        raw_score = (z_score / 2.0) * 100
        return min(100.0, max(0.0, raw_score))

    def load_season_models(self, season):
        """
        Load models trained on data prior to 'season'.
        E.g. if season=2022, load models trained on 1997-2021.
        Assumes models are saved with suffix '_{season}'.
        """
        print(f"Loading models for Season {season}...")
        
        # Determine model suffix/path based on season
        # Our training script saves models per season.
        # e.g. "points_2022" is trained on data UP TO 2022 (inclusive? or exclusive?).
        # Walk-forward logic: Train on 1997-2021, Validate on 2022.
        # So for predicting 2022 games, we need the model trained on 1997-2021.
        # Let's assume the model saved as '2021' is the one trained on data up to 2021.
        
        train_season = season - 1
        
        for target in self.targets:
            # Load Ensemble
            pred = EnsemblePredictor()
            # Models are in subdirectories: models/points/xgb_model_2021.pkl
            pred.model_dir = os.path.join(self.models_dir, target)
            
            # Try loading specific season model
            try:
                pred.load_models(suffix=f"_{train_season}") 
                self.predictors[target] = pred
            except Exception as e:
                print(f"Error loading {target} model for {train_season}: {e}")
                # Fallback to latest? Or skip?
                # For backtest validity, we should probably skip or warn.
                pass
                
        # Load FT-Transformer
        ft_path = os.path.join(self.models_dir, f"global_ft_{train_season}", "ft_transformer.pt")
        print(f"DEBUG: Attempting to load FT from {ft_path}")
        if os.path.exists(ft_path):
            try:
                cat_cols = self.processor.get_cat_cols()
                print(f"DEBUG: cat_cols for FT: {len(cat_cols)} columns")
                cardinalities = [len(self.processor.label_encoders[col].classes_) for col in cat_cols]
                
                ft = FTTransformerFeatureExtractor(cardinalities, embed_dim=16, device='cpu')
                ft.load(ft_path)
                self.ft_extractors['global'] = ft
                print(f"DEBUG: FT-Transformer loaded successfully for {train_season}")
            except Exception as e:
                print(f"DEBUG: Failed to init/load FT: {e}")
        else:
            print(f"Warning: FT-Transformer not found for {train_season} at {ft_path}")

    def generate_bets(self, predictions_df, bankroll=1000, confidence_threshold=10, kelly_fraction=0.25, min_ev=0.0):
        # 1. Filter by Minutes (Strict)
        min_col = 'pred_minutes' if 'pred_minutes' in predictions_df.columns else 'minutes'
        if min_col not in predictions_df.columns:
            mask_min = pd.Series([True] * len(predictions_df))
        else:
            mask_min = predictions_df[min_col] >= 20
            
        candidates = predictions_df[mask_min].copy()
        
        bets = []
        rmses = {'points': 4.5, 'rebounds': 2.0, 'assists': 1.8, 'three_pointers': 0.8}
        
        for idx, row in candidates.iterrows():
            for target in self.targets:
                line = row.get(f'line_{target}')
                odds = row.get(f'odds_{target}', -110)
                
                if line is None:
                    # Map full target name to dataset column abbreviation
                    # dataset: prior_pts, prior_reb, prior_ast, prior_three_pointers (maybe?)
                    abbr_map = {
                        'points': 'prior_pts',
                        'rebounds': 'prior_reb', 
                        'assists': 'prior_ast',
                        'three_pointers': 'prior_three_pointers' # Verify this one exists or fallback
                    }
                    
                    prior_col = abbr_map.get(target)
                    
                    if prior_col and prior_col in row:
                        line = row[prior_col]
                    else:
                        # Fallback for three pointers if not in prior
                        # or if prior_col is null
                        continue 
                        
                pred = row[f'pred_{target}']
                rmse = rmses.get(target, 4.5)
                
                ev, win_prob = self.calculate_ev(row, target, line, odds)
                conf = self.calculate_confidence(pred, line, rmse)
                
                # Filters
                if ev > min_ev and conf >= confidence_threshold:
                    bets.append({
                        'player': row['player_name'],
                        'team': row.get('playerteamName', 'N/A'),
                        'game_id': row.get('gameId', 'N/A'),
                        'target': target,
                        'line': line,
                        'prediction': pred,
                        'ev': ev,
                        'win_prob': win_prob,
                        'odds': odds,
                        'confidence': conf
                    })
                    
        bets_df = pd.DataFrame(bets)
        if bets_df.empty:
            return pd.DataFrame()
            
        # 3. Ranking
        bets_df = bets_df.sort_values('ev', ascending=False)
        
        # 4. Dynamic Kelly
        def get_kelly(row):
            b = 0.909 
            if row['odds'] < 0:
                b = 100 / abs(row['odds'])
            else:
                b = row['odds'] / 100
            p = row['win_prob']
            q = 1 - p
            f = (b * p - q) / b
            return max(0, f)
            
        bets_df['kelly_fraction'] = bets_df.apply(get_kelly, axis=1)
        
        # Scale by Confidence & User Fraction
        bets_df['adj_kelly'] = bets_df['kelly_fraction'] * (bets_df['confidence'] / 100.0)
        
        # Drawdown Protection (Placeholder)
        drawdown_pct = 0.0
        if drawdown_pct > 0.10:
            bets_df['adj_kelly'] *= 0.5
            
        bets_df['stake_pct'] = bets_df['adj_kelly'] * kelly_fraction # Use passed fraction (default 0.25 = 1/4)
        bets_df['stake_amt'] = bets_df['stake_pct'] * bankroll
        
        return bets_df

    def generate_parlays(self, bets_df, num_parlays=5):
        # Generate 5 parlays from top selections
        # Constraint: Uncorrelated legs (Max 1 leg per Game ID)
        
        parlays = []
        # Pool of best bets (Top 20)
        pool = bets_df.head(20).copy()
        
        if len(pool) < 2:
            return pd.DataFrame()
            
        import random
        
        attempts = 0
        while len(parlays) < num_parlays and attempts < 50:
            attempts += 1
            num_legs = random.randint(2, 5)
            
            # Iterative selection to ensure unique games
            current_parlay = []
            used_games = set()
            
            # Shuffle pool to randomize start
            shuffled_pool = pool.sample(frac=1)
            
            for _, bet in shuffled_pool.iterrows():
                if len(current_parlay) >= num_legs:
                    break
                    
                game_id = bet.get('game_id')
                # If game_id missing, assume unique? Or skip?
                # If we have game_id, check usage.
                if game_id and game_id != 'N/A':
                    if game_id in used_games:
                        continue
                    used_games.add(game_id)
                
                current_parlay.append(bet)
                
            if len(current_parlay) < 2:
                continue
                
            # Create Parlay Object
            legs = pd.DataFrame(current_parlay)
            
            combined_prob = legs['win_prob'].prod()
            # Odds calc: Convert all to decimal, multiply, convert back to American?
            # Or just store decimal.
            combined_dec_odds = legs['odds'].apply(lambda x: (1 + 100/abs(x)) if x < 0 else (1 + x/100)).prod()
            
            # EV
            parlay_ev = (combined_prob * (combined_dec_odds - 1)) - (1 - combined_prob)
            
            parlays.append({
                'legs': legs['player'].tolist(),
                'targets': legs['target'].tolist(),
                'combined_odds': combined_dec_odds, # Decimal
                'combined_prob': combined_prob,
                'ev': parlay_ev,
                'num_legs': len(legs)
            })
            
        return pd.DataFrame(parlays)

    def generate_round_robins(self, bets_df, num_rr=5):
        # Generate Round Robins (e.g. 3 bets, all 2-leg combos)
        rrs = []
        top_bets = bets_df.head(5) # Top 5 for RR
        
        if len(top_bets) < 3:
            return pd.DataFrame()
            
        # Example: 3x2 (3 selections, parlay size 2)
        from itertools import combinations
        
        combos = list(combinations(top_bets.index, 2))
        
        for idx_list in combos:
            legs = top_bets.loc[list(idx_list)]
            rrs.append({
                'type': '2-leg RR',
                'legs': legs['player'].tolist(),
                'combined_odds': combined_odds,
                'ev': parlay_ev
            })
            
        return pd.DataFrame(rrs)

    def backtest(self, start_season=2020, end_season=2026, confidence_threshold=10, kelly_fraction=0.25, min_ev=0.0):
        # Backtest Strategy with Season-Aware Model Loading
        print(f"Starting Backtest ({start_season}-{end_season}) | Conf: {confidence_threshold} | Kelly: {kelly_fraction} | MinEV: {min_ev}")
        
        # Filter data by season
        mask = (self.processor.df['season'] >= start_season) & (self.processor.df['season'] <= end_season)
        df_backtest = self.processor.df[mask].copy()
        
        # Get unique dates
        dates = df_backtest['date'].unique()
        dates = sorted(dates)
        
        bankroll = 1000
        history = []
        current_season = None
        
        for date in dates:
            # Determine season for this date
            # Ensure date is comparable if needed
            season = df_backtest[df_backtest['date'] == date]['season'].iloc[0]
            
            # Load correct model if season changes
            if season != current_season:
                self.load_season_models(season)
                current_season = season
            
            # Generate Predictions
            preds = self.generate_predictions(date)
            if preds is None or preds.empty:
                continue
                
            bets = self.generate_bets(preds, bankroll=bankroll, 
                                      confidence_threshold=confidence_threshold, 
                                      kelly_fraction=kelly_fraction, 
                                      min_ev=min_ev)
            
            if bets.empty:
                history.append({'date': date, 'bankroll': bankroll, 'pnl': 0, 'roi': 0})
                continue
                
            # Evaluate Bets
            day_pnl = 0
            day_stake = 0
            
            for _, bet in bets.iterrows():
                player = bet['player']
                target = bet['target']
                line = bet['line']
                
                # Get actual
                actual_row = df_backtest[(df_backtest['date'] == date) & (df_backtest['player_name'] == player)]
                if actual_row.empty:
                    continue
                    
                actual = actual_row[target].iloc[0]
                
                # Determine Win/Loss
                won = False
                if bet['prediction'] > line: # We bet OVER
                    if actual > line: won = True
                else: # We bet UNDER
                    if actual < line: won = True
                        
                # Calculate PnL
                stake = bet['stake_amt']
                day_stake += stake
                
                if won:
                    # Profit
                    dec_odds = (1 + bet['odds']/100) if bet['odds'] > 0 else (1 + 100/abs(bet['odds']))
                    profit = stake * (dec_odds - 1)
                    day_pnl += profit
                else:
                    day_pnl -= stake
            
            bankroll += day_pnl
            roi = (day_pnl / day_stake) if day_stake > 0 else 0
            history.append({'date': date, 'bankroll': bankroll, 'pnl': day_pnl, 'roi': roi})
            print(f"Date: {date} | Season: {season} | PnL: ${day_pnl:.2f} | Bankroll: ${bankroll:.2f}", end='\r')
            
        print("\nBacktest Complete.")
        
        # Calculate Metrics
        hist_df = pd.DataFrame(history)
        if hist_df.empty:
            return {'roi': 0, 'sharpe': 0, 'drawdown': 0, 'final_bankroll': 1000}
            
        total_roi = (bankroll - 1000) / 1000
        
        # Sharpe (Daily Returns)
        # We need daily % return relative to bankroll? Or just PnL?
        # Sharpe usually excess return / std dev.
        # Let's use daily PnL / Starting Bankroll as "daily return" approximation?
        # Or better: ln(today_br / yesterday_br)
        hist_df['daily_return'] = hist_df['bankroll'].pct_change().fillna(0)
        sharpe = hist_df['daily_return'].mean() / hist_df['daily_return'].std() * np.sqrt(252) if hist_df['daily_return'].std() != 0 else 0
        
        # Max Drawdown
        hist_df['peak'] = hist_df['bankroll'].cummax()
        hist_df['drawdown'] = (hist_df['bankroll'] - hist_df['peak']) / hist_df['peak']
        max_drawdown = hist_df['drawdown'].min()
        
        return {
            'roi': total_roi,
            'sharpe': sharpe,
            'drawdown': max_drawdown,
            'final_bankroll': bankroll,
            'history': hist_df
        }

        return {
            'roi': total_roi,
            'sharpe': sharpe,
            'drawdown': max_drawdown,
            'final_bankroll': bankroll,
            'history': hist_df
        }

    def generate_daily_outputs(self):
        """
        Reads live predictions, generates betting picks, and outputs them to
        a markdown file and a ledger.
        """
        predictions_df = pd.read_csv('predictions/live_ensemble_2025.csv')
        bets_df = self.generate_bets(predictions_df)
        parlays_df = self.generate_parlays(bets_df)

        today = datetime.datetime.now().strftime('%m.%d.%y')
        filename = f"{today} - Riq's Picks.md"

        with open(filename, 'w') as f:
            f.write(f"# {today} - Riq's Picks\n\n")
            f.write("## System Version: V4 Ensemble\n\n")
            f.write("*Disclaimer: These picks are model-driven and not guaranteed wins. Bet responsibly.*\n\n")

            f.write("### Straight Bets (Top Plays)\n\n")
            f.write("| Player | Prop Type | Line | Odds | Model Projection | Edge | Confidence Tier | Suggested Stake |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")
            for _, row in bets_df.iterrows():
                f.write(f"| {row['player']} | {row['target']} | {row['line']} | {row['odds']} | {row['prediction']:.2f} | {row['ev']:.2f} | {row['confidence']:.2f} | {row['stake_amt']:.2f} |\n")

            if not parlays_df.empty:
                f.write("\n### Parlays\n\n")
                for _, row in parlays_df.iterrows():
                    f.write(f"**{row['num_legs']}-Leg Parlay ({row['combined_odds']:.2f})**\n")
                    for i in range(row['num_legs']):
                        f.write(f"- **{row['legs'][i]}** ({row['targets'][i]})\n")
                    f.write("\n*No same-game legs, no correlated stats.*\n\n")

        # Create Ledger
        if not bets_df.empty:
            ledger_df = bets_df.copy()
            ledger_df['run_date'] = datetime.datetime.now().strftime('%Y-%m-%d')
            ledger_df['bet_type'] = 'straight'
            ledger_df['pick_identifier'] = ledger_df['run_date'] + '-' + ledger_df['player'] + '-' + ledger_df['target']
            ledger_df = ledger_df[['run_date', 'player', 'target', 'line', 'odds', 'prediction', 'ev', 'confidence', 'bet_type', 'stake_amt', 'pick_identifier']]
            ledger_df.rename(columns={'target': 'prop_type', 'stake_amt': 'stake_sizing', 'prediction': 'model_prediction', 'ev': 'edge', 'confidence': 'confidence_tier'}, inplace=True)

            ledger_path = 'data/ledger.csv'
            if os.path.exists(ledger_path):
                ledger_df.to_csv(ledger_path, mode='a', header=False, index=False)
            else:
                ledger_df.to_csv(ledger_path, index=False)
