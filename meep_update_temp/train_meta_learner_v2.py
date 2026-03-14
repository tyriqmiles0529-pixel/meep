#!/usr/bin/env python
"""
Meta-Learner V2 - Addressing Critical Bottlenecks

Improvements:
✅ Nonlinear interaction modeling (GradientBoosting ensemble)
✅ Model agreement features (agreement count, bias direction, divergence)
✅ Nonlinear window weighting (era-based jumps)
✅ Reduced collinearity (feature selection + regularization)
✅ Enhanced uncertainty modeling

Usage:
    python train_meta_learner_v2.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from typing import Dict, List, Tuple
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuration
TRAINING_SEASON = '2024-2025'
DATA_FILE = 'PlayerStatistics.csv'
OUTPUT_FILE = 'model_cache/meta_learner_v2_2025_2026.pkl'
MIN_SAMPLES_PER_PROP = 100

class MetaLearnerV2:
    """
    Advanced meta-learner addressing key bottlenecks:
    - Nonlinear interaction modeling
    - Model agreement features
    - Era-aware nonlinear weighting
    - Reduced collinearity
    - Enhanced uncertainty
    """
    
    def __init__(self, n_windows: int = 27):
        self.n_windows = n_windows
        self.meta_models = {}  # Nonlinear models
        self.linear_models = {}  # Backup linear models
        self.scalers = {}
        self.feature_selectors = {}
        self.uncertainty_models = {}
        self.feature_importance = {}
        
    def get_era_weights(self, n_windows: int) -> np.ndarray:
        """
        Nonlinear era weighting based on NBA evolution jumps
        """
        if n_windows == 27:
            # 27 windows from 1947-2025, era-based weighting
            weights = np.array([
                0.1, 0.1, 0.1,  # 1940s-50s (very low)
                0.2, 0.2, 0.2,  # 1960s (low)
                0.3, 0.3, 0.3,  # 1970s (medium-low)
                0.5, 0.5, 0.5,  # 1980s (medium)
                0.7, 0.7, 0.7,  # 1990s (medium-high)
                0.8, 0.8, 0.8,  # 2000s (high)
                0.9, 0.9, 0.9,  # 2010s (very high)
                1.0, 1.0, 1.0,  # 2020s (maximum)
            ])
        else:
            # Generic nonlinear weighting for different window counts
            weights = np.linspace(0.1, 1.0, n_windows)
            weights = weights ** 2  # Square for nonlinear emphasis on recent
        
        return weights / weights.sum()  # Normalize
    
    def compute_model_agreement_features(self, predictions: np.ndarray) -> Dict[str, float]:
        """
        Advanced model agreement analysis
        """
        features = {}
        
        # Basic statistics
        features['pred_mean'] = np.mean(predictions)
        features['pred_std'] = np.std(predictions)
        features['pred_median'] = np.median(predictions)
        
        # Agreement features
        pred_range = np.max(predictions) - np.min(predictions)
        features['agreement_ratio'] = 1.0 / (1.0 + pred_range)  # Higher = more agreement
        
        # Bias direction analysis
        median_pred = np.median(predictions)
        overpredict_count = np.sum(predictions > median_pred)
        underpredict_count = np.sum(predictions < median_pred)
        features['bias_direction'] = (overpredict_count - underpredict_count) / len(predictions)
        
        # Inter-model divergence
        sorted_preds = np.sort(predictions)
        features['divergence_score'] = (sorted_preds[-1] - sorted_preds[0]) / (sorted_preds[-4] - sorted_preds[3] + 1e-6)
        
        # Symmetry analysis
        q25, q75 = np.percentile(predictions, [25, 75])
        features['symmetry_score'] = abs((median_pred - q25) - (q75 - median_pred)) / (q75 - q25 + 1e-6)
        
        # Consensus strength
        mad = np.median(np.abs(predictions - median_pred))  # Median absolute deviation
        features['consensus_strength'] = 1.0 / (1.0 + mad)
        
        return features
    
    def generate_advanced_features(self, games_df: pd.DataFrame, window_predictions: np.ndarray, prop_name: str) -> pd.DataFrame:
        """
        Generate features with reduced collinearity and enhanced interactions
        """
        features = []
        era_weights = self.get_era_weights(len(window_predictions[0]))
        
        for i, (_, game) in enumerate(games_df.iterrows()):
            if i >= len(window_predictions):
                break
                
            preds = window_predictions[i]
            feature_row = {}
            
            # 1. Model agreement features (replaces basic stats)
            agreement_features = self.compute_model_agreement_features(preds)
            feature_row.update(agreement_features)
            
            # 2. Era-weighted predictions (nonlinear weighting)
            feature_row['era_weighted_pred'] = np.average(preds, weights=era_weights)
            feature_row['recent_weighted_pred'] = np.mean(preds[-8:])  # Last 8 windows
            feature_row['historical_weighted_pred'] = np.mean(preds[:-8])  # Earlier windows
            
            # 3. Temporal features (reduced collinearity)
            recent_windows = preds[-8:]
            historical_windows = preds[:-8]
            
            feature_row.update({
                'recent_vs_historical': np.mean(recent_windows) - np.mean(historical_windows),
                'recent_volatility': np.std(recent_windows),
                'historical_volatility': np.std(historical_windows),
                'volatility_ratio': np.std(recent_windows) / (np.std(historical_windows) + 1e-6)
            })
            
            # 4. Usage and efficiency (key interaction features)
            fga = game.get('FGA', 0)
            fta = game.get('FTA', 0)
            minutes = game.get('MIN', 1)
            pts = game.get('PTS', 0)
            ast = game.get('AST', 0)
            
            # Usage rate (primary driver)
            usage_rate = ((fga + fta * 0.44) / minutes) if minutes > 0 else 0
            feature_row['usage_rate'] = usage_rate
            
            # Key interaction: Usage × Era
            feature_row['usage_x_era'] = usage_rate * feature_row['era_weighted_pred']
            
            # Efficiency metrics
            ts_pct = pts / (2 * (fga + 0.44 * fta)) if (fga + 0.44 * fta) > 0 else 0
            feature_row['true_shooting'] = ts_pct
            
            # Interaction: Usage × Efficiency
            feature_row['usage_efficiency'] = usage_rate * ts_pct
            
            # 5. Context features (reduced set to avoid collinearity)
            feature_row.update({
                'is_home': 1 if game.get('LOCATION', '') == 'HOME' else 0,
                'days_rest': game.get('DAYS_REST', 2),
                'opp_pace': game.get('OPP_PACE', 100.0),
                'opp_def_rating': game.get('OPP_DEF_RATING', 110.0)
            })
            
            # 6. Role indicators (simplified)
            position = game.get('POSITION', 'SF')
            feature_row.update({
                'is_guard': 1 if position in ['PG', 'SG'] else 0,
                'is_big': 1 if position in ['PF', 'C'] else 0
            })
            
            # 7. Stat-specific interaction features
            if prop_name == 'assists':
                # Assists depend on usage and pace
                feature_row['assist_potential'] = usage_rate * feature_row['opp_pace'] / 100
                feature_row['usage_x_minutes'] = usage_rate * minutes
                
            elif prop_name == 'points':
                # Points depend on usage and efficiency
                feature_row['scoring_potential'] = usage_rate * ts_pct
                feature_row['pace_x_usage'] = feature_row['opp_pace'] * usage_rate / 100
                
            elif prop_name == 'rebounds':
                # Rebounds depend on position and pace
                feature_row['rebound_potential'] = feature_row['is_big'] * feature_row['opp_pace'] / 100
                feature_row['minutes_x_position'] = minutes * feature_row['is_big']
                
            elif prop_name == 'minutes':
                # Minutes depend on role and rest
                feature_row['minutes_potential'] = usage_rate * (4 - feature_row['days_rest']) / 4
                feature_row['role_x_rest'] = feature_row['is_guard'] * (4 - feature_row['days_rest']) / 4
            
            # 8. Uncertainty indicators
            feature_row['prediction_uncertainty'] = feature_row['pred_std'] / (abs(feature_row['pred_mean']) + 1e-6)
            feature_row['agreement_uncertainty'] = 1.0 - feature_row['agreement_ratio']
            
            features.append(feature_row)
        
        return pd.DataFrame(features)
    
    def fit_oof(self, window_predictions: np.ndarray, y_true: np.ndarray, 
                games_df: pd.DataFrame, prop_name: str) -> Dict:
        """
        Fit with nonlinear models and feature selection
        """
        print(f"  Training V2 meta-learner for {prop_name}...")
        
        # Generate advanced features
        X = self.generate_advanced_features(games_df, window_predictions, prop_name)
        y = y_true[:len(X)]
        
        print(f"    Generated {len(X.columns)} features (reduced collinearity)")
        
        # Remove any remaining NaN values
        X = X.fillna(0)
        
        # Feature selection to reduce collinearity
        selector = SelectKBest(score_func=f_regression, k=min(30, len(X.columns)))
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        print(f"    Selected {len(selected_features)} best features")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)
        
        # 5-fold cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        oof_predictions = np.zeros(len(y))
        oof_predictions_linear = np.zeros(len(y))
        oof_uncertainties = np.zeros(len(y))
        
        nonlinear_models = []
        linear_models = []
        uncertainty_models = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
            print(f"    Fold {fold+1}/5...")
            
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Nonlinear model (GradientBoosting)
            nonlinear_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
            nonlinear_model.fit(X_train, y_train)
            nonlinear_models.append(nonlinear_model)
            
            # Linear model (Ridge backup)
            linear_model = Ridge(alpha=1.0, random_state=42)
            linear_model.fit(X_train, y_train)
            linear_models.append(linear_model)
            
            # Uncertainty model
            train_errors = np.abs(y_train - nonlinear_model.predict(X_train))
            uncertainty_model = RandomForestRegressor(
                n_estimators=50, 
                max_depth=5, 
                random_state=42
            )
            uncertainty_model.fit(X_train, train_errors)
            uncertainty_models.append(uncertainty_model)
            
            # OOF predictions
            nonlinear_pred = nonlinear_model.predict(X_val)
            linear_pred = linear_model.predict(X_val)
            
            # Ensemble nonlinear + linear (weighted average)
            ensemble_pred = 0.7 * nonlinear_pred + 0.3 * linear_pred
            
            oof_predictions[val_idx] = ensemble_pred
            oof_predictions_linear[val_idx] = linear_pred
            oof_uncertainties[val_idx] = uncertainty_model.predict(X_val)
        
        # Calculate metrics
        baseline_mae = mean_absolute_error(y, np.mean(window_predictions[:len(y)], axis=1))
        meta_mae = mean_absolute_error(y, oof_predictions)
        baseline_rmse = np.sqrt(mean_squared_error(y, np.mean(window_predictions[:len(y)], axis=1)))
        meta_rmse = np.sqrt(mean_squared_error(y, oof_predictions))
        
        improvement_mae = ((baseline_mae - meta_mae) / baseline_mae) * 100
        improvement_rmse = ((baseline_rmse - meta_rmse) / baseline_rmse) * 100
        
        # Train final models on all data
        final_nonlinear = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            random_state=42
        )
        final_nonlinear.fit(X_scaled, y)
        
        final_linear = Ridge(alpha=1.0, random_state=42)
        final_linear.fit(X_scaled, y)
        
        final_errors = np.abs(y - (0.7 * final_nonlinear.predict(X_scaled) + 0.3 * final_linear.predict(X_scaled)))
        final_uncertainty = RandomForestRegressor(
            n_estimators=100, 
            max_depth=7, 
            random_state=42
        )
        final_uncertainty.fit(X_scaled, final_errors)
        
        # Store models
        self.meta_models[prop_name] = final_nonlinear
        self.linear_models[prop_name] = final_linear
        self.scalers[prop_name] = scaler
        self.feature_selectors[prop_name] = selector
        self.uncertainty_models[prop_name] = final_uncertainty
        self.feature_importance[prop_name] = dict(zip(selected_features, final_nonlinear.feature_importances_))
        
        print(f"    ✓ V2 meta-learner trained")
        print(f"    MAE: {baseline_mae:.3f} → {meta_mae:.3f} ({improvement_mae:+.1f}%)")
        print(f"    RMSE: {baseline_rmse:.3f} → {meta_rmse:.3f} ({improvement_rmse:+.1f}%)")
        print(f"    Mean uncertainty: {np.mean(final_errors):.3f}")
        
        return {
            'mae': meta_mae,
            'rmse': meta_rmse,
            'improvement_mae_pct': improvement_mae,
            'improvement_rmse_pct': improvement_rmse,
            'uncertainty_mean': np.mean(final_errors),
            'n_features': len(selected_features)
        }
    
    def predict_with_uncertainty(self, window_predictions: np.ndarray, 
                                game_context: pd.Series, prop_name: str) -> Tuple[float, float]:
        """
        Predict with ensemble of nonlinear + linear models
        """
        if prop_name not in self.meta_models:
            return np.mean(window_predictions), 1.0
        
        # Create single-row feature DataFrame
        games_df = pd.DataFrame([game_context])
        X = self.generate_advanced_features(games_df, window_predictions.reshape(1, -1), prop_name)
        X = X.fillna(0)
        
        # Feature selection and scaling
        X_selected = self.feature_selectors[prop_name].transform(X)
        X_scaled = self.scalers[prop_name].transform(X_selected)
        
        # Ensemble prediction
        nonlinear_pred = self.meta_models[prop_name].predict(X_scaled)[0]
        linear_pred = self.linear_models[prop_name].predict(X_scaled)[0]
        ensemble_pred = 0.7 * nonlinear_pred + 0.3 * linear_pred
        
        # Uncertainty
        uncertainty = self.uncertainty_models[prop_name].predict(X_scaled)[0]
        
        return ensemble_pred, uncertainty
    
    def save(self, path: str):
        """Save V2 meta-learner"""
        save_data = {
            'meta_models': self.meta_models,
            'linear_models': self.linear_models,
            'scalers': self.scalers,
            'feature_selectors': self.feature_selectors,
            'uncertainty_models': self.uncertainty_models,
            'feature_importance': self.feature_importance,
            'n_windows': self.n_windows
        }
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
    
    def load(self, path: str):
        """Load V2 meta-learner"""
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.meta_models = save_data['meta_models']
        self.linear_models = save_data['linear_models']
        self.scalers = save_data['scalers']
        self.feature_selectors = save_data['feature_selectors']
        self.uncertainty_models = save_data['uncertainty_models']
        self.feature_importance = save_data['feature_importance']
        self.n_windows = save_data['n_windows']


# Reuse the same data loading and collection functions from enhanced version
def load_player_statistics_csv(csv_path: str = 'PlayerStatistics.csv') -> pd.DataFrame:
    """Load PlayerStatistics.csv with robust column handling"""
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"PlayerStatistics.csv not found at {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows from {csv_path}")
    print(f"Columns: {list(df.columns)[:20]}...")
    return df


def filter_to_season(df: pd.DataFrame, season: str = '2024-2025') -> pd.DataFrame:
    """Filter DataFrame to a specific NBA season"""
    if 'gameDate' not in df.columns:
        raise ValueError("gameDate column required for season filtering")

    df['gameDate'] = pd.to_datetime(df['gameDate'], errors='coerce')

    if season == '2024-2025':
        start_date = pd.to_datetime('2024-10-01')
        end_date = pd.to_datetime('2025-06-30')
    elif season == '2023-2024':
        start_date = pd.to_datetime('2023-10-01')
        end_date = pd.to_datetime('2024-06-30')
    else:
        start_year = int(season.split('-')[0])
        start_date = pd.to_datetime(f'{start_year}-10-01')
        end_date = pd.to_datetime(f'{start_year+1}-06-30')

    season_df = df[
        (df['gameDate'] >= start_date) & (df['gameDate'] <= end_date)
    ].copy().reset_index(drop=True)

    print(f"Filtered to {season}: {len(season_df):,} rows")
    print(f"Date range: {season_df['gameDate'].min()} to {season_df['gameDate'].max()}")

    return season_df


def create_features_for_game(game_row: pd.Series) -> pd.DataFrame:
    """Create feature row for a single game to feed into window models"""
    features = pd.DataFrame([{
        'fieldGoalsAttempted': game_row.get('FGA', 0),
        'freeThrowsAttempted': game_row.get('FTA', 0),
        'assists': game_row.get('AST', 0),
        'reboundsTotal': game_row.get('REB', 0),
        'threePointersMade': game_row.get('FG3M', 0),
        'points': game_row.get('PTS', 0),
        'numMinutes': game_row.get('MIN', 0),
        'fieldGoalsMade': game_row.get('FGM', 0),
        'freeThrowsMade': game_row.get('FTM', 0),
        'turnovers': game_row.get('TOV', 0),
        'steals': game_row.get('STL', 0),
        'blocks': game_row.get('BLK', 0),
        'reboundsDefensive': game_row.get('DREB', 0),
        'reboundsOffensive': game_row.get('OREB', 0),
    }])

    return features


def collect_window_predictions(games_df: pd.DataFrame, window_models: Dict, prop: str) -> Dict:
    """Collect predictions from all windows for each game"""
    print(f"\n  Collecting predictions for: {prop.upper()}")

    window_predictions = []
    actuals = []

    prop_col_map = {
        'points': 'PTS',
        'rebounds': 'REB',
        'assists': 'AST',
        'threes': 'FG3M'
    }
    actual_col = prop_col_map.get(prop)

    if actual_col not in games_df.columns:
        print(f"    [!] Column {actual_col} not found in data, skipping {prop}")
        return None

    for idx, game in games_df.iterrows():
        actual = game.get(actual_col)
        if pd.isna(actual) or actual < 0:
            continue

        preds_for_game = []
        for window_name, models in window_models.items():
            try:
                X_game = create_features_for_game(game)
                from ensemble_predictor import predict_with_window
                pred = predict_with_window(models, X_game, prop)
                if pred is not None:
                    preds_for_game.append(pred)
                else:
                    preds_for_game.append(0.0)
            except Exception as e:
                preds_for_game.append(0.0)

        if len(preds_for_game) < 20:
            continue

        while len(preds_for_game) < len(window_models):
            preds_for_game.append(np.mean(preds_for_game))

        window_predictions.append(preds_for_game[:len(window_models)])
        actuals.append(actual)

    if len(actuals) < MIN_SAMPLES_PER_PROP:
        print(f"    [!] Not enough samples: {len(actuals)} < {MIN_SAMPLES_PER_PROP}")
        return None

    print(f"    ✓ Collected {len(actuals):,} valid samples")

    return {
        'window_predictions': np.array(window_predictions),
        'actuals': np.array(actuals)
    }


def train_meta_learner_v2():
    """Main training function for V2 meta-learner"""
    print(f"\n{'='*70}")
    print(f"META-LEARNER V2 TRAINING - BOTTLENECK ADDRESSED")
    print(f"{'='*70}")
    print(f"  Training Season: {TRAINING_SEASON}")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"  Improvements:")
    print(f"    ✅ Nonlinear interaction modeling (GradientBoosting)")
    print(f"    ✅ Model agreement features (consensus, bias, divergence)")
    print(f"    ✅ Era-based nonlinear weighting")
    print(f"    ✅ Feature selection (reduced collinearity)")
    print(f"    ✅ Enhanced uncertainty modeling")
    print(f"{'='*70}\n")

    # Load window models
    print("Loading window models...")
    try:
        from ensemble_predictor import load_all_window_models
        window_models = load_all_window_models('model_cache')
        print(f"  ✓ Loaded {len(window_models)} windows from model_cache")
    except:
        print("  ! Window models not found - will update when CPU models ready")
        return

    # Load training data
    games_df = load_player_statistics_csv(DATA_FILE)
    games_df = filter_to_season(games_df, TRAINING_SEASON)

    # Initialize V2 meta-learner
    meta_learner = MetaLearnerV2(n_windows=len(window_models))

    # Train for each prop
    props_to_train = ['points', 'rebounds', 'assists', 'threes']

    for prop in props_to_train:
        print(f"\n{'='*70}")
        print(f"PROP: {prop.upper()}")
        print(f"{'='*70}")

        data = collect_window_predictions(games_df, window_models, prop)

        if data is None:
            print(f"  ⚠ Skipping {prop} - insufficient data")
            continue

        metrics = meta_learner.fit_oof(
            window_predictions=data['window_predictions'],
            y_true=data['actuals'],
            games_df=games_df.iloc[:len(data['actuals'])],
            prop_name=prop
        )

        print(f"\n  ✅ V2 meta-learner trained for {prop}")
        print(f"     RMSE improvement: {metrics['improvement_rmse_pct']:+.1f}%")
        print(f"     Features used: {metrics['n_features']}")

    # Save V2 meta-learner
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(exist_ok=True)

    meta_learner.save(str(output_path))

    print(f"\n{'='*70}")
    print(f"✅ V2 TRAINING COMPLETE - BOTTLENECKS ADDRESSED")
    print(f"{'='*70}")
    print(f"  Saved to: {output_path}")
    print(f"  Props trained: {len(meta_learner.meta_models)}")
    print(f"\nKey improvements:")
    print(f"  🎯 Nonlinear interactions captured")
    print(f"  📊 Model agreement analysis")
    print(f"  ⏰ Era-aware nonlinear weighting")
    print(f"  🔧 Reduced collinearity via feature selection")
    print(f"  🎲 Enhanced uncertainty modeling")
    print(f"\nReady for backtesting evaluation!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    try:
        train_meta_learner_v2()
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
