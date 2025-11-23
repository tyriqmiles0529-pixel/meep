#!/usr/bin/env python
"""
Enhanced Context-Aware Meta-Learner for NBA Props

Implements advanced features:
- Advanced usage and efficiency features
- Opponent-adjusted context features  
- Temporal smoothness with rolling averages
- Uncertainty estimation for volatile stats
- Era-specific context handling

Usage:
    python train_meta_learner_enhanced.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from typing import Dict, List, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuration
TRAINING_SEASON = '2024-2025'  # Last complete season
DATA_FILE = 'PlayerStatistics.csv'  # Kaggle CSV
OUTPUT_FILE = 'model_cache/meta_learner_enhanced_2025_2026.pkl'
MIN_SAMPLES_PER_PROP = 100  # Minimum games needed to train

class EnhancedMetaLearner:
    """
    Enhanced meta-learner with advanced features and uncertainty estimation
    """
    
    def __init__(self, n_windows: int = 27):
        self.n_windows = n_windows
        self.meta_models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.uncertainty_models = {}  # For confidence estimation
        
    def generate_advanced_features(self, games_df: pd.DataFrame, window_predictions: np.ndarray) -> pd.DataFrame:
        """
        Generate advanced features for meta-learner combining window predictions with context
        """
        features = []
        
        for i, (_, game) in enumerate(games_df.iterrows()):
            if i >= len(window_predictions):
                break
                
            feature_row = {}
            preds = window_predictions[i]
            
            # 1. Window prediction statistics
            feature_row.update({
                'pred_mean': np.mean(preds),
                'pred_median': np.median(preds),
                'pred_std': np.std(preds),
                'pred_min': np.min(preds),
                'pred_max': np.max(preds),
                'pred_range': np.max(preds) - np.min(preds),
                'pred_q25': np.percentile(preds, 25),
                'pred_q75': np.percentile(preds, 75),
                'pred_iqr': np.percentile(preds, 75) - np.percentile(preds, 25)
            })
            
            # 2. Recent vs Historical window split
            recent_windows = preds[-10:]  # Last 10 windows (most recent)
            historical_windows = preds[:-10]  # Earlier windows
            
            feature_row.update({
                'recent_mean': np.mean(recent_windows),
                'historical_mean': np.mean(historical_windows),
                'recent_vs_historical': np.mean(recent_windows) - np.mean(historical_windows),
                'recent_trend': np.polyfit(range(len(recent_windows)), recent_windows, 1)[0] if len(recent_windows) > 1 else 0
            })
            
            # 3. Advanced usage and efficiency features
            if 'USG_PCT' in game:
                feature_row['usage_rate'] = game['USG_PCT']
            else:
                # Calculate usage proxy from shot attempts
                fga = game.get('FGA', 0)
                fta = game.get('FTA', 0) 
                minutes = game.get('MIN', 1)
                feature_row['usage_rate'] = ((fga + fta * 0.44) / minutes) if minutes > 0 else 0
            
            # 4. Efficiency features
            if 'TS_PCT' in game:
                feature_row['true_shooting'] = game['TS_PCT']
            else:
                pts = game.get('PTS', 0)
                fga = game.get('FGA', 1)
                fta = game.get('FTA', 0)
                ts_pct = pts / (2 * (fga + 0.44 * fta)) if (fga + 0.44 * fta) > 0 else 0
                feature_row['true_shooting'] = ts_pct
            
            # 5. Rate stats (per 36 minutes)
            minutes = game.get('MIN', 1)
            if minutes > 0:
                feature_row.update({
                    'points_per_36': (game.get('PTS', 0) / minutes) * 36,
                    'assists_per_36': (game.get('AST', 0) / minutes) * 36,
                    'rebounds_per_36': (game.get('REB', 0) / minutes) * 36,
                    'threes_per_36': (game.get('FG3M', 0) / minutes) * 36
                })
            else:
                feature_row.update({
                    'points_per_36': 0, 'assists_per_36': 0, 
                    'rebounds_per_36': 0, 'threes_per_36': 0
                })
            
            # 6. Opponent-adjusted context
            if 'OPP_PACE' in game:
                feature_row['opp_pace'] = game['OPP_PACE']
            else:
                feature_row['opp_pace'] = 100.0  # League average default
                
            if 'OPP_DEF_RATING' in game:
                feature_row['opp_def_rating'] = game['OPP_DEF_RATING']
            else:
                feature_row['opp_def_rating'] = 110.0  # League average default
            
            # 7. Temporal smoothness features
            # Player recent form (last 5 games)
            player_id = game.get('playerId')
            if player_id is not None:
                player_games = games_df[games_df['playerId'] == player_id].sort_values('gameDate')
                current_idx = player_games[player_games.index == game.name].index[0] if game.name in player_games.index else -1
                
                if current_idx > 0:
                    # Get last 5 games before current
                    start_idx = max(0, current_idx - 5)
                    recent_games = player_games.iloc[start_idx:current_idx]
                    
                    if len(recent_games) > 0:
                        feature_row.update({
                            'recent_points_trend': np.polyfit(range(len(recent_games)), recent_games['PTS'], 1)[0] if len(recent_games) > 1 else 0,
                            'recent_assists_trend': np.polyfit(range(len(recent_games)), recent_games['AST'], 1)[0] if len(recent_games) > 1 else 0,
                            'recent_minutes_avg': recent_games['MIN'].mean(),
                            'recent_usage_avg': recent_games.get('USG_PCT', 0.2).mean()
                        })
                    else:
                        feature_row.update({
                            'recent_points_trend': 0, 'recent_assists_trend': 0,
                            'recent_minutes_avg': game.get('MIN', 30),
                            'recent_usage_avg': game.get('USG_PCT', 0.2)
                        })
                else:
                    feature_row.update({
                        'recent_points_trend': 0, 'recent_assists_trend': 0,
                        'recent_minutes_avg': game.get('MIN', 30),
                        'recent_usage_avg': game.get('USG_PCT', 0.2)
                    })
            else:
                feature_row.update({
                    'recent_points_trend': 0, 'recent_assists_trend': 0,
                    'recent_minutes_avg': game.get('MIN', 30),
                    'recent_usage_avg': game.get('USG_PCT', 0.2)
                })
            
            # 8. Era context for window predictions
            # Weight recent windows more heavily for modern NBA patterns
            era_weights = np.linspace(0.5, 1.5, len(preds))  # Recent windows get higher weight
            feature_row['era_weighted_pred'] = np.average(preds, weights=era_weights)
            
            # 9. Game context features
            feature_row.update({
                'is_home': 1 if game.get('LOCATION', '') == 'HOME' else 0,
                'days_rest': game.get('DAYS_REST', 2),
                'back_to_back': 1 if game.get('DAYS_REST', 2) == 0 else 0
            })
            
            # 10. Player role indicators
            position = game.get('POSITION', 'SF')
            feature_row.update({
                'is_guard': 1 if position in ['PG', 'SG'] else 0,
                'is_forward': 1 if position in ['SF', 'PF'] else 0,
                'is_center': 1 if position == 'C' else 0
            })
            
            features.append(feature_row)
        
        return pd.DataFrame(features)
    
    def fit_oof(self, window_predictions: np.ndarray, y_true: np.ndarray, 
                games_df: pd.DataFrame, prop_name: str) -> Dict:
        """
        Fit meta-learner with out-of-fold predictions and uncertainty estimation
        """
        print(f"  Training enhanced meta-learner for {prop_name}...")
        
        # Generate advanced features
        X = self.generate_advanced_features(games_df, window_predictions)
        y = y_true[:len(X)]  # Ensure alignment
        
        print(f"    Generated {len(X.columns)} advanced features")
        
        # Remove any remaining NaN values
        X = X.fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 5-fold cross-validation for robust training
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        oof_predictions = np.zeros(len(y))
        oof_uncertainties = np.zeros(len(y))
        
        # Train ensemble models
        models = []
        uncertainty_models = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
            print(f"    Fold {fold+1}/5...")
            
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Main prediction model
            model = Ridge(alpha=1.0, random_state=42)
            model.fit(X_train, y_train)
            models.append(model)
            
            # Uncertainty model (predicts error magnitude)
            train_errors = np.abs(y_train - model.predict(X_train))
            uncertainty_model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
            uncertainty_model.fit(X_train, train_errors)
            uncertainty_models.append(uncertainty_model)
            
            # OOF predictions
            oof_predictions[val_idx] = model.predict(X_val)
            oof_uncertainties[val_idx] = uncertainty_model.predict(X_val)
        
        # Calculate metrics
        baseline_mae = mean_absolute_error(y, np.mean(window_predictions[:len(y)], axis=1))
        meta_mae = mean_absolute_error(y, oof_predictions)
        baseline_rmse = np.sqrt(mean_squared_error(y, np.mean(window_predictions[:len(y)], axis=1)))
        meta_rmse = np.sqrt(mean_squared_error(y, oof_predictions))
        
        improvement_mae = ((baseline_mae - meta_mae) / baseline_mae) * 100
        improvement_rmse = ((baseline_rmse - meta_rmse) / baseline_rmse) * 100
        
        # Train final models on all data
        final_model = Ridge(alpha=1.0, random_state=42)
        final_model.fit(X_scaled, y)
        
        final_errors = np.abs(y - final_model.predict(X_scaled))
        final_uncertainty = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42)
        final_uncertainty.fit(X_scaled, final_errors)
        
        # Store models
        self.meta_models[prop_name] = final_model
        self.scalers[prop_name] = scaler
        self.uncertainty_models[prop_name] = final_uncertainty
        self.feature_importance[prop_name] = dict(zip(X.columns, np.abs(final_model.coef_)))
        
        print(f"    ✓ Enhanced meta-learner trained")
        print(f"    MAE: {baseline_mae:.3f} → {meta_mae:.3f} ({improvement_mae:+.1f}%)")
        print(f"    RMSE: {baseline_rmse:.3f} → {meta_rmse:.3f} ({improvement_rmse:+.1f}%)")
        print(f"    Mean uncertainty: {np.mean(final_errors):.3f}")
        
        return {
            'mae': meta_mae,
            'rmse': meta_rmse,
            'improvement_mae_pct': improvement_mae,
            'improvement_rmse_pct': improvement_rmse,
            'uncertainty_mean': np.mean(final_errors)
        }
    
    def predict_with_uncertainty(self, window_predictions: np.ndarray, 
                                game_context: pd.Series, prop_name: str) -> Tuple[float, float]:
        """
        Predict with uncertainty estimation
        
        Returns:
            Tuple of (prediction, uncertainty_score)
        """
        if prop_name not in self.meta_models:
            return np.mean(window_predictions), 1.0
        
        # Create single-row feature DataFrame
        games_df = pd.DataFrame([game_context])
        X = self.generate_advanced_features(games_df, window_predictions.reshape(1, -1))
        X = X.fillna(0)
        
        # Scale and predict
        X_scaled = self.scalers[prop_name].transform(X)
        prediction = self.meta_models[prop_name].predict(X_scaled)[0]
        uncertainty = self.uncertainty_models[prop_name].predict(X_scaled)[0]
        
        return prediction, uncertainty
    
    def save(self, path: str):
        """Save enhanced meta-learner"""
        save_data = {
            'meta_models': self.meta_models,
            'scalers': self.scalers,
            'uncertainty_models': self.uncertainty_models,
            'feature_importance': self.feature_importance,
            'n_windows': self.n_windows
        }
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
    
    def load(self, path: str):
        """Load enhanced meta-learner"""
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.meta_models = save_data['meta_models']
        self.scalers = save_data['scalers']
        self.uncertainty_models = save_data['uncertainty_models']
        self.feature_importance = save_data['feature_importance']
        self.n_windows = save_data['n_windows']


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

    # NBA season logic
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


def generate_features_for_prediction(df_games):
    """Generate features from PlayerStatistics.csv that match training features"""
    # Sort by player and date
    df = df_games.sort_values(['playerId', 'gameDate']).copy()

    # Basic stats (already in CSV)
    features = pd.DataFrame(index=df.index)

    # Direct stats
    for col in ['points', 'assists', 'reboundsTotal', 'threePointersMade',
                'numMinutes', 'fieldGoalsAttempted', 'fieldGoalsMade',
                'freeThrowsAttempted', 'freeThrowsMade', 'turnovers',
                'steals', 'blocks', 'reboundsDefensive', 'reboundsOffensive']:
        if col in df.columns:
            features[col] = df[col].fillna(0)

    # Rolling averages (L3, L5, L7, L10)
    for window in [3, 5, 7, 10]:
        for stat in ['points', 'assists', 'reboundsTotal', 'threePointersMade', 'numMinutes']:
            if stat in df.columns:
                features[f'{stat}_L{window}_avg'] = df.groupby('playerId')[stat].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                ).fillna(0)

    # Shooting percentages
    if 'fieldGoalsMade' in df.columns and 'fieldGoalsAttempted' in df.columns:
        features['fg_pct'] = (df['fieldGoalsMade'] / df['fieldGoalsAttempted'].replace(0, 1)).fillna(0)

    if 'freeThrowsMade' in df.columns and 'freeThrowsAttempted' in df.columns:
        features['ft_pct'] = (df['freeThrowsMade'] / df['freeThrowsAttempted'].replace(0, 1)).fillna(0)

    # Usage proxy
    if 'fieldGoalsAttempted' in df.columns and 'freeThrowsAttempted' in df.columns:
        features['usage'] = (df['fieldGoalsAttempted'].fillna(0) +
                           df['freeThrowsAttempted'].fillna(0) * 0.44)

    # Per-minute stats
    if 'numMinutes' in df.columns:
        minutes_safe = df['numMinutes'].replace(0, 1)
        for stat in ['points', 'assists', 'reboundsTotal']:
            if stat in df.columns:
                features[f'{stat}_per_min'] = (df[stat] / minutes_safe).fillna(0)

    # Home/away
    if 'home' in df.columns:
        features['home'] = df['home'].fillna(0).astype(int)

    # Days rest
    if 'gameDate' in df.columns:
        df['gameDate'] = pd.to_datetime(df['gameDate'])
        features['days_rest'] = df.groupby('playerId')['gameDate'].diff().dt.days.fillna(2).clip(0, 7)

    # Fill any remaining NaN
    features = features.fillna(0)

    return features


def load_training_data(season: str = TRAINING_SEASON) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load PlayerStatistics.csv, filter to target season, and generate features"""
    print(f"\n{'='*70}")
    print(f"LOADING TRAINING DATA: {season}")
    print(f"{'='*70}")

    # Load the Kaggle CSV
    df_raw = load_player_statistics_csv(DATA_FILE)

    # Filter to the target season
    df_raw = filter_to_season(df_raw, season)

    # Generate features for window models
    print("\n[*] Generating features for window models...")
    df_features = generate_features_for_prediction(df_raw)
    print(f"Generated {len(df_features.columns)} features")

    return df_raw, df_features


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
        # Get actual outcome
        actual = game.get(actual_col)
        if pd.isna(actual) or actual < 0:
            continue

        # Get predictions from each window
        preds_for_game = []
        for window_name, models in window_models.items():
            try:
                # Create features
                X_game = create_features_for_game(game)

                # Predict (using ensemble_predictor)
                from ensemble_predictor import predict_with_window
                pred = predict_with_window(models, X_game, prop)
                if pred is not None:
                    preds_for_game.append(pred)
                else:
                    preds_for_game.append(0.0)
            except Exception as e:
                preds_for_game.append(0.0)

        # Need predictions from most windows
        if len(preds_for_game) < 20:
            continue

        # Pad to expected number if some failed
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


def train_enhanced_meta_learner():
    """Main training function for enhanced meta-learner"""
    print(f"\n{'='*70}")
    print(f"ENHANCED META-LEARNER TRAINING")
    print(f"{'='*70}")
    print(f"  Training Season: {TRAINING_SEASON}")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"  Features: Advanced usage, opponent context, temporal smoothness, uncertainty")
    print(f"{'='*70}\n")

    # 1. Load window models (will use CPU models when training finishes)
    print("Loading window models...")
    try:
        from ensemble_predictor import load_all_window_models
        window_models = load_all_window_models('model_cache')
        print(f"  ✓ Loaded {len(window_models)} windows from model_cache")
    except:
        print("  ! Window models not found in model_cache, will try nba-models-cpu volume")
        # For now, create dummy data structure - update when CPU models are ready
        window_models = {}
        print("  ⚠ Will need to update with actual CPU models after training finishes")
        return

    # 2. Load training data
    games_df, _ = load_training_data(TRAINING_SEASON)

    # 3. Initialize enhanced meta-learner
    meta_learner = EnhancedMetaLearner(n_windows=len(window_models))

    # 4. Train for each prop type
    props_to_train = ['points', 'rebounds', 'assists', 'threes']

    for prop in props_to_train:
        print(f"\n{'='*70}")
        print(f"PROP: {prop.upper()}")
        print(f"{'='*70}")

        # Collect window predictions
        data = collect_window_predictions(games_df, window_models, prop)

        if data is None:
            print(f"  ⚠ Skipping {prop} - insufficient data")
            continue

        # Train enhanced meta-learner with OOF
        metrics = meta_learner.fit_oof(
            window_predictions=data['window_predictions'],
            y_true=data['actuals'],
            games_df=games_df.iloc[:len(data['actuals'])],
            prop_name=prop
        )

        print(f"\n  ✅ Enhanced meta-learner trained for {prop}")
        print(f"     RMSE improvement: {metrics['improvement_rmse_pct']:+.1f}%")
        print(f"     Mean uncertainty: {metrics['uncertainty_mean']:.3f}")

    # 5. Save enhanced meta-learner
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(exist_ok=True)

    meta_learner.save(str(output_path))

    print(f"\n{'='*70}")
    print(f"✅ ENHANCED TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Saved to: {output_path}")
    print(f"  Props trained: {len(meta_learner.meta_models)}")
    print(f"  Advanced features: Usage rates, opponent context, temporal trends, uncertainty")
    print(f"\nNext steps:")
    print(f"  1. Upload to Modal: modal volume put nba-models {output_path}")
    print(f"  2. Run backtesting with enhanced ensemble")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    try:
        train_enhanced_meta_learner()
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
