#!/usr/bin/env python3
"""
Example: Using all ensemble models together.

This script demonstrates:
1. Training Ridge, Elo, Four Factors, and meta-learner models
2. Loading and using them for inference
3. Comparing model performance
4. Adding exhaustion features to player predictions

Run this AFTER training has completed (after train_auto.py has been run with ensemble integration).
"""

import numpy as np
import pandas as pd
import pickle
from typing import Dict, List

# Import ensemble components
from ensemble_models import (
    RidgeScoreDiffModel,
    EloRating,
    FourFactorsModel,
    LogisticEnsembler,
    add_exhaustion_features
)


# ============================================================================
# EXAMPLE 1: Load Trained Ensemble Models
# ============================================================================

def load_ensemble_models() -> Dict:
    """Load all trained ensemble models from disk."""
    print("Loading ensemble models...")
    
    try:
        ridge_model = pickle.load(open('ridge_score_diff_model.pkl', 'rb'))
        print("✓ Ridge model loaded")
    except FileNotFoundError:
        print("✗ Ridge model not found")
        ridge_model = None
    
    try:
        elo_model = pickle.load(open('elo_model.pkl', 'rb'))
        print("✓ Elo model loaded")
    except FileNotFoundError:
        print("✗ Elo model not found")
        elo_model = None
    
    try:
        ff_model = pickle.load(open('four_factors_model.pkl', 'rb'))
        print("✓ Four Factors model loaded")
    except FileNotFoundError:
        print("✗ Four Factors model not found (optional)")
        ff_model = None
    
    try:
        meta_learner = pickle.load(open('meta_learner.pkl', 'rb'))
        print("✓ Meta-learner loaded")
    except FileNotFoundError:
        print("✗ Meta-learner not found")
        meta_learner = None
    
    try:
        lgb_classifier = pickle.load(open('moneyline_model.pkl', 'rb'))
        print("✓ LightGBM classifier loaded")
    except FileNotFoundError:
        print("✗ LightGBM classifier not found")
        lgb_classifier = None
    
    return {
        'ridge': ridge_model,
        'elo': elo_model,
        'four_factors': ff_model,
        'meta_learner': meta_learner,
        'lgb': lgb_classifier
    }


# ============================================================================
# EXAMPLE 2: Game-Level Predictions (Moneyline Probability)
# ============================================================================

def predict_game_ensemble(
    game_row: pd.Series,
    models: Dict,
    game_features: List[str]
) -> Dict[str, float]:
    """
    Generate ensemble predictions for a single game.
    
    Args:
        game_row: Row from games dataframe with features
        models: Dictionary of loaded models
        game_features: List of feature column names
    
    Returns:
        Dictionary with predictions from each model and ensemble
    """
    ridge = models['ridge']
    elo = models['elo']
    ff = models['four_factors']
    meta = models['meta_learner']
    lgb = models['lgb']
    
    # Prepare features
    X_game = game_row[game_features].values.astype('float32').reshape(1, -1)
    
    predictions = {}
    
    # Ridge prediction
    if ridge is not None:
        ridge_margin = ridge.predict(X_game)[0]
        ridge_prob = ridge.predict_proba(X_game)[0]
        predictions['ridge'] = {
            'margin': float(ridge_margin),
            'prob': float(ridge_prob)
        }
    
    # Elo prediction
    if elo is not None:
        elo_prob = elo.expected_win_prob(
            game_row['elo_home'] + 70.0,
            game_row['elo_away']
        )
        predictions['elo'] = {
            'elo_home': float(game_row['elo_home']),
            'elo_away': float(game_row['elo_away']),
            'prob': float(elo_prob)
        }
    
    # Four Factors prediction
    if ff is not None:
        try:
            X_ff = game_row[[
                'home_efg_prior', 'home_tov_pct_prior', 'home_orb_pct_prior', 'home_ftr_prior',
                'away_efg_prior', 'away_tov_pct_prior', 'away_orb_pct_prior', 'away_ftr_prior'
            ]].values.astype('float32').reshape(1, -1)
            ff_margin = ff.predict(X_ff)[0]
            ff_prob = ff.predict_proba(X_ff)[0]
            predictions['four_factors'] = {
                'margin': float(ff_margin),
                'prob': float(ff_prob)
            }
        except:
            pass
    
    # LGB prediction
    if lgb is not None:
        lgb_prob = lgb.predict_proba(X_game)[0, 1]
        predictions['lgb'] = {
            'prob': float(lgb_prob)
        }
    
    # Ensemble prediction (meta-learner)
    if meta is not None and ridge is not None and elo is not None and lgb is not None:
        ridge_prob = predictions['ridge']['prob']
        elo_prob = predictions['elo']['prob']
        ff_prob = predictions.get('four_factors', {}).get('prob', 0.5)
        lgb_prob = predictions['lgb']['prob']
        
        X_meta = np.array([[ridge_prob, elo_prob, ff_prob, lgb_prob]])
        ensemble_prob = meta.predict_proba(X_meta)[0]
        
        weights = meta.get_weights()
        predictions['ensemble'] = {
            'prob': float(ensemble_prob),
            'weights': weights
        }
    
    return predictions


# ============================================================================
# EXAMPLE 3: Player Prop Predictions with Exhaustion Features
# ============================================================================

def add_exhaustion_to_player_stats(
    player_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Add exhaustion features to player statistics dataframe.
    
    Args:
        player_df: Player game log dataframe
    
    Returns:
        DataFrame with exhaustion features added
    """
    print("Adding exhaustion features to player data...")
    
    player_df = add_exhaustion_features(player_df)
    
    # Fill NaN values with reasonable defaults
    player_df['season_fatigue'] = player_df['season_fatigue'].fillna(0.5)
    player_df['heavy_usage'] = player_df['heavy_usage'].fillna(0.0)
    player_df['consecutive_b2b'] = player_df['consecutive_b2b'].fillna(0.0)
    player_df['rest_accumulated'] = player_df['rest_accumulated'].fillna(200.0)
    
    print(f"✓ Added exhaustion features:")
    print(f"  - season_fatigue (0-1, normalized by 82 games)")
    print(f"  - heavy_usage (1 if avg minutes > 30, else 0)")
    print(f"  - consecutive_b2b (running count of B2B games)")
    print(f"  - rest_accumulated (cumulative rest days)")
    
    return player_df


# ============================================================================
# EXAMPLE 4: Compare Model Performance
# ============================================================================

def compare_ensemble_performance(
    games_df: pd.DataFrame,
    models: Dict,
    game_features: List[str]
) -> None:
    """
    Compare predictions from all models on a dataset.
    """
    from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
    
    print("\nComparing ensemble performance...")
    
    ridge = models['ridge']
    elo = models['elo']
    ff = models['four_factors']
    meta = models['meta_learner']
    lgb = models['lgb']
    
    # Target: home team wins
    y_true = (games_df['home_score'] > games_df['away_score']).astype(int).values
    
    results = {}
    
    # Ridge
    if ridge is not None:
        X = games_df[game_features].values.astype('float32')
        ridge_probs = ridge.predict_proba(X)
        results['ridge'] = {
            'logloss': log_loss(y_true, ridge_probs),
            'brier': brier_score_loss(y_true, ridge_probs),
            'auc': roc_auc_score(y_true, ridge_probs)
        }
    
    # Elo
    if elo is not None:
        elo_probs = np.array([
            elo.expected_win_prob(
                games_df.iloc[i]['elo_home'] + 70.0,
                games_df.iloc[i]['elo_away']
            )
            for i in range(len(games_df))
        ])
        results['elo'] = {
            'logloss': log_loss(y_true, elo_probs),
            'brier': brier_score_loss(y_true, elo_probs),
            'auc': roc_auc_score(y_true, elo_probs)
        }
    
    # LGB
    if lgb is not None:
        X = games_df[game_features].values.astype('float32')
        lgb_probs = lgb.predict_proba(X)[:, 1]
        results['lgb'] = {
            'logloss': log_loss(y_true, lgb_probs),
            'brier': brier_score_loss(y_true, lgb_probs),
            'auc': roc_auc_score(y_true, lgb_probs)
        }
    
    # Ensemble
    if meta is not None and ridge is not None and elo is not None and lgb is not None:
        ridge_probs = ridge.predict_proba(X)
        elo_probs = np.array([
            elo.expected_win_prob(
                games_df.iloc[i]['elo_home'] + 70.0,
                games_df.iloc[i]['elo_away']
            )
            for i in range(len(games_df))
        ])
        ff_probs = np.full(len(games_df), 0.5)
        if ff is not None:
            try:
                X_ff = games_df[[
                    'home_efg_prior', 'home_tov_pct_prior', 'home_orb_pct_prior', 'home_ftr_prior',
                    'away_efg_prior', 'away_tov_pct_prior', 'away_orb_pct_prior', 'away_ftr_prior'
                ]].values.astype('float32')
                ff_probs = ff.predict_proba(X_ff)
            except:
                pass
        lgb_probs = lgb.predict_proba(X)[:, 1]
        
        X_meta = np.column_stack([ridge_probs, elo_probs, ff_probs, lgb_probs])
        ensemble_probs = meta.predict_proba(X_meta)
        
        results['ensemble'] = {
            'logloss': log_loss(y_true, ensemble_probs),
            'brier': brier_score_loss(y_true, ensemble_probs),
            'auc': roc_auc_score(y_true, ensemble_probs)
        }
    
    # Print results
    print("\n" + "─" * 70)
    print(f"{'Model':<15} | {'Logloss':<10} | {'Brier':<10} | {'AUC':<8}")
    print("─" * 70)
    for name, metrics in sorted(results.items()):
        print(f"{name:<15} | {metrics['logloss']:<10.4f} | {metrics['brier']:<10.4f} | {metrics['auc']:<8.4f}")
    print("─" * 70)


# ============================================================================
# EXAMPLE 5: Get Meta-Learner Weights
# ============================================================================

def print_ensemble_weights(models: Dict) -> None:
    """Print the learned weights of the meta-learner."""
    meta = models['meta_learner']
    
    if meta is None:
        print("✗ Meta-learner not loaded")
        return
    
    weights = meta.get_weights()
    
    print("\nMeta-Learner Weights (relative importance of sub-models):")
    print("─" * 50)
    total_weight = sum(abs(w) for w in weights.values())
    
    for model_name, weight in sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True):
        if total_weight > 0:
            pct = abs(weight) / total_weight * 100
        else:
            pct = 0
        print(f"  {model_name:<15}: {weight:>8.4f}  ({pct:>5.1f}%)")
    print("─" * 50)


# ============================================================================
# MAIN DEMO
# ============================================================================

if __name__ == '__main__':
    print("╔" + "=" * 68 + "╗")
    print("║" + "  NBA Ensemble Models Demo  ".center(68) + "║")
    print("╚" + "=" * 68 + "╝\n")
    
    # Load models
    models = load_ensemble_models()
    
    print("\n" + "─" * 70)
    print("LOADED MODELS:")
    print("─" * 70)
    for name, model in models.items():
        status = "✓" if model is not None else "✗"
        print(f"  {status} {name.ljust(15)}: {type(model).__name__}")
    
    # Example game prediction
    print("\n" + "─" * 70)
    print("EXAMPLE: Game Prediction")
    print("─" * 70)
    
    # Create a dummy game
    dummy_game = pd.Series({
        'home_tid': '1610612738',
        'away_tid': '1610612739',
        'elo_home': 1550.0,
        'elo_away': 1480.0,
        'home_advantage': 1.0,
        'neutral_site': 0.0,
        'home_recent_pace': 1.02,
        'away_recent_pace': 0.98,
        'home_off_strength': 1.05,
        'home_def_strength': 1.02,
        'away_off_strength': 0.95,
        'away_def_strength': 0.98,
        'home_recent_winrate': 0.65,
        'away_recent_winrate': 0.45,
        'match_off_edge': 0.10,
        'match_def_edge': 0.04,
        'match_pace_sum': 2.00,
        'winrate_diff': 0.20,
        'home_days_rest': 2.0,
        'away_days_rest': 2.0,
        'home_b2b': 0.0,
        'away_b2b': 0.0,
        'home_injury_impact': 0.0,
        'away_injury_impact': 0.0,
        'season_end_year': 2025.0,
        'season_decade': 2020.0,
        'elo_diff': 70.0,
        'home_efg_prior': 0.53,
        'home_tov_pct_prior': 0.13,
        'home_orb_pct_prior': 0.24,
        'home_ftr_prior': 0.25,
        'away_efg_prior': 0.50,
        'away_tov_pct_prior': 0.15,
        'away_orb_pct_prior': 0.21,
        'away_ftr_prior': 0.23,
    })
    
    game_features = [col for col in dummy_game.index if col not in ['home_tid', 'away_tid', 'home_score', 'away_score', 'gid']]
    
    predictions = predict_game_ensemble(dummy_game, models, game_features)
    
    print("\nPredictions for example game:")
    for model_name, pred in predictions.items():
        print(f"\n  {model_name.upper()}:")
        for key, val in pred.items():
            if isinstance(val, dict):
                for k, v in val.items():
                    print(f"    {k}: {v:.4f}")
            else:
                print(f"    {key}: {val:.4f}")
    
    # Print meta-learner weights
    print_ensemble_weights(models)
    
    # Example exhaustion features
    print("\n" + "─" * 70)
    print("EXAMPLE: Adding Exhaustion Features")
    print("─" * 70)
    
    dummy_players = pd.DataFrame({
        'playerId': ['player1', 'player1', 'player1'],
        'season_end_year': [2025, 2025, 2025],
        'min_prev_mean10': [28, 29, 31],
        'player_b2b': [0, 1, 1],
        'days_rest': [2, 1, 1],
    })
    
    dummy_players_with_exhaustion = add_exhaustion_to_player_stats(dummy_players)
    print("\n✓ Exhaustion features added successfully")
    print(dummy_players_with_exhaustion[['playerId', 'season_fatigue', 'heavy_usage', 'consecutive_b2b', 'rest_accumulated']])
    
    print("\n" + "─" * 70)
    print("Demo complete!")
    print("─" * 70 + "\n")
