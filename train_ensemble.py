"""
Integration of ensemble models into train_auto.py workflow.

Functions to train Ridge, Elo, Four Factors, and Logistic Regression meta-learner.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from ensemble_models import (
    RidgeScoreDiffModel,
    EloRating,
    FourFactorsModel,
    LogisticEnsembler,
    add_exhaustion_features
)


def train_ridge_score_diff(
    games_df: pd.DataFrame,
    game_features: List[str],
    game_defaults: Dict[str, float],
    verbose: bool = False,
    seed: int = 42
) -> Tuple[RidgeScoreDiffModel, Dict[str, float]]:
    """
    Train L2 Ridge regression on game score differentials.
    
    Args:
        games_df: Game-level dataframe with features and scores
        game_features: List of feature column names
        game_defaults: Default values for missing features
        verbose: Print metrics
        seed: Random seed
    
    Returns:
        (trained_model, metrics_dict)
    """
    # Prepare features
    X_full = games_df[game_features].apply(pd.to_numeric, errors='coerce')
    for col, default in game_defaults.items():
        if col in X_full.columns:
            X_full[col] = X_full[col].fillna(default)
    X_full = X_full.astype('float32')
    
    # Target: home score - away score
    y = (games_df['home_score'].values - games_df['away_score'].values).astype(float)
    
    # Time-based split (80/20)
    split = max(1, int(len(X_full) * 0.8))
    X_tr, X_val = X_full.iloc[:split], X_full.iloc[split:]
    y_tr, y_val = y[:split], y[split:]
    
    # Train model
    ridge = RidgeScoreDiffModel(alpha=1.0)
    train_metrics = ridge.fit(X_tr, y_tr)
    
    # Validation metrics
    y_pred_val = ridge.predict(X_val)
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    val_rmse = float(np.sqrt(mean_squared_error(y_val, y_pred_val)))
    val_mae = float(mean_absolute_error(y_val, y_pred_val))
    
    metrics = {
        **train_metrics,
        'val_rmse': val_rmse,
        'val_mae': val_mae,
        'train_size': len(X_tr),
        'val_size': len(X_val)
    }
    
    if verbose:
        print(f"\n── Ridge Score Diff Model ──")
        print(f"  Train RMSE: {train_metrics['rmse']:.3f}, MAE: {train_metrics['mae']:.3f}")
        print(f"  Val RMSE: {val_rmse:.3f}, MAE: {val_mae:.3f}")
        print(f"  Residual Std: {train_metrics['residual_std']:.3f}")
    
    return ridge, metrics


def train_elo_model(
    games_df: pd.DataFrame,
    k_factor: float = 20.0,
    home_advantage: float = 70.0,
    verbose: bool = False
) -> Tuple[EloRating, pd.DataFrame, Dict[str, any]]:
    """
    Build Elo ratings for all teams.
    
    Args:
        games_df: Game-level dataframe (must include home_tid, away_tid, home_score, away_score)
        k_factor: K-factor for rating updates
        home_advantage: Home advantage rating boost
        verbose: Print info
    
    Returns:
        (elo_model, games_df_with_elo_features, metrics_dict)
    """
    elo = EloRating(k_factor=k_factor, home_advantage=home_advantage)
    
    # Add Elo features to games
    games_with_elo, elo_final = elo.add_elo_features(games_df.copy())
    
    # Compute accuracy of Elo predictions
    games_with_elo['elo_pred_prob'] = games_with_elo.apply(
        lambda row: elo.expected_win_prob(
            row['elo_home'] + home_advantage,
            row['elo_away']
        ),
        axis=1
    )
    games_with_elo['elo_pred_win'] = games_with_elo['elo_pred_prob'] > 0.5
    games_with_elo['actual_win'] = games_with_elo['home_score'] > games_with_elo['away_score']
    
    accuracy = (games_with_elo['elo_pred_win'] == games_with_elo['actual_win']).mean()
    
    from sklearn.metrics import log_loss, brier_score_loss
    ll = log_loss(games_with_elo['actual_win'], games_with_elo['elo_pred_prob'])
    brier = brier_score_loss(games_with_elo['actual_win'], games_with_elo['elo_pred_prob'])
    
    metrics = {
        'accuracy': float(accuracy),
        'logloss': float(ll),
        'brier': float(brier),
        'n_teams': len(elo_final.ratings),
        'final_ratings': dict(elo_final.ratings)
    }
    
    if verbose:
        print(f"\n── Elo Rating Model ──")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Logloss: {ll:.3f}, Brier: {brier:.3f}")
        print(f"  Teams: {len(elo_final.ratings)}")
    
    return elo_final, games_with_elo, metrics


def train_four_factors_model(
    games_df: pd.DataFrame,
    verbose: bool = False,
    seed: int = 42
) -> Tuple[Optional[FourFactorsModel], Dict[str, float]]:
    """
    Train Four Factors model.
    
    NOTE: Requires detailed box score data (FG, 3P, FGA, FTA, TOV, ORB, DRB).
    If unavailable, returns None and placeholder metrics.
    
    Args:
        games_df: Game-level dataframe
        verbose: Print info
        seed: Random seed
    
    Returns:
        (model_or_none, metrics_dict)
    """
    # Check if we have four factors features
    ff_cols = ['home_efg_prior', 'home_tov_pct_prior', 'home_orb_pct_prior', 'home_ftr_prior',
               'away_efg_prior', 'away_tov_pct_prior', 'away_orb_pct_prior', 'away_ftr_prior']
    
    has_ff_data = all(col in games_df.columns for col in ff_cols)
    
    if not has_ff_data:
        if verbose:
            print(f"\n── Four Factors Model ──")
            print(f"  WARNING: Box score data not available. Skipping Four Factors model.")
        return None, {'status': 'skipped', 'reason': 'no_box_score_data'}
    
    # Build four factors features
    X_ff = games_df[[
        'home_efg_prior', 'home_tov_pct_prior', 'home_orb_pct_prior', 'home_ftr_prior',
        'away_efg_prior', 'away_tov_pct_prior', 'away_orb_pct_prior', 'away_ftr_prior'
    ]].apply(pd.to_numeric, errors='coerce').fillna(0).astype('float32')
    
    y = (games_df['home_score'].values - games_df['away_score'].values).astype(float)
    
    # Time-based split
    split = max(1, int(len(X_ff) * 0.8))
    X_tr, X_val = X_ff.iloc[:split], X_ff.iloc[split:]
    y_tr, y_val = y[:split], y[split:]
    
    # Train
    ff_model = FourFactorsModel()
    train_metrics = ff_model.fit(X_tr, y_tr)
    
    # Validation
    y_pred_val = ff_model.predict(X_val)
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    val_rmse = float(np.sqrt(mean_squared_error(y_val, y_pred_val)))
    val_mae = float(mean_absolute_error(y_val, y_pred_val))
    
    metrics = {
        **train_metrics,
        'val_rmse': val_rmse,
        'val_mae': val_mae
    }
    
    if verbose:
        print(f"\n── Four Factors Model ──")
        print(f"  Train RMSE: {train_metrics['rmse']:.3f}, MAE: {train_metrics['mae']:.3f}")
        print(f"  Val RMSE: {val_rmse:.3f}, MAE: {val_mae:.3f}")
    
    return ff_model, metrics


def train_logistic_ensembler(
    games_df: pd.DataFrame,
    ridge_model: RidgeScoreDiffModel,
    elo_model: EloRating,
    ff_model: Optional[FourFactorsModel],
    lgb_classifier: any,  # LightGBM classifier
    game_features: List[str],
    game_defaults: Dict[str, float],
    refit_frequency: int = 20,
    verbose: bool = False,
    seed: int = 42
) -> Tuple[LogisticEnsembler, np.ndarray, Dict[str, any]]:
    """
    Train Logistic Regression meta-learner that blends all sub-models.
    
    Args:
        games_df: Game-level dataframe with all features
        ridge_model: Trained Ridge model
        elo_model: Trained Elo model
        ff_model: Trained Four Factors model (can be None)
        lgb_classifier: Trained LightGBM classifier
        game_features: List of feature column names
        game_defaults: Default values for features
        refit_frequency: Refit meta-learner every N games
        verbose: Print info
        seed: Random seed
    
    Returns:
        (meta_learner, oof_probs, metrics_dict)
    """
    # Prepare target
    y_target = (games_df['home_score'].values > games_df['away_score'].values).astype(int)
    
    # Get sub-model predictions
    X_full = games_df[game_features].apply(pd.to_numeric, errors='coerce')
    for col, default in game_defaults.items():
        if col in X_full.columns:
            X_full[col] = X_full[col].fillna(default)
    X_full = X_full.astype('float32')
    
    # Ridge predictions
    ridge_margins = ridge_model.predict(X_full)
    ridge_probs = ridge_model.predict_proba(X_full)
    
    # Elo predictions
    elo_probs = np.array([
        elo_model.expected_win_prob(
            games_df.iloc[i]['elo_home'] + 70,
            games_df.iloc[i]['elo_away']
        )
        for i in range(len(games_df))
    ])
    
    # Four Factors predictions
    if ff_model is not None:
        X_ff = games_df[[
            'home_efg_prior', 'home_tov_pct_prior', 'home_orb_pct_prior', 'home_ftr_prior',
            'away_efg_prior', 'away_tov_pct_prior', 'away_orb_pct_prior', 'away_ftr_prior'
        ]].apply(pd.to_numeric, errors='coerce').fillna(0).astype('float32')
        ff_probs = ff_model.predict_proba(X_ff, residual_std=15.0)
    else:
        ff_probs = np.full(len(games_df), 0.5)
    
    # LGB predictions
    lgb_probs = lgb_classifier.predict_proba(X_full)[:, 1]
    
    # Stack predictions: shape (n_games, 4)
    X_meta = np.column_stack([ridge_probs, elo_probs, ff_probs, lgb_probs])
    
    # Train meta-learner
    ensembler = LogisticEnsembler(refit_frequency=refit_frequency, random_state=seed)
    history = ensembler.fit(X_meta, y_target)
    
    # Get OOF probabilities from training
    oof_probs = np.full(len(games_df), np.nan, dtype=float)
    refit_intervals = list(range(0, len(X_meta), refit_frequency))
    for i in range(1, len(refit_intervals)):
        val_start = refit_intervals[i]
        val_end = refit_intervals[i + 1] if i + 1 < len(refit_intervals) else len(X_meta)
        X_val = X_meta[val_start:val_end]
        oof_probs[val_start:val_end] = ensembler.predict_proba(X_val)
    
    weights = ensembler.get_weights()
    
    if verbose:
        print(f"\n── Logistic Regression Ensembler ──")
        print(f"  Refits: {history['refits']}")
        print(f"  OOF Logloss: {history['oof_logloss']:.3f}")
        print(f"  OOF Brier: {history['oof_brier']:.3f}")
        print(f"  Sub-model weights:")
        print(f"    Ridge: {weights['ridge']:.3f}")
        print(f"    Elo: {weights['elo']:.3f}")
        print(f"    Four Factors: {weights['four_factors']:.3f}")
        print(f"    LGB: {weights['lgb']:.3f}")
    
    metrics = {
        **history,
        'weights': weights,
        'refit_frequency': refit_frequency
    }
    
    return ensembler, oof_probs, metrics


def compare_models(
    games_df: pd.DataFrame,
    ridge_probs: np.ndarray,
    elo_probs: np.ndarray,
    ff_probs: Optional[np.ndarray],
    lgb_probs: np.ndarray,
    ensemble_probs: np.ndarray,
    verbose: bool = False
) -> Dict[str, Dict[str, float]]:
    """
    Compare performance of all models on validation set.
    """
    from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
    
    y_target = (games_df['home_score'].values > games_df['away_score'].values).astype(int)
    
    results = {}
    
    for name, probs in [
        ('ridge', ridge_probs),
        ('elo', elo_probs),
        ('ff', ff_probs) if ff_probs is not None else None,
        ('lgb', lgb_probs),
        ('ensemble', ensemble_probs)
    ]:
        if probs is None:
            continue
        
        valid_mask = ~np.isnan(probs)
        if valid_mask.sum() == 0:
            continue
        
        y_valid = y_target[valid_mask]
        p_valid = probs[valid_mask]
        
        ll = log_loss(y_valid, p_valid)
        brier = brier_score_loss(y_valid, p_valid)
        auc = roc_auc_score(y_valid, p_valid)
        
        results[name] = {
            'logloss': float(ll),
            'brier': float(brier),
            'auc': float(auc)
        }
    
    if verbose:
        print(f"\n── Model Comparison (Validation) ──")
        for name, metrics in results.items():
            print(f"  {name.upper():12} | LL: {metrics['logloss']:.3f} | Brier: {metrics['brier']:.3f} | AUC: {metrics['auc']:.3f}")
    
    return results
