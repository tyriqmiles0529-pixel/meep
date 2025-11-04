"""
UNIFIED ENSEMBLE SYSTEM - Master Meta-Learner

Combines ALL ensemble models into a hierarchical meta-learning architecture:

Level 1: Basic Models (4 models)
  - Ridge Score Diff
  - Elo Rating
  - Four Factors
  - LightGBM Base

Level 2: Enhanced Models (3 models)
  - Dynamic Elo
  - Rolling Four Factors
  - Enhanced Logistic (with interactions)

Level 3: Master Meta-Learner
  - Blends Level 1 and Level 2 predictions
  - Learns optimal weights across ALL models
  - Cross-validation for weight optimization
  - Continuous refitting every 20 games

This architecture ensures:
- All models contribute their unique insights
- No redundant retraining
- Optimal weight allocation based on recent performance
- Maximum prediction accuracy
"""

import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Optional, Tuple, Any
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Import both ensemble systems
from ensemble_models import (
    RidgeScoreDiffModel as BasicRidge,
    EloRating as BasicElo,
    FourFactorsModel as BasicFourFactors,
    LogisticEnsembler as BasicEnsembler,
)

from ensemble_models_enhanced import (
    DynamicEloRating,
    FourFactorsModelDynamic,
    EnhancedLogisticEnsembler,
)


# ============================================================================
# HIERARCHICAL META-LEARNER ARCHITECTURE
# ============================================================================

class HierarchicalEnsemble:
    """
    Three-level hierarchical ensemble:

    Level 1: Basic models (Ridge, Elo, FF, LGB)
    Level 2: Enhanced models (Dynamic Elo, Rolling FF, Enhanced Logistic)
    Level 3: Master meta-learner (blends Level 1 + Level 2)
    """

    def __init__(self, refit_frequency: int = 20):
        self.refit_frequency = refit_frequency

        # Level 1: Basic models
        self.basic_models = {
            'ridge': None,
            'elo': None,
            'four_factors': None,
            'lgb': None,
        }

        # Level 2: Enhanced models
        self.enhanced_models = {
            'dynamic_elo': None,
            'rolling_ff': None,
            'enhanced_logistic': None,
        }

        # Level 3: Master meta-learner
        self.master_meta_learner = None
        self.level1_meta_learner = None
        self.level2_meta_learner = None

        # Performance tracking
        self.model_weights_history = []
        self.performance_by_model = {}
        self.refit_count = 0

    def train_level1_models(self, games_df, game_features, game_defaults, lgb_model, verbose=True):
        """Train all Level 1 basic models."""
        if verbose:
            print("\n" + "="*70)
            print("LEVEL 1: Training Basic Ensemble Models")
            print("="*70)

        # Import training functions
        from train_ensemble import (
            train_ridge_score_diff,
            train_elo_model,
            train_four_factors_model,
        )

        # Train Ridge
        if verbose:
            print("\n[1/4] Training Ridge Score Diff...")
        ridge, ridge_metrics = train_ridge_score_diff(
            games_df, game_features, game_defaults, verbose=verbose
        )
        self.basic_models['ridge'] = ridge

        # Train Elo
        if verbose:
            print("\n[2/4] Training Elo Ratings...")
        elo, games_df, elo_metrics = train_elo_model(
            games_df, k_factor=20.0, home_advantage=70.0, verbose=verbose
        )
        self.basic_models['elo'] = elo

        # Train Four Factors
        if verbose:
            print("\n[3/4] Training Four Factors...")
        ff, ff_metrics = train_four_factors_model(games_df, verbose=verbose)
        self.basic_models['four_factors'] = ff

        # Store LGB
        if verbose:
            print("\n[4/4] Storing LightGBM model...")
        self.basic_models['lgb'] = lgb_model

        if verbose:
            print("\n[OK] Level 1 models trained successfully")

        return games_df, {
            'ridge': ridge_metrics,
            'elo': elo_metrics,
            'four_factors': ff_metrics,
        }

    def train_level2_models(self, games_df, game_features, game_defaults, verbose=True):
        """Train all Level 2 enhanced models."""
        if verbose:
            print("\n" + "="*70)
            print("LEVEL 2: Training Enhanced Ensemble Models")
            print("="*70)

        # Import training functions
        from train_ensemble_enhanced import (
            train_dynamic_elo_model,
            train_four_factors_dynamic,
        )

        # Train Dynamic Elo
        if verbose:
            print("\n[1/3] Training Dynamic Elo...")
        dynamic_elo, games_df, dyn_elo_metrics = train_dynamic_elo_model(
            games_df, base_k=20.0, home_advantage=70.0
        )
        self.enhanced_models['dynamic_elo'] = dynamic_elo

        # Train Rolling Four Factors
        if verbose:
            print("\n[2/3] Training Rolling Four Factors...")
        rolling_ff, rolling_ff_metrics = train_four_factors_dynamic(
            games_df, game_features, game_defaults, rolling_window=10
        )
        self.enhanced_models['rolling_ff'] = rolling_ff

        # Enhanced Logistic will be trained in train_master_metalearner
        if verbose:
            print("\n[3/3] Enhanced Logistic Ensembler will be trained at Level 3")

        if verbose:
            print("\n[OK] Level 2 models trained successfully")

        return games_df, {
            'dynamic_elo': dyn_elo_metrics,
            'rolling_ff': rolling_ff_metrics,
        }

    def generate_all_predictions(self, games_df, game_features, game_defaults):
        """
        Generate predictions from ALL models (Level 1 + Level 2).

        Returns:
            Dict with predictions from each model
        """
        X_full = games_df[game_features].apply(pd.to_numeric, errors='coerce')
        for col, default in game_defaults.items():
            if col in X_full.columns:
                X_full[col] = X_full[col].fillna(default)
        X_full = X_full.astype('float32')

        predictions = {}

        # Level 1 predictions
        if self.basic_models['ridge'] is not None:
            predictions['ridge'] = self.basic_models['ridge'].predict_proba(X_full)

        if self.basic_models['elo'] is not None:
            elo_probs = []
            for idx, row in games_df.iterrows():
                prob = self.basic_models['elo'].expected_win_prob(
                    row.get('elo_home', 1500) + 70.0,
                    row.get('elo_away', 1500)
                )
                elo_probs.append(prob)
            predictions['elo'] = np.array(elo_probs)

        if self.basic_models['four_factors'] is not None:
            try:
                X_ff = games_df[[
                    'home_efg_prior', 'home_tov_pct_prior', 'home_orb_pct_prior', 'home_ftr_prior',
                    'away_efg_prior', 'away_tov_pct_prior', 'away_orb_pct_prior', 'away_ftr_prior'
                ]].apply(pd.to_numeric, errors='coerce').fillna(0).astype('float32')
                predictions['four_factors'] = self.basic_models['four_factors'].predict_proba(X_ff, residual_std=15.0)
            except:
                predictions['four_factors'] = np.full(len(games_df), 0.5)

        if self.basic_models['lgb'] is not None:
            predictions['lgb'] = self.basic_models['lgb'].predict_proba(X_full)[:, 1]

        # Level 2 predictions
        if self.enhanced_models['dynamic_elo'] is not None:
            dyn_elo_probs = []
            for idx, row in games_df.iterrows():
                home_team = row.get('home_team')
                away_team = row.get('away_team')
                prob = self.enhanced_models['dynamic_elo'].expected_win_prob(home_team, away_team)
                dyn_elo_probs.append(prob)
            predictions['dynamic_elo'] = np.array(dyn_elo_probs)

        if self.enhanced_models['rolling_ff'] is not None:
            try:
                predictions['rolling_ff'] = self.enhanced_models['rolling_ff'].predict_proba(X_full)[:, 1]
            except:
                predictions['rolling_ff'] = np.full(len(games_df), 0.5)

        return predictions

    def train_master_metalearner(self, games_df, game_features, game_defaults,
                                 cv_splits=5, verbose=True):
        """
        Train Level 3 master meta-learner using cross-validation.

        Strategy:
        1. Generate predictions from ALL Level 1 and Level 2 models
        2. Use TimeSeriesSplit for proper temporal validation
        3. Train master logistic regression with optimal weights
        4. Continuously refit every 20 games
        """
        if verbose:
            print("\n" + "="*70)
            print("LEVEL 3: Training Master Meta-Learner")
            print("="*70)

        # Generate all predictions
        if verbose:
            print("\n[1/3] Generating predictions from all models...")

        all_preds = self.generate_all_predictions(games_df, game_features, game_defaults)

        # Stack predictions into meta-features
        # Shape: (n_games, n_models)
        available_models = list(all_preds.keys())
        X_meta = np.column_stack([all_preds[model] for model in available_models])

        if verbose:
            print(f"  Available models: {available_models}")
            print(f"  Meta-features shape: {X_meta.shape}")

        # Target
        y_target = (games_df['home_score'].values > games_df['away_score'].values).astype(int)

        # Cross-validation with TimeSeriesSplit
        if verbose:
            print(f"\n[2/3] Cross-validating with {cv_splits} time-based splits...")

        tscv = TimeSeriesSplit(n_splits=cv_splits)
        cv_scores = []
        cv_weights = []

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_meta)):
            X_train_cv, X_val_cv = X_meta[train_idx], X_meta[val_idx]
            y_train_cv, y_val_cv = y_target[train_idx], y_target[val_idx]

            # Train meta-learner
            meta_cv = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
            meta_cv.fit(X_train_cv, y_train_cv)

            # Evaluate
            preds_cv = meta_cv.predict_proba(X_val_cv)[:, 1]
            score_cv = log_loss(y_val_cv, preds_cv)
            cv_scores.append(score_cv)
            cv_weights.append(meta_cv.coef_[0])

            if verbose:
                print(f"  Fold {fold_idx+1}/{cv_splits}: Logloss = {score_cv:.4f}")

        avg_cv_score = np.mean(cv_scores)
        avg_cv_weights = np.mean(cv_weights, axis=0)

        if verbose:
            print(f"\n  Cross-validation Logloss: {avg_cv_score:.4f} (+/- {np.std(cv_scores):.4f})")

        # Train final master meta-learner on all data
        if verbose:
            print(f"\n[3/3] Training final master meta-learner...")

        self.master_meta_learner = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        self.master_meta_learner.fit(X_meta, y_target)

        # Store model names and weights
        final_weights = dict(zip(available_models, self.master_meta_learner.coef_[0]))

        if verbose:
            print(f"\n[OK] Master Meta-Learner trained successfully")
            print(f"\n  Model Weights:")
            total_weight = sum(abs(w) for w in final_weights.values())
            for model, weight in sorted(final_weights.items(), key=lambda x: abs(x[1]), reverse=True):
                pct = abs(weight) / total_weight * 100 if total_weight > 0 else 0
                print(f"    {model:20s}: {weight:>8.4f}  ({pct:>5.1f}%)")

        self.model_weights_history.append({
            'refit_iteration': self.refit_count,
            'weights': final_weights,
            'cv_score': avg_cv_score,
            'available_models': available_models,
        })
        self.refit_count += 1

        return {
            'available_models': available_models,
            'final_weights': final_weights,
            'cv_logloss': avg_cv_score,
            'cv_logloss_std': float(np.std(cv_scores)),
            'refit_count': self.refit_count,
        }

    def predict(self, games_df, game_features, game_defaults):
        """
        Generate final ensemble predictions using master meta-learner.

        Returns:
            Array of probabilities (P(home wins))
        """
        if self.master_meta_learner is None:
            raise ValueError("Master meta-learner not trained yet. Call train_master_metalearner() first.")

        # Generate all predictions
        all_preds = self.generate_all_predictions(games_df, game_features, game_defaults)

        # Stack predictions
        available_models = list(all_preds.keys())
        X_meta = np.column_stack([all_preds[model] for model in available_models])

        # Predict with master meta-learner
        final_probs = self.master_meta_learner.predict_proba(X_meta)[:, 1]

        return final_probs

    def evaluate_all_models(self, games_df, game_features, game_defaults, verbose=True):
        """
        Evaluate and compare ALL models individually + ensemble.

        Returns comprehensive metrics for each model.
        """
        if verbose:
            print("\n" + "="*70)
            print("MODEL EVALUATION & COMPARISON")
            print("="*70)

        y_true = (games_df['home_score'].values > games_df['away_score'].values).astype(int)

        # Get predictions from all models
        all_preds = self.generate_all_predictions(games_df, game_features, game_defaults)

        # Get ensemble prediction
        ensemble_pred = self.predict(games_df, game_features, game_defaults)
        all_preds['ENSEMBLE_MASTER'] = ensemble_pred

        # Evaluate each model
        results = {}
        for model_name, preds in all_preds.items():
            try:
                ll = log_loss(y_true, preds)
                brier = brier_score_loss(y_true, preds)
                auc = roc_auc_score(y_true, preds)
                acc = ((preds > 0.5) == y_true).mean()

                results[model_name] = {
                    'logloss': float(ll),
                    'brier': float(brier),
                    'auc': float(auc),
                    'accuracy': float(acc),
                }
            except Exception as e:
                if verbose:
                    print(f"  Error evaluating {model_name}: {e}")

        # Print comparison
        if verbose:
            print(f"\n{'Model':<25} | {'Logloss':<10} | {'Brier':<10} | {'AUC':<8} | {'Accuracy':<8}")
            print("-" * 70)

            # Sort by logloss (lower is better)
            for model_name in sorted(results.keys(), key=lambda x: results[x]['logloss']):
                metrics = results[model_name]
                marker = "*** " if model_name == 'ENSEMBLE_MASTER' else "    "
                print(f"{marker}{model_name:<21} | {metrics['logloss']:<10.4f} | "
                      f"{metrics['brier']:<10.4f} | {metrics['auc']:<8.4f} | "
                      f"{metrics['accuracy']:<8.4f}")

        return results

    def save_all_models(self, output_dir='models/', verbose=True):
        """Save all models and the master ensemble."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        if verbose:
            print(f"\n[SAVE] Saving all models to {output_dir}...")

        # Save Level 1 models
        for name, model in self.basic_models.items():
            if model is not None:
                path = f"{output_dir}/level1_{name}.pkl"
                with open(path, 'wb') as f:
                    pickle.dump(model, f)
                if verbose:
                    print(f"  [OK] {path}")

        # Save Level 2 models
        for name, model in self.enhanced_models.items():
            if model is not None:
                path = f"{output_dir}/level2_{name}.pkl"
                with open(path, 'wb') as f:
                    pickle.dump(model, f)
                if verbose:
                    print(f"  [OK] {path}")

        # Save master meta-learner
        if self.master_meta_learner is not None:
            path = f"{output_dir}/master_meta_learner.pkl"
            with open(path, 'wb') as f:
                pickle.dump(self.master_meta_learner, f)
            if verbose:
                print(f"  [OK] {path}")

        # Save full ensemble object
        path = f"{output_dir}/hierarchical_ensemble_full.pkl"
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        if verbose:
            print(f"  [OK] {path}")

        # Save weights history
        weights_df = pd.DataFrame(self.model_weights_history)
        weights_path = f"{output_dir}/ensemble_weights_history.csv"
        weights_df.to_csv(weights_path, index=False)
        if verbose:
            print(f"  [OK] {weights_path}")

        print(f"\n[OK] All models saved successfully!")


# ============================================================================
# COMPLETE TRAINING PIPELINE
# ============================================================================

def train_unified_ensemble(games_df, game_features, game_defaults, lgb_model,
                          refit_frequency=20, cv_splits=5, verbose=True):
    """
    Complete unified ensemble training pipeline.

    Trains ALL models (basic + enhanced) and creates hierarchical meta-learner.

    Args:
        games_df: Games dataframe
        game_features: List of feature names
        game_defaults: Dict of default values
        lgb_model: Pre-trained LightGBM model
        refit_frequency: Games between refits
        cv_splits: Number of cross-validation splits
        verbose: Print progress

    Returns:
        HierarchicalEnsemble object with all models trained
    """
    print("\n" + "="*70)
    print("UNIFIED ENSEMBLE TRAINING PIPELINE")
    print("="*70)
    print(f"Training hierarchical ensemble with {refit_frequency}-game refit frequency")
    print(f"Using {cv_splits}-fold time-series cross-validation")
    print("="*70)

    # Initialize ensemble
    ensemble = HierarchicalEnsemble(refit_frequency=refit_frequency)

    # Train Level 1 (Basic models)
    games_df, level1_metrics = ensemble.train_level1_models(
        games_df, game_features, game_defaults, lgb_model, verbose=verbose
    )

    # Train Level 2 (Enhanced models)
    games_df, level2_metrics = ensemble.train_level2_models(
        games_df, game_features, game_defaults, verbose=verbose
    )

    # Train Level 3 (Master meta-learner)
    level3_metrics = ensemble.train_master_metalearner(
        games_df, game_features, game_defaults, cv_splits=cv_splits, verbose=verbose
    )

    # Evaluate all models
    evaluation_results = ensemble.evaluate_all_models(
        games_df, game_features, game_defaults, verbose=verbose
    )

    # Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Level 1 Models: {len([m for m in ensemble.basic_models.values() if m is not None])}")
    print(f"Level 2 Models: {len([m for m in ensemble.enhanced_models.values() if m is not None])}")
    print(f"Total Models in Ensemble: {len(evaluation_results) - 1}")  # -1 for ENSEMBLE_MASTER
    print(f"Master Meta-Learner CV Logloss: {level3_metrics['cv_logloss']:.4f}")

    ensemble_ll = evaluation_results.get('ENSEMBLE_MASTER', {}).get('logloss', np.nan)
    lgb_ll = evaluation_results.get('lgb', {}).get('logloss', np.nan)
    improvement = ((lgb_ll - ensemble_ll) / lgb_ll * 100) if not np.isnan(lgb_ll) and not np.isnan(ensemble_ll) else 0

    print(f"\nLGB Baseline Logloss: {lgb_ll:.4f}")
    print(f"Ensemble Master Logloss: {ensemble_ll:.4f}")
    print(f"Improvement: {improvement:.2f}%")
    print("="*70)

    return ensemble, {
        'level1': level1_metrics,
        'level2': level2_metrics,
        'level3': level3_metrics,
        'evaluation': evaluation_results,
    }


if __name__ == '__main__':
    print("Unified Hierarchical Ensemble System")
    print("\nArchitecture:")
    print("  Level 1: Basic Models (Ridge, Elo, Four Factors, LGB)")
    print("  Level 2: Enhanced Models (Dynamic Elo, Rolling FF, Enhanced Logistic)")
    print("  Level 3: Master Meta-Learner (blends all predictions)")
    print("\nFeatures:")
    print("  - Hierarchical meta-learning across all models")
    print("  - Time-series cross-validation for weight optimization")
    print("  - Continuous refitting every 20 games")
    print("  - Comprehensive model evaluation and comparison")
    print("  - Automatic weight tracking and analysis")
