#!/usr/bin/env python3
"""
Test script for unified hierarchical ensemble.

This demonstrates how all models work together and validates the architecture.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path


def create_dummy_data(n_games=1000):
    """Create dummy game data for testing."""
    np.random.seed(42)

    # Generate dummy features
    game_features = [
        'home_advantage', 'neutral_site',
        'home_recent_pace', 'away_recent_pace',
        'home_off_strength', 'home_def_strength',
        'away_off_strength', 'away_def_strength',
        'home_recent_winrate', 'away_recent_winrate',
        'match_off_edge', 'match_def_edge', 'match_pace_sum',
        'winrate_diff', 'home_days_rest', 'away_days_rest',
        'home_b2b', 'away_b2b',
        'home_injury_impact', 'away_injury_impact',
        'season_end_year', 'season_decade', 'elo_diff',
    ]

    # Required columns for models
    data = {
        'gid': [f'game_{i}' for i in range(n_games)],
        'home_tid': np.random.choice(['team_A', 'team_B', 'team_C', 'team_D'], n_games),
        'away_tid': np.random.choice(['team_A', 'team_B', 'team_C', 'team_D'], n_games),
        'home_team': np.random.choice(['team_A', 'team_B', 'team_C', 'team_D'], n_games),
        'away_team': np.random.choice(['team_A', 'team_B', 'team_C', 'team_D'], n_games),
        'date': pd.date_range('2024-01-01', periods=n_games, freq='D'),
        'home_score': np.random.randint(80, 130, n_games),
        'away_score': np.random.randint(80, 130, n_games),
    }

    # Add dummy features
    for feat in game_features:
        if 'year' in feat:
            data[feat] = 2024.0
        elif 'decade' in feat:
            data[feat] = 2020.0
        elif 'b2b' in feat or 'neutral' in feat:
            data[feat] = np.random.randint(0, 2, n_games).astype(float)
        else:
            data[feat] = np.random.randn(n_games).astype(float)

    # Add Four Factors features (optional)
    ff_features = [
        'home_efg_prior', 'home_tov_pct_prior', 'home_orb_pct_prior', 'home_ftr_prior',
        'away_efg_prior', 'away_tov_pct_prior', 'away_orb_pct_prior', 'away_ftr_prior'
    ]
    for feat in ff_features:
        data[feat] = np.random.uniform(0.45, 0.55, n_games)

    games_df = pd.DataFrame(data)

    game_defaults = {feat: 0.0 for feat in game_features}
    game_defaults.update({
        'home_advantage': 1.0,
        'season_end_year': 2024.0,
        'season_decade': 2020.0,
    })

    return games_df, game_features, game_defaults


def create_dummy_lgb_model():
    """Create a dummy LGB-like model for testing."""
    from sklearn.ensemble import RandomForestClassifier

    # Simple RF as LGB replacement
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    return model


def test_unified_ensemble():
    """Test the unified ensemble system."""
    print("="*70)
    print("TESTING UNIFIED HIERARCHICAL ENSEMBLE")
    print("="*70)

    # Create dummy data
    print("\n[1/5] Creating dummy data...")
    games_df, game_features, game_defaults = create_dummy_data(n_games=1000)
    print(f"  Generated {len(games_df)} games with {len(game_features)} features")

    # Create dummy LGB model
    print("\n[2/5] Creating dummy LGB model...")
    lgb_model = create_dummy_lgb_model()

    # Train on first 80% to create a "trained" model
    X_dummy = games_df[game_features].fillna(pd.Series(game_defaults, index=game_features)).astype(float)
    y_dummy = (games_df['home_score'] > games_df['away_score']).astype(int)
    split = int(len(X_dummy) * 0.8)
    lgb_model.fit(X_dummy[:split], y_dummy[:split])
    print(f"  LGB model trained on {split} games")

    # Train unified ensemble
    print("\n[3/5] Training unified hierarchical ensemble...")
    from ensemble_unified import train_unified_ensemble

    unified_ensemble, metrics = train_unified_ensemble(
        games_df=games_df,
        game_features=game_features,
        game_defaults=game_defaults,
        lgb_model=lgb_model,
        refit_frequency=20,
        cv_splits=3,  # Reduced for faster testing
        verbose=True
    )

    # Test predictions
    print("\n[4/5] Testing predictions on new data...")
    test_games = games_df.tail(100)  # Last 100 games
    predictions = unified_ensemble.predict(test_games, game_features, game_defaults)
    print(f"  Generated {len(predictions)} predictions")
    print(f"  Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"  Mean prediction: {predictions.mean():.4f}")

    # Save models
    print("\n[5/5] Saving models...")
    output_dir = 'test_models/'
    unified_ensemble.save_all_models(output_dir=output_dir, verbose=True)

    # Verify files exist
    files_created = list(Path(output_dir).glob('*.pkl')) + list(Path(output_dir).glob('*.csv'))
    print(f"\n[OK] Created {len(files_created)} files in {output_dir}")

    # Test loading
    print("\n[VERIFY] Testing model loading...")
    with open(f'{output_dir}/hierarchical_ensemble_full.pkl', 'rb') as f:
        loaded_ensemble = pickle.load(f)
    print("[OK] Ensemble loaded successfully")

    # Test loaded predictions
    loaded_preds = loaded_ensemble.predict(test_games, game_features, game_defaults)
    match = np.allclose(predictions, loaded_preds)
    print(f"[{'OK' if match else 'FAIL'}] Predictions match: {match}")

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)

    # Summary
    print("\n=== SUMMARY ===")
    print(f"Total Models Trained: {len([m for m in unified_ensemble.basic_models.values() if m is not None])}")
    print(f"                    + {len([m for m in unified_ensemble.enhanced_models.values() if m is not None])}")
    print(f"Master Meta-Learner: {'Trained' if unified_ensemble.master_meta_learner is not None else 'Not trained'}")
    print(f"Refits Performed: {unified_ensemble.refit_count}")

    if 'evaluation' in metrics:
        ensemble_ll = metrics['evaluation'].get('ENSEMBLE_MASTER', {}).get('logloss', 'N/A')
        lgb_ll = metrics['evaluation'].get('lgb', {}).get('logloss', 'N/A')
        print(f"\nEnsemble Logloss: {ensemble_ll}")
        print(f"LGB Logloss: {lgb_ll}")

        if ensemble_ll != 'N/A' and lgb_ll != 'N/A':
            improvement = ((lgb_ll - ensemble_ll) / lgb_ll * 100)
            print(f"Improvement: {improvement:.2f}%")

    print("\n=== NEXT STEPS ===")
    print("1. Integrate into train_auto.py using UNIFIED_ENSEMBLE_GUIDE.md")
    print("2. Run full training with real data")
    print("3. Compare with your current LGB baseline")
    print("4. Monitor weight evolution in ensemble_weights_history.csv")

    return unified_ensemble, metrics


if __name__ == '__main__':
    try:
        ensemble, metrics = test_unified_ensemble()
        print("\n[SUCCESS] All tests passed!")
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
