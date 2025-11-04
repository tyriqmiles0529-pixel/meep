"""
Test ensemble training on a single window (2022-2026)
Quick validation before running full training.
"""

import sys
from pathlib import Path
from train_ensemble_players import train_ensemble_for_window, load_player_data_for_window

def main():
    print("="*70)
    print("TESTING ENSEMBLE ON SINGLE WINDOW (2022-2026)")
    print("="*70)

    # Setup paths
    kaggle_cache = Path.home() / ".cache" / "kagglehub" / "datasets" / \
                  "eoinamoore" / "historical-nba-data-and-player-box-scores" / \
                  "versions" / "258"
    player_stats_path = kaggle_cache / "PlayerStatistics.csv"
    lgbm_models_dir = Path("models")
    cache_dir = Path("model_cache")
    cache_dir.mkdir(exist_ok=True)

    if not player_stats_path.exists():
        print(f"ERROR: Player stats not found at {player_stats_path}")
        print("Run train_auto.py first")
        return

    # Test window: 2022-2026 (current season)
    window_info = {
        'seasons': [2022, 2023, 2024, 2025, 2026],
        'start_year': 2022,
        'end_year': 2026,
        'is_current': True
    }

    print(f"\nTest window: {window_info['start_year']}-{window_info['end_year']}")
    print(f"Seasons: {window_info['seasons']}")

    try:
        # Train ensemble for this window
        ensembles = train_ensemble_for_window(
            window_info,
            player_stats_path,
            lgbm_models_dir,
            cache_dir,
            verbose=True
        )

        # Display results
        print("\n" + "="*70)
        print("ENSEMBLE TRAINING RESULTS")
        print("="*70)

        for stat_name, info in ensembles.items():
            print(f"\n{stat_name.upper()}:")
            print(f"  RMSE: {info['rmse']:.3f}")
            print(f"  MAE: {info['mae']:.3f}")
            print(f"  Samples: {info['n_samples']:,}")

        print("\n" + "="*70)
        print("✅ TEST SUCCESSFUL!")
        print("="*70)
        print("\nEnsemble models saved to:")
        print(f"  model_cache/player_ensemble_2022_2026.pkl")
        print(f"  model_cache/player_ensemble_2022_2026_meta.json")

        print("\n💡 Next steps:")
        print("1. Run full training: python train_ensemble_players.py --verbose")
        print("2. Backtest ensemble vs LightGBM-only")
        print("3. Compare improvement in RMSE/MAE")

    except Exception as e:
        print("\n" + "="*70)
        print("❌ TEST FAILED")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
