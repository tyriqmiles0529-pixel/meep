"""
Compare Ensemble vs Baseline Performance
Analyzes ensemble model metrics vs backtest baseline metrics.
"""

import json
from pathlib import Path

def load_ensemble_metrics(window_start: int, window_end: int):
    """Load ensemble metrics from metadata file."""
    meta_path = Path(f"model_cache/player_ensemble_{window_start}_{window_end}_meta.json")

    if not meta_path.exists():
        print(f"ERROR: Ensemble metadata not found at {meta_path}")
        return None

    with open(meta_path, 'r') as f:
        return json.load(f)

def load_backtest_results(backtest_file: str):
    """Load backtest results from JSON file."""
    backtest_path = Path(backtest_file)

    if not backtest_path.exists():
        print(f"ERROR: Backtest results not found at {backtest_path}")
        return None

    with open(backtest_path, 'r') as f:
        return json.load(f)

def compare_metrics():
    """Compare ensemble vs baseline metrics."""

    print("="*70)
    print("ENSEMBLE vs BASELINE COMPARISON")
    print("="*70)

    # Load ensemble metrics (2017-2021 window covers 2020-2021)
    # Load ensemble metrics (2022-2026 window covers 2022-2026)
    ensemble_2017 = load_ensemble_metrics(2017, 2021)
    ensemble_2022 = load_ensemble_metrics(2022, 2026)

    # Load backtest results
    backtest = load_backtest_results("backtest_ensemble_2020_2026.json")

    if not backtest:
        print("\nRun backtest first:")
        print("python comprehensive_backtest.py --start-season 2020 --end-season 2026 --output backtest_ensemble_2020_2026.json --verbose")
        return

    # Extract baseline metrics from backtest
    baseline_metrics = {}
    if 'backtest_results' in backtest and 'player_props' in backtest['backtest_results']:
        baseline_metrics = backtest['backtest_results']['player_props']
    elif 'player_props' in backtest:
        baseline_metrics = backtest['player_props']

    print("\n" + "="*70)
    print("PLAYER PROP PREDICTIONS (2020-2026)")
    print("="*70)

    stats = ['points', 'rebounds', 'assists', 'threes', 'minutes']

    for stat in stats:
        print(f"\n{stat.upper()}:")
        print("-" * 50)

        # Baseline from backtest
        if stat in baseline_metrics:
            baseline_rmse = baseline_metrics[stat].get('rmse', 0)
            baseline_mae = baseline_metrics[stat].get('mae', 0)
            baseline_r2 = baseline_metrics[stat].get('r_squared', 0)
        else:
            baseline_rmse = 0
            baseline_mae = 0
            baseline_r2 = 0

        # Ensemble from 2022-2026 window (most recent)
        if ensemble_2022 and stat in ensemble_2022['metrics']:
            ensemble_rmse = ensemble_2022['metrics'][stat]['rmse']
            ensemble_mae = ensemble_2022['metrics'][stat]['mae']
            ensemble_samples = ensemble_2022['metrics'][stat]['n_samples']
        else:
            ensemble_rmse = 0
            ensemble_mae = 0
            ensemble_samples = 0

        # Calculate improvement
        if baseline_rmse > 0:
            rmse_improvement = (baseline_rmse - ensemble_rmse) / baseline_rmse * 100
        else:
            rmse_improvement = 0

        if baseline_mae > 0:
            mae_improvement = (baseline_mae - ensemble_mae) / baseline_mae * 100
        else:
            mae_improvement = 0

        print(f"  Baseline RMSE:     {baseline_rmse:.3f}")
        print(f"  Ensemble RMSE:     {ensemble_rmse:.3f}")
        print(f"  RMSE Improvement:  {rmse_improvement:+.1f}%")
        print()
        print(f"  Baseline MAE:      {baseline_mae:.3f}")
        print(f"  Ensemble MAE:      {ensemble_mae:.3f}")
        print(f"  MAE Improvement:   {mae_improvement:+.1f}%")
        print()
        print(f"  Baseline R²:       {baseline_r2:.3f}")
        print(f"  Ensemble Samples:  {ensemble_samples:,}")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    # Calculate overall improvement
    total_improvements = []
    for stat in stats:
        if stat in baseline_metrics and ensemble_2022 and stat in ensemble_2022['metrics']:
            baseline_rmse = baseline_metrics[stat].get('rmse', 0)
            ensemble_rmse = ensemble_2022['metrics'][stat]['rmse']
            if baseline_rmse > 0:
                improvement = (baseline_rmse - ensemble_rmse) / baseline_rmse * 100
                total_improvements.append((stat, improvement))

    if total_improvements:
        avg_improvement = sum(imp for _, imp in total_improvements) / len(total_improvements)
        print(f"\nAverage RMSE Improvement: {avg_improvement:+.1f}%")
        print("\nBreakdown by stat:")
        for stat, imp in sorted(total_improvements, key=lambda x: x[1], reverse=True):
            emoji = "✅" if imp > 0 else "❌"
            print(f"  {emoji} {stat:10s}: {imp:+.1f}%")

    print("\n" + "="*70)
    print("ENSEMBLE MODELS AVAILABLE")
    print("="*70)

    # Show all available ensemble windows
    cache_dir = Path("model_cache")
    ensemble_files = sorted(cache_dir.glob("player_ensemble_*_meta.json"))

    for meta_file in ensemble_files:
        with open(meta_file, 'r') as f:
            meta = json.load(f)

        print(f"\nWindow {meta['start_year']}-{meta['end_year']}:")
        print(f"  Trained: {meta['trained_date'][:10]}")
        print(f"  Seasons: {', '.join(map(str, meta['seasons']))}")
        print(f"  Current: {meta.get('is_current_season', False)}")

        # Show metrics summary
        if 'metrics' in meta:
            metrics = meta['metrics']
            avg_rmse = sum(m['rmse'] for m in metrics.values()) / len(metrics)
            print(f"  Avg RMSE: {avg_rmse:.3f}")

if __name__ == "__main__":
    compare_metrics()
