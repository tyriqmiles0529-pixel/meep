"""
Compare Meta-Context Ensemble vs Original Ensemble vs Baseline

Shows three-way comparison:
1. Baseline (rolling 10-game average)
2. Original Ensemble (5 components with simple context)
3. Meta-Context Ensemble (15 features with learned weights)
"""

import json
from pathlib import Path


def main():
    print("="*70)
    print("META-CONTEXT ENSEMBLE COMPARISON")
    print("="*70)

    # Load baseline backtest results
    backtest_path = Path("backtest_ensemble_2020_2026.json")
    if backtest_path.exists():
        with open(backtest_path, 'r') as f:
            backtest = json.load(f)
        baseline_metrics = backtest.get('backtest_results', {}).get('player_props', {})
    else:
        print("\nERROR: Baseline backtest not found")
        print("Run: python comprehensive_backtest.py --start-season 2020 --end-season 2026 --output backtest_ensemble_2020_2026.json --verbose")
        return

    # Load original ensemble metadata
    orig_meta_path = Path("model_cache/player_ensemble_2022_2026_meta.json")
    if orig_meta_path.exists():
        with open(orig_meta_path, 'r') as f:
            orig_meta = json.load(f)
        orig_metrics = orig_meta.get('metrics', {})
    else:
        print("\nWARNING: Original ensemble metadata not found")
        orig_metrics = {}

    # Load meta-context ensemble metadata
    meta_context_path = Path("model_cache/player_ensemble_meta_context_2022_2026_meta.json")
    if meta_context_path.exists():
        with open(meta_context_path, 'r') as f:
            meta_context = json.load(f)
        meta_metrics = meta_context.get('metrics', {})
    else:
        print("\nERROR: Meta-context ensemble not found")
        print("Run: python train_meta_context_test.py")
        return

    print("\n" + "="*70)
    print("THREE-WAY COMPARISON (2020-2026 Baseline, 2022-2026 Ensembles)")
    print("="*70)

    stats = ['points', 'rebounds', 'assists', 'threes', 'minutes']

    for stat in stats:
        print(f"\n{stat.upper()}:")
        print("-" * 70)

        # Baseline (from 2020-2026 backtest)
        if stat in baseline_metrics:
            baseline_rmse = baseline_metrics[stat].get('rmse', 0)
            baseline_mae = baseline_metrics[stat].get('mae', 0)
        else:
            baseline_rmse = 0
            baseline_mae = 0

        # Original ensemble (trained on 2022-2026)
        if stat in orig_metrics:
            orig_rmse = orig_metrics[stat].get('rmse', 0)
            orig_mae = orig_metrics[stat].get('mae', 0)
        else:
            orig_rmse = 0
            orig_mae = 0

        # Meta-context ensemble (trained on 2022-2026)
        if stat in meta_metrics:
            meta_rmse = meta_metrics[stat].get('rmse', 0)
            meta_mae = meta_metrics[stat].get('mae', 0)
        else:
            meta_rmse = 0
            meta_mae = 0

        # Calculate improvements
        if baseline_rmse > 0:
            orig_improvement = (baseline_rmse - orig_rmse) / baseline_rmse * 100
            meta_improvement = (baseline_rmse - meta_rmse) / baseline_rmse * 100
        else:
            orig_improvement = 0
            meta_improvement = 0

        if orig_rmse > 0:
            meta_vs_orig = (orig_rmse - meta_rmse) / orig_rmse * 100
        else:
            meta_vs_orig = 0

        # Display
        print(f"  Baseline RMSE:          {baseline_rmse:.3f}")
        print(f"  Original Ensemble RMSE: {orig_rmse:.3f}  ({orig_improvement:+.1f}% vs baseline)")
        print(f"  Meta-Context RMSE:      {meta_rmse:.3f}  ({meta_improvement:+.1f}% vs baseline, {meta_vs_orig:+.1f}% vs original)")
        print()
        print(f"  Baseline MAE:           {baseline_mae:.3f}")
        print(f"  Original Ensemble MAE:  {orig_mae:.3f}")
        print(f"  Meta-Context MAE:       {meta_mae:.3f}")

    print("\n" + "="*70)
    print("SUMMARY: RMSE IMPROVEMENTS")
    print("="*70)

    # Calculate averages
    orig_improvements = []
    meta_improvements = []
    meta_vs_orig_improvements = []

    for stat in stats:
        if stat in baseline_metrics and stat in orig_metrics and stat in meta_metrics:
            baseline_rmse = baseline_metrics[stat]['rmse']
            orig_rmse = orig_metrics[stat]['rmse']
            meta_rmse = meta_metrics[stat]['rmse']

            if baseline_rmse > 0:
                orig_imp = (baseline_rmse - orig_rmse) / baseline_rmse * 100
                meta_imp = (baseline_rmse - meta_rmse) / baseline_rmse * 100
                orig_improvements.append((stat, orig_imp))
                meta_improvements.append((stat, meta_imp))

            if orig_rmse > 0:
                meta_vs_orig_imp = (orig_rmse - meta_rmse) / orig_rmse * 100
                meta_vs_orig_improvements.append((stat, meta_vs_orig_imp))

    if orig_improvements:
        print("\nOriginal Ensemble vs Baseline:")
        avg_orig = sum(imp for _, imp in orig_improvements) / len(orig_improvements)
        print(f"  Average improvement: {avg_orig:+.1f}%")
        for stat, imp in sorted(orig_improvements, key=lambda x: x[1], reverse=True):
            emoji = "✅" if imp > 0 else "❌"
            print(f"    {emoji} {stat:10s}: {imp:+.1f}%")

    if meta_improvements:
        print("\nMeta-Context Ensemble vs Baseline:")
        avg_meta = sum(imp for _, imp in meta_improvements) / len(meta_improvements)
        print(f"  Average improvement: {avg_meta:+.1f}%")
        for stat, imp in sorted(meta_improvements, key=lambda x: x[1], reverse=True):
            emoji = "✅" if imp > 0 else "❌"
            print(f"    {emoji} {stat:10s}: {imp:+.1f}%")

    if meta_vs_orig_improvements:
        print("\nMeta-Context vs Original Ensemble:")
        avg_diff = sum(imp for _, imp in meta_vs_orig_improvements) / len(meta_vs_orig_improvements)
        print(f"  Average improvement: {avg_diff:+.1f}%")
        for stat, imp in sorted(meta_vs_orig_improvements, key=lambda x: x[1], reverse=True):
            emoji = "✅" if imp > 0 else "❌"
            print(f"    {emoji} {stat:10s}: {imp:+.1f}%")

    print("\n" + "="*70)
    print("LEARNED CONTEXT WEIGHTS")
    print("="*70)

    # Show learned weights for each stat
    if meta_context_path.exists():
        print("\nMeta-learner coefficients show which features matter most.")
        print("(See training output for full weight breakdown)")

    print("\n" + "="*70)
    print("DECISION")
    print("="*70)

    if meta_vs_orig_improvements:
        avg_gain = sum(imp for _, imp in meta_vs_orig_improvements) / len(meta_vs_orig_improvements)

        if avg_gain >= 1.0:
            print(f"\n✅ META-CONTEXT WINS: {avg_gain:+.1f}% better than original ensemble")
            print("   → Proceed to Phase 2 (5-year backtest)")
        elif avg_gain >= 0.3:
            print(f"\n⚠️  MARGINAL GAIN: {avg_gain:+.1f}% better than original ensemble")
            print("   → Consider proceeding to Phase 2 to validate on larger dataset")
        else:
            print(f"\n❌ NO SIGNIFICANT GAIN: {avg_gain:+.1f}% vs original ensemble")
            print("   → Stick with original ensemble (simpler and equally good)")


if __name__ == "__main__":
    main()
