"""
Check Enhanced Selector Window Preferences and Phase 1 Features

This script:
1. Loads the enhanced selector
2. Shows which windows it picks for different player scenarios
3. Verifies Phase 1 features are in the 2022-2026 window
4. Checks if selector is already preferring the new window
"""

import pickle
import json
import os
from pathlib import Path
import pandas as pd
import numpy as np

print("="*70)
print("ENHANCED SELECTOR & PHASE 1 FEATURES CHECK")
print("="*70)

CACHE_DIR = Path("model_cache")

# ============================================================================
# 1. Check Enhanced Selector
# ============================================================================

print("\n" + "="*70)
print("1. ENHANCED SELECTOR ANALYSIS")
print("="*70)

selector_file = CACHE_DIR / "dynamic_selector_enhanced.pkl"
selector_meta_file = CACHE_DIR / "dynamic_selector_enhanced_meta.json"

if not selector_file.exists():
    print("\n[X] Enhanced selector not found!")
    print(f"   Expected: {selector_file}")
    print("\n   Run this to train selector:")
    print("   python train_dynamic_selector_enhanced.py --verbose")
else:
    print("\n[OK] Loading enhanced selector...")

    with open(selector_file, 'rb') as f:
        selector = pickle.load(f)

    with open(selector_meta_file, 'r') as f:
        meta = json.load(f)

    print(f"\n  Trained: {meta.get('trained_date', 'Unknown')}")
    print(f"  Test accuracy: {meta.get('test_accuracy', {}).get('points', 'N/A')}")
    print(f"  Improvement vs cherry-pick: {meta.get('improvement_vs_cherrypick', {}).get('points', 'N/A')}")

    # Check which windows are available
    if 'points' in selector:
        selector_obj = selector['points']
        windows_list = selector_obj.get('windows_list', [])
        print(f"\n  Available windows: {windows_list}")

        # Simulate different player scenarios to see which window gets picked
        print("\n" + "-"*70)
        print("  WINDOW SELECTION TEST (Points)")
        print("-"*70)

        # Test scenarios: (games_played, recent_avg, recent_std, trend)
        test_scenarios = [
            ("Rookie (10 games)", [10, 12.0, 3.0, 0.5, 0.15, 0.8, 1.0, 0.05, 0.02, 0.3]),
            ("Veteran (200 games)", [200, 18.5, 4.2, -0.2, 0.12, 0.6, 1.2, 0.08, 0.03, 0.25]),
            ("Star (300 games)", [300, 28.0, 6.5, 0.8, 0.08, 0.4, 1.5, 0.12, 0.04, 0.35]),
            ("Consistent role player", [150, 10.2, 2.1, 0.0, 0.20, 0.5, 0.9, 0.06, 0.02, 0.22]),
            ("Volatile scorer", [100, 15.5, 8.5, 1.2, 0.55, 0.3, 1.3, 0.18, 0.05, 0.40]),
        ]

        scaler = selector_obj['scaler']
        model = selector_obj['selector']

        window_counts = {w: 0 for w in windows_list}

        for scenario_name, features in test_scenarios:
            # Scale features
            X = np.array(features).reshape(1, -1)
            X_scaled = scaler.transform(X)

            # Predict window
            window_idx = model.predict(X_scaled)[0]
            selected_window = windows_list[window_idx]
            window_counts[selected_window] += 1

            print(f"\n  {scenario_name:30s} -> {selected_window}")

        print("\n" + "-"*70)
        print("  SUMMARY: Window selection frequency")
        print("-"*70)
        for window, count in sorted(window_counts.items(), key=lambda x: x[1], reverse=True):
            pct = count / len(test_scenarios) * 100
            bar = "#" * int(pct / 10)
            print(f"  {window:15s}: {count}/5 ({pct:4.1f}%) {bar}")

        most_picked = max(window_counts.items(), key=lambda x: x[1])
        print(f"\n  [*] Most preferred window: {most_picked[0]} ({most_picked[1]}/5 scenarios)")

# ============================================================================
# 2. Check Phase 1 Features in 2022-2026 Window
# ============================================================================

print("\n\n" + "="*70)
print("2. PHASE 1 FEATURES VERIFICATION (2022-2026 Window)")
print("="*70)

window_2022 = CACHE_DIR / "player_models_2022_2026.pkl"

if not window_2022.exists():
    print("\n[X] 2022-2026 window not found!")
    print(f"   Expected: {window_2022}")
else:
    print("\n[OK] Loading 2022-2026 window...")

    with open(window_2022, 'rb') as f:
        models_2022 = pickle.load(f)

    # Check points model features
    if 'points' in models_2022:
        points_model = models_2022['points']['model']

        # Get feature names
        if hasattr(points_model, 'feature_name_'):
            feature_names = points_model.feature_name_
        elif hasattr(points_model, 'feature_names_in_'):
            feature_names = points_model.feature_names_in_
        else:
            feature_names = []

        print(f"\n  Total features: {len(feature_names)}")

        # Check for Phase 1 features
        phase1_keywords = [
            'ts_pct',                    # True Shooting %
            'fieldGoalsAttempted',       # FGA
            'threePointersAttempted',    # 3PA
            'freeThrowsAttempted',       # FTA
            'rate_fga',                  # FGA per minute
            'rate_3pa',                  # 3PA per minute
            'rate_fta',                  # FTA per minute
            'three_pct',                 # 3P%
            'ft_pct',                    # FT%
        ]

        print("\n" + "-"*70)
        print("  PHASE 1 FEATURE DETECTION")
        print("-"*70)

        found_features = {}
        for keyword in phase1_keywords:
            matches = [f for f in feature_names if keyword in f]
            found_features[keyword] = matches

            if matches:
                print(f"\n  [+] {keyword:25s}: {len(matches)} features")
                for feat in matches:
                    print(f"      - {feat}")
            else:
                print(f"\n  [-] {keyword:25s}: NOT FOUND")

        # Summary
        total_phase1 = sum(len(v) for v in found_features.values())
        print("\n" + "-"*70)
        print(f"  TOTAL PHASE 1 FEATURES: {total_phase1}")
        print("-"*70)

        if total_phase1 >= 15:
            print("\n  [OK] SUCCESS! Phase 1 features are active in 2022-2026 window")
        elif total_phase1 > 0:
            print(f"\n  [!] PARTIAL: Only {total_phase1}/19 expected features found")
        else:
            print("\n  [X] FAILED: No Phase 1 features detected!")
            print("      The window may have been trained before Phase 1 changes.")

        # Feature importance (if available)
        if hasattr(points_model, 'feature_importances_'):
            importances = points_model.feature_importances_

            # Create importance dataframe
            feat_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)

            # Check if Phase 1 features are important
            print("\n" + "-"*70)
            print("  TOP 20 MOST IMPORTANT FEATURES")
            print("-"*70)

            for i, row in feat_df.head(20).iterrows():
                is_phase1 = any(kw in row['feature'] for kw in phase1_keywords)
                marker = " [*] PHASE 1" if is_phase1 else ""
                print(f"  {row['feature']:35s}: {row['importance']:6.1f}{marker}")

            # Count Phase 1 features in top 20
            phase1_in_top20 = sum(1 for i, row in feat_df.head(20).iterrows()
                                 if any(kw in row['feature'] for kw in phase1_keywords))

            print(f"\n  Phase 1 features in top 20: {phase1_in_top20}/20")

            if phase1_in_top20 >= 5:
                print("  [OK] Phase 1 features are highly important!")
            elif phase1_in_top20 > 0:
                print("  [i] Phase 1 features have moderate importance")
            else:
                print("  [!] Phase 1 features not in top 20")

# ============================================================================
# 3. Compare with Old Windows
# ============================================================================

print("\n\n" + "="*70)
print("3. OLD vs NEW WINDOW COMPARISON")
print("="*70)

window_2017 = CACHE_DIR / "player_models_2017_2021.pkl"

if window_2017.exists() and window_2022.exists():
    print("\n[OK] Comparing 2017-2021 vs 2022-2026 windows...")

    with open(window_2017, 'rb') as f:
        models_2017 = pickle.load(f)

    print("\n" + "-"*70)
    print("  FEATURE COUNT COMPARISON")
    print("-"*70)

    for stat_name in ['points', 'rebounds', 'assists', 'threes']:
        if stat_name in models_2017 and stat_name in models_2022:
            model_2017 = models_2017[stat_name]['model']
            model_2022 = models_2022[stat_name]['model']

            # Get feature counts
            if hasattr(model_2017, 'feature_name_'):
                feat_2017 = len(model_2017.feature_name_)
            else:
                feat_2017 = 0

            if hasattr(model_2022, 'feature_name_'):
                feat_2022 = len(model_2022.feature_name_)
            else:
                feat_2022 = 0

            diff = feat_2022 - feat_2017
            marker = "[*] NEW FEATURES" if diff > 0 else ""
            print(f"\n  {stat_name.upper():10s}:")
            print(f"    2017-2021: {feat_2017} features")
            print(f"    2022-2026: {feat_2022} features  (+{diff}) {marker}")
else:
    print("\n  [i] Can't compare - old window not available")

# ============================================================================
# 4. Recommendations
# ============================================================================

print("\n\n" + "="*70)
print("4. RECOMMENDATIONS")
print("="*70)

print("\nBased on the analysis above:")

if total_phase1 >= 15:
    print("\n[OK] Phase 1 implementation is ACTIVE and working!")
    print("   Your 2022-2026 window has the new features.")

    if selector_file.exists():
        if most_picked[0] == '2022-2026':
            print("\n[OK] Enhanced selector is ALREADY preferring the new window!")
            print("   You're getting the benefit of Phase 1 features.")
            print("\n   NEXT STEPS:")
            print("   1. Monitor prediction accuracy for 1-2 weeks")
            print("   2. If performance is good, you may not need to retrain old windows")
            print("   3. Consider Strategy A from WINDOW_STRATEGY_ANALYSIS.md (recent only)")
        else:
            print(f"\n[!] Enhanced selector prefers {most_picked[0]} window")
            print("   This may reduce the benefit of Phase 1 features.")
            print("\n   OPTIONS:")
            print("   A. Retrain enhanced selector (it will learn new window is better)")
            print("   B. Retrain old windows with Phase 1 features (fair comparison)")
            print("   C. Force use of 2022-2026 window in riq_analyzer.py")
    else:
        print("\n   Enhanced selector not trained yet - that's OK!")
        print("   The 2022-2026 window will be used by default.")
else:
    print("\n[!] Phase 1 features not detected in 2022-2026 window")
    print("   The window may need to be retrained.")
    print("\n   TO FIX:")
    print("   1. Delete: model_cache/player_models_2022_2026.pkl")
    print("   2. Run: python train_auto.py --verbose")

print("\n" + "="*70)
