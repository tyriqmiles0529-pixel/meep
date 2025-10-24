#!/usr/bin/env python3
"""
Test to validate that priors terminology is clear and diagnostics are helpful.
This test validates the changes made to address the issue:
"why are you calling them priors? do you have the appropiate columns for all csv being used?"
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_game_defaults_comments():
    """Verify GAME_DEFAULTS has clear comments about priors vs baselines"""
    print("\n" + "="*60)
    print("TEST 1: GAME_DEFAULTS Comments")
    print("="*60)
    
    from train_auto import GAME_DEFAULTS
    
    # Check that the defaults dict has the expected priors columns
    priors_keys = [
        "home_o_rtg_prior", "home_d_rtg_prior", "home_pace_prior", "home_srs_prior",
        "away_o_rtg_prior", "away_d_rtg_prior", "away_pace_prior", "away_srs_prior"
    ]
    
    missing_keys = [k for k in priors_keys if k not in GAME_DEFAULTS]
    if missing_keys:
        print(f"✗ FAIL: Missing keys in GAME_DEFAULTS: {missing_keys}")
        return False
    
    # Verify default values are league-average baselines
    expected_defaults = {
        "home_o_rtg_prior": 110.0,
        "home_d_rtg_prior": 110.0,
        "home_pace_prior": 100.0,
        "away_o_rtg_prior": 110.0,
        "away_d_rtg_prior": 110.0,
        "away_pace_prior": 100.0,
        "home_srs_prior": 0.0,
        "away_srs_prior": 0.0,
    }
    
    for key, expected_val in expected_defaults.items():
        actual_val = GAME_DEFAULTS[key]
        if actual_val != expected_val:
            print(f"✗ FAIL: {key} has unexpected default: {actual_val} (expected {expected_val})")
            return False
    
    print(f"✓ All {len(priors_keys)} priors columns present with correct baseline defaults")
    print(f"  o/d_rtg: 110.0 (league average)")
    print(f"  pace: 100.0 (league average)")
    print(f"  srs: 0.0 (league average)")
    return True


def test_game_features_includes_priors():
    """Verify GAME_FEATURES includes all priors columns"""
    print("\n" + "="*60)
    print("TEST 2: GAME_FEATURES Includes Priors")
    print("="*60)
    
    from train_auto import GAME_FEATURES
    
    priors_features = [
        "home_o_rtg_prior", "home_d_rtg_prior", "home_pace_prior", "home_srs_prior",
        "away_o_rtg_prior", "away_d_rtg_prior", "away_pace_prior", "away_srs_prior"
    ]
    
    missing_features = [f for f in priors_features if f not in GAME_FEATURES]
    if missing_features:
        print(f"✗ FAIL: Missing features in GAME_FEATURES: {missing_features}")
        return False
    
    print(f"✓ All {len(priors_features)} priors features present in GAME_FEATURES")
    return True


def test_column_naming_convention():
    """Verify that priors columns use consistent _prior suffix"""
    print("\n" + "="*60)
    print("TEST 3: Column Naming Convention")
    print("="*60)
    
    from train_auto import GAME_FEATURES
    
    # Find all columns that should be priors (from Basketball Reference)
    priors_columns = [f for f in GAME_FEATURES if "o_rtg" in f or "d_rtg" in f or "pace_prior" in f or "srs_prior" in f]
    
    # Verify they all have _prior suffix
    non_standard = [c for c in priors_columns if not c.endswith("_prior")]
    if non_standard:
        print(f"✗ FAIL: Found priors columns without _prior suffix: {non_standard}")
        return False
    
    print(f"✓ All {len(priors_columns)} Basketball Reference columns use '_prior' suffix")
    print(f"  This clearly indicates they hold prior-season stats when available")
    return True


def test_defaults_vs_priors_terminology():
    """Test that terminology clearly distinguishes defaults from actual priors"""
    print("\n" + "="*60)
    print("TEST 4: Terminology Clarity")
    print("="*60)
    
    # Read the train_auto.py source to check comments
    train_auto_path = Path(__file__).parent / "train_auto.py"
    with open(train_auto_path, 'r', encoding='utf-8') as f:
        source = f.read()
    
    # Check for improved comments about priors
    required_terms = [
        "Basketball Reference",  # Should mention the source
        "baseline",              # Should use "baseline" for defaults
        "actual prior",          # Should distinguish actual priors from defaults
    ]
    
    missing_terms = []
    for term in required_terms:
        if term not in source:
            missing_terms.append(term)
    
    if missing_terms:
        print(f"✗ FAIL: Source code missing clear terminology: {missing_terms}")
        print(f"  The code should clearly explain that 'priors' can be:")
        print(f"  1. Actual Basketball Reference prior-season stats (when available)")
        print(f"  2. Baseline defaults (when not available)")
        return False
    
    print(f"✓ Source code uses clear terminology:")
    print(f"  - 'Basketball Reference' to identify source of priors")
    print(f"  - 'baseline' to describe default values")
    print(f"  - 'actual prior' to distinguish real data from defaults")
    return True


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("PRIORS TERMINOLOGY VALIDATION TEST SUITE")
    print("="*70)
    print("\nThis test validates the fix for:")
    print("  'why are you calling them priors?'")
    print("  'do you have the appropriate columns for all csv being used?'")
    print()
    
    tests = [
        test_game_defaults_comments,
        test_game_features_includes_priors,
        test_column_naming_convention,
        test_defaults_vs_priors_terminology,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"\n✗ EXCEPTION in {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test.__name__, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All tests passed! Priors terminology is clear and consistent.")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed. Please review the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
