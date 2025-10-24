#!/usr/bin/env python3
"""
Simple source code validation for priors terminology improvements.
This test doesn't import train_auto.py, just reads it as text to validate changes.
"""

from pathlib import Path

def validate_terminology():
    """Check that the source code has clear terminology about priors"""
    print("="*70)
    print("VALIDATING PRIORS TERMINOLOGY IN SOURCE CODE")
    print("="*70)
    
    train_auto_path = Path(__file__).parent / "train_auto.py"
    with open(train_auto_path, 'r', encoding='utf-8') as f:
        source = f.read()
    
    checks = {
        "Basketball Reference priors mentioned": "Basketball Reference prior-season stats" in source,
        "Baseline defaults explained": "baseline defaults" in source,
        "Actual priors distinguished": "actual prior" in source,
        "League-average baseline noted": "league-average" in source,
        "teamTricode requirement documented": "teamTricode" in source,
        "Merge failure scenarios explained": "merge fails" in source,
        "Clear diagnostic for missing priors": "NO games have actual team priors" in source,
        "Root cause identification": "Root cause" in source,
        "User-friendly guidance": "Solution:" in source or "To enable priors:" in source,
    }
    
    print("\nTERMINOLOGY CHECKS:")
    passed = 0
    for check_name, check_result in checks.items():
        status = "✓" if check_result else "✗"
        print(f"  {status} {check_name}")
        if check_result:
            passed += 1
    
    print(f"\nRESULT: {passed}/{len(checks)} checks passed")
    
    if passed == len(checks):
        print("\n✅ SUCCESS: All terminology improvements are present!")
        print("\nThe code now clearly explains that '_prior' columns:")
        print("  1. Hold actual Basketball Reference prior-season stats when available")
        print("  2. Use baseline defaults (110.0 o/d_rtg, 100.0 pace, 0.0 srs) when not")
        print("  3. Require teamTricode column in TeamStatistics.csv to merge")
        return True
    else:
        print(f"\n❌ FAILED: {len(checks) - passed} checks did not pass")
        return False


def validate_defaults_values():
    """Check that GAME_DEFAULTS has correct baseline values for priors"""
    print("\n" + "="*70)
    print("VALIDATING GAME_DEFAULTS BASELINE VALUES")
    print("="*70)
    
    train_auto_path = Path(__file__).parent / "train_auto.py"
    with open(train_auto_path, 'r', encoding='utf-8') as f:
        source = f.read()
    
    # Extract GAME_DEFAULTS section
    start = source.find("GAME_DEFAULTS: Dict[str, float] = {")
    end = source.find("}", start) + 1
    
    if start == -1 or end == -1:
        print("✗ Could not find GAME_DEFAULTS dictionary")
        return False
    
    defaults_section = source[start:end]
    
    expected_priors = {
        '"home_o_rtg_prior": 110.0': "home offensive rating baseline",
        '"home_d_rtg_prior": 110.0': "home defensive rating baseline",
        '"home_pace_prior": 100.0': "home pace baseline",
        '"away_o_rtg_prior": 110.0': "away offensive rating baseline",
        '"away_d_rtg_prior": 110.0': "away defensive rating baseline",
        '"away_pace_prior": 100.0': "away pace baseline",
        '"home_srs_prior": 0.0': "home SRS baseline",
        '"away_srs_prior": 0.0': "away SRS baseline",
    }
    
    print("\nBASELINE VALUE CHECKS:")
    passed = 0
    for expected, description in expected_priors.items():
        if expected in defaults_section:
            print(f"  ✓ {description}: {expected.split(':')[1].strip()}")
            passed += 1
        else:
            print(f"  ✗ {description}: NOT FOUND")
    
    print(f"\nRESULT: {passed}/{len(expected_priors)} baseline values correct")
    
    return passed == len(expected_priors)


def validate_diagnostic_output():
    """Check that diagnostic output is helpful and clear"""
    print("\n" + "="*70)
    print("VALIDATING DIAGNOSTIC OUTPUT")
    print("="*70)
    
    train_auto_path = Path(__file__).parent / "train_auto.py"
    with open(train_auto_path, 'r', encoding='utf-8') as f:
        source = f.read()
    
    diagnostic_checks = {
        "Emoji indicator for priors section": "🏀 TEAM PRIORS" in source,
        "Explains what columns contain": "These columns hold actual prior-season stats" in source,
        "Shows when defaults are used": "or baseline defaults" in source,
        "Lists specific default values": "110.0 o/d_rtg, 100.0 pace, 0.0 srs" in source,
        "Clear warning when no priors": "NO games have actual team priors" in source,
        "Identifies root cause": "Missing 'teamTricode' column" in source,
        "Provides solution": "Ensure TeamStatistics.csv has a 'teamTricode' column" in source,
        "Checks for abbreviations": "No team abbreviations available" in source,
    }
    
    print("\nDIAGNOSTIC OUTPUT CHECKS:")
    passed = 0
    for check_name, check_result in diagnostic_checks.items():
        status = "✓" if check_result else "✗"
        print(f"  {status} {check_name}")
        if check_result:
            passed += 1
    
    print(f"\nRESULT: {passed}/{len(diagnostic_checks)} diagnostic improvements present")
    
    return passed == len(diagnostic_checks)


def main():
    print("\n" + "="*70)
    print("PRIORS TERMINOLOGY FIX VALIDATION")
    print("="*70)
    print("\nThis validates the fix for the issue:")
    print("  'why are you calling them priors?'")
    print("  'do you have the appropiate columns for all csv being used?'")
    print()
    
    results = []
    
    # Run validation checks
    results.append(("Terminology", validate_terminology()))
    results.append(("Baseline Values", validate_defaults_values()))
    results.append(("Diagnostic Output", validate_diagnostic_output()))
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    if passed == total:
        print(f"\n✅ All {total} validation checks passed!")
        print("\nThe fix successfully addresses the issue by:")
        print("  1. Clarifying that '_prior' columns hold actual priors OR baseline defaults")
        print("  2. Documenting when each is used (based on teamTricode availability)")
        print("  3. Providing clear diagnostics to identify why priors aren't merging")
        print("  4. Offering concrete solutions to fix missing priors")
        return 0
    else:
        print(f"\n❌ {total - passed} validation check(s) failed")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
