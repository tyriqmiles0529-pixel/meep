#!/usr/bin/env python3
"""Quick test to verify API keys are loaded correctly"""
import sys

print("🔑 Testing API Key Setup")
print("=" * 60)

# Test API-Sports key
print("\n1. Testing API-Sports.io key...")
try:
    from api_keys import API_SPORTS_KEY
    if API_SPORTS_KEY and len(API_SPORTS_KEY) > 10:
        print(f"   ✅ API-Sports key loaded: {API_SPORTS_KEY[:10]}...{API_SPORTS_KEY[-4:]}")
    else:
        print("   ❌ API-Sports key is empty or invalid")
except ImportError:
    print("   ❌ Could not import from api_keys.py")
    print("   Make sure api_keys.py exists in this directory")

# Test Kaggle key
print("\n2. Testing Kaggle key...")
try:
    from api_keys import KAGGLE_KEY, KAGGLE_USERNAME
    if KAGGLE_KEY and len(KAGGLE_KEY) > 10:
        print(f"   ✅ Kaggle key loaded: {KAGGLE_KEY[:10]}...{KAGGLE_KEY[-4:]}")
        if KAGGLE_USERNAME:
            print(f"   ✅ Kaggle username: {KAGGLE_USERNAME}")
        else:
            print("   ⚠️  Kaggle username not set (optional)")
    else:
        print("   ❌ Kaggle key is empty or invalid")
except ImportError:
    print("   ❌ Could not import from api_keys.py")

# Test NBA analyzer import
print("\n3. Testing NBA analyzer can load keys...")
try:
    # Simulate what nba_prop_analyzer_fixed.py does
    from api_keys import API_SPORTS_KEY as TEST_KEY
    print(f"   ✅ NBA analyzer will use: {TEST_KEY[:10]}...{TEST_KEY[-4:]}")
except ImportError:
    print("   ❌ NBA analyzer won't be able to load keys")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("✅ If all checks passed, you're ready to run:")
print("   python nba_prop_analyzer_fixed.py")
print("   python train_auto.py")
print("\n❌ If any checks failed, edit api_keys.py and add your keys")
print("=" * 60)
