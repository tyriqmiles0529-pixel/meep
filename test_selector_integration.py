"""
Test Enhanced Selector Integration in riq_analyzer.py

This script simulates a prediction to verify the enhanced selector is working.
"""

import sys
import os

# Enable debug mode
os.environ['DEBUG_MODE'] = '1'

# Import after setting DEBUG_MODE
from riq_analyzer import ModelPredictor
import pandas as pd
import numpy as np

print("=" * 70)
print("ENHANCED SELECTOR INTEGRATION TEST")
print("=" * 70)

# Initialize model predictor
print("\n1. Loading models...")
MODEL = ModelPredictor()

# Check if selector loaded
print(f"\n2. Enhanced selector loaded: {MODEL.enhanced_selector is not None}")
if MODEL.enhanced_selector:
    print(f"   Stats available: {list(MODEL.enhanced_selector.keys())}")
    print(f"   Windows loaded: {len(MODEL.selector_windows)}")
    if MODEL.selector_windows:
        print(f"   Window names: {list(MODEL.selector_windows.keys())}")
else:
    print("   ❌ Enhanced selector NOT loaded!")
    print("   Make sure model_cache/dynamic_selector_enhanced.pkl exists")
    sys.exit(1)

# Create test player history (simulating Stephen Curry's last 10 games)
print("\n3. Creating test player history...")
test_history = pd.DataFrame({
    'points': [28.5, 30.2, 25.8, 32.1, 27.4, 29.8, 26.3, 31.5, 28.9, 27.2],
    'assists': [6.5, 7.2, 5.8, 6.9, 7.5, 6.1, 5.9, 7.8, 6.4, 7.0],
    'rebounds': [5.2, 4.8, 6.1, 5.5, 4.9, 5.7, 5.3, 4.6, 5.9, 5.1],
    'threes': [4.0, 5.0, 3.0, 6.0, 4.0, 5.0, 3.0, 5.0, 4.0, 4.0],
    'minutes': [34.2, 35.8, 32.5, 36.1, 33.7, 35.2, 33.9, 36.5, 34.8, 35.1]
})

print(f"   Created {len(test_history)} games of test data")
print(f"   Columns: {list(test_history.columns)}")
print(f"   Points average: {test_history['points'].mean():.1f}")

# Create dummy features (not used by selector, but needed for LightGBM fallback)
from riq_analyzer import build_player_features
feats = build_player_features(pd.DataFrame(), test_history)

# Test prediction with enhanced selector
print("\n4. Testing enhanced selector prediction...")
print("-" * 70)

for stat in ['points', 'assists', 'rebounds', 'threes']:
    print(f"\n{stat.upper()}:")
    
    # This should trigger the enhanced selector
    prediction = MODEL.predict_with_ensemble(stat, feats, player_history=test_history)
    
    if prediction is not None:
        print(f"   ✅ Enhanced selector returned: {prediction:.2f}")
        print(f"   Baseline (simple average): {test_history[stat].mean():.2f}")
        print(f"   Difference: {prediction - test_history[stat].mean():+.2f}")
    else:
        print(f"   ❌ Enhanced selector failed, falling back to LightGBM...")
        # Try LightGBM fallback
        lgb_pred = MODEL.predict(stat, feats)
        if lgb_pred:
            print(f"   LightGBM prediction: {lgb_pred:.2f}")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)

print("\nExpected output:")
print("  ✅ Should see '🎯 SELECTOR: [window name]' for each stat")
print("  ✅ Should see '✅ ENHANCED PREDICTION: [value]' for each stat")
print("  ✅ Predictions should differ from simple averages")
print("\nIf you see '❌' errors, check the debug messages above.")
print("Common issues:")
print("  - Selector file not found")
print("  - Window ensembles not loaded")
print("  - Player history format incorrect")
