"""
Quick test to verify riq_analyzer feature count fix
"""
import pickle
import pandas as pd
from riq_analyzer import build_player_features, build_minutes_features

# Load models
points_model = pickle.load(open('models/points_model.pkl', 'rb'))
minutes_model = pickle.load(open('models/minutes_model.pkl', 'rb'))

# Expected feature counts
points_expected = points_model.n_features_
minutes_expected = minutes_model.n_features_

print("=" * 70)
print("FEATURE COUNT VERIFICATION")
print("=" * 70)

# Create dummy dataframes
df_empty = pd.DataFrame()

# Test points model features
print("\n1. POINTS MODEL")
print(f"   Expected features: {points_expected}")
try:
    feats = build_player_features(df_empty, df_empty)
    print(f"   Generated features: {len(feats.columns)}")
    print(f"   Status: {'✅ MATCH' if len(feats.columns) == points_expected else '❌ MISMATCH'}")
    if len(feats.columns) != points_expected:
        print(f"   Difference: {len(feats.columns) - points_expected}")
except Exception as e:
    print(f"   ❌ ERROR: {e}")

# Test minutes model features
print("\n2. MINUTES MODEL")
print(f"   Expected features: {minutes_expected}")
try:
    feats = build_minutes_features(df_empty, df_empty)
    print(f"   Generated features: {len(feats.columns)}")
    print(f"   Status: {'✅ MATCH' if len(feats.columns) == minutes_expected else '❌ MISMATCH'}")
    if len(feats.columns) != minutes_expected:
        print(f"   Difference: {len(feats.columns) - minutes_expected}")
except Exception as e:
    print(f"   ❌ ERROR: {e}")

# Test prediction
print("\n3. PREDICTION TEST")
try:
    feats_points = build_player_features(df_empty, df_empty)
    pred = points_model.predict(feats_points)
    print(f"   Points prediction: {pred[0]:.2f}")
    print(f"   Status: ✅ SUCCESS")
except Exception as e:
    print(f"   ❌ ERROR: {e}")

try:
    feats_mins = build_minutes_features(df_empty, df_empty)
    pred = minutes_model.predict(feats_mins)
    print(f"   Minutes prediction: {pred[0]:.2f}")
    print(f"   Status: ✅ SUCCESS")
except Exception as e:
    print(f"   ❌ ERROR: {e}")

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)
