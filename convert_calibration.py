"""
Convert IsotonicRegression calibrators to riq_analyzer bin format
"""

import pickle
import numpy as np
from pathlib import Path

print("Converting calibration curves to riq_analyzer format...")

# Load IsotonicRegression calibrators
with open('model_cache/calibration_curves.pkl', 'rb') as f:
    calibrators = pickle.load(f)

# Convert to bin format
calibration_data = {}

for stat_type, iso_regressor in calibrators.items():
    print(f"\nProcessing {stat_type.upper()}...")
    
    # Get the X and Y points from IsotonicRegression
    X_thresholds = iso_regressor.X_thresholds_
    y_thresholds = iso_regressor.y_thresholds_
    
    # Create bins (X values) and vals (calibrated probabilities)
    bins = X_thresholds.tolist()
    vals = y_thresholds.tolist()
    
    # Add boundary points for better interpolation
    if bins[0] > 0.01:
        bins.insert(0, 0.01)
        vals.insert(0, vals[0])
    if bins[-1] < 0.99:
        bins.append(0.99)
        vals.append(vals[-1])
    
    calibration_data[stat_type] = {
        "bins": bins,
        "vals": vals
    }
    
    print(f"  Created {len(bins)} calibration points")
    print(f"  Sample mappings:")
    test_probs = [0.55, 0.65, 0.75, 0.85, 0.95]
    for p in test_probs:
        calibrated = iso_regressor.predict([p])[0]
        print(f"    {p*100:.0f}% → {calibrated*100:.1f}%")

# Save in riq_analyzer format
output_file = Path("calibration.pkl")
with open(output_file, 'wb') as f:
    pickle.dump(calibration_data, f)

print(f"\n✅ Calibration data saved to: {output_file}")
print(f"   Stats calibrated: {', '.join(calibration_data.keys())}")
print("\n✅ riq_analyzer.py will now use these calibrations automatically!")
