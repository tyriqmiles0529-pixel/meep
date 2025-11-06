#!/usr/bin/env python3
"""
Quick test to verify the type fix will work for .isin() filtering.
"""

import pandas as pd
import numpy as np

print("Testing type compatibility for .isin() filtering fix\n")
print("="*70)

# Simulate the original bug
print("1. ORIGINAL BUG (float64 season vs int set):")
test_series_float = pd.Series([2007.0, 2008.0, 2009.0, 2010.0, 2011.0])
test_set_int = {2007, 2008, 2009, 2010, 2011}

result_buggy = test_series_float.isin(test_set_int)
print(f"   Series dtype: {test_series_float.dtype}")
print(f"   Set type: {type(list(test_set_int)[0])}")
print(f"   .isin() result: {result_buggy.tolist()}")
print(f"   Matched: {result_buggy.sum()} / {len(test_series_float)}")

# Test the fix with Int64
print("\n2. FIX ATTEMPT #1 (Int64 season vs int set):")
test_series_int64 = test_series_float.astype('Int64')
print(f"   Series dtype: {test_series_int64.dtype}")
print(f"   Set type: {type(list(test_set_int)[0])}")
result_fixed = test_series_int64.isin(test_set_int)
print(f"   .isin() result: {result_fixed.tolist()}")
print(f"   Matched: {result_fixed.sum()} / {len(test_series_int64)}")

# Test with regular int conversion
print("\n3. FIX ATTEMPT #2 (int64 season vs int set):")
test_series_int = test_series_float.astype(int)
print(f"   Series dtype: {test_series_int.dtype}")
print(f"   Set type: {type(list(test_set_int)[0])}")
result_fixed2 = test_series_int.isin(test_set_int)
print(f"   .isin() result: {result_fixed2.tolist()}")
print(f"   Matched: {result_fixed2.sum()} / {len(test_series_int)}")

# Test with NaN values (realistic scenario)
print("\n4. REALISTIC TEST (with NaN values):")
test_series_with_nan = pd.Series([2007.0, np.nan, 2009.0, 2010.0, np.nan])
test_series_with_nan_fixed = test_series_with_nan.astype('Int64')
print(f"   Original: {test_series_with_nan.tolist()}")
print(f"   After Int64: {test_series_with_nan_fixed.tolist()}")
print(f"   Dtype: {test_series_with_nan_fixed.dtype}")
result_with_nan = test_series_with_nan_fixed.isin(test_set_int)
print(f"   .isin() result: {result_with_nan.tolist()}")
print(f"   Matched: {result_with_nan.sum()} / {len(test_series_with_nan_fixed)}")

print("\n" + "="*70)
print("VERDICT:")
print("="*70)
if result_fixed.sum() == len(test_series_int64):
    print("\n✅ FIX WORKS! Int64 dtype successfully matches int set")
    print("   The .astype('Int64') conversion resolves the type mismatch")
else:
    print("\n❌ FIX FAILED! Int64 dtype does not match int set")
    print("   Need to try a different approach")

if result_with_nan.sum() == 3:  # Should match 3 non-NaN values
    print("✅ FIX HANDLES NaN! Nullable Int64 preserves NaN values correctly")
else:
    print("❌ FIX BREAKS WITH NaN! Nullable Int64 has issues")

print("="*70)
