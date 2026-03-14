#!/usr/bin/env python3
"""Check if lines 1-17 are valid JSON"""
import json

with open('NBA_COLAB_SIMPLE.ipynb', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Try just the first 17 lines
partial_17 = ''.join(lines[:17])
print("Trying first 17 lines...")
try:
    json.loads(partial_17 + ']}}')  # Close the array and objects
    print("Lines 1-17: Valid (when properly closed)")
except json.JSONDecodeError as e:
    print(f"Lines 1-17: Invalid - {e}")

# Now try with line 18
partial_18 = ''.join(lines[:18])
print("\nTrying first 18 lines...")
try:
    json.loads(partial_18 + ']}}')  # Close the array and objects
    print("Lines 1-18: Valid (when properly closed)")
except json.JSONDecodeError as e:
    print(f"Lines 1-18: Invalid - {e}")
    print(f"  Line {e.lineno}, col {e.colno}")

# Show line 18 with character positions
print("\nLine 18 with positions:")
line18 = lines[17]
for i, char in enumerate(line18[:50]):
    print(f"{i:2d}: {repr(char)}")
