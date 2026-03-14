#!/usr/bin/env python3
"""Fix line 43 - add missing \\n"""

with open('NBA_COLAB_SIMPLE.ipynb', 'r', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

# Fix line 43 (index 42)
print(f"Line 43 BEFORE: {repr(lines[42])}")

# Line 43 should end with \\n"
if lines[42].rstrip('\n\r').endswith('"'):
    # Add the missing \\n before the closing quote
    lines[42] = lines[42].rstrip('\n\r')[:-1] + '\\\\n"\n'

print(f"Line 43 AFTER:  {repr(lines[42])}")

# Write back
with open('NBA_COLAB_SIMPLE.ipynb', 'w', encoding='utf-8', newline='') as f:
    f.writelines(lines)

# Verify
import json
try:
    with open('NBA_COLAB_SIMPLE.ipynb', 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"\nSUCCESS! Notebook is now valid with {len(data.get('cells', []))} cells")
except json.JSONDecodeError as e:
    print(f"\nStill invalid: {e}")
