#!/usr/bin/env python3
"""Fix malformed JSON in NBA_COLAB_SIMPLE.ipynb"""

import re
import sys

# Set UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

# Read the file as raw text
with open('NBA_COLAB_SIMPLE.ipynb', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

# Find the malformed line around position 688
# The issue is likely a string that wasn't properly closed or has invalid escape sequences
print("Content around position 680-700:")
problem_area = content[680:700]
print(f"Text: {problem_area}")
print(f"Repr: {repr(problem_area)}")
print()

# Show lines 15-20
lines = content.split('\n')
print("Lines around line 18:")
for i in range(15, 22):
    if i < len(lines):
        line = lines[i]
        print(f"Line {i+1:2d} ({len(line):3d} chars): {line[:120]}")
