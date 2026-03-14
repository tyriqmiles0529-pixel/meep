#!/usr/bin/env python3
"""Show context around line 18"""
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('NBA_COLAB_SIMPLE.ipynb', 'r', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

print("Lines 1-25:")
for i in range(0, min(25, len(lines))):
    line = lines[i].rstrip('\n')
    # Show line number, length, and content
    print(f"{i+1:3d} ({len(line):3d}): {repr(line)}")