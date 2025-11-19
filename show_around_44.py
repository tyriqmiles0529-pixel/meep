#!/usr/bin/env python3

with open('NBA_COLAB_SIMPLE.ipynb', 'r', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

print("Lines 40-48:")
for i in range(39, 48):
    print(f"{i+1:3d}: {repr(lines[i].rstrip())}")