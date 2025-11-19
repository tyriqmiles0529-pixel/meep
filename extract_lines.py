#!/usr/bin/env python3
"""Extract problem lines to a file"""

with open('NBA_COLAB_SIMPLE.ipynb', 'r', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

with open('lines_15_20.txt', 'w', encoding='utf-8') as out:
    for i in range(14, 20):
        out.write(f"=== LINE {i+1} ===\n")
        out.write(lines[i])
        out.write("\n")

print("Written to lines_15_20.txt")