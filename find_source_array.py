#!/usr/bin/env python3
"""Find where the 'source' array should close"""

with open('NBA_COLAB_SIMPLE.ipynb', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find all lines with array markers
print("Array structure (first 50 lines):")
bracket_depth = 0
for i in range(min(50, len(lines))):
    line = lines[i].rstrip()

    # Calculate bracket depth before this line
    open_before = line.count('[')
    close_before = line.count(']')

    # Show the line
    indent_marker = "  " * bracket_depth
    print(f"{i+1:3d} [{bracket_depth}]: {indent_marker}{line[:70]}")

    # Update depth
    bracket_depth += open_before - close_before

print(f"\nFinal bracket depth: {bracket_depth}")
