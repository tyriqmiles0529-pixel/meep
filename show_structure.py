#!/usr/bin/env python3
"""Show the structure around line 17"""

with open('NBA_COLAB_SIMPLE.ipynb', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print("Lines 14-22 with special markers:")
for i in range(13, 22):
    line = lines[i].rstrip('\n')
    indent = len(line) - len(line.lstrip())
    # Count brackets
    opens = line.count('[') + line.count('{')
    closes = line.count(']') + line.count('}')

    marker = ""
    if ']' in line:
        marker = " <-- CLOSES ARRAY"
    if '[' in line:
        marker = " <-- OPENS ARRAY"

    print(f"{i+1:3d} (indent={indent}, [{opens} ]{closes}): {line[:80]}{marker}")
