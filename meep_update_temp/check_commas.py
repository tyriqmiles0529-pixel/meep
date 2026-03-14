#!/usr/bin/env python3
"""Check for missing commas"""

with open('NBA_COLAB_SIMPLE.ipynb', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print("Lines 10-25 - checking for trailing commas:")
for i in range(9, 25):
    line = lines[i]
    stripped = line.rstrip('\n\r')
    last_char = stripped[-1] if stripped else ''

    marker = ""
    if last_char not in [',', '{', '[']:
        marker = f" <-- MISSING COMMA? (ends with {repr(last_char)})"

    print(f"{i+1:3d}: {repr(stripped[-50:])}{marker}")
