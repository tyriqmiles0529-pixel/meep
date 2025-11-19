#!/usr/bin/env python3
"""Diagnose the exact issue"""

with open('NBA_COLAB_SIMPLE.ipynb', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

# Replace emoji with simple chars first
replacements = {
    '→': '->',
    '✅': '[OK]',
    '❌': '[X]',
    '📊': '[DATA]',
    '🚀': '*',
    '📝': '*',
    '📈': '[GRAPH]',
    '⚡': '*',
    '🎯': '*',
    '🏀': '*',
    '⏱️': '[TIMER]',
    '📦': '*',
    '⬇️': '[DOWN]',
    '🔧': '*',
    '⚙️': '*',
    '➕': '+',
    '🔍': '*',
    '💡': '*',
    '🏃': '[RUNNING]',
}

for old_char, new_char in replacements.items():
    content = content.replace(old_char, new_char)

# Look at the specific area around position 706
print("Characters around position 700-730:")
for i in range(690, 730):
    char = content[i]
    print(f"  {i}: {repr(char)} {ord(char):04x}")
