#!/usr/bin/env python3
"""Fix NBA_COLAB_SIMPLE.ipynb by replacing problematic Unicode characters"""

import json
import sys

# Read the raw file
with open('NBA_COLAB_SIMPLE.ipynb', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

# Replace problematic Unicode characters with safe ASCII equivalents
# → (U+2192) with ->
# ✅ (U+2705) with [OK]
# ❌ (U+274C) with [X]
# 📊 (U+1F4CA) with [CHART]
# 🚀 (U+1F680) with [ROCKET]
# 📝 (U+1F4DD) with [NOTE]
# 📈 (U+1F4C8) with [GRAPH]
# ⚡ (U+26A1) with [LIGHTNING]
# 🎯 (U+1F3AF) with [TARGET]
# 🏀 (U+1F3C0) with [BASKETBALL]
# ⏱️ (U+23F1) with [TIMER]
# 📦 (U+1F4E6) with [PACKAGE]
# ⬇️ (U+2B07) with [DOWN]
# 🔧 (U+1F527) with [WRENCH]
# ⚙️ (U+2699) with [GEAR]

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

# Now try to parse as JSON to verify it's valid
try:
    data = json.loads(content)
    print("✓ JSON is now valid!")
    print(f"✓ Found {len(data.get('cells', []))} cells")

    # Write the fixed version
    with open('NBA_COLAB_SIMPLE.ipynb', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=1, ensure_ascii=False)

    print("✓ Fixed notebook saved!")

except json.JSONDecodeError as e:
    print(f"[X] JSON still invalid: {e}")
    print(f"  At line {e.lineno}, column {e.colno}")
    print(f"  Position {e.pos}")

    # Show the problematic area
    lines = content.split('\n')
    if e.lineno <= len(lines):
        print(f"\n  Problem line: {lines[e.lineno-1][:100]}")

    # Show exact characters around the error position
    print(f"\n  Around position {e.pos}:")
    start = max(0, e.pos - 50)
    end = min(len(content), e.pos + 50)
    print(f"  {repr(content[start:end])}")
