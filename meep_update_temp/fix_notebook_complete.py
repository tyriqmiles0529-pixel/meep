#!/usr/bin/env python3
"""Complete fix for NBA_COLAB_SIMPLE.ipynb - remove problematic Unicode"""

# Read raw bytes and decode
with open('NBA_COLAB_SIMPLE.ipynb', 'rb') as f:
    raw_bytes = f.read()

# Decode as UTF-8
content = raw_bytes.decode('utf-8', errors='replace')

# Replace all problematic Unicode with ASCII equivalents
replacements = [
    ('→', '->'),
    ('✅', '[x]'),
    ('❌', '[X]'),
    ('📊', '*'),
    ('🚀', '*'),
    ('📝', '*'),
    ('📈', '*'),
    ('⚡', '*'),
    ('🎯', '*'),
    ('🏀', '*'),
    ('⏱️', '*'),
    ('⏱', '*'),
    ('📦', '*'),
    ('⬇️', '*'),
    ('⬇', '*'),
    ('🔧', '*'),
    ('⚙️', '*'),
    ('⚙', '*'),
    ('➕', '+'),
    ('🔍', '*'),
    ('💡', '*'),
    ('🏃', '*'),
    ('\ufeff', ''),  # BOM
]

for old, new in replacements:
    content = content.replace(old, new)

# Write back as UTF-8
with open('NBA_COLAB_SIMPLE.ipynb', 'w', encoding='utf-8', newline='\n') as f:
    f.write(content)

# Verify it's valid JSON now
import json
try:
    with open('NBA_COLAB_SIMPLE.ipynb', 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"SUCCESS! Notebook is now valid JSON with {len(data.get('cells', []))} cells")

    # Re-write with proper formatting
    with open('NBA_COLAB_SIMPLE.ipynb', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=1, ensure_ascii=False)

    print("Notebook has been fixed and reformatted!")

except json.JSONDecodeError as e:
    print(f"ERROR: Still invalid JSON")
    print(f"  {e}")
    print(f"  Line {e.lineno}, column {e.colno}, position {e.pos}")

    # Show the error area
    lines = content.split('\n')
    if e.lineno-1 < len(lines):
        print(f"  Problem line: {lines[e.lineno-1][:100]}")
