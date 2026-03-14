#!/usr/bin/env python3
"""Remove emoji from notebook to make it Colab-compatible"""
import json

# Load the notebook
with open('NBA_COLAB_SIMPLE.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Replace emoji in all cells
replacements = [
    ('🏀', '[NBA]'),
    ('✅', '[x]'),
    ('→', '->'),
    ('❌', '[X]'),
    ('📊', '[DATA]'),
    ('🚀', '*'),
    ('📝', '*'),
    ('📈', '*'),
    ('⚡', '*'),
    ('🎯', '*'),
    ('⏱️', '[TIME]'),
    ('⏱', '[TIME]'),
    ('📦', '*'),
    ('⬇️', 'v'),
    ('⬇', 'v'),
    ('🔧', '*'),
    ('⚙️', '*'),
    ('⚙', '*'),
    ('➕', '+'),
    ('🔍', '*'),
    ('💡', '*'),
    ('🏃', '*'),
]

cells_modified = 0
for cell in notebook['cells']:
    if 'source' in cell:
        # source can be a string or list of strings
        if isinstance(cell['source'], str):
            # Convert string to list
            cell['source'] = [cell['source']]

        # Now process as list
        modified = False
        new_source = []
        for line in cell['source']:
            original = line
            for old, new in replacements:
                line = line.replace(old, new)
            new_source.append(line)
            if line != original:
                modified = True

        cell['source'] = new_source
        if modified:
            cells_modified += 1

print(f"Modified {cells_modified} cells")

# Save the cleaned notebook
with open('NBA_COLAB_SIMPLE.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("Saved cleaned notebook!")

# Verify it's still valid
with open('NBA_COLAB_SIMPLE.ipynb', 'r', encoding='utf-8') as f:
    test = json.load(f)
print(f"Verified: {len(test['cells'])} cells")
