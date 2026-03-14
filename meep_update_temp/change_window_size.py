#!/usr/bin/env python
"""
Change player model window size from 5 years to 3 years.

This reduces memory usage per window:
- 5-year windows: ~2M rows each
- 3-year windows: ~1.2M rows each (40% reduction)
"""

import sys

# Get desired window size from command line or default to 3
new_window_size = int(sys.argv[1]) if len(sys.argv) > 1 else 3

with open('train_auto.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace both occurrences
changes = 0

# First occurrence (around line 4730 - game models)
old1 = '            window_size = 5'
new1 = f'            window_size = {new_window_size}'
if old1 in content:
    content = content.replace(old1, new1, 1)  # Only first occurrence
    changes += 1
    print(f"[OK] Changed game model window size to {new_window_size}")

# Second occurrence (around line 5510 - player models)
if content.count('window_size = 5') > 0:
    content = content.replace('        window_size = 5', f'        window_size = {new_window_size}')
    changes += 1
    print(f"[OK] Changed player model window size to {new_window_size}")

with open('train_auto.py', 'w', encoding='utf-8') as f:
    f.write(content)

print(f"\nTotal changes: {changes}")
print(f"Window size: 5 years → {new_window_size} years")
print(f"Memory per window: ~2M rows → ~{2 * new_window_size / 5:.1f}M rows")
print(f"Number of windows: ~5 → ~{25 // new_window_size + 1}")
