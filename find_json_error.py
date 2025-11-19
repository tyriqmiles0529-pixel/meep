#!/usr/bin/env python3
"""Find exactly where JSON breaks by parsing incrementally"""
import json

with open('NBA_COLAB_SIMPLE.ipynb', 'r', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

# Try parsing progressively more lines to find where it breaks
for num_lines in range(1, len(lines) + 1):
    partial = ''.join(lines[:num_lines])
    try:
        # Try to parse - if it's incomplete but valid so far, it will fail with "Expecting value"
        json.loads(partial)
        print(f"Lines 1-{num_lines}: Valid complete JSON")
    except json.JSONDecodeError as e:
        if "Expecting" in str(e) and ("value" in str(e) or "property name" in str(e)):
            # Incomplete JSON is OK
            pass
        else:
            # This is a real syntax error
            print(f"Lines 1-{num_lines}: SYNTAX ERROR")
            print(f"  Error: {e}")
            print(f"  Line {e.lineno}, column {e.colno}")
            if num_lines >= e.lineno:
                print(f"  Problem line {e.lineno}: {repr(lines[e.lineno-1][:100])}")
            break
