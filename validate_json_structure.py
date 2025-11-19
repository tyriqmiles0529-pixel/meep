#!/usr/bin/env python3
"""Validate JSON structure line by line"""
import json

with open('NBA_COLAB_SIMPLE.ipynb', 'r', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

# Check each line for proper string closing
print("Checking each line for unclosed strings:")
for i in range(0, 20):
    line = lines[i].strip()
    if line.startswith('"') and not line.endswith(','):
        if not line.endswith('}') and not line.endswith(']'):
            print(f"Line {i+1}: Missing comma? {repr(line[:50])}")

# Try parsing lines 1-17
test_json = ''.join(lines[:17])
print("\n=== Testing lines 1-17 ===")
print(repr(test_json[-100:]))

#Try to identify the specific issue
import re
# Look for unclosed quotes
quote_pattern = r'"[^"\\]*(?:\\.[^"\\]*)*"'
for i in range(0, 20):
    line = lines[i].rstrip()
    # Count quotes
    quotes = line.count('"')
    escaped_quotes = line.count('\\"')
    actual_quotes = quotes - escaped_quotes

    if actual_quotes % 2 != 0:
        print(f"Line {i+1}: ODD number of quotes ({actual_quotes}) - {repr(line[:60])}")
