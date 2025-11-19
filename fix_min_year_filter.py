#!/usr/bin/env python
"""
Fix: Apply --min-year filter for Parquet files (not just CSV)
"""

with open('train_auto.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the condition to include Parquet files
old_code = '''        elif getattr(args, 'min_year', None) and not is_parquet:
            # Only filter here for CSV files (Parquet already filtered during chunked load)
            min_year = args.min_year'''

new_code = '''        elif getattr(args, 'min_year', None):
            # Filter by year for BOTH CSV and Parquet files
            min_year = args.min_year'''

if old_code in content:
    content = content.replace(old_code, new_code)
    print("[OK] Fixed: min-year filter now applies to Parquet files")
else:
    print("[SKIP] Could not find code to patch (different version?)")

with open('train_auto.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Now --min-year will filter Parquet data after loading")
