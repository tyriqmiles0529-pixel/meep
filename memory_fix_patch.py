#!/usr/bin/env python
"""
Patch script to fix the memory issue in train_auto.py

The problem: ~190 columns with many duplicate string columns consuming 10+GB
The solution: Drop redundant columns during loading

Apply this patch by running:
    python memory_fix_patch.py
"""

import re

# Read the original file
with open('train_auto.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Define the old code block to replace
old_code = '''                print(f"- Total rows in file: {total_rows:,}")
                print(f"- Parquet file has {num_row_groups} row groups")

                # Group row groups into 5 chunks
                num_chunks = 5'''

# Define the new code with column selection
new_code = '''                print(f"- Total rows in file: {total_rows:,}")
                print(f"- Parquet file has {num_row_groups} row groups")

                # CRITICAL: Select only essential columns (saves 10+ GB RAM!)
                all_columns = [field.name for field in parquet_file.schema_arrow]

                # Drop duplicate/redundant columns that bloat memory
                columns_to_drop = set()
                for col in all_columns:
                    # Drop duplicate columns from merged Basketball Reference tables
                    if '_dup' in col:  # player_per100_dup, team_shoot_dup, etc.
                        columns_to_drop.add(col)
                    # Drop redundant league columns (all "NBA")
                    elif col in ['adv_lg', 'per100_lg', 'shoot_lg', 'pbp_lg']:
                        columns_to_drop.add(col)
                    # Drop duplicate name/id columns (keep firstName, lastName, personId)
                    elif col in ['player', 'player_name', 'player_id']:
                        columns_to_drop.add(col)
                    # Drop redundant age/pos/game columns (keep from adv_ table only)
                    elif col in ['per100_age', 'shoot_age', 'pbp_age',
                                 'per100_pos', 'shoot_pos', 'pbp_pos',
                                 'per100_g', 'per100_gs', 'per100_mp',
                                 'shoot_g', 'shoot_gs', 'shoot_mp',
                                 'pbp_g', 'pbp_gs', 'pbp_mp']:
                        columns_to_drop.add(col)

                columns_to_load = [c for c in all_columns if c not in columns_to_drop]

                dropped_count = len(columns_to_drop)
                print(f"- MEMORY OPTIMIZATION: Dropping {dropped_count} redundant columns")
                if dropped_count > 0:
                    print(f"    Examples: {list(columns_to_drop)[:8]}...")
                print(f"- Loading {len(columns_to_load)} essential columns (was {len(all_columns)})")

                # Group row groups into 5 chunks
                num_chunks = 5'''

# Replace
if old_code in content:
    content = content.replace(old_code, new_code)
    print("[OK] Patched: Added column selection logic")
else:
    print("[SKIP] Could not find the code block to patch (already patched?)")

# Also patch the read_row_groups call to use columns parameter
old_read = '                        chunk = parquet_file.read_row_groups(rg_indices).to_pandas()'
new_read = '                        # Read only specific row groups AND specific columns (memory efficient!)\n                        chunk = parquet_file.read_row_groups(rg_indices, columns=columns_to_load).to_pandas()'

if old_read in content:
    content = content.replace(old_read, new_read)
    print("[OK] Patched: read_row_groups now uses columns parameter")
else:
    print("[SKIP] Could not patch read_row_groups (already patched?)")

# Write the patched file
with open('train_auto.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("\nPatch complete! The train_auto.py file has been updated.")
print("Expected memory reduction: ~10GB (from dropping redundant string columns)")
