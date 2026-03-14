#!/usr/bin/env python
"""
Add post-load year filter to reduce memory before feature engineering.
This allows loading full dataset but filtering to 2002+ before heavy processing.
"""

# Read the file
with open('train_auto.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace
old_code = '''                optimized_mb = agg_df.memory_usage(deep=True).sum() / 1024**2
                print(f"- Loaded {len(agg_df):,} rows")
                if min_year_filter:
                    print(f"- Filtered to {min_year_filter}+ during loading (reduced from {total_rows:,} rows)")
                print(f"- Memory after chunked optimization: {optimized_mb:.1f} MB ({optimized_mb/1024:.1f} GB)")

            except ImportError:'''

new_code = '''                optimized_mb = agg_df.memory_usage(deep=True).sum() / 1024**2
                print(f"- Loaded {len(agg_df):,} rows")
                if min_year_filter:
                    print(f"- Filtered to {min_year_filter}+ during loading (reduced from {total_rows:,} rows)")
                print(f"- Memory after chunked optimization: {optimized_mb:.1f} MB ({optimized_mb/1024:.1f} GB)")

                # POST-LOAD YEAR FILTER: Apply if not done during chunk loading
                if min_year_filter and len(agg_df) > 0:
                    year_col = None
                    for col_name in ['season', 'game_year', 'season_end_year', 'year']:
                        if col_name in agg_df.columns:
                            year_col = col_name
                            break

                    if year_col:
                        rows_before = len(agg_df)
                        agg_df = agg_df[agg_df[year_col] >= min_year_filter].copy()
                        rows_after = len(agg_df)
                        if rows_after < rows_before:
                            gc.collect()
                            new_mb = agg_df.memory_usage(deep=True).sum() / 1024**2
                            print(f"- POST-LOAD FILTER: {rows_before:,} -> {rows_after:,} rows (kept {min_year_filter}+)")
                            print(f"- Memory after year filter: {new_mb:.1f} MB ({new_mb/1024:.1f} GB)")

            except ImportError:'''

if old_code in content:
    content = content.replace(old_code, new_code)
    print("[OK] Added post-load year filter")
else:
    print("[SKIP] Could not find code to patch")

# Write back
with open('train_auto.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Patch applied. Use --min-year 2002 to filter after loading.")
