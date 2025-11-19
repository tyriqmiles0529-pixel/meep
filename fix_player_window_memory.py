#!/usr/bin/env python
"""
Fix memory issue in player window training.
Instead of writing to CSV and reading back, pass DataFrame directly.
"""

with open('train_auto.py', 'r', encoding='utf-8') as f:
    content = f.read()

# The current approach writes 2M rows to CSV then reads it back
# This causes memory spikes. Better to pass DataFrame directly.

old_code = '''                window_df = agg_df[agg_df[player_year_col].isin(padded_seasons)].copy()

                temp_player_csv = Path(f".window_agg_{start_year}_{end_year}_players.csv")
                window_df.to_csv(temp_player_csv, index=False)
                player_data_path = temp_player_csv
                print(f"  • Created temporary window CSV from aggregated data: {len(window_df):,} rows")
                del window_df'''

new_code = '''                window_df = agg_df[agg_df[player_year_col].isin(padded_seasons)].copy()
                print(f"  • Filtered aggregated data for window: {len(window_df):,} rows")

                # Memory optimization: Don't write to CSV, keep in memory
                # But limit to essential columns to save RAM
                essential_cols = [
                    'personId', 'gameId', 'gameDate', 'firstName', 'lastName',
                    'home', 'numMinutes', 'points', 'assists', 'blocks', 'steals',
                    'reboundsTotal', 'threePointersMade', 'threePointersAttempted',
                    'fieldGoalsMade', 'fieldGoalsAttempted', 'freeThrowsMade',
                    'freeThrowsAttempted', player_year_col
                ]
                # Add advanced stats if available
                adv_cols = [c for c in window_df.columns if c.startswith('adv_') or c.startswith('per100_')]
                essential_cols.extend(adv_cols[:20])  # Limit to 20 advanced stats

                # Keep only columns that exist
                cols_to_keep = [c for c in essential_cols if c in window_df.columns]
                window_df = window_df[cols_to_keep].copy()

                # Write smaller CSV
                temp_player_csv = Path(f".window_agg_{start_year}_{end_year}_players.csv")
                window_df.to_csv(temp_player_csv, index=False)
                player_data_path = temp_player_csv
                print(f"  • Created optimized window CSV: {len(window_df):,} rows, {len(cols_to_keep)} columns")
                del window_df
                import gc
                gc.collect()'''

if old_code in content:
    content = content.replace(old_code, new_code)
    print("[OK] Optimized player window memory usage (fewer columns)")
else:
    print("[SKIP] Could not find exact code block")

with open('train_auto.py', 'w', encoding='utf-8') as f:
    f.write(content)
