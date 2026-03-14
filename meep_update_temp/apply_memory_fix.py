#!/usr/bin/env python
"""Apply memory optimization fix"""

with open('train_auto.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the line with "Created temporary window CSV"
for i, line in enumerate(lines):
    if 'print(f"  • Created temporary window CSV from aggregated data:' in line:
        # Replace the block (lines around this)
        # Look for window_df = agg_df... above and del window_df below
        start_idx = i - 4  # Should be around here
        end_idx = i + 2    # del window_df

        # Find exact start (window_df = agg_df...)
        for j in range(i-1, max(0, i-10), -1):
            if 'window_df = agg_df[agg_df[player_year_col].isin(padded_seasons)].copy()' in lines[j]:
                start_idx = j
                break

        # Find exact end (del window_df)
        for j in range(i+1, min(len(lines), i+5)):
            if 'del window_df' in lines[j]:
                end_idx = j + 1
                break

        # New optimized code
        indent = '                '
        new_lines = [
            f'{indent}window_df = agg_df[agg_df[player_year_col].isin(padded_seasons)].copy()\n',
            f'{indent}print(f"  • Filtered aggregated data for window: {{len(window_df):,}} rows")\n',
            f'\n',
            f'{indent}# Memory optimization: Keep only essential columns\n',
            f'{indent}essential_cols = [\n',
            f'{indent}    \'personId\', \'gameId\', \'gameDate\', \'firstName\', \'lastName\',\n',
            f'{indent}    \'home\', \'numMinutes\', \'points\', \'assists\', \'blocks\', \'steals\',\n',
            f'{indent}    \'reboundsTotal\', \'threePointersMade\', \'threePointersAttempted\',\n',
            f'{indent}    \'fieldGoalsMade\', \'fieldGoalsAttempted\', \'freeThrowsMade\',\n',
            f'{indent}    \'freeThrowsAttempted\', player_year_col\n',
            f'{indent}]\n',
            f'{indent}adv_cols = [c for c in window_df.columns if c.startswith(\'adv_\')][:10]\n',
            f'{indent}essential_cols.extend(adv_cols)\n',
            f'{indent}cols_to_keep = [c for c in essential_cols if c in window_df.columns]\n',
            f'{indent}window_df = window_df[cols_to_keep].copy()\n',
            f'\n',
            f'{indent}temp_player_csv = Path(f".window_agg_{{start_year}}_{{end_year}}_players.csv")\n',
            f'{indent}window_df.to_csv(temp_player_csv, index=False)\n',
            f'{indent}player_data_path = temp_player_csv\n',
            f'{indent}print(f"  • Optimized CSV: {{len(window_df):,}} rows, {{len(cols_to_keep)}} columns")\n',
            f'{indent}del window_df\n',
            f'{indent}gc.collect()\n',
        ]

        # Replace
        lines = lines[:start_idx] + new_lines + lines[end_idx:]
        print(f"[OK] Applied memory optimization at line {i+1}")
        break
else:
    print("[SKIP] Could not find target line")

with open('train_auto.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)
