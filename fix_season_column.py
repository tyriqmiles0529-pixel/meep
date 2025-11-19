#!/usr/bin/env python
"""Fix: Use 'season' column instead of 'season_end_year' for player models"""

with open('train_auto.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace all instances of season_end_year with flexible column detection
old_code = '''        all_seasons = sorted([int(s) for s in seasons_df["season_end_year"].dropna().unique()])
        max_year = int(seasons_df["season_end_year"].max())'''

new_code = '''        # Find the year column (could be season_end_year, season, or game_year)
        year_col = None
        for col_name in ['season_end_year', 'season', 'game_year', 'year']:
            if col_name in seasons_df.columns:
                year_col = col_name
                break
        if year_col is None:
            raise KeyError(f"No year column found. Available: {list(seasons_df.columns)}")

        all_seasons = sorted([int(s) for s in seasons_df[year_col].dropna().unique()])
        max_year = int(seasons_df[year_col].max())'''

if old_code in content:
    content = content.replace(old_code, new_code)
    print("[OK] Fixed season column detection in player model training")
else:
    print("[SKIP] Could not find exact code block")

with open('train_auto.py', 'w', encoding='utf-8') as f:
    f.write(content)
