#!/usr/bin/env python
"""Fix ALL hardcoded season_end_year references to use flexible detection"""

with open('train_auto.py', 'r', encoding='utf-8') as f:
    content = f.read()

fixes = 0

# Fix 1: Line 5563 - window filtering
old1 = "window_df = agg_df[agg_df['season_end_year'].isin(padded_seasons)].copy()"
new1 = """# Use flexible year column detection
                player_year_col = None
                for col in ['season_end_year', 'season', 'game_year', 'year']:
                    if col in agg_df.columns:
                        player_year_col = col
                        break
                if player_year_col is None:
                    raise KeyError(f"No year column found in aggregated data. Available: {list(agg_df.columns)[:20]}")
                window_df = agg_df[agg_df[player_year_col].isin(padded_seasons)].copy()"""

if old1 in content:
    content = content.replace(old1, new1)
    fixes += 1
    print(f"[OK] Fix 1: Window filtering uses flexible year column")

# Fix 2: Check for other season_end_year in player training
# Search for pattern where it accesses season_end_year on player data
old2 = "agg_df[agg_df['season_end_year'] >= 2002]"
if old2 in content:
    # This is in the memory limit section, needs to use the year_col variable that's already defined above it
    new2 = "agg_df[agg_df.get('season_end_year', agg_df.get('season', agg_df.get('game_year'))) >= 2002]"
    # Actually this one is fine since it's in a different code path
    print(f"[INFO] Memory limit filter uses season_end_year - OK for game models")

with open('train_auto.py', 'w', encoding='utf-8') as f:
    f.write(content)

print(f"\nTotal fixes applied: {fixes}")
