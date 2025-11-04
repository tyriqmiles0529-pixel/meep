"""
Fix indentation in train_auto.py player training section
The code is indented too far - needs to be dedented to align properly
"""

with open('train_auto.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Process lines 4110-4555
# Target: Make the player training code align with 4-space base indent inside main()
fixed_lines = []

for i, line in enumerate(lines, 1):
    if 4110 <= i <= 4555:
        # Count leading spaces
        stripped = line.lstrip(' ')
        if not stripped or stripped.startswith('\n'):
            # Empty or whitespace-only line
            fixed_lines.append(line)
            continue

        leading_spaces = len(line) - len(stripped)

        # Determine correct indentation based on content
        # Base level inside "if players_path and players_path.exists():" = 8 spaces

        if 'print(_sec(' in line or '# Prepare current season' in line:
            # First level inside the if block = 8 spaces
            fixed_lines.append(' ' * 8 + stripped)
        elif 'current_player_df = fetch_current' in line:
            # Also 8 spaces
            fixed_lines.append(' ' * 8 + stripped)
        elif 'if current_player_df is not None' in line:
            # 8 spaces
            fixed_lines.append(' ' * 8 + stripped)
        elif 'else:' in line and i in [4141, 4551]:
            # else at 8 spaces
            fixed_lines.append(' ' * 8 + stripped)
        elif stripped.startswith('temp_player_csv') or stripped.startswith('hist_players') or stripped.startswith('combined_players') or stripped.startswith('player_data_path'):
            # Inside if block = 12 spaces
            fixed_lines.append(' ' * 12 + stripped)
        elif stripped.startswith('print(f"- Added'):
            # Inside if block = 12 spaces
            fixed_lines.append(' ' * 12 + stripped)
        elif 'for idx, window_info in enumerate' in line:
            # for loop = 8 spaces
            fixed_lines.append(' ' * 8 + stripped)
        elif 'window_seasons = set' in line or 'start_year = ' in line or 'end_year = ' in line or 'cache_path = ' in line or 'is_current = ' in line:
            # Inside for loop = 12 spaces
            fixed_lines.append(' ' * 12 + stripped)
        elif 'if os.path.exists(cache_path)' in line:
            # if inside for = 12 spaces
            fixed_lines.append(' ' * 12 + stripped)
        elif 'continue' in line and i > 4180 and i < 4190:
            # continue inside if = 16 spaces
            fixed_lines.append(' ' * 16 + stripped)
        elif 'context_window = ' in line or 'oof_window = ' in line or 'padded_seasons = ' in line or 'priors_window = ' in line:
            # Window filtering = 12 spaces
            fixed_lines.append(' ' * 12 + stripped)
        elif 'frames = build_players_from_playerstats' in line:
            # 12 spaces
            fixed_lines.append(' ' * 12 + stripped)
        elif stripped.startswith('player_data_path,') or stripped.startswith('context_window,') or stripped.startswith('verbose=') or stripped.startswith('priors_players=') or stripped.startswith('window_seasons='):
            # Function call continuation = 16 spaces
            fixed_lines.append(' ' * 16 + stripped)
        elif 'if not historical_player_props.empty' in line or 'window_games = games_df' in line:
            # 12 spaces
            fixed_lines.append(' ' * 12 + stripped)
        elif 'if "date" in window_games' in line or 'min_date = ' in line or 'max_date = ' in line or 'window_props = historical' in line:
            # Nested if = 16 spaces
            fixed_lines.append(' ' * 16 + stripped)
        elif 'for stat_name, stat_df in frames.items' in line:
            # for inside if = 16 spaces (or 20 depending on nesting)
            # Let's check context - if it's deeply nested, use 20
            if i > 4290:
                fixed_lines.append(' ' * 20 + stripped)
            else:
                fixed_lines.append(' ' * 16 + stripped)
        elif stripped.startswith('if stat_df is None') or stripped.startswith('prop_type_map = ') or stripped.startswith('prop_type = '):
            # Inside for loop = 20 spaces
            fixed_lines.append(' ' * 20 + stripped)
        elif stripped.startswith("'points':") or stripped.startswith("'rebounds':") or stripped.startswith("'assists':") or stripped.startswith("'threes':") or stripped.startswith("'minutes':"):
            # Dict items = 24 spaces
            fixed_lines.append(' ' * 24 + stripped)
        elif 'minutes_model, m_metrics = ' in line or 'points_model, points_sigma' in line or 'rebounds_model' in line or 'assists_model' in line or 'threes_model' in line:
            # Training calls = 12 spaces
            fixed_lines.append(' ' * 12 + stripped)
        elif 'window_models = {' in line:
            # 12 spaces
            fixed_lines.append(' ' * 12 + stripped)
        elif "'minutes':" in line and 'minutes_model' in line:
            # Dict items = 16 spaces
            fixed_lines.append(' ' * 16 + stripped)
        elif 'with open(cache_path' in line or 'with open(cache_meta' in line:
            # 12 spaces
            fixed_lines.append(' ' * 12 + stripped)
        elif 'pickle.dump(' in line and i > 4430:
            # Inside with = 16 spaces
            fixed_lines.append(' ' * 16 + stripped)
        elif 'meta = {' in line and i > 4440:
            # 12 spaces
            fixed_lines.append(' ' * 12 + stripped)
        elif "'seasons':" in line or "'start_year':" in line or "'end_year':" in line or "'trained_date':" in line:
            # Dict items = 16 spaces
            fixed_lines.append(' ' * 16 + stripped)
        elif 'json.dump(' in line:
            # 16 spaces
            fixed_lines.append(' ' * 16 + stripped)
        elif 'del context_window' in line or 'del minutes_model' in line or 'gc.collect()' in line:
            # 12 spaces
            fixed_lines.append(' ' * 12 + stripped)
        elif 'latest_window = max' in line or 'latest_cache = ' in line:
            # 8 spaces
            fixed_lines.append(' ' * 8 + stripped)
        elif 'if os.path.exists(latest_cache)' in line:
            # 8 spaces
            fixed_lines.append(' ' * 8 + stripped)
        elif 'with open(latest_cache' in line:
            # 12 spaces
            fixed_lines.append(' ' * 12 + stripped)
        elif 'latest_models = pickle.load' in line:
            # 16 spaces
            fixed_lines.append(' ' * 16 + stripped)
        elif 'for stat_name in [' in line and i > 4500:
            # 12 spaces
            fixed_lines.append(' ' * 12 + stripped)
        elif 'if stat_name in latest_models' in line and i > 4500:
            # 16 spaces
            fixed_lines.append(' ' * 16 + stripped)
        elif 'model_path = models_dir' in line or 'sigma_model_path = ' in line:
            # 20 spaces
            fixed_lines.append(' ' * 20 + stripped)
        elif "if 'temp_player_csv' in locals" in line or "if 'temp_combined_csv' in locals" in line:
            # 8 spaces
            fixed_lines.append(' ' * 8 + stripped)
        elif 'temp_player_csv.unlink' in line or 'temp_combined_csv.unlink' in line:
            # 12 spaces
            fixed_lines.append(' ' * 12 + stripped)
        elif 'player_metrics = latest_models' in line:
            # 12 spaces
            fixed_lines.append(' ' * 12 + stripped)
        else:
            # Default: try to preserve relative indentation
            # If currently at 20+ spaces, probably nested deep, use 16
            if leading_spaces >= 20:
                fixed_lines.append(' ' * 16 + stripped)
            elif leading_spaces >= 16:
                fixed_lines.append(' ' * 12 + stripped)
            elif leading_spaces >= 12:
                fixed_lines.append(' ' * 8 + stripped)
            else:
                fixed_lines.append(line)
    else:
        fixed_lines.append(line)

# Write back
with open('train_auto.py', 'w', encoding='utf-8') as f:
    f.writelines(fixed_lines)

print("[OK] Fixed indentation in train_auto.py")
print("Run: python -m py_compile train_auto.py")
