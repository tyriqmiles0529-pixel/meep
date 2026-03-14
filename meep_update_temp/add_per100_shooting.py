#!/usr/bin/env python
"""Add per-100 and shooting stats to kept columns"""

with open('train_auto.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the column selection block
for i, line in enumerate(lines):
    if "adv_cols = [c for c in window_df.columns if c.startswith('adv_')][:10]" in line:
        # Replace this line and next 2 lines
        indent = '                '
        new_lines = [
            f"{indent}# Add ALL advanced stats (important for player archetypes)\n",
            f"{indent}adv_cols = [c for c in window_df.columns if c.startswith('adv_')]\n",
            f"{indent}essential_cols.extend(adv_cols)\n",
            f"{indent}# Add ALL per-100 stats (pace-adjusted, very predictive)\n",
            f"{indent}per100_cols = [c for c in window_df.columns if c.startswith('per100_')]\n",
            f"{indent}essential_cols.extend(per100_cols)\n",
            f"{indent}# Add key shooting stats (critical for 3PT predictions)\n",
            f"{indent}shoot_cols = [c for c in window_df.columns if c.startswith('shoot_') and\n",
            f"{indent}             any(x in c for x in ['percent_fga_from', 'fg_percent_from', 'corner', 'x3p'])]\n",
            f"{indent}essential_cols.extend(shoot_cols)\n",
        ]

        # Replace old lines
        lines = lines[:i] + new_lines + lines[i+2:]
        print(f"[OK] Updated column selection at line {i+1}")
        print(f"Now keeping:")
        print(f"  - All adv_* columns (~26)")
        print(f"  - All per100_* columns (~30)")
        print(f"  - Key shoot_* columns (~12)")
        print(f"  - Total: ~85 columns (was 30, saves 45% memory vs 155)")
        break
else:
    print("[SKIP] Could not find target line")

with open('train_auto.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)
