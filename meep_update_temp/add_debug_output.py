#!/usr/bin/env python
"""Add debug output to show parsed arguments"""

with open('train_auto.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find line with "args = ap.parse_args()"
for i, line in enumerate(lines):
    if 'args = ap.parse_args()' in line and i < len(lines) - 1:
        # Insert debug output after this line
        indent = '    '  # 4 spaces
        debug_lines = [
            f'\n',
            f'{indent}# DEBUG: Show critical arguments to verify they were passed\n',
            f'{indent}print("=" * 70)\n',
            f'{indent}print("PARSED ARGUMENTS:")\n',
            f'{indent}print(f"  --min-year: {{args.min_year}}")\n',
            f'{indent}print(f"  --aggregated-data: {{args.aggregated_data}}")\n',
            f'{indent}print(f"  --add-rolling-features: {{args.add_rolling_features}}")\n',
            f'{indent}print("=" * 70)\n',
        ]

        # Insert after the parse_args line
        for j, debug_line in enumerate(debug_lines):
            lines.insert(i + 1 + j, debug_line)

        print(f"[OK] Added debug output after line {i+1}")
        break
else:
    print("[SKIP] Could not find parse_args line")

with open('train_auto.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)
