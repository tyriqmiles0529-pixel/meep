"""
Quick script to fix the indentation in train_auto.py lines 4110-4550
The player training section has incorrect indentation (too many spaces)
"""

with open('train_auto.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Fix lines 4110-4550: dedent by 12 spaces (3 levels)
fixed_lines = []
for i, line in enumerate(lines, 1):
    if 4110 <= i <= 4550:
        # If line starts with at least 12 spaces of indentation beyond base, dedent by 12
        if line.startswith('                    '):  # 20 spaces
            fixed_lines.append('        ' + line[20:])  # Keep 8 spaces
        elif line.startswith('                '):  # 16 spaces
            fixed_lines.append('    ' + line[16:])  # Keep 4 spaces
        elif line.startswith('            '):  # 12 spaces
            fixed_lines.append(line[12:])  # Remove 12 spaces
        else:
            fixed_lines.append(line)
    else:
        fixed_lines.append(line)

# Write back
with open('train_auto.py', 'w', encoding='utf-8') as f:
    f.writelines(fixed_lines)

print("[OK] Fixed indentation in train_auto.py (lines 4110-4550)")
