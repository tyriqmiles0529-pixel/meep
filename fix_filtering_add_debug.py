#!/usr/bin/env python
"""
Add debugging to show which arguments were received
"""

with open('train_auto.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add debug output after argument parsing
old_code = '''    args = ap.parse_args()

    main()'''

new_code = '''    args = ap.parse_args()

    # DEBUG: Show all parsed arguments
    print("=" * 70)
    print("PARSED ARGUMENTS:")
    print("=" * 70)
    for key, value in vars(args).items():
        if value is not None and value is not False:
            print(f"  --{key.replace('_', '-')}: {value}")
    print("=" * 70)
    print()

    main()'''

if old_code in content:
    content = content.replace(old_code, new_code)
    print("[OK] Added argument debugging")
else:
    print("[SKIP] Could not find parse_args location")

with open('train_auto.py', 'w', encoding='utf-8') as f:
    f.write(content)
