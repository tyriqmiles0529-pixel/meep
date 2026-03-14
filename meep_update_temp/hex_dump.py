#!/usr/bin/env python3
"""Hex dump around position 700"""

with open('NBA_COLAB_SIMPLE.ipynb', 'rb') as f:
    data = f.read()

print(f"Total file size: {len(data)} bytes")
print(f"\nBytes 680-720:")

for i in range(680, 720):
    byte = data[i]
    char = chr(byte) if 32 <= byte < 127 else '.'
    print(f"{i:4d}: 0x{byte:02x} ({byte:3d}) {repr(char)}")
