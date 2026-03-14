#!/usr/bin/env python3
import zipfile
import os

print('Compressing PlayerStatistics.csv...')
with zipfile.ZipFile('PlayerStatistics.csv.zip', 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
    zf.write('PlayerStatistics.csv')

size_mb = os.path.getsize('PlayerStatistics.csv.zip') / 1024 / 1024
print(f'✅ Created PlayerStatistics.csv.zip: {size_mb:.1f} MB')
