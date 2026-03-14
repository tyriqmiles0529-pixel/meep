#!/usr/bin/env python3
"""
Clear All Caches - Fresh Start

Clears model cache and training data to force fresh download.
Safe to run while training happens in cloud.
"""

import os
import shutil
from pathlib import Path

print("\n" + "="*80)
print("CLEARING CACHES".center(80))
print("="*80)

# Directories to clear
cache_dirs = [
    'model_cache',
    'data',
    '__pycache__',
    'priors_data'
]

# Files to clear (optional - comment out if you want to keep)
cache_files = [
    # 'bets_ledger.pkl',  # Uncomment to clear bet history
    # 'bets_ledger_backup.pkl',
]

total_size = 0
cleared_items = []

# Clear directories
for cache_dir in cache_dirs:
    if os.path.exists(cache_dir):
        # Calculate size
        dir_size = sum(f.stat().st_size for f in Path(cache_dir).rglob('*') if f.is_file())
        total_size += dir_size
        
        # Remove directory
        shutil.rmtree(cache_dir)
        cleared_items.append(f"📁 {cache_dir}/ ({dir_size / 1e6:.1f} MB)")
        print(f"✅ Cleared: {cache_dir}/ ({dir_size / 1e6:.1f} MB)")

# Clear files
for cache_file in cache_files:
    if os.path.exists(cache_file):
        file_size = os.path.getsize(cache_file)
        total_size += file_size
        
        os.remove(cache_file)
        cleared_items.append(f"📄 {cache_file} ({file_size / 1e6:.1f} MB)")
        print(f"✅ Cleared: {cache_file} ({file_size / 1e6:.1f} MB)")

print(f"\n" + "="*80)
print(f"✅ CACHE CLEARING COMPLETE")
print(f"="*80)
print(f"\n📊 Summary:")
print(f"   Items cleared: {len(cleared_items)}")
print(f"   Space freed: {total_size / 1e6:.1f} MB ({total_size / 1e9:.2f} GB)")

if cleared_items:
    print(f"\n📋 Cleared:")
    for item in cleared_items:
        print(f"   {item}")
else:
    print(f"\n💡 No caches found (already clean)")

print(f"\n💡 Next steps:")
print(f"   1. Training in cloud will download fresh data")
print(f"   2. Models will be trained from scratch")
print(f"   3. Download trained models from cloud")
print(f"   4. Run predictions normally")

print(f"\n" + "="*80 + "\n")
