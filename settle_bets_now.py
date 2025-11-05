#!/usr/bin/env python3
"""
Quick Bet Settlement - Run While Training in Cloud

Settles all unsettled bets using NBA API.
Safe to run while training happens in Colab.
"""

import os
import sys
import pickle
import time
from datetime import datetime

print("\n" + "="*80)
print("SETTLING PREVIOUS PREDICTIONS".center(80))
print("="*80)

# Check if ledger exists
if not os.path.exists('bets_ledger.pkl'):
    print("\n❌ No bets ledger found!")
    print("   Run riq_analyzer.py first to create predictions.")
    sys.exit(1)

# Load ledger
with open('bets_ledger.pkl', 'rb') as f:
    ledger = pickle.load(f)

bets = ledger.get('bets', [])
unsettled = [b for b in bets if not b.get('settled', False)]

if not unsettled:
    print("\n✅ All bets already settled!")
    sys.exit(0)

print(f"\n📊 Bets Summary:")
print(f"   Total: {len(bets):,}")
print(f"   Settled: {len(bets) - len(unsettled):,}")
print(f"   Unsettled: {len(unsettled):,}")

print(f"\n🔄 Starting settlement process...")
print(f"   This will take ~{len(unsettled) * 2} seconds (rate limited)")

# Import evaluation script
try:
    from evaluate import fetch_actual_results
except ImportError:
    print("\n❌ Could not import evaluate.py")
    print("   Make sure evaluate.py is in the same folder")
    sys.exit(1)

# Settle bets
print(f"\n📥 Fetching results from NBA API...")
settled_count = fetch_actual_results(verbose=True, max_fetches=len(unsettled))

# Reload ledger to see results
with open('bets_ledger.pkl', 'rb') as f:
    ledger = pickle.load(f)

bets = ledger.get('bets', [])
remaining_unsettled = [b for b in bets if not b.get('settled', False)]

print(f"\n" + "="*80)
print(f"✅ SETTLEMENT COMPLETE")
print(f"="*80)
print(f"\n📊 Results:")
print(f"   Settled this run: {settled_count:,}")
print(f"   Remaining unsettled: {len(remaining_unsettled):,}")

# Calculate win rate for newly settled
if settled_count > 0:
    settled_this_run = [b for b in bets if b.get('settled') and b.get('result') in ['win', 'loss']][-settled_count:]
    wins = sum(1 for b in settled_this_run if b['result'] == 'win')
    if settled_this_run:
        win_rate = wins / len(settled_this_run) * 100
        print(f"\n🎯 Performance (newly settled):")
        print(f"   Win rate: {win_rate:.1f}%")
        print(f"   Wins: {wins}")
        print(f"   Losses: {len(settled_this_run) - wins}")

# Next steps
if remaining_unsettled:
    print(f"\n💡 {len(remaining_unsettled)} bets still unsettled (games not played yet)")
    print(f"   Run this script again after those games finish")
else:
    print(f"\n🎉 All bets settled! Run analyze_ledger.py for full analysis")

print(f"\n" + "="*80 + "\n")
