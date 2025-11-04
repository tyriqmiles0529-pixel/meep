"""
Reset ledger to unsettled state and clear fetch progress
Use this if you want to re-fetch all predictions from scratch
"""

import pickle
import shutil
from pathlib import Path

print("="*70)
print("RESET FETCH CACHE")
print("="*70)

# Backup current ledger
ledger_file = Path("bets_ledger.pkl")
backup_file = Path("bets_ledger_before_reset.pkl")

if ledger_file.exists():
    shutil.copy(ledger_file, backup_file)
    print(f"\n✅ Backed up current ledger to: {backup_file}")

# Load ledger
with open(ledger_file, 'rb') as f:
    ledger_data = pickle.load(f)

bets = ledger_data['bets'] if isinstance(ledger_data, dict) else ledger_data

# Count current state
settled_count = sum(1 for bet in bets if bet.get('settled', False))
print(f"\nCurrent state:")
print(f"  Total predictions: {len(bets)}")
print(f"  Settled: {settled_count}")
print(f"  Unsettled: {len(bets) - settled_count}")

# Reset all to unsettled
for bet in bets:
    bet['settled'] = False
    if 'actual' in bet:
        del bet['actual']
    if 'won' in bet:
        del bet['won']

print(f"\n✅ Reset all {len(bets)} predictions to unsettled")

# Save reset ledger
if isinstance(ledger_data, dict):
    ledger_data['bets'] = bets
    updated_ledger = ledger_data
else:
    updated_ledger = bets

with open(ledger_file, 'wb') as f:
    pickle.dump(updated_ledger, f)

print(f"✅ Saved reset ledger")

# Delete fetch progress
progress_file = Path("fetch_progress.json")
if progress_file.exists():
    progress_file.unlink()
    print(f"✅ Deleted fetch progress file")

print("\n" + "="*70)
print("RESET COMPLETE")
print("="*70)
print("\nYou can now run fetch_bet_results_incremental.py with clean slate")
print("\nTo restore old ledger if needed:")
print(f"  copy {backup_file} {ledger_file}")
