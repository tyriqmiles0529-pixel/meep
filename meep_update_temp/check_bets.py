import pickle
import os

if os.path.exists('bets_ledger.pkl'):
    with open('bets_ledger.pkl', 'rb') as f:
        ledger = pickle.load(f)
    
    bets = ledger.get('bets', [])
    unsettled = [b for b in bets if not b.get('settled', False)]
    settled = [b for b in bets if b.get('settled', False)]
    
    print(f"Total bets: {len(bets)}")
    print(f"Unsettled: {len(unsettled)}")
    print(f"Settled: {len(settled)}")
    
    if unsettled:
        print(f"\nOldest unsettled bet: {unsettled[0].get('date', 'unknown')}")
        print(f"Newest unsettled bet: {unsettled[-1].get('date', 'unknown')}")
else:
    print("No bets ledger found")
