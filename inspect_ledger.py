"""
Inspect bets_ledger.pkl structure
"""
import pickle
import pprint
from pathlib import Path

ledger_file = Path("bets_ledger.pkl")

if ledger_file.exists():
    print("=" * 70)
    print("BETS LEDGER INSPECTION")
    print("=" * 70)
    
    with open(ledger_file, 'rb') as f:
        ledger = pickle.load(f)
    
    print(f"\nLedger type: {type(ledger)}")
    
    if hasattr(ledger, '__len__'):
        print(f"Total entries: {len(ledger)}")
    
    if isinstance(ledger, dict):
        print(f"\nKeys: {list(ledger.keys())[:10]}")
        print("\nSample entries:")
        for i, (key, value) in enumerate(list(ledger.items())[:3]):
            print(f"\n{i+1}. Key: {key}")
            print(f"   Value type: {type(value)}")
            if isinstance(value, dict):
                print("   Fields:")
                pprint.pprint(value, indent=4)
            else:
                print(f"   Value: {value}")
    
    elif isinstance(ledger, list):
        print(f"\nSample entries:")
        for i, entry in enumerate(ledger[:3]):
            print(f"\n{i+1}. Entry type: {type(entry)}")
            if isinstance(entry, dict):
                print("   Fields:")
                pprint.pprint(entry, indent=4)
            else:
                print(f"   Entry: {entry}")
    
    else:
        print(f"\nLedger structure:")
        pprint.pprint(ledger)
        
else:
    print("bets_ledger.pkl not found")
