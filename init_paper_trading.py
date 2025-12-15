from paper_ledger import PaperLedger
import os

# Remove old ledger and create new one with $25 starting bankroll
if os.path.exists('paper_ledger.json'):
    os.remove('paper_ledger.json')

ledger = PaperLedger(initial_bankroll=25.0)
ledger.save()

print("=== PAPER TRADING INITIALIZED ===")
print(f"Starting Bankroll: $25.00")
print(f"Ledger File: paper_ledger.json")
print()
print("Run paper trading daily with:")
print("  python3 run_paper_trading.py")
