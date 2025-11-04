"""
Fetch ALL Bet Results - Multi-Run Script

This script runs fetch_bet_results.py multiple times with delays
to get around the 50 API call limit and fetch ALL predictions.

It will:
1. Run fetch_bet_results.py
2. Wait 60 seconds (NBA API rate limit reset)
3. Repeat until all predictions are fetched

Estimated time: 5-10 minutes for 1,500 predictions
"""

import subprocess
import time
from pathlib import Path
import pickle
import pandas as pd

print("=" * 70)
print("FETCH ALL PREDICTIONS - Multi-Run Script")
print("=" * 70)

# Check if fetch script exists
fetch_script = Path("fetch_bet_results.py")
if not fetch_script.exists():
    print("\nERROR: fetch_bet_results.py not found!")
    exit(1)

# Check ledger
ledger_file = Path("bets_ledger.pkl")
if not ledger_file.exists():
    print("\nERROR: bets_ledger.pkl not found!")
    exit(1)

def get_unsettled_count():
    """Get count of unsettled predictions"""
    try:
        with open(ledger_file, 'rb') as f:
            ledger_data = pickle.load(f)
        
        if isinstance(ledger_data, dict) and 'bets' in ledger_data:
            bets = ledger_data['bets']
        else:
            bets = ledger_data if isinstance(ledger_data, list) else [ledger_data]
        
        df = pd.DataFrame(bets)
        unsettled = (df['settled'] == False).sum()
        settled = (df['settled'] == True).sum()
        
        return unsettled, settled, len(bets)
    except Exception as e:
        print(f"Error reading ledger: {e}")
        return None, None, None

# Initial status
print("\nInitial Status:")
unsettled, settled, total = get_unsettled_count()
if unsettled is None:
    print("  Could not read ledger!")
    exit(1)

print(f"  Total predictions: {total:,}")
print(f"  Settled: {settled:,}")
print(f"  Unsettled: {unsettled:,}")

if unsettled == 0:
    print("\n✅ All predictions already settled!")
    exit(0)

# Calculate estimated runs needed
# Each run processes ~50 unique players, each player has ~3-4 predictions
# So each run settles ~150-200 predictions
estimated_runs = max(1, unsettled // 150)
estimated_time = estimated_runs * 1.5  # 1.5 min per run (including delay)

print(f"\nEstimated:")
print(f"  Runs needed: {estimated_runs}")
print(f"  Total time: ~{estimated_time:.1f} minutes")

input("\nPress ENTER to start fetching...")

# Run fetch multiple times
run_number = 1
max_runs = 20  # Safety limit

while run_number <= max_runs:
    print("\n" + "=" * 70)
    print(f"RUN {run_number}")
    print("=" * 70)
    
    # Check status before run
    unsettled_before, settled_before, total_before = get_unsettled_count()
    
    if unsettled_before == 0:
        print("✅ All predictions settled!")
        break
    
    print(f"Before run: {unsettled_before:,} unsettled predictions remaining")
    
    # Run fetch script
    try:
        print(f"\nRunning fetch_bet_results.py...")
        
        # Use the virtual environment Python
        python_exe = Path('.venv/Scripts/python.exe')
        if not python_exe.exists():
            python_exe = 'python'  # Fallback
        
        result = subprocess.run(
            [str(python_exe), 'fetch_bet_results.py'],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per run
        )
        
        # Show output
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:", result.stderr)
        
    except subprocess.TimeoutExpired:
        print("⚠️  Run timed out after 5 minutes")
    except Exception as e:
        print(f"❌ Error running fetch script: {e}")
        break
    
    # Check status after run
    unsettled_after, settled_after, total_after = get_unsettled_count()
    
    if unsettled_after is None:
        print("⚠️  Could not read ledger after run")
        break
    
    updates_made = settled_after - settled_before
    
    print(f"\n{'='*70}")
    print(f"RUN {run_number} RESULTS:")
    print(f"  Before: {settled_before:,} settled")
    print(f"  After:  {settled_after:,} settled")
    print(f"  Updates: +{updates_made:,} predictions")
    print(f"  Remaining: {unsettled_after:,} unsettled")
    print(f"{'='*70}")
    
    if updates_made == 0:
        print("\n⚠️  No new predictions updated. Possible reasons:")
        print("  - All games in ledger haven't been played yet")
        print("  - Players not found in NBA database")
        print("  - API issues")
        print("\nStopping fetch runs.")
        break
    
    if unsettled_after == 0:
        print("\n🎉 ALL PREDICTIONS FETCHED!")
        break
    
    # Check if we should continue
    if run_number < max_runs:
        print(f"\nWaiting 60 seconds before next run...")
        print(f"(Rate limit cooldown)")
        
        for i in range(60, 0, -10):
            print(f"  {i} seconds remaining...", end='\r')
            time.sleep(10)
        
        print("\n")
    
    run_number += 1

# Final summary
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

unsettled_final, settled_final, total_final = get_unsettled_count()

print(f"\nTotal predictions: {total_final:,}")
print(f"Settled: {settled_final:,} ({settled_final/total_final*100:.1f}%)")
print(f"Unsettled: {unsettled_final:,} ({unsettled_final/total_final*100:.1f}%)")

if settled_final > settled:
    improvement = settled_final - settled
    print(f"\n✅ Fetched {improvement:,} new results!")
    
    print(f"\nNext steps:")
    print(f"  1. python analyze_ledger.py  # Analyze performance")
    print(f"  2. python recalibrate_models.py  # Recalibrate with {settled_final} predictions")
else:
    print(f"\n⚠️  No new results fetched")
    print(f"\nPossible reasons:")
    print(f"  - Games haven't been played yet")
    print(f"  - Players not in NBA database")
    print(f"  - All available results already fetched")

print("\n" + "=" * 70)
print("COMPLETE")
print("=" * 70)
