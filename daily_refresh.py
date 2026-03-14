import sys
from pathlib import Path
import subprocess
from fetch_season_data import fetch_season
from update_feature_matrix import update_dataset

def run_cleanup():
    """Run the cleanup_master script to ensure identity column consistency."""
    print(">>> [REFRESH] Cleaning up master data headers...")
    try:
        subprocess.run([sys.executable, "cleanup_master.py"], check=True)
    except Exception as e:
        print(f">>> [WARN] Cleanup failed (can usually be ignored): {e}")

from datetime import datetime, timedelta

def daily_refresh():
    print("\n" + "="*60)
    print("NBA PREDICTOR - AUTOMATED DAILY REFRESH")
    print("="*60)
    
    season = "2025-26"
    master = "final_feature_matrix_with_per_min_1997_onward.csv"
    
    # 1. Fetch latest logs from NBA API
    print(f"\n[1/3] Fetching latest logs for {season}...")
    daily_file = fetch_season(season)
    
    if not daily_file:
        print(">>> [FALLBACK] NBA API stalled. Attempting ESPN Bridge Sync...")
        try:
            from scripts.espn_sync import run_sync
            # Sync last 14 days as a safety net to cover any gaps
            end_date = datetime.now()
            start_date = end_date - timedelta(days=14)
            run_sync(start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"))
            print(">>> [SUCCESS] Fallback ESPN Sync completed.")
            # We don't return False here because run_sync handles the update_dataset part
            # But for the sake of the rest of the script, we need a skip flag
            daily_file = "ALREADY_UPDATED"
        except Exception as e:
            print(f">>> [ERR] Initial fetch and Fallback both failed: {e}")
            return False
            
    # 2. Update master feature matrix
    if daily_file != "ALREADY_UPDATED":
        print(f"\n[2/3] Updating master feature matrix: {master}")
        try:
            update_dataset(daily_file, master)
        except Exception as e:
            print(f">>> [ERR] Failed to update feature matrix: {e}")
            return False
        
    # 3. Final Header Cleanup
    print("\n[3/4] Running final identity standardization...")
    run_cleanup()
    
    # 4. Update Player Archetypes
    print("\n[4/4] Updating player style archetypes...")
    try:
        subprocess.run([sys.executable, "generate_archetypes.py"], check=True)
    except Exception as e:
        print(f">>> [WARN] Archetype update failed: {e}")
    
    print("\n" + "="*60)
    print("REFRESH COMPLETE: Data is now up to date.")
    print("You can now run: python predict_live_FINAL.py --betting")
    print("="*60 + "\n")
    return True

if __name__ == "__main__":
    daily_refresh()
