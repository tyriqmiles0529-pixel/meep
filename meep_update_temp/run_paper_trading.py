import argparse
import subprocess
import os
from datetime import datetime, timedelta
import pandas as pd

def run_command(cmd):
    print(f"Running: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        print(f"Error executing command: {cmd}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yesterday', type=str, help="Date for results (YYYY-MM-DD). Defaults to actual yesterday.")
    parser.add_argument('--today', type=str, help="Date for predictions (YYYY-MM-DD). Defaults to actual today.")
    parser.add_argument('--skip_update', action='store_true', help="Skip fetching/pdating yesterday's data")
    args = parser.parse_args()
    
    # Dates
    if args.yesterday:
        date_yst = args.yesterday
    else:
        date_yst = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
    if args.today:
        date_today = args.today
    else:
        date_today = datetime.now().strftime('%Y-%m-%d')
        
    print(f"--- Paper Trading Automation ---")
    print(f"Results Date: {date_yst}")
    print(f"Prediction Date: {date_today}")
    
    # 1. Update Data (Yesterday's Results)
    if not args.skip_update:
        print("\n[Step 1] Fetching Completed Games (Yesterday)...")
        if not run_command(f"python3 fetch_new_games.py --date {date_yst}"): return
        
        file_yst = f"daily_games_{date_yst}.csv"
        if os.path.exists(file_yst):
            print("\n[Step 2] Updating Feature Matrix...")
            if not run_command(f"python3 update_feature_matrix.py --daily {file_yst}"): return
        else:
            print(f"Warning: {file_yst} not found. Maybe no games yesterday?")
            
    # 2. Resolve Bets
    print("\n[Step 3] Resolving Pending Bets...")
    # 'betting_strategy.py --resolve' uses the master file (now updated)
    if not run_command("python3 betting_strategy.py --resolve --paper"): return
    
    # 3. Predict Today
    print("\n[Step 4] Fetching Upcoming Games (Today)...")
    if not run_command(f"python3 fetch_new_games.py --date {date_today}"): return
    
    file_today = f"daily_games_{date_today}.csv"
    if os.path.exists(file_today):
        print("\n[Step 5] Generating Predictions & Placing Bets...")
        # We need to tell betting_strategy to predict specific games.
        # We'll pass the daily file.
        # We need to update betting_strategy.py to handle --predict_file
        # Also pass provider
        provider_arg = "--provider rapid-api" # Default to rapid-api now
        if not run_command(f"python3 betting_strategy.py --paper --predict_file {file_today} {provider_arg}"): return
    else:
        print(f"Warning: {file_today} not found. No games today?")
        
    print("\n--- Automation Complete ---")

if __name__ == "__main__":
    main()
