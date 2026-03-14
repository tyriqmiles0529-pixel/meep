
import argparse
import subprocess
import sys
import datetime
import os

# --- PHASE K OPERATIONS CONTROLLER ---
# Automates the daily Institutional Betting Cycle.
# Usage:
#   python daily_ops.py morning  -> Run daily picks (run_phase_i.py)
#   python daily_ops.py night    -> Run CLV tracker (clv_tracker.py)
#   python daily_ops.py weekly   -> Run Audits (monitor_systems.py + monte_carlo)

LOG_FILE = "operations.log"

def log(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {message}"
    print(full_msg)
    with open(LOG_FILE, "a", encoding='utf-8') as f:
        f.write(full_msg + "\n")

def run_script(script_name, args=None):
    if args is None: args = []
    log(f"Starting {script_name} {' '.join(args)}...")
    try:
        # Run script using the current Python interpreter
        full_cmd = [sys.executable, script_name] + args
        result = subprocess.run(
            full_cmd, 
            capture_output=True, 
            text=True, 
            cwd=os.path.dirname(os.path.abspath(__file__)) # Run in correct dir
        )
        
        if result.returncode == 0:
            log(f"SUCCESS: {script_name} completed.")
            # Optionally log stdout if needed, for now just success
            # log(f"Output:\n{result.stdout}")
        else:
            log(f"FAILURE: {script_name} failed with code {result.returncode}.")
            log(f"Error Output:\n{result.stderr}")
            if result.stdout: log(f"Std Output:\n{result.stdout}")
            return False
    except Exception as e:
        log(f"EXCEPTION: Could not run {script_name}: {e}")
        return False
    return True

def morning_routine():
    log("=== INITIATING PHASE K MORNING ROUTINE (PICKS) ===")
    
    # 0. Data Update (Dynamic)
    yesterday_str = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    daily_csv = f"daily_games_{yesterday_str}.csv"
    master_csv = "final_feature_matrix_with_per_min_1997_onward.csv"
    
    log("[STEP 1/4] Fetching Data (Yesterday's Games)...")
    if not run_script("fetch_new_games.py", ["--date", yesterday_str]):
        log("[WARN] Data fetch failed. Features might be stale.")
        # We don't return here because maybe we have data from a previous manual run?
        
    log("[STEP 2/4] Updating Feature Matrix...")
    if not run_script("update_feature_matrix.py", ["--daily", daily_csv, "--master", master_csv]):
        log("[WARN] Feature update failed.")
        
    log("[STEP 3/4] Generating Predictions...")
    # Generate predictions using the MASTER csv (now updated)
    if not run_script("predict_live_FINAL.py", ["--betting", "--output", "predictions/live_ensemble_2025.csv", "--aggregated-data", master_csv]):
        log("[FAIL] Prediction generation failed. Cannot proceed to picks.")
        return

    log("[STEP 4/4] Generating Picks (Phase S2)...")
    success = run_script("run_phase_i.py")
    
    if success:
        log("[PASS] Picks generated. Check the date-stamped Markdown report.")
    else:
        log("[FAIL] Picks generation failed. Check logs.")
    log("=== MORNING ROUTINE COMPLETE ===")

def night_routine():
    log("=== INITIATING PHASE K NIGHT ROUTINE (CLV) ===")
    success = run_script("clv_tracker.py")
    if success:
        log("[PASS] CLV updated for pending bets.")
    else:
        log("[FAIL] CLV tracking failed.")
    log("=== NIGHT ROUTINE COMPLETE ===")

def weekly_audit():
    log("=== INITIATING PHASE K WEEKLY AUDIT ===")
    
    # 1. Monitoring / Drift
    s1 = run_script("monitor_systems.py")
    
    # 2. Re-run Simulations
    s2 = run_script("monte_carlo_engine.py")
    
    if s1 and s2:
        log("[PASS] Audit & Simulation complete. Check 'performance_audit.md'.")
    else:
        log("[FAIL] Audit encountered errors.")
        
    log("=== WEEKLY AUDIT COMPLETE ===")

def full_daily_cycle():
    log("=== INITIATING FULL DAILY CYCLE (CLV + PICKS) ===")
    
    # 1. Update CLV for existing/yesterday's bets FIRST
    # (Prevents overwriting today's new bets immediately with opening lines)
    night_routine()
    
    # 2. Generate new picks for today
    morning_routine()
    
    log("=== FULL DAILY CYCLE COMPLETE ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase K Operations Controller")
    parser.add_argument("mode", choices=["morning", "night", "weekly", "combo"], help="Operation mode (morning=picks, night=clv, weekly=audit, combo=both)")
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
        
    args = parser.parse_args()
    
    if args.mode == "morning":
        morning_routine()
    elif args.mode == "night":
        night_routine()
    elif args.mode == "weekly":
        weekly_audit()
    elif args.mode == "combo":
        full_daily_cycle()
