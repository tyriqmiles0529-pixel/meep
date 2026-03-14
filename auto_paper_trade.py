
import subprocess
import time
import datetime
import os
import sys

# Configuration
PREDICTIONS_OUTPUT = "predictions/live_ensemble_2025.csv"
DATA_AGGREGATED = "final_feature_matrix_with_per_min_1997_onward.csv" 
MODELS_DIR = "models"
LOG_FILE = "paper_trading.log"

def log(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {message}"
    print(full_msg)
    with open(LOG_FILE, "a") as f:
        f.write(full_msg + "\n")

def run_command(cmd_list, description):
    log(f"Starting: {description}")
    try:
        # TIMEOUT added to prevent hangs
        result = subprocess.run(cmd_list, capture_output=True, text=True, timeout=1200)
        if result.returncode == 0:
            log(f"SUCCESS: {description}")
            return True
        else:
            log(f"FAILURE: {description} (Code {result.returncode})")
            log(f"STDERR: {result.stderr}")
            log(f"STDOUT: {result.stdout}")
            return False
    except subprocess.TimeoutExpired:
        log(f"TIMEOUT: {description} exceeded 20 minutes.")
        return False
    except Exception as e:
        log(f"EXCEPTION: {description} - {e}")
        return False

def update_daily_data():
    """Fetch yesterday's game data and update the feature matrix."""
    yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    daily_file = f"daily_games_{yesterday}.csv"
    
    # 1. Fetch Yesterday's Data
    cmd_fetch = [sys.executable, "fetch_new_games.py", "--date", yesterday]
    if not run_command(cmd_fetch, f"Fetch Games for {yesterday}"):
        log("SKIP: Data update failed at fetch step.")
        return False

    # 2. Update Feature Matrix
    cmd_update = [
        sys.executable, "update_feature_matrix.py", 
        "--daily", daily_file,
        "--master", DATA_AGGREGATED
    ]
    if not run_command(cmd_update, "Update Feature Matrix"):
        log("SKIP: Feature matrix update failed.")
        return False
        
    return True

def morning_cycle():
    log("=== [AUTO] STARTING MORNING CYCLE (PAPER) ===")
    
    # 0. Data Ingestion (Daily Update)
    # Critical for dynamic "Live" model behavior
    update_daily_data()
    
    # 1. Generate Predictions (Shared Engine)
    cmd_predict = [
        sys.executable, "predict_live_FINAL.py",
        "--betting", 
        "--output", PREDICTIONS_OUTPUT,  
        "--aggregated-data", DATA_AGGREGATED,
        "--models-dir", MODELS_DIR
    ]
    if not run_command(cmd_predict, "Generate Predictions"):
        log("CRITICAL: Prediction generation failed.")
        return

    # 2. Run Paper Phase I (Picks & Logging to paper_ledger.csv)
    cmd_phase_i = [sys.executable, "paper_phase_i.py"]
    run_command(cmd_phase_i, "Paper Phase S Pick Generation")
    
    log("=== [AUTO] MORNING CYCLE COMPLETE ===")

def night_cycle():
    log("=== [AUTO] STARTING NIGHT CYCLE (PAPER) ===")
    
    # 1. CLV Tracker (Paper Ledger)
    cmd_clv = [sys.executable, "paper_clv_tracker.py"]
    run_command(cmd_clv, "Paper CLV & Outcome Tracking")
    
    log("=== [AUTO] NIGHT CYCLE COMPLETE ===")

def main():
    log("--- BACKGROUND PAPER TRADING ACTIVE (DYNAMIC MODE) ---")
    log("Routine: Fetch Data @ Startup | Morning Cycle @ 10:30 AM | Night Cycle @ 02:00 AM")
    
    last_run_date = None
    last_night_run_date = None

    log("--- LOOP STARTED ---")

    while True:
        try:
            now = datetime.datetime.now()
            current_date = now.strftime("%Y-%m-%d")
            
            # Heartbeat Logging (approx hourly)
            if now.minute == 0 and now.second < 40:
                print(f"[{now.strftime('%H:%M:%S')}] Heartbeat... (Last Run: {last_run_date})")

            # ---------------------------------------------------------
            # MORNING SCHEDULE (Pick Generation)
            # ---------------------------------------------------------
            is_time_for_morning = (now.hour > 10) or (now.hour == 10 and now.minute >= 30)
            
            if is_time_for_morning and last_run_date != current_date:
                morning_cycle()
                last_run_date = current_date 

            # ---------------------------------------------------------
            # NIGHT SCHEDULE (CLV / Grading)
            # ---------------------------------------------------------
            if now.hour >= 2 and now.hour < 10 and last_night_run_date != current_date:
                night_cycle()
                last_night_run_date = current_date

            time.sleep(30)
            
        except Exception as e:
            log(f"CRITICAL LOOP ERROR: {e}")
            time.sleep(60) 

if __name__ == "__main__":
    main()
