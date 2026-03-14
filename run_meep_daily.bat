@echo off
:: Navigate to project directory
cd /d "C:\Users\tmiles11\nba_predictor\meep\meep"

echo.
echo ============================================================
echo   MEEP TERMINAL - AUTOMATED DAILY OPS ROUTINE
echo ============================================================
echo [%date% %time%] INITIATING REFRESH...

:: Step 0: Performance Evaluation (Previous Day)
echo.
echo "[0/3] Grading previous day's picks..."
python paper_grader.py

:: Step 1: Data Refresh (Historical Sync)
echo.
echo "[1/3] Refreshing Historical Data & Syncing ESPN..."
python daily_refresh.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Data refresh failed. Check logs.
    pause
    exit /b %ERRORLEVEL%
)

:: Step 2: Prediction Generation
echo.
echo [2/3] Generating Live Inference Set (Predictions)...
python predict_live_FINAL.py --betting --aggregated-data "final_feature_matrix_with_per_min_1997_onward.csv" --output-wide "predictions/live_ensemble_2025.csv"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Prediction generation failed.
    pause
    exit /b %ERRORLEVEL%
)

:: Step 3: Pick Generation & Ledger Update
echo.
echo "[3/3] Generating Final Picks & Updating Ledger..."
python run_phase_i.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Pick generation failed.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ============================================================
echo   DAILY CYCLE COMPLETE: Check the latest .md report.
echo ============================================================
echo.
pause
