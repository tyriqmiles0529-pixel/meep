@echo off
REM Clear all window caches for NBA predictor
REM This forces retraining of all windows on next run

echo Clearing window caches...

REM Clear game ensemble window caches
if exist model_cache (
    echo Removing game ensemble window caches...
    del /Q model_cache\*.pkl 2>nul
    del /Q model_cache\*.json 2>nul
    echo   - Deleted ensemble_*.pkl and ensemble_*_meta.json
)

REM Clear player model window caches (if they exist)
if exist model_cache\player_models_*.pkl (
    echo Removing player model window caches...
    del /Q model_cache\player_models_*.pkl 2>nul
    del /Q model_cache\player_models_*_meta.json 2>nul
    echo   - Deleted player_models_*.pkl
)

REM Clear temp files
if exist .combined_players_temp.csv (
    del .combined_players_temp.csv
    echo   - Deleted temp player files
)
if exist .current_season_players_temp.csv (
    del .current_season_players_temp.csv
)

echo.
echo Cache clearing complete!
echo Next training run will retrain all windows from scratch.
