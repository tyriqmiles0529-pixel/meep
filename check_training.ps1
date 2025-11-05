# Quick status checker
Write-Host "Training Status Check - 01:54:36" -ForegroundColor Cyan
Write-Host ""

$pythonRunning = Get-Process python -ErrorAction SilentlyContinue
if ($pythonRunning) {
    Write-Host "✅ Training is running" -ForegroundColor Green
    Write-Host "Last 20 lines of output:" -ForegroundColor Yellow
    Get-Content training_output.log -Tail 20
} else {
    Write-Host "❌ Training stopped" -ForegroundColor Red
    Write-Host ""
    Write-Host "Player windows created:" -ForegroundColor Yellow
    ls model_cache\player_models*.pkl 2> | ForEach-Object { 
        Write-Host "  ✅ " -ForegroundColor Green
    }
}
