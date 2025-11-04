# CHANGE these paths as needed
$RepoDir  = "C:\Users\tmiles11\nba_predictor"
$VenvPy   = Join-Path $RepoDir "venv\Scripts\python.exe"
if (-not (Test-Path $VenvPy)) {
  # Fallback to system Python if venv python not found
  $VenvPy = "python"
}

# Force UTF-8 console and Python I/O to avoid UnicodeEncodeError on emojis
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($true)
$OutputEncoding = [Console]::OutputEncoding
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$LogDir   = Join-Path $RepoDir "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogFile  = Join-Path $LogDir ("pipeline_{0}.log" -f (Get-Date -Format "yyyy-MM-dd_HH-mm-ss"))

# Optional: set env vars (or keep credentials in your scripts)
# $env:KAGGLE_KEY = "YOUR_KAGGLE_KEY"
# $env:KAGGLE_USERNAME = "YOUR_KAGGLE_USERNAME"
# $env:API_SPORTS_KEY = "YOUR_API_SPORTS_IO_KEY"

Set-Location $RepoDir

"=== $(Get-Date -Format o) START ===" | Tee-Object -FilePath $LogFile -Append

try {
  "[1/2] Training models..." | Tee-Object -FilePath $LogFile -Append
  & $VenvPy -X utf8 -u "train_auto.py" | Tee-Object -FilePath $LogFile -Append
  if ($LASTEXITCODE -ne 0) { throw "train_auto.py failed with exit code $LASTEXITCODE" }

  "[2/2] Running analyzer..." | Tee-Object -FilePath $LogFile -Append
  & $VenvPy -X utf8 -u "riq_analyzer.py" | Tee-Object -FilePath $LogFile -Append
  if ($LASTEXITCODE -ne 0) { throw "riq_analyzer.py failed with exit code $LASTEXITCODE" }

  "=== $(Get-Date -Format o) DONE ===" | Tee-Object -FilePath $LogFile -Append
  exit 0
}
catch {
  "ERROR: $_" | Tee-Object -FilePath $LogFile -Append
  exit 1
}