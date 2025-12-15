# PowerShell Script to Prepare Deployment to GCE VM

# 1. Create a zip of the codebase (excluding data and heavy models)
$exclude = "*.csv", "*.pkl", "*.zip", "__pycache__", ".git", "models", "historical_data"
Compress-Archive -Path "C:\Users\tmiles11\nba_predictor\meep\meep\*" -DestinationPath "nba_codebase_latest.zip" -Update

Write-Host "Codebase zipped to nba_codebase_latest.zip"

# 2. Instructions for User
Write-Host "`nTo deploy to your VM, run the following in your Google Cloud SDK Shell:"
Write-Host "---------------------------------------------------------------------"
Write-Host "1. Upload Code:"
Write-Host "   gcloud compute scp nba_codebase_latest.zip INSTANCE_NAME:~/nba_predictor/ --zone=YOUR_ZONE"
Write-Host ""
Write-Host "2. SSH into VM:"
Write-Host "   gcloud compute ssh INSTANCE_NAME --zone=YOUR_ZONE"
Write-Host ""
Write-Host "3. Unzip and Run (Inside VM):"
Write-Host "   unzip -o nba_codebase_latest.zip -d ~/nba_predictor/"
Write-Host "   cd ~/nba_predictor/"
Write-Host "   screen -S training python train_all_targets.py --epochs 1000"
Write-Host "---------------------------------------------------------------------"
