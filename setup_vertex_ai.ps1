# ========================================
# Vertex AI Setup Script for NBA Predictor
# ========================================
# Run this in PowerShell
# Right-click and "Run with PowerShell"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "🏀 NBA Predictor - Vertex AI Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# ========================================
# STEP 1: Check Prerequisites
# ========================================
Write-Host "📋 Step 1: Checking prerequisites..." -ForegroundColor Yellow
Write-Host ""

# Check if gcloud is installed
try {
    $gcloudVersion = gcloud --version 2>&1 | Select-String "Google Cloud SDK"
    Write-Host "✅ Google Cloud SDK installed: $gcloudVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Google Cloud SDK not found!" -ForegroundColor Red
    Write-Host "   Install from: https://cloud.google.com/sdk/docs/install" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit
}

Write-Host ""

# ========================================
# STEP 2: Login and Select Project
# ========================================
Write-Host "🔐 Step 2: Google Cloud Authentication" -ForegroundColor Yellow
Write-Host ""
Write-Host "Opening browser for login..." -ForegroundColor Cyan
gcloud auth login

Write-Host ""
Write-Host "📂 Available Projects:" -ForegroundColor Cyan
gcloud projects list --format="table(projectId,name)"

Write-Host ""
$PROJECT_ID = Read-Host "Enter your PROJECT_ID from the list above"

if ([string]::IsNullOrWhiteSpace($PROJECT_ID)) {
    Write-Host "❌ No project ID entered. Exiting." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit
}

Write-Host ""
Write-Host "Setting active project to: $PROJECT_ID" -ForegroundColor Cyan
gcloud config set project $PROJECT_ID

Write-Host "✅ Project set!" -ForegroundColor Green
Write-Host ""

# ========================================
# STEP 3: Enable Required APIs
# ========================================
Write-Host "🔧 Step 3: Enabling required APIs (this takes 2-3 minutes)..." -ForegroundColor Yellow
Write-Host ""

Write-Host "Enabling Vertex AI API..." -ForegroundColor Cyan
gcloud services enable aiplatform.googleapis.com

Write-Host "Enabling Container Registry API..." -ForegroundColor Cyan
gcloud services enable containerregistry.googleapis.com

Write-Host "Enabling Secret Manager API..." -ForegroundColor Cyan
gcloud services enable secretmanager.googleapis.com

Write-Host "Enabling Cloud Storage API..." -ForegroundColor Cyan
gcloud services enable storage.googleapis.com

Write-Host "Enabling Cloud Build API..." -ForegroundColor Cyan
gcloud services enable cloudbuild.googleapis.com

Write-Host "✅ All APIs enabled!" -ForegroundColor Green
Write-Host ""

# ========================================
# STEP 4: Create Storage Bucket
# ========================================
Write-Host "📦 Step 4: Creating Cloud Storage bucket..." -ForegroundColor Yellow
Write-Host ""

$INITIALS = Read-Host "Enter your initials (e.g., 'tm' for Tyriq Miles)"
$BUCKET_NAME = "nba-predictor-models-$INITIALS"

Write-Host "Creating bucket: gs://$BUCKET_NAME" -ForegroundColor Cyan
$bucketExists = gsutil ls gs://$BUCKET_NAME 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "⚠️  Bucket already exists, skipping creation" -ForegroundColor Yellow
} else {
    gsutil mb -l us-central1 gs://$BUCKET_NAME
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Bucket created!" -ForegroundColor Green
    } else {
        Write-Host "❌ Failed to create bucket. Check the error above." -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit
    }
}

Write-Host ""

# ========================================
# STEP 5: Set Up Kaggle Credentials
# ========================================
Write-Host "🔑 Step 5: Setting up Kaggle credentials..." -ForegroundColor Yellow
Write-Host ""
Write-Host "To get your Kaggle API credentials:" -ForegroundColor Cyan
Write-Host "1. Go to https://www.kaggle.com/" -ForegroundColor White
Write-Host "2. Click your profile picture → Settings" -ForegroundColor White
Write-Host "3. Scroll to 'API' section" -ForegroundColor White
Write-Host "4. Click 'Create New Token'" -ForegroundColor White
Write-Host "5. Open the downloaded kaggle.json file" -ForegroundColor White
Write-Host ""

$KAGGLE_USERNAME = Read-Host "Enter your Kaggle username"
$KAGGLE_KEY = Read-Host "Enter your Kaggle API key" -AsSecureString

if ([string]::IsNullOrWhiteSpace($KAGGLE_USERNAME)) {
    Write-Host "❌ No Kaggle username entered. Exiting." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit
}

# Convert SecureString to plain text for gcloud
$BSTR = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($KAGGLE_KEY)
$KAGGLE_KEY_PLAIN = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($BSTR)

Write-Host ""
Write-Host "Creating secrets in Secret Manager..." -ForegroundColor Cyan

# Create username secret
$usernameExists = gcloud secrets describe kaggle-username 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "⚠️  Secret 'kaggle-username' already exists, updating..." -ForegroundColor Yellow
    echo $KAGGLE_USERNAME | gcloud secrets versions add kaggle-username --data-file=-
} else {
    echo $KAGGLE_USERNAME | gcloud secrets create kaggle-username --data-file=-
}

# Create key secret
$keyExists = gcloud secrets describe kaggle-key 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "⚠️  Secret 'kaggle-key' already exists, updating..." -ForegroundColor Yellow
    echo $KAGGLE_KEY_PLAIN | gcloud secrets versions add kaggle-key --data-file=-
} else {
    echo $KAGGLE_KEY_PLAIN | gcloud secrets create kaggle-key --data-file=-
}

Write-Host "✅ Kaggle credentials stored in Secret Manager!" -ForegroundColor Green
Write-Host ""

# ========================================
# STEP 6: Build Container with Cloud Build
# ========================================
Write-Host "🐳 Step 6: Building container with Cloud Build..." -ForegroundColor Yellow
Write-Host ""
Write-Host "This will take 5-10 minutes. Please wait..." -ForegroundColor Cyan
Write-Host ""

$IMAGE_URI = "gcr.io/$PROJECT_ID/nba-trainer:latest"

Write-Host "Building image: $IMAGE_URI" -ForegroundColor Cyan
gcloud builds submit --tag $IMAGE_URI --timeout=20m .

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Container built and pushed successfully!" -ForegroundColor Green
} else {
    Write-Host "❌ Build failed. Check the error above." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit
}

Write-Host ""

# ========================================
# STEP 7: Generate Configuration File
# ========================================
Write-Host "📝 Step 7: Generating configuration file..." -ForegroundColor Yellow
Write-Host ""

$CONFIG_FILE = "vertex_ai_config_READY.txt"
$configContent = @"
========================================
🏀 VERTEX AI TRAINING JOB CONFIGURATION
========================================

Copy and paste these values into Vertex AI console:
https://console.cloud.google.com/vertex-ai/training

========================================
CONTAINER SETTINGS
========================================

Container image:
$IMAGE_URI

Model output directory:
gs://$BUCKET_NAME/models/

========================================
ARGUMENTS (add one per line in UI)
========================================

--dataset
eoinamoore/historical-nba-data-and-player-box-scores
--verbose
--fresh
--lgb-log-period
50
--neural-epochs
30
--neural-device
gpu
--enable-window-ensemble
--models-dir
/gcs/$BUCKET_NAME/models/

========================================
ENVIRONMENT VARIABLES
========================================

Variable 1:
  Name: KAGGLE_USERNAME
  Value type: Secret
  Secret: kaggle-username
  Version: latest

Variable 2:
  Name: KAGGLE_KEY
  Value type: Secret
  Secret: kaggle-key
  Version: latest

========================================
COMPUTE CONFIGURATION
========================================

Region: us-central1
Machine type: n1-highmem-8
Accelerator type: NVIDIA_TESLA_T4
Number of accelerators: 1
Boot disk size: 100 GB
Maximum run time: 6 hours

========================================
AFTER TRAINING COMPLETES
========================================

Download models with:
gsutil -m cp -r gs://$BUCKET_NAME/models/ ./trained-models/

Or view in browser:
https://console.cloud.google.com/storage/browser/$BUCKET_NAME/models

========================================
ESTIMATED COST
========================================

Training time: 1-1.5 hours
Cost per run: ~`$2-3 USD

========================================
"@

$configContent | Out-File -FilePath $CONFIG_FILE -Encoding UTF8

Write-Host "✅ Configuration saved to: $CONFIG_FILE" -ForegroundColor Green
Write-Host ""

# ========================================
# FINAL STEPS
# ========================================
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "✅ SETUP COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "📋 Next Steps:" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. Open the configuration file:" -ForegroundColor White
Write-Host "   $CONFIG_FILE" -ForegroundColor Cyan
Write-Host ""
Write-Host "2. Go to Vertex AI Training:" -ForegroundColor White
Write-Host "   https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=$PROJECT_ID" -ForegroundColor Cyan
Write-Host ""
Write-Host "3. Click CREATE and fill in the form using values from the config file" -ForegroundColor White
Write-Host ""
Write-Host "4. Click START TRAINING and wait 1-1.5 hours" -ForegroundColor White
Write-Host ""
Write-Host "5. Download trained models when complete:" -ForegroundColor White
Write-Host "   gsutil -m cp -r gs://$BUCKET_NAME/models/ ./trained-models/" -ForegroundColor Cyan
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan

Write-Host ""
Write-Host "Opening configuration file..." -ForegroundColor Yellow
Start-Sleep -Seconds 2
notepad $CONFIG_FILE

Read-Host "Press Enter to exit"
