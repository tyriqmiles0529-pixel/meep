# Vertex AI Setup - Step by Step Commands

Copy and paste these commands into PowerShell one at a time.

---

## STEP 1: Check Google Cloud SDK

```powershell
gcloud --version
```

If not installed, download from: https://cloud.google.com/sdk/docs/install

---

## STEP 2: Login to Google Cloud

```powershell
gcloud auth login
```

This will open your browser for authentication.

---

## STEP 3: List Your Projects

```powershell
gcloud projects list --format="table(projectId,name)"
```

---

## STEP 4: Set Your Project

```powershell
gcloud config set project e-copilot-476507-p1
```

---

## STEP 5: Enable Required APIs

This takes 2-3 minutes. Run each command:

```powershell
gcloud services enable aiplatform.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable secretmanager.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

---

## STEP 6: Create Storage Bucket

Replace `tm` with your initials:

```powershell
gsutil mb -l us-central1 gs://nba-predictor-models-tm
```

If bucket already exists, you'll see a message - that's fine, continue.

---

## STEP 7: Set Up Kaggle Credentials

First, get your Kaggle API credentials:
1. Go to https://www.kaggle.com/
2. Click your profile picture -> Settings
3. Scroll to API section
4. Click "Create New Token"
5. Open the downloaded kaggle.json file
6. Copy your username and key

Now create the secrets (replace YOUR_USERNAME and YOUR_KEY):

```powershell
echo YOUR_USERNAME | gcloud secrets create kaggle-username --data-file=-
```

```powershell
echo YOUR_KEY | gcloud secrets create kaggle-key --data-file=-
```

If secrets already exist, update them instead:

```powershell
echo YOUR_USERNAME | gcloud secrets versions add kaggle-username --data-file=-
echo YOUR_KEY | gcloud secrets versions add kaggle-key --data-file=-
```

---

## STEP 8: Build Container with Cloud Build

This takes 5-10 minutes:

```powershell
gcloud builds submit --tag gcr.io/e-copilot-476507-p1/nba-trainer:latest --timeout=20m .
```

Wait for this to complete before continuing.

---

## STEP 9: Verify Container Was Built

```powershell
gcloud container images list --repository=gcr.io/e-copilot-476507-p1
```

You should see `gcr.io/e-copilot-476507-p1/nba-trainer` in the list.

---

## STEP 10: Create Training Job Configuration

Now you need to go to the Vertex AI console:

https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=e-copilot-476507-p1

Click **CREATE** and fill in these values:

### Container Settings

**Container image:**
```
gcr.io/e-copilot-476507-p1/nba-trainer:latest
```

**Model output directory:**
```
gs://nba-predictor-models-tm/models/
```
(Replace `tm` with your initials from Step 6)

### Arguments (add one per line)

```
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
/gcs/nba-predictor-models-tm/models/
```
(Replace `tm` in the last line with your initials)

### Environment Variables

Add two environment variables:

**Variable 1:**
- Name: `KAGGLE_USERNAME`
- Value type: Secret
- Secret: `kaggle-username`
- Version: `latest`

**Variable 2:**
- Name: `KAGGLE_KEY`
- Value type: Secret
- Secret: `kaggle-key`
- Version: `latest`

### Compute Configuration

- Region: `us-central1`
- Machine type: `n1-highmem-8`
- Accelerator type: `NVIDIA_TESLA_T4`
- Number of accelerators: `1`
- Boot disk size: `100 GB`
- Maximum run time: `6 hours`

---

## STEP 11: Start Training

Click **START TRAINING** in the Vertex AI console.

Training will take approximately 1-1.5 hours.

---

## STEP 12: Monitor Training

Watch the logs in the Vertex AI console. Look for:

```
✅ TabNet training complete
✅ LightGBM training complete
✅ Ensemble training complete
📊 Validation MAE: X.XX
```

---

## STEP 13: Download Trained Models

After training completes, download the models (replace `tm` with your initials):

```powershell
gsutil -m cp -r gs://nba-predictor-models-tm/models/ ./trained-models/
```

---

## Troubleshooting

### Error: "Kaggle authentication failed"

Check your secrets:
```powershell
gcloud secrets versions access latest --secret=kaggle-username
gcloud secrets versions access latest --secret=kaggle-key
```

### Error: "Out of memory"

Use a larger machine type: `n1-highmem-16`

### Error: "GPU not available"

Verify accelerator is attached in the compute configuration, or use `--neural-device cpu` instead.

---

## Cost Estimate

- Training time: 1-1.5 hours
- Cost per run: ~$2-3 USD
- Machine: n1-highmem-8 + T4 GPU

---

## Next Steps After Training

1. Test the new models locally
2. Compare accuracy vs old models
3. Deploy to production if improved
4. Schedule monthly retraining

---

**Questions?** Check the logs in Vertex AI console first, then review VERTEX_AI_CONFIG.md for more details.
