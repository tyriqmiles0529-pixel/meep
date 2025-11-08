# 🚀 Vertex AI Training Configuration Guide

## Quick Setup (3 Steps)

### 1. **Container Image**
```
gcr.io/YOUR-PROJECT-ID/nba-trainer:latest
```

**To build and push:**
```bash
# Edit vertex_ai_setup.sh and set your PROJECT_ID
# Then run:
bash vertex_ai_setup.sh
```

---

### 2. **Model Output Directory**
```
gs://nba-predictor-models/models/
```

**What gets saved here:**
- `models/points/*.pkl` - Points prediction models
- `models/assists/*.pkl` - Assists prediction models
- `models/rebounds/*.pkl` - Rebounds prediction models
- `models/threes/*.pkl` - Threes prediction models
- `model_cache/` - Cached ensemble models

---

### 3. **Arguments** (Enter EXACTLY as shown, one per line)

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
/gcs/nba-predictor-models/models/
```

---

## Machine Configuration

**Recommended for your workload:**

### Option A: Fast Training (Recommended)
- **Machine type:** n1-highmem-8
- **Accelerator:** 1 x NVIDIA Tesla T4
- **Training time:** ~1-1.5 hours
- **Cost:** ~$2-3 per run

### Option B: Budget Training
- **Machine type:** n1-standard-4
- **Accelerator:** None (CPU only)
- **Training time:** ~3-4 hours
- **Cost:** ~$0.50 per run
- **Note:** Add `--neural-device cpu` to arguments

### Option C: Maximum Speed
- **Machine type:** n1-highmem-16
- **Accelerator:** 1 x NVIDIA Tesla V100
- **Training time:** ~30-45 min
- **Cost:** ~$5-8 per run

---

## Environment Variables (Secrets)

**Kaggle API Credentials:**

If using Secret Manager (recommended):
1. Go to Secret Manager in Cloud Console
2. Create two secrets:
   - Name: `kaggle-username`, Value: your Kaggle username
   - Name: `kaggle-key`, Value: your Kaggle API key
3. Grant Vertex AI service account access to secrets

**In Vertex AI training job config:**
- Add environment variables:
  - `KAGGLE_USERNAME` → Reference secret: `kaggle-username`
  - `KAGGLE_KEY` → Reference secret: `kaggle-key`

**Alternative (less secure):**
Hardcode in Dockerfile (not recommended for production):
```dockerfile
ENV KAGGLE_USERNAME="your-username"
ENV KAGGLE_KEY="your-api-key"
```

---

## Full Training Job Configuration

### Container Settings
```yaml
Container image: gcr.io/YOUR-PROJECT-ID/nba-trainer:latest
Model output directory: gs://nba-predictor-models/models/
```

### Arguments (one per line)
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
/gcs/nba-predictor-models/models/
```

### Compute Resources
```
Machine type: n1-highmem-8
Accelerator type: NVIDIA_TESLA_T4
Accelerator count: 1
```

### Advanced Settings
```
Service account: [default Vertex AI service account]
Network: default
Maximum run time: 6 hours
```

---

## Optional Arguments (Hyperparameter Tuning)

If you want to use Vertex AI HyperTune to find optimal settings:

### Hyperparameters to Tune
```
--decay
[0.90, 0.95, 0.97, 0.99]

--neural-epochs
[20, 30, 40, 50]

--player-season-cutoff
[1998, 2002, 2005]
```

### HyperTune Metric
```
Metric name: validation_mae_points
Goal: MINIMIZE
```

**Note:** This runs multiple training jobs in parallel. Cost scales with number of trials.

---

## Post-Training

### Download Models
```bash
# Download all trained models
gsutil -m cp -r gs://nba-predictor-models/models/ ./local-models/

# Or download specific stat
gsutil -m cp -r gs://nba-predictor-models/models/points/ ./
```

### Use in Predictions
```python
# In riq_analyzer.py, models auto-load from:
MODEL_DIR = "models/"  # or "local-models/" if downloaded

# Or mount Cloud Storage directly:
# gcsfuse nba-predictor-models /mnt/gcs
# MODEL_DIR = "/mnt/gcs/models/"
```

---

## Monitoring

### View Training Logs
1. Go to Vertex AI > Training
2. Click on your training job
3. Click "View Logs"
4. Look for:
   ```
   ✅ TabNet training complete
   ✅ LightGBM training complete
   ✅ Ensemble training complete
   📊 Validation MAE: X.XX
   ```

### Check Output
```bash
# List saved models
gsutil ls gs://nba-predictor-models/models/

# Should see:
# gs://nba-predictor-models/models/points/
# gs://nba-predictor-models/models/assists/
# gs://nba-predictor-models/models/rebounds/
# gs://nba-predictor-models/models/threes/
```

---

## Troubleshooting

### Error: "Kaggle authentication failed"
**Solution:** Check that secrets are correctly set:
```bash
gcloud secrets versions access latest --secret="kaggle-username"
gcloud secrets versions access latest --secret="kaggle-key"
```

### Error: "Out of memory"
**Solutions:**
1. Use n1-highmem-8 or higher
2. Add `--n-jobs 4` to arguments (reduce parallelism)

### Error: "GPU not available"
**Solutions:**
1. Check accelerator is attached
2. Verify image has CUDA support
3. Try `--neural-device cpu` as fallback

### Error: "Cannot write to /gcs/..."
**Solution:** Mount path should be `/gcs/BUCKET/...` not `gs://BUCKET/...`

---

## Cost Estimates

| Configuration | Training Time | Cost per Run |
|---------------|---------------|--------------|
| CPU only (n1-standard-4) | 3-4 hours | ~$0.50 |
| GPU (n1-highmem-8 + T4) | 1-1.5 hours | ~$2-3 |
| GPU (n1-highmem-16 + V100) | 30-45 min | ~$5-8 |

**Monthly cost (train weekly):**
- CPU: ~$2/month
- GPU (T4): ~$8-12/month
- GPU (V100): ~$20-30/month

---

## Next Steps After Training

1. **Download models** → `gsutil cp -r gs://nba-predictor-models/models/ ./`
2. **Test locally** → Run predictions with new models
3. **Compare accuracy** → Old vs new win rates
4. **Deploy to production** → Update riq_analyzer.py model paths
5. **Schedule retraining** → Monthly or when performance degrades

---

## Complete Example Command

```bash
# Build image
docker build -f Dockerfile.vertexai -t gcr.io/my-project/nba-trainer:latest .

# Push to Container Registry
docker push gcr.io/my-project/nba-trainer:latest

# Create training job via gcloud (alternative to UI)
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=nba-training-$(date +%Y%m%d-%H%M%S) \
  --worker-pool-spec=machine-type=n1-highmem-8,replica-count=1,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,container-image-uri=gcr.io/my-project/nba-trainer:latest \
  --args="--dataset,eoinamoore/historical-nba-data-and-player-box-scores,--verbose,--fresh,--neural-device,gpu,--models-dir,/gcs/nba-predictor-models/models/"
```

---

**Questions?** Check the logs first, then review this guide. Most issues are:
1. Kaggle credentials not set
2. Wrong output directory format
3. Insufficient memory
