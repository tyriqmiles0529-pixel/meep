# NBA Meta-Learner GCE Deployment Guide

This guide shows how to migrate your NBA season window training from Modal to Google Compute Engine with cost optimization and production-ready features.

## Overview

The GCE deployment provides:
- **80% cost savings** with preemptible instances
- **Auto-shutdown** after training completion
- **Checkpointing** to handle instance preemption
- **Cloud Monitoring** integration
- **Error handling** and retry logic
- **Automated deployment** with startup scripts

## Files Created

| File | Purpose |
|------|---------|
| `gce_train_meta_v4_robust.py` | Main training script with retry/checkpointing |
| `gce_train_meta_v4.py` | Simpler training script version |
| `create_instance_prod.sh` | Production instance creation with cost controls |
| `create_instance.sh` | Basic instance creation |
| `setup_gcs.sh` | Google Cloud Storage bucket setup |
| `gce_startup.sh` | Instance startup automation |
| `requirements.txt` | Python dependencies |
| `cleanup_gcs.sh` | GCS cleanup script |

## Quick Start

### 0. Make Scripts Executable

```bash
# Make all shell scripts executable
chmod +x *.sh
```

### 1. Set up Google Cloud

```bash
# Install gcloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init

# Set your project
export PROJECT_ID="your-project-id"
gcloud config set project $PROJECT_ID
```

### 2. Set up GCS Bucket

```bash
# Create bucket and folder structure
./setup_gcs.sh $PROJECT_ID nba-models

# Upload existing models (if you have them)
gsutil cp player_models_*.pkl gs://nba-models/models/
gsutil cp player_models_*_meta.json gs://nba-models/models/
```

### 3. Set up Kaggle Credentials

```bash
# Create secret with Kaggle credentials
echo '{"username":"YOUR_KAGGLE_USERNAME","key":"YOUR_KAGGLE_KEY"}' | \
gcloud secrets create kaggle-credentials --data-file=-

# Or set environment variables
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_key"
```

### 4. Create and Launch Instance

```bash
# Production instance (recommended - 80% cost savings)
./create_instance_prod.sh $PROJECT_ID nba-trainer-1 us-central1-a \
  https://github.com/tyriqmiles0529-pixel/meep.git nba-models kaggle-credentials experiments/v4_full.yaml true

# Standard instance (for long-running jobs)
./create_instance_prod.sh $PROJECT_ID nba-trainer-1 us-central1-a \
  https://github.com/tyriqmiles0529-pixel/meep.git nba-models kaggle-credentials experiments/v4_full.yaml false
```

### 5. Monitor Training

```bash
# Connect to instance
gcloud compute ssh nba-trainer-1 --zone=us-central1-a

# Monitor training
sudo /opt/nba-predictor/monitor.sh

# View logs
tail -f /tmp/training.log

# View logs in GCS
gsutil ls gs://nba-models/logs/
```

## Cost Management

### Preemptible Instances (Recommended)
- **80% cost savings** compared to standard instances
- Auto-shutdown after training
- Checkpointing handles preemption gracefully
- Perfect for batch training jobs

### Instance Types
- `n1-standard-16`: 16 vCPUs, 60GB RAM (~$0.76/hour standard, ~$0.15/hour preemptible)
- `n1-standard-32`: 32 vCPUs, 120GB RAM for larger datasets

### Budget Alerts
1. Go to [Cloud Billing](https://console.cloud.google.com/billing)
2. Create budget for your project
3. Set alerts at 50%, 90%, 100% of budget

## IAM Permissions Required

The service account needs these roles:
- `roles/storage.admin` (for GCS access)
- `roles/secretmanager.secretAccessor` (for Kaggle credentials)
- `roles/monitoring.metricWriter` (for Cloud Monitoring)
- `roles/compute.instanceAdmin` (for instance management)

```bash
# Assign required roles
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$(gcloud config get-value account)" \
  --role="roles/storage.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$(gcloud config get-value account)" \
  --role="roles/secretmanager.secretAccessor"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$(gcloud config get-value account)" \
  --role="roles/monitoring.metricWriter"
```

## Training Configurations

### All Components (Default)
```bash
--config experiments/v4_full.yaml
```

### Feature Ablation Studies
```bash
# Residual correction only
--config experiments/v4_residual_only.yaml

# Temporal memory only  
--config experiments/v4_temporal_memory_only.yaml

# Player embeddings only
--config experiments/v4_player_embeddings_only.yaml
```

## Monitoring and Debugging

### Cloud Monitoring Metrics
- `custom.googleapis.com/nba/training_success`
- `custom.googleapis.com/nba/training_failure`

### Log Locations
- **Instance logs**: `/tmp/training.log`
- **GCS logs**: `gs://your-bucket/logs/`
- **Checkpoints**: `gs://your-bucket/checkpoints/`

### Common Issues

**Issue**: "No model files found in GCS bucket"
```bash
# Upload your models first
gsutil cp player_models_*.pkl gs://your-bucket/models/
```

**Issue**: "Kaggle credentials not found"
```bash
# Check secret exists
gcloud secrets describe kaggle-credentials

# Test credentials
gcloud secrets versions access latest --secret=kaggle-credentials
```

**Issue**: Instance gets preempted
- This is normal for preemptible instances
- Training resumes from last checkpoint
- Check GCS checkpoints for progress

## Cleanup

### Clean up GCS Storage
```bash
# Remove old logs (keep last 30 days)
gsutil ls gs://nba-models/logs/ | grep "$(date -d '30 days ago' +%Y%m%d)" | \
xargs gsutil rm

# Remove old checkpoints
gsutil rm -r gs://nba-models/checkpoints/

# Or use the cleanup script
./cleanup_gcs.sh nba-models 30
```

### Delete Instance
```bash
gcloud compute instances delete nba-trainer-1 --zone=us-central1-a
```

### Delete Bucket (when completely done)
```bash
gsutil rm -r gs://nba-models
```

## Performance Tips

1. **Use SSD boot disks** for faster I/O
2. **Choose the right region** closest to your data sources
3. **Monitor memory usage** - upgrade to n1-standard-32 if needed
4. **Use checkpoints** for long training jobs
5. **Schedule training** during off-peak hours for better availability

## Migration from Modal

| Modal Feature | GCE Equivalent |
|---------------|-----------------|
| `@app.function` | Standard Python function |
| Modal Volumes | Google Cloud Storage |
| Modal Secrets | Secret Manager |
| `modal.run` | `gcloud compute ssh` + script execution |
| Modal GPU | GCE GPU instances (add `--accelerator-type`) |

## Support

For issues:
1. Check instance logs: `tail -f /tmp/training.log`
2. Check GCS logs: `gsutil ls gs://your-bucket/logs/`
3. Verify IAM permissions
4. Check Cloud Monitoring metrics

## Next Steps

1. Run initial training with all components
2. Compare performance with feature ablation configs
3. Analyze results to identify redundant components
4. Deploy optimized model to production
