# Google Compute Engine Deployment Guide

## Overview
This guide shows how to migrate your NBA Meta-Learner V4 training from Modal to Google Compute Engine.

## Prerequisites
- Google Cloud account with gcloud CLI installed
- Kaggle account with API credentials
- Window model files (player_models_*.pkl) from your previous training

## Step 1: Create GCE Instance

```bash
# Create a high-memory instance (16 vCPUs, 32GB RAM)
gcloud compute instances create nba-trainer \
    --machine-type=n2-highmem-8 \
    --image-family=ubuntu-2004-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=100GB \
    --zone=us-central1-a
```

## Step 2: Upload Files to GCE

```bash
# Upload the training package
gcloud compute scp gce_train_meta.py nba-trainer:~/
gcloud compute scp gce_requirements.txt nba-trainer:~/
gcloud compute scp gce_setup.sh nba-trainer:~/
gcloud compute scp --recurse experiments/ nba-trainer:~/
gcloud compute scp --recurse shared/ nba-trainer:~/

# Upload required Python modules
gcloud compute scp ensemble_predictor.py nba-trainer:~/
gcloud compute scp train_meta_learner_v4.py nba-trainer:~/
gcloud compute scp hybrid_multi_task.py nba-trainer:~/
gcloud compute scp optimization_features.py nba-trainer:~/
gcloud compute scp phase7_features.py nba-trainer:~/
gcloud compute scp rolling_features.py nba-trainer:~/
gcloud compute scp meta_learner_ensemble.py nba-trainer:~/

# Upload window model files (IMPORTANT!)
gcloud compute scp player_models_*.pkl nba-trainer:~/
gcloud compute scp player_models_*_meta.json nba-trainer:~/
```

## Step 3: Setup GCE Environment

```bash
# SSH into the instance
gcloud compute ssh nba-trainer

# Run setup script
chmod +x gce_setup.sh
./gce_setup.sh

# Activate virtual environment
source nba_env/bin/activate
```

## Step 4: Set Kaggle Credentials

```bash
# Set your Kaggle API credentials
export KAGGLE_USERNAME='your_kaggle_username'
export KAGGLE_KEY='your_kaggle_api_key'

# Optional: Add to .bashrc for persistence
echo "export KAGGLE_USERNAME='your_kaggle_username'" >> ~/.bashrc
echo "export KAGGLE_KEY='your_kaggle_api_key'" >> ~/.bashrc
```

## Step 5: Run Training

```bash
# Run the training
python gce_train_meta.py

# Or with custom config
python gce_train_meta.py --config experiments/v4_full.yaml
```

## Step 6: Download Results

```bash
# Download the trained model
gcloud compute scp nba-trainer:~/meta_learner_v4_all_components.pkl ./

# Download logs if needed
gcloud compute scp --recurse nba-trainer:~/logs/ ./
```

## Resource Recommendations

### Instance Types
- **Minimum**: n2-highmem-8 (8 vCPUs, 32GB RAM) - $0.50/hour
- **Recommended**: n2-highmem-16 (16 vCPUs, 64GB RAM) - $1.00/hour
- **Large**: n2-highmem-32 (32 vCPUs, 128GB RAM) - $2.00/hour

### Storage
- **Boot disk**: 100GB SSD (for OS and dependencies)
- **Additional**: 200GB if you want to store multiple model versions

### Cost Optimization
- Use preemptible instances for 60-80% cost savings
- Shut down instance when not training
- Use Cloud Storage for model archiving

## Troubleshooting

### Common Issues

1. **Missing window model files**
   ```
   ERROR: No player_models_*.pkl files found
   ```
   Solution: Upload your window model files from Modal/local storage

2. **Kaggle authentication failed**
   ```
   ERROR: 401 - Unauthorized
   ```
   Solution: Verify KAGGLE_USERNAME and KAGGLE_KEY are correct

3. **Memory errors**
   ```
   MemoryError: Unable to allocate array
   ```
   Solution: Use a larger instance type or reduce max_samples in script

4. **Missing dependencies**
   ```
   ModuleNotFoundError: No module named 'xxx'
   ```
   Solution: Ensure all requirements are installed with pip install -r gce_requirements.txt

### Performance Tips

1. **Monitor resources**
   ```bash
   htop  # CPU and memory usage
   df -h # Disk usage
   ```

2. **Enable parallel processing**
   - The script automatically uses all available CPU cores
   - For faster I/O, consider using local SSD

3. **Optimize for repeated runs**
   - Keep the downloaded CSV file locally
   - Use screen/tmux for long-running sessions

## Migration from Modal

| Modal Feature | GCE Equivalent |
|---------------|-----------------|
| @app.function | Standalone Python function |
| modal.Volume | Local disk or Cloud Storage |
| modal.Secret | Environment variables |
| 16 CPU, 32GB RAM | n2-highmem-8 instance |
| Timeout handling | No timeout limits |
| Automatic scaling | Manual instance sizing |

## Next Steps

1. Test with a smaller dataset first
2. Monitor training progress and resource usage
3. Consider automating with Cloud Scheduler for regular training
4. Set up monitoring and alerting for production use
