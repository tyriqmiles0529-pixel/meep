# Modal Setup Guide - Complete Migration

## Prerequisites

### 1. Create Modal Account
Go to https://modal.com and sign up (free $30 credit)

### 2. Install Modal CLI
```bash
pip install modal
```

### 3. Authenticate
```bash
modal setup
```

This will open your browser to get your API token. Follow the prompts.

## API Keys Needed

### Modal Token (Required)
- **What**: Your Modal authentication token
- **Where to get**: Automatically created during `modal setup`
- **Stored at**: `~/.modal.toml`

### Kaggle API (For downloading data)
- **What**: Kaggle API credentials to download datasets
- **Where to get**: https://www.kaggle.com/settings → API → "Create New API Token"
- **Downloads**: `kaggle.json` file
- **Contents**:
  ```json
  {
    "username": "your_kaggle_username",
    "key": "your_kaggle_api_key"
  }
  ```

### GitHub Personal Access Token (Optional, for private repos)
- **What**: Token to clone private GitHub repos in Modal
- **Where to get**: GitHub → Settings → Developer settings → Personal access tokens
- **Permissions needed**: `repo` (full control of private repositories)
- **Not needed if**: Your repo is public

## Project Structure on Modal

```
Modal Volumes:
├── nba-data/              # Persistent data storage
│   ├── csv_dir/           # All 7 Basketball Reference CSVs
│   │   ├── PlayerStatistics.csv
│   │   ├── Player Advanced.csv
│   │   ├── Player Per 100 Poss.csv
│   │   ├── Player Play-By-Play.csv
│   │   └── Player Shooting.csv
│   └── aggregated_nba_data.parquet (optional)
│
└── nba-models/            # Trained model cache
    ├── player_models_1947_1949.pkl
    ├── player_models_1950_1952.pkl
    └── ...
```

## Step-by-Step Setup

### Step 1: Set Kaggle Credentials in Modal

```bash
# Set Kaggle credentials as Modal secrets
modal secret create kaggle-secret \
  KAGGLE_USERNAME=your_username \
  KAGGLE_KEY=your_api_key
```

Or create via Python:
```python
import modal

# Create secret
modal.Secret.create(
    "kaggle-secret",
    KAGGLE_USERNAME="your_username",
    KAGGLE_KEY="your_api_key"
)
```

### Step 2: Upload Data to Modal

See `modal_upload_data.py` (created below)

### Step 3: Run Training

See `modal_train.py` (created below)

## Files to Create

All files are created in the sections below:
1. `modal_upload_data.py` - Upload Kaggle data to Modal
2. `modal_train.py` - Main training script
3. `modal_predict.py` - Inference script (future)

## Secrets Management

### View Your Secrets
```bash
modal secret list
```

### Delete a Secret
```bash
modal secret delete kaggle-secret
```

### Update a Secret
```bash
modal secret create kaggle-secret \
  KAGGLE_USERNAME=new_username \
  KAGGLE_KEY=new_key \
  --force  # Overwrites existing
```

## Cost Estimation

### Data Upload (One-time)
- Upload ~2GB CSV data: **FREE** (no GPU needed)
- Time: ~5-10 minutes

### Training (27 windows)
- GPU: A10G ($1.10/hour)
- RAM: 64GB
- Estimated time: 10-15 hours
- **Total cost: ~$11-16**

### Storage
- Modal volumes: **FREE** for first 10GB
- Your data + models: ~5GB
- **Cost: $0**

## Monitoring

### View Logs
```bash
modal app logs nba-training
```

### List Running Functions
```bash
modal app list
```

### Stop All Functions
```bash
modal app stop nba-training
```

## Common Issues

### "ModuleNotFoundError: No module named 'shared'"
**Solution**: Make sure to include your code files in the Modal image:
```python
image = modal.Image.debian_slim().pip_install(...).copy_local_dir(".", "/root")
```

### "Volume not found"
**Solution**: Create volumes first:
```bash
modal volume create nba-data
modal volume create nba-models
```

### "Out of memory"
**Solution**: Increase memory in function decorator:
```python
@stub.function(memory=131072)  # 128GB
```

## Next Steps

1. ✅ Sign up for Modal
2. ✅ Run `modal setup`
3. ✅ Set Kaggle credentials
4. ✅ Run `modal_upload_data.py` to upload data
5. ✅ Run `modal_train.py` to start training
6. ✅ Monitor with `modal app logs nba-training`
7. ✅ Download models when complete

See the notebook files below for complete implementation!
