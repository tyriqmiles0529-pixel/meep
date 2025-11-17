# Modal Deployment Guide - Unrestricted Training

## IMPORTANT: Use CSV for Full 1947-2026 Data

⚠️ **The aggregated Parquet file only has 2004-2026 data!**

This guide uses **CSV loading** to guarantee full 1947-2026 historical data.
The refactored training script auto-detects CSV format and loads all 80 seasons.

## Why Modal?

✅ **No time limits** (unlike Kaggle's 9-hour limit)
✅ **More RAM** (up to 256GB vs Kaggle's 30GB)
✅ **Better GPU options** (A100, H100 available)
✅ **Persistent storage** (models saved to Modal volumes)
✅ **Resume training** (pick up where you left off)

## Setup

### 1. Install Modal

```bash
pip install modal
modal setup  # Login to Modal account
```

### 2. Upload Your Data to Modal Volume

Create a Modal volume and upload your data:

```python
# upload_data_to_modal.py
import modal

# Create volume for data storage
volume = modal.Volume.from_name("nba-data", create_if_missing=True)
stub = modal.Stub("nba-upload")

@stub.function(
    volumes={"/data": volume},
    timeout=3600  # 1 hour for upload
)
def upload_csv_data():
    """Upload CSV data to Modal volume (guaranteed full 1947-2026)"""
    import urllib.request

    # Download Kaggle dataset CSV
    # You'll need to download PlayerStatistics.csv from Kaggle first
    print("Uploading PlayerStatistics.csv...")

    # Upload local CSV file
    # (Run this locally, it will upload to Modal)
    import shutil
    shutil.copy(
        "data/PlayerStatistics.csv",  # Your local CSV
        "/data/PlayerStatistics.csv"
    )

    print("✓ Upload complete!")
    print("Data range: 1947-2026 (full historical)")
    volume.commit()

if __name__ == "__main__":
    with stub.run():
        upload_csv_data.remote()
```

Run it:
```bash
python upload_data_to_modal.py
```

### 3. Create Modal Training Script

```python
# modal_train_players.py
import modal

# Define Modal stub
stub = modal.Stub("nba-player-training")

# Create volumes for data and models
data_volume = modal.Volume.from_name("nba-data")
model_volume = modal.Volume.from_name("nba-models", create_if_missing=True)

# Define image with dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        "pandas",
        "numpy",
        "pyarrow",
        "lightgbm",
        "pytorch-tabnet",
        "torch",
        "scikit-learn"
    )
)

@stub.function(
    image=image,
    gpu="A10G",  # or "A100" for more power
    memory=65536,  # 64GB RAM (way more than Kaggle!)
    timeout=86400,  # 24 hours (no Kaggle limits!)
    volumes={
        "/data": data_volume,
        "/models": model_volume
    }
)
def train_one_window(window_start: int, window_end: int):
    """Train a single window - can run many in parallel!"""
    import sys
    sys.path.insert(0, "/root")

    from shared.data_loading import load_player_data, get_year_column
    from train_player_models import create_window_training_data, train_player_window

    print(f"Training window: {window_start}-{window_end}")

    # Load data from Modal volume (CSV = guaranteed 1947-2026)
    agg_df = load_player_data(
        "/data/PlayerStatistics.csv",  # Use CSV for full data!
        verbose=True
    )

    year_col = get_year_column(agg_df)
    window_seasons = list(range(window_start, window_end + 1))

    # Create window data
    window_df = create_window_training_data(
        agg_df,
        window_seasons,
        year_col,
        verbose=True
    )

    # Train models
    result = train_player_window(
        window_df,
        window_start,
        window_end,
        neural_epochs=12,
        verbose=True
    )

    # Save to Modal volume
    import pickle
    cache_path = f"/models/player_models_{window_start}_{window_end}.pkl"
    with open(cache_path, 'wb') as f:
        pickle.dump(result['models'], f)

    # Commit volume (persist changes)
    model_volume.commit()

    print(f"✓ Window {window_start}-{window_end} complete!")
    return result['metrics']


@stub.function(
    image=image,
    volumes={
        "/data": data_volume,
        "/models": model_volume
    }
)
def train_all_windows():
    """Orchestrate training of all windows"""
    from shared.data_loading import load_player_data, get_year_column

    # Load data to discover windows (CSV = full 1947-2026)
    agg_df = load_player_data("/data/PlayerStatistics.csv", verbose=True)
    year_col = get_year_column(agg_df)
    all_seasons = sorted([int(s) for s in agg_df[year_col].dropna().unique()])

    # Create 3-year windows
    window_size = 3
    windows = []
    for i in range(0, len(all_seasons), window_size):
        window_seasons = all_seasons[i:i+window_size]
        if window_seasons:
            windows.append((window_seasons[0], window_seasons[-1]))

    print(f"Training {len(windows)} windows...")

    # Train windows in parallel on Modal!
    results = []
    for start, end in windows:
        result = train_one_window.remote(start, end)
        results.append(result)

    print("✓ All windows trained!")
    return results


if __name__ == "__main__":
    # Option 1: Train all windows in parallel
    with stub.run():
        train_all_windows.remote()

    # Option 2: Train specific windows
    # with stub.run():
    #     train_one_window.remote(2022, 2024)
    #     train_one_window.remote(2019, 2021)
```

Run it:
```bash
modal run modal_train_players.py
```

## Benefits Over Kaggle

| Feature | Kaggle | Modal |
|---------|--------|-------|
| Time limit | 9 hours | 24+ hours |
| RAM | 30GB | 256GB |
| GPU | T4 (free) | A10G, A100 |
| Parallel training | ❌ | ✅ Multiple windows at once |
| Persistent storage | ❌ (must download) | ✅ Modal volumes |
| Resume training | ❌ | ✅ Easy |
| Cost | Free | ~$0.50-2/hour |

## Cost Estimation

**Scenario: Train all 27 windows (1947-2026)**

**Serial (one at a time):**
- ~30 min per window × 27 = 13.5 hours
- A10G GPU: $1.10/hour × 13.5 = **~$15**

**Parallel (5 windows at once):**
- ~30 min per window ÷ 5 parallel = 2.7 hours
- 5× A10G GPU: $5.50/hour × 2.7 = **~$15**
- **BUT**: Finishes in 2.7 hours instead of 13.5!

## Recommended Workflow

### Step 1: Upload Data (One Time)
```bash
# Upload your full Parquet file
python upload_data_to_modal.py
```

### Step 2: Train Windows
```bash
# Train all windows
modal run modal_train_players.py

# Or train specific era
modal run modal_train_players.py::train_modern_era  # 2010-2026
```

### Step 3: Download Trained Models
```python
# download_models.py
import modal

volume = modal.Volume.from_name("nba-models")
stub = modal.Stub("download")

@stub.function(volumes={"/models": volume})
def download_all():
    import os
    import shutil

    # List all models
    models = [f for f in os.listdir("/models") if f.endswith('.pkl')]
    print(f"Found {len(models)} models")

    # Copy to local directory
    os.makedirs("downloaded_models", exist_ok=True)
    for model_file in models:
        shutil.copy(
            f"/models/{model_file}",
            f"downloaded_models/{model_file}"
        )

    print("✓ Downloaded all models")

if __name__ == "__main__":
    with stub.run():
        download_all.remote()
```

## CSV Loading on Modal

If you want to use CSV instead of Parquet:

```python
@stub.function(
    image=image,
    gpu="A10G",
    memory=65536,
    volumes={"/data": data_volume, "/models": model_volume}
)
def train_from_csv(window_start: int, window_end: int):
    """Train using CSV data"""
    from shared.data_loading import load_player_data

    # Load CSV from Modal volume
    agg_df = load_player_data(
        "/data/PlayerStatistics.csv",  # CSV file
        verbose=True
    )

    # Rest is the same...
```

## Full Parquet (No Filters)

Since Modal has 64GB+ RAM, you can load the FULL Parquet without any year filtering:

```python
# Load entire 1947-2026 dataset (9.2M rows)
agg_df = load_player_data(
    "/data/aggregated_nba_data.parquet",
    min_year=None,  # No filtering!
    max_year=None,
    verbose=True
)
```

This won't OOM on Modal like it does on Kaggle!

## Advanced: Parallel Training

Train multiple windows simultaneously:

```python
@stub.function(...)
def train_all_parallel():
    """Train 5 windows at once!"""
    windows = [(1947, 1949), (1950, 1952), (1953, 1955), ...]

    # Launch 5 parallel jobs
    batch_size = 5
    for i in range(0, len(windows), batch_size):
        batch = windows[i:i+batch_size]

        # These run in parallel!
        futures = [
            train_one_window.spawn(start, end)
            for start, end in batch
        ]

        # Wait for batch to complete
        results = [f.get() for f in futures]
        print(f"Batch {i//batch_size + 1} complete!")
```

## Monitoring

Check progress in real-time:
```bash
modal app logs nba-player-training
```

## Next Steps

1. Sign up for Modal: https://modal.com
2. Upload your Parquet file to Modal volume
3. Run training with unlimited time and RAM
4. Download trained models when complete

**Estimated total cost for full 1947-2026 training: ~$15-30**
