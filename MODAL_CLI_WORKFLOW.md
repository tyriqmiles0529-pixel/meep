# Modal CLI Workflow - Run Everything Locally!

## How It Works

You write Python scripts **locally** on your computer, then run them **remotely** on Modal's cloud infrastructure using the `modal` CLI.

```
Your Laptop              Modal Cloud
┌─────────────┐         ┌──────────────────┐
│             │         │                  │
│  Python     │ modal   │  A10G GPU        │
│  Script  ───┼────────>│  64GB RAM        │
│             │  run    │  Runs your code  │
│             │<────────│  Returns output  │
└─────────────┘         └──────────────────┘
```

## Workflow

### Step 1: Write Script Locally (Your Laptop)

```python
# modal_train.py (on your laptop)
import modal

app = modal.App("my-app")

@app.function(gpu="A10G", memory=65536)
def train_model():
    print("Training on Modal's A10G GPU!")
    # Your training code here
    return "Model trained!"

@app.local_entrypoint()
def main():
    result = train_model.remote()  # Runs on Modal
    print(f"Result: {result}")      # Prints on your laptop
```

### Step 2: Run From Your Laptop

```bash
# This runs ON MODAL (not your laptop!)
modal run modal_train.py
```

What happens:
1. Modal CLI uploads your code to Modal cloud
2. Code runs on Modal's A10G GPU with 64GB RAM
3. Output streams back to your laptop terminal in real-time
4. You see all logs locally!

### Step 3: Monitor Locally

```bash
# All running in your local terminal!
modal app logs my-app      # See logs
modal app list             # List running functions
modal app stop my-app      # Stop training
```

## Complete Example: Upload & Train

### File 1: Upload Data (Runs Once)

```python
# modal_upload_data.py
import modal

app = modal.App("upload")
volume = modal.Volume.from_name("nba-data", create_if_missing=True)

@app.function(volumes={"/data": volume})
def upload_local_file(filepath: str):
    """Upload file from YOUR laptop to Modal"""
    import shutil
    # filepath is on YOUR laptop
    # /data/ is on Modal
    shutil.copy(filepath, "/data/PlayerStatistics.csv")
    volume.commit()  # Save to Modal
    print("✓ Uploaded!")

@app.local_entrypoint()
def main():
    # This runs the upload
    upload_local_file.remote("C:/Users/you/Downloads/PlayerStatistics.csv")
```

Run from your laptop:
```bash
modal run modal_upload_data.py
```

### File 2: Train (Runs on Modal GPU)

```python
# modal_train.py
import modal

app = modal.App("train")
volume = modal.Volume.from_name("nba-data")

@app.function(
    gpu="A10G",          # Modal's GPU
    memory=65536,        # 64GB RAM on Modal
    volumes={"/data": volume}
)
def train():
    import pandas as pd

    # Load data from Modal volume
    df = pd.read_csv("/data/PlayerStatistics.csv")
    print(f"Loaded {len(df)} rows on Modal GPU!")

    # Train model...
    print("Training on A10G...")

    return "Done!"

@app.local_entrypoint()
def main():
    result = train.remote()  # Runs on Modal, returns to laptop
    print(f"Training complete: {result}")
```

Run from your laptop:
```bash
modal run modal_train.py
```

Output appears in your local terminal!

## Key Commands

All run from your **local** terminal:

### Run a Script (On Modal Cloud)
```bash
modal run script.py
```

### Monitor Running Apps
```bash
modal app list             # Show running apps
modal app logs my-app      # Stream logs to laptop
modal app stop my-app      # Stop remote execution
```

### Manage Volumes
```bash
modal volume list          # List volumes
modal volume get nba-data PlayerStatistics.csv  # Download file to laptop!
```

### Manage Secrets
```bash
modal secret create kaggle-secret KAGGLE_KEY=abc123
modal secret list
```

## File Transfer

### Upload Local File to Modal

```python
@app.function(volumes={"/data": volume})
def upload():
    import shutil
    shutil.copy("local_file.csv", "/data/file.csv")
    volume.commit()
```

### Download Modal File to Laptop

```bash
# Download entire volume
modal volume get nba-models .

# Download specific file
modal volume get nba-models player_models_2022_2024.pkl
```

## Your Workflow

```bash
# 1. Write scripts locally (VS Code, etc.)
code modal_train.py

# 2. Run on Modal (from laptop terminal)
modal run modal_train.py

# 3. Monitor (from laptop terminal)
modal app logs nba-training

# 4. Download results (to laptop)
modal volume get nba-models .
```

**You never leave your laptop!** All heavy computation runs on Modal's cloud.

## Advantages

| Your Laptop | Modal Cloud |
|-------------|-------------|
| Write code | Execute code |
| See output | Provide GPU/RAM |
| Control execution | Store data/models |
| Download results | Handle scaling |

**Cost**: Only pay for Modal execution time (~$1/hour for A10G)
**Your laptop**: Stays cool and battery-friendly!

## Next Step

1. Install Modal: `pip install modal`
2. Setup: `modal setup`
3. Create `modal_train.py` locally
4. Run: `modal run modal_train.py`
5. Watch training happen on Modal from your laptop!

That's it! No Jupyter notebooks required - just local Python scripts running remotely.
