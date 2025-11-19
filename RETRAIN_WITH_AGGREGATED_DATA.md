# Retraining Guide: Aggregated Data + TabNet Embeddings

## Overview

You need to:
1. Upload `aggregated_nba_data.csv.gzip` to Kaggle as a dataset
2. Download it locally
3. Retrain models with full TabNet 24-dim embeddings
4. Validate embeddings are working correctly

---

## Step 1: Upload Aggregated Data to Kaggle

### In Your Kaggle Notebook (where aggregated data was created):

```python
# Cell 1: Verify aggregated file exists
import os
file_path = '/kaggle/working/aggregated_nba_data.csv.gzip'
if os.path.exists(file_path):
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"✓ Found aggregated data: {file_size_mb:.2f} MB")
else:
    print("❌ Aggregated data not found! Run aggregation script first.")
```

```python
# Cell 2: Create dataset metadata
import json

metadata = {
    "title": "Aggregated NBA Player Statistics (2002-2026)",
    "id": "tyriqmiles/aggregated-nba-data",
    "licenses": [{"name": "CC0-1.0"}],
    "keywords": ["basketball", "nba", "sports", "player-props", "feature-engineering"],
    "resources": [
        {
            "path": "aggregated_nba_data.csv.gzip",
            "description": "Pre-aggregated NBA player statistics with 150+ engineered features including rolling averages, team context, opponent matchups, Basketball Reference priors, momentum features, and advanced stats. Optimized for player prop predictions (2002-2026)."
        }
    ],
    "description": """# Aggregated NBA Player Statistics Dataset

## Features (150+):
- **Phase 1**: Shot volume (FGA, 3PA, FTA) + efficiency (TS%, eFG%, FT%)
- **Phase 2**: Team context (pace, off/def strength, recent form)
- **Phase 3**: Advanced stats (usage rate, rebound rate, assist rate)
- **Phase 4**: Positional encoding, starter status
- **Phase 5**: Home/away performance splits
- **Phase 6**: Momentum features (trend detection, acceleration, streaks)
- **Phase 7**: Fatigue (B2B, rest days), variance/consistency, ceiling/floor
- **Priors**: Basketball Reference O-Rtg, D-Rtg, Four Factors

## Use Cases:
- NBA player prop predictions (points, rebounds, assists, 3PM, minutes)
- Daily fantasy sports (DFS) optimization
- Sports betting analytics
- Research on player performance modeling

## Data Quality:
- ~1.6M player-game rows (2002-2026)
- All features pre-computed and leakage-safe
- Ready for TabNet + LightGBM hybrid models

## Source:
Aggregated from [Historical NBA Data](https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores)
with Basketball Reference statistical priors.
"""
}

with open('dataset-metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("✓ Metadata created")
```

```python
# Cell 3: Upload to Kaggle
!kaggle datasets create -p /kaggle/working

# If dataset already exists, update instead:
# !kaggle datasets version -p /kaggle/working -m "Updated with latest features"
```

**You'll get a URL like**: `https://www.kaggle.com/datasets/tyriqmiles/aggregated-nba-data`

---

## Step 2: Download Aggregated Data Locally

### On Your Local Machine:

```bash
# Make sure data directory exists
cd C:\Users\tmiles11\nba_predictor
mkdir data 2>nul

# Download the aggregated dataset
python -c "import kaggle; kaggle.api.dataset_download_files('tyriqmiles/aggregated-nba-data', path='./data', unzip=True)"
```

**Verify download**:
```bash
python -c "import os; print(f'File size: {os.path.getsize(\"./data/aggregated_nba_data.csv.gzip\") / (1024**2):.2f} MB')"
```

Expected output: `File size: ~400-450 MB`

---

## Step 3: Retrain Models with TabNet Embeddings

### Option A: Train Locally (if you have 16+ GB RAM)

```bash
# Install dependencies
pip install pytorch-tabnet torch lightgbm pandas numpy scikit-learn

# Run training with neural hybrid
python train_auto.py \
    --dataset ./data/aggregated_nba_data.csv.gzip \
    --use-neural \
    --neural-epochs 30 \
    --neural-device auto \
    --verbose \
    --fresh
```

**Expected training time**:
- TabNet training: 30-60 minutes per prop
- LightGBM training: 5-10 minutes per prop
- **Total**: ~3-4 hours for all 5 props

---

### Option B: Train in Google Colab (Recommended for GPU)

**File**: `NBA_COLAB_SIMPLE.ipynb`

```python
# Cell 1: Setup
!pip install -q pytorch-tabnet torch lightgbm kaggle

# Configure Kaggle credentials
import os
from google.colab import userdata

os.environ['KAGGLE_USERNAME'] = 'tyriqmiles'
os.environ['KAGGLE_KEY'] = userdata.get('KAGGLE_KEY')  # Store key in Colab secrets
```

```python
# Cell 2: Download aggregated data
!kaggle datasets download -d tyriqmiles/aggregated-nba-data
!gunzip aggregated_nba_data.csv.gzip
```

```python
# Cell 3: Upload training script
from google.colab import files
uploaded = files.upload()  # Upload train_auto.py, neural_hybrid.py, etc.
```

```python
# Cell 4: Train with GPU
!python train_auto.py \
    --dataset ./aggregated_nba_data.csv \
    --use-neural \
    --neural-epochs 30 \
    --neural-device gpu \
    --verbose \
    --fresh \
    --output-dir ./models
```

```python
# Cell 5: Verify embeddings are working
import pickle
import numpy as np

# Load a trained model
with open('./models/points_hybrid_2022_2026.pkl', 'rb') as f:
    model = pickle.load(f)

# Check TabNet embedding dimension
if hasattr(model, 'tabnet'):
    # Create dummy input (match your feature count)
    dummy_X = np.random.randn(10, 150).astype(np.float32)

    # Get embeddings
    _, embeddings = model.tabnet.predict(dummy_X, return_embeddings=True)

    print(f"✓ TabNet embeddings: {embeddings.shape}")
    print(f"  Expected: (10, 24)")
    print(f"  Actual: {embeddings.shape}")

    if embeddings.shape[1] == 24:
        print("\n✅ 24-dimensional embeddings working correctly!")
    else:
        print(f"\n⚠️  Unexpected embedding dimension: {embeddings.shape[1]}")
else:
    print("❌ TabNet not found in model")
```

```python
# Cell 6: Download trained models
!zip -r models.zip models/
from google.colab import files
files.download('models.zip')
```

---

## Step 4: Validate Embedding Quality

### Test 1: Embedding Dimension Check

```python
import pickle
import numpy as np
import pandas as pd

# Load model
with open('./models/points_hybrid_2022_2026.pkl', 'rb') as f:
    model = pickle.load(f)

# Load sample data
df = pd.read_csv('./data/aggregated_nba_data.csv.gzip', nrows=100)

# Prepare features (match training features)
feature_cols = [c for c in df.columns if c not in ['points', 'rebounds', 'assists', 'threes', 'minutes']]
X_sample = df[feature_cols]

# Get embeddings
if hasattr(model, 'tabnet'):
    _, embeddings = model.tabnet.predict(X_sample.values.astype(np.float32), return_embeddings=True)

    print(f"✓ Embedding shape: {embeddings.shape}")
    print(f"✓ Expected: (100, 24)")
    print(f"✓ Embedding mean: {embeddings.mean():.4f}")
    print(f"✓ Embedding std: {embeddings.std():.4f}")

    # Check for degenerate embeddings (all zeros)
    if np.abs(embeddings).sum() < 1e-6:
        print("❌ WARNING: Embeddings are all zeros!")
    else:
        print("✅ Embeddings have meaningful values")
```

---

### Test 2: Embedding Visualization (t-SNE)

```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load more data for better visualization
df = pd.read_csv('./data/aggregated_nba_data.csv.gzip', nrows=1000)
X_sample = df[feature_cols]
y_sample = df['points']

# Get embeddings
_, embeddings = model.tabnet.predict(X_sample.values.astype(np.float32), return_embeddings=True)

# Reduce to 2D with t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot colored by points scored
plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    embeddings_2d[:, 0],
    embeddings_2d[:, 1],
    c=y_sample,
    cmap='viridis',
    alpha=0.6,
    s=50
)
plt.colorbar(scatter, label='Points Scored')
plt.title('TabNet 24-dim Embeddings (projected to 2D via t-SNE)')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.tight_layout()
plt.savefig('embeddings_tsne.png', dpi=150)
plt.show()

print("✓ Embedding visualization saved to embeddings_tsne.png")
```

**Expected result**: Points should cluster by value (high scorers separate from low scorers)

---

### Test 3: Embedding Feature Importance

```python
# Check how much LightGBM uses the embeddings vs raw features
import lightgbm as lgb

# Get feature importance
if hasattr(model, 'lgbm'):
    importance = model.lgbm.feature_importance(importance_type='gain')

    # Assuming last 24 features are embeddings
    n_raw = len(feature_cols)
    n_emb = 24

    raw_importance = importance[:n_raw].sum()
    emb_importance = importance[n_raw:n_raw+n_emb].sum()
    total = importance.sum()

    print(f"Feature Importance Breakdown:")
    print(f"  Raw features ({n_raw}): {raw_importance/total*100:.1f}%")
    print(f"  Embeddings (24): {emb_importance/total*100:.1f}%")

    if emb_importance / total > 0.15:
        print(f"\n✅ Embeddings contribute {emb_importance/total*100:.1f}% (healthy)")
    else:
        print(f"\n⚠️  Embeddings only contribute {emb_importance/total*100:.1f}% (low)")
```

**Healthy range**: Embeddings should contribute **15-40%** of total importance

---

## Step 5: Compare Performance (With vs Without Embeddings)

### Baseline Test: LightGBM Only

```python
# Train baseline LightGBM (no TabNet)
from sklearn.model_selection import train_test_split
import lightgbm as lgb

df = pd.read_csv('./data/aggregated_nba_data.csv.gzip')
X = df[feature_cols]
y = df['points']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# LightGBM baseline
lgb_baseline = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42
)
lgb_baseline.fit(X_train, y_train)

y_pred_baseline = lgb_baseline.predict(X_val)
rmse_baseline = np.sqrt(mean_squared_error(y_val, y_pred_baseline))
mae_baseline = mean_absolute_error(y_val, y_pred_baseline)

print(f"LightGBM Baseline (no embeddings):")
print(f"  RMSE: {rmse_baseline:.3f}")
print(f"  MAE: {mae_baseline:.3f}")
```

### Hybrid Test: TabNet + LightGBM

```python
# Your hybrid model
y_pred_hybrid = model.predict(X_val)
rmse_hybrid = np.sqrt(mean_squared_error(y_val, y_pred_hybrid))
mae_hybrid = mean_absolute_error(y_val, y_pred_hybrid)

print(f"\nHybrid Model (TabNet + LightGBM):")
print(f"  RMSE: {rmse_hybrid:.3f}")
print(f"  MAE: {mae_hybrid:.3f}")

print(f"\nImprovement:")
print(f"  RMSE: {(rmse_baseline - rmse_hybrid)/rmse_baseline*100:+.1f}%")
print(f"  MAE: {(mae_baseline - mae_hybrid)/mae_baseline*100:+.1f}%")
```

**Expected improvement**: 5-15% better RMSE/MAE with embeddings

---

## Troubleshooting

### Issue 1: "TabNet embeddings are all zeros"

**Cause**: TabNet didn't train properly (NaN loss, early stopping)

**Fix**:
```python
# Check TabNet training logs
# Look for:
# - Loss decreasing (should go from ~10 → ~3)
# - No NaN warnings
# - At least 10 epochs completed

# Retrain with more epochs and lower learning rate
model.tabnet_params['optimizer_params']['lr'] = 0.01  # Lower from 0.02
model.fit(X_train, y_train, epochs=50)  # More epochs
```

---

### Issue 2: "Embeddings contribute <5% importance"

**Cause**: Embeddings are redundant (raw features already capture patterns)

**Fix**:
```python
# Increase TabNet's embedding power
model.tabnet_params['n_d'] = 32  # Increase from 24
model.tabnet_params['n_steps'] = 5  # More attention steps
model.tabnet_params['lambda_sparse'] = 1e-5  # Less sparsity (was 1e-4)
```

---

### Issue 3: "Out of memory during training"

**Cause**: TabNet + full dataset too large

**Fix**:
```python
# Use batch processing
CHUNK_SIZE = 100000

for chunk in pd.read_csv('./data/aggregated_nba_data.csv.gzip', chunksize=CHUNK_SIZE):
    # Process chunk
    X_chunk = chunk[feature_cols]
    y_chunk = chunk['points']

    # Partial fit (requires warm-start)
    model.fit(X_chunk, y_chunk, epochs=10)
```

Or use Colab with GPU (recommended)

---

## Success Criteria

✅ **Embeddings working correctly if**:
- Embedding shape is `(n_samples, 24)`
- Embeddings are NOT all zeros (mean ≠ 0, std > 0.1)
- t-SNE visualization shows clustering by target value
- Embeddings contribute 15-40% of LightGBM importance
- Hybrid model beats baseline by 5-15%

---

## Timeline

**Estimated time**:
1. Upload aggregated data to Kaggle: **5 min**
2. Download locally: **2 min**
3. Retrain in Colab (GPU): **2-3 hours**
4. Validate embeddings: **15 min**
5. Download models: **5 min**

**Total: ~3-4 hours** (mostly waiting for training)

---

## Next Steps After Retraining

Once models are trained with proper embeddings:
1. ✅ Validate 24-dim embeddings working
2. ✅ Verify improvement over baseline
3. → Complete `predict_live.py` feature engineering
4. → Test predictions on last week's games
5. → Update notebooks with new models
6. → Begin backtesting on October-November 2024

---

## Quick Commands Summary

```bash
# 1. Download aggregated data
python -c "import kaggle; kaggle.api.dataset_download_files('tyriqmiles/aggregated-nba-data', path='./data', unzip=True)"

# 2. Train with embeddings (local)
python train_auto.py --dataset ./data/aggregated_nba_data.csv.gzip --use-neural --neural-epochs 30 --verbose --fresh

# 3. Validate embeddings
python -c "
import pickle
with open('./models/points_hybrid_2022_2026.pkl', 'rb') as f:
    model = pickle.load(f)
print('Embedding check:', hasattr(model, 'tabnet'))
"

# 4. Extract models
unzip models.zip -d ./models/
```

Let me know when you're ready to upload the aggregated data to Kaggle, and I can help with any issues during retraining!
