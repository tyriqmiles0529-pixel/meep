# Meta-Learner Training on Kaggle (GPU Required)

**Problem**: Window models were trained on GPU and can't run on Modal's CPU-only containers.

**Solution**: Train meta-learner on Kaggle with free GPU, then upload trained model to Modal.

---

## Why Kaggle Instead of Modal?

1. **Device Compatibility**: TabNet models trained on GPU → need GPU for inference
2. **Free GPU**: Kaggle provides free T4 GPU (16GB VRAM)
3. **Data Already There**: PlayerStatistics.csv is already on Kaggle dataset
4. **Simple Upload**: Once trained, upload .pkl file to Modal for production use

---

## Step-by-Step Workflow

### 1. Prepare Window Models

First, download window models from Modal to your local machine:

```bash
# Download all window models from Modal
modal volume get nba-models model_cache/ ./kaggle_models/

# You should see:
# - player_models_1947_1949.pkl
# - player_models_1950_1952.pkl
# - ... (27 total)
```

### 2. Create Kaggle Dataset for Window Models

1. Go to: https://www.kaggle.com/datasets
2. Click "New Dataset"
3. Upload all `.pkl` files from `./kaggle_models/`
4. Name: "nba-window-models"
5. Make it private (models contain proprietary training)
6. Click "Create"

### 3. Create Kaggle Dataset for Code Files

Upload these files as a dataset:

```bash
# Files to upload:
- ensemble_predictor.py
- meta_learner_ensemble.py
- hybrid_multi_task.py
- optimization_features.py
- phase7_features.py
- rolling_features.py
- shared/ (entire directory)
- priors_data/ (entire directory)
```

**Steps:**
1. Create new dataset: "nba-predictor-code"
2. Upload all files above
3. Make it private
4. Click "Create"

### 4. Create Kaggle Notebook

1. Go to: https://www.kaggle.com/code
2. Click "New Notebook"
3. Click "File" → "Upload Notebook"
4. Upload: `train_meta_learner_kaggle.ipynb`

### 5. Configure Notebook Settings

**Important settings:**

- **Accelerator**: GPU T4 x2 (or P100)
  - Click "Settings" → "Accelerator" → "GPU T4 x2"

- **Internet**: ON
  - Click "Settings" → "Internet" → ON

- **Dataset Dependencies**:
  1. Click "Add Data" → "Your Datasets"
  2. Add "nba-window-models"
  3. Add "nba-predictor-code"
  4. Add "historical-nba-data-and-player-box-scores" (Eoin Moore's dataset)

### 6. Run the Notebook

Click "Run All" and wait ~30-60 minutes.

**Expected output:**
```
Loading 27 window models...
✓ Loaded 27 windows

COLLECTING PREDICTIONS: POINTS
==================================================
  Processed 500/5000 games... (non-zero preds: 27/27)
  Processed 1000/5000 games... (non-zero preds: 27/27)
  ...
  ✓ Collected 4,873 samples

TRAINING META-LEARNER
==================================================
  points      :  4,873 samples, +12.3% improvement
  rebounds    :  4,901 samples, +8.7% improvement
  assists     :  4,865 samples, +15.1% improvement
  threes      :  4,822 samples, +9.4% improvement

✅ Saved: meta_learner_2025_2026.pkl
```

### 7. Download Trained Meta-Learner

1. Click "Output" tab (bottom right)
2. Download `meta_learner_2025_2026.pkl` (~50MB)
3. Save to your local machine

### 8. Upload to Modal

```bash
# Upload to Modal volume
modal volume put nba-models meta_learner_2025_2026.pkl

# Verify upload
modal volume ls nba-models

# Should see:
# - meta_learner_2025_2026.pkl (50MB)
# - player_models_*.pkl (27 files)
```

### 9. Test with Analyzer

```bash
# Run analyzer on Modal (will now use meta-learner)
modal run modal_analyzer.py

# Expected output:
# [OK] Loaded meta-learner from /models/meta_learner_2025_2026.pkl
# [OK] 27 window models ready
# ...
```

---

## Troubleshooting

### Q: "Window predictions all zero"
**A**: Window models failed to load. Check:
1. All 27 .pkl files uploaded to Kaggle dataset?
2. Dataset added as dependency in notebook?
3. GPU enabled in notebook settings?

### Q: "Not enough samples"
**A**: 2024-2025 season has limited data. Try:
1. Increase `sample_size` from 5000 to 10000
2. Include 2023-2024 season: `season_start_year = [2023, 2024]`

### Q: "GPU out of memory"
**A**: Reduce batch size:
1. In notebook, change `sample_size=5000` to `sample_size=2000`
2. Or use "GPU T4 x1" instead of "T4 x2"

### Q: "Can I use Colab instead?"
**A**: Yes! Same process:
1. Upload notebook to Colab
2. Upload window models to Google Drive
3. Mount Drive: `from google.colab import drive; drive.mount('/content/drive')`
4. Change paths: `model_cache_dir = "/content/drive/MyDrive/nba_models/"`

---

## Monthly Update Workflow

When new data is available (monthly):

1. **Run on Kaggle**: Train meta-learner with latest data
2. **Download**: Get new `meta_learner_YYYY_YYYY.pkl`
3. **Upload to Modal**: `modal volume put nba-models meta_learner_YYYY_YYYY.pkl`
4. **Update code**: Change `meta_learner_2025_2026.pkl` → `meta_learner_2025_2026.pkl` in `ensemble_predictor.py`

---

## Cost Comparison

| Platform | GPU | Cost | Training Time |
|----------|-----|------|---------------|
| **Kaggle** | T4 (free) | $0 | 30-60 min |
| **Colab** | T4 (free) | $0 | 30-60 min |
| **Modal** | T4 | $0.60/hr | 30 min = $0.30 |
| **Modal** | A100 | $4.00/hr | 15 min = $1.00 |

**Recommendation**: Use Kaggle (free, easy, reliable)
