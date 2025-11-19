# Ready to Train - Quick Start Guide

✅ **Status**: NBA_COLAB_SIMPLE.ipynb updated to use aggregated CSV

---

## What's Ready

### ✅ Training System
- **train_auto.py**: Verified with ALL Phase 1-7 features (235 total)
- **neural_hybrid.py**: TabNet + LightGBM architecture ready
- **NBA_COLAB_SIMPLE.ipynb**: Updated to use `aggregated_nba_data.csv.gzip`

### ✅ Prediction System
- **predict_live_FINAL.py**: Complete with 150+ features, SHAP, betting integration
- **The Odds API**: Merged from riq_analyzer.py

### ⏳ Pending
- backtest_engine.py (can wait)
- Riq_Machine.ipynb (can wait)
- Evaluate_Predictions.ipynb (can wait)

---

## Steps to Start Training NOW

### Step 1: Upload Aggregated Data to Kaggle (5 minutes)

```bash
# On your local machine, create a Kaggle dataset

# Option A: Via Kaggle website
# 1. Go to https://www.kaggle.com/datasets
# 2. Click "New Dataset"
# 3. Upload aggregated_nba_data.csv.gzip
# 4. Title: "NBA Aggregated Training Data"
# 5. Make it private

# Option B: Via Kaggle CLI (if you have it)
cd C:\Users\tmiles11\nba_predictor\data
# Create a dataset-metadata.json file:
# {
#   "title": "NBA Aggregated Training Data",
#   "id": "tyriqmiles/nba-aggregated-training-data",
#   "licenses": [{"name": "CC0-1.0"}]
# }
kaggle datasets create -p .
```

### Step 2: Open Colab and Run Training (3-4 hours background)

1. **Upload Notebook to Colab**:
   - Go to https://colab.research.google.com/
   - File → Upload notebook
   - Select `NBA_COLAB_SIMPLE.ipynb`

2. **Set GPU Runtime**:
   - Runtime → Change runtime type
   - Hardware accelerator: GPU
   - GPU type: A100 (fastest) or L4

3. **Run Training Cell**:
   - The notebook will prompt to upload `aggregated_nba_data.csv.gzip`
   - Or download from your Kaggle dataset (if you uploaded there)
   - Training will run 3-4 hours in background

4. **Monitor Progress**:
   - Keep browser tab open
   - Training output will show progress
   - Models save incrementally (won't lose progress if interrupted)

### Step 3: Validate Embeddings (After Training)

Run the validation cell to verify:
- ✅ 24-dimensional embeddings extracted correctly
- ✅ TabNet + LightGBM hybrid working
- ✅ Uncertainty models trained

### Step 4: Download Models

Run the download cell to get `nba_models_trained.zip`

---

## Expected Training Output

```
======================================================================
🚀 TRAINING WITH AGGREGATED DATA (v4.0)
======================================================================

📊 Dataset: aggregated_nba_data.csv.gzip
   • Pre-computed 150+ features (Phase 1-7)
   • Basketball Reference priors merged
   • 2002-2026 seasons
   • ~125K player-games

🧠 Neural hybrid: TabNet (24-dim embeddings) + LightGBM
⚡ Training optimization:
   • No feature engineering needed (already done!)
   • Direct load → train workflow
   • Incremental model saving

⏱️  Expected time: ~3 hours (A100)

💡 Training 5 player props: minutes, points, rebounds, assists, threes
   Expected: Points MAE ~2.2-2.4 (vs baseline 2.5-2.7)
   Features: 235 total (150+ engineered + 68 priors + splits)
   Embeddings: 24-dimensional TabNet attention vectors

======================================================================

🔄 Loading aggregated data...
   Loaded 125,487 player-games with 235 features

🏋️ Training MINUTES model...
   TabNet training... [15 min]
   LightGBM training on raw + embeddings... [2 min]
   Sigma model training... [1 min]
   ✅ Minutes model saved

🏋️ Training POINTS model...
   TabNet training... [15 min]
   LightGBM training on raw + embeddings... [2 min]
   Sigma model training... [1 min]
   ✅ Points model saved

   MAE: 2.31 (baseline: 2.65) ← 12.8% improvement!
   RMSE: 3.12 (baseline: 3.45)
   R²: 0.68

... (3 more props)

======================================================================
✅ TRAINING COMPLETE!
======================================================================

📊 Models trained on aggregated data with 235 features
🧠 Neural hybrid: TabNet 24-dim embeddings + LightGBM

Models saved:
  • models/minutes_hybrid_2002_2026.pkl
  • models/points_hybrid_2002_2026.pkl
  • models/rebounds_hybrid_2002_2026.pkl
  • models/assists_hybrid_2002_2026.pkl
  • models/threes_hybrid_2002_2026.pkl

Next: Run the 'Download Models' cell
```

---

## What to Do While Training Runs (3-4 hours)

### Option 1: Let it run, come back later
Just keep browser tab open and check back in 3-4 hours.

### Option 2: Work on other tasks
- Complete backtest_engine.py
- Update Riq_Machine.ipynb
- Update Evaluate_Predictions.ipynb
- Test predict_live_FINAL.py locally (with old models)

### Option 3: Monitor and debug
- Watch training output for errors
- Check embedding validation after each model
- Ensure GPU utilization is high (watch Colab metrics)

---

## After Training: Next Steps

### Immediate (Day 1)
1. ✅ Download trained models
2. Test predict_live_FINAL.py with new models
3. Validate embeddings are 24-dim
4. Run predictions on last week's games (Nov 1-8)

### Week 1 (Phase 1)
5. Verify MAE < 2.5 for points
6. Check 80% intervals cover 78-82% of actuals
7. Test SHAP explanations make sense

### Week 2 (Phase 2)
8. Full backtest Oct-Nov 2024
9. Analyze weaknesses
10. Calibrate safe margin for betting

---

## Configuration Used in Training

```python
# train_auto.py arguments
--dataset /content/aggregated_nba_data.csv.gzip  # Pre-computed features
--use-neural                                      # Enable TabNet
--neural-epochs 30                                # 30 epochs (sweet spot)
--neural-device gpu                               # Use GPU
--verbose                                         # Show progress
--fresh                                           # Train from scratch
--skip-game-models                                # Only player models
```

### TabNet Hyperparameters (in neural_hybrid.py)
```python
n_d = 24                    # Decision layer width (24-dim embeddings)
n_a = 24                    # Attention layer width
n_steps = 4                 # Sequential attention steps
gamma = 1.5                 # Feature reuse coefficient
lambda_sparse = 1e-4        # Sparsity regularization
momentum = 0.3              # Batch norm momentum
learning_rate = 2e-2        # AdamW learning rate
batch_size = 2048           # Large batch for stability
```

### LightGBM Hyperparameters (in train_auto.py)
```python
num_leaves = 31             # Leaf complexity
learning_rate = 0.05        # Boosting learning rate
n_estimators = 500          # Number of trees
min_child_samples = 20      # Min samples per leaf
subsample = 0.8             # Row sampling
colsample_bytree = 0.8      # Column sampling
```

---

## Troubleshooting

### "FileNotFoundError: aggregated_nba_data.csv.gzip not found"
- Make sure you uploaded the file to Colab
- Check file is in `/content/` directory
- Re-run upload cell if needed

### "CUDA out of memory"
- Runtime → Disconnect and delete runtime
- Restart with fresh GPU
- Or reduce batch_size in neural_hybrid.py

### "Models not saving"
- Check `/content/nba_predictor/models/` directory exists
- Models save incrementally (one at a time)
- Even if training crashes, you keep completed models

### "Training taking longer than expected"
- A100: 3 hours
- L4: 3.5 hours
- T4: 4 hours
- CPU: Don't even try (20+ hours)

### "Low accuracy (MAE > 3.0)"
- Check aggregated data has all 235 features
- Verify embeddings are 24-dim (run validation cell)
- Check for NaN values in data
- Ensure train_auto.py uses optimization_features.py

---

## Files Involved

### Training Files (in GitHub repo)
```
nba_predictor/
├── train_auto.py                    # Main training script (uses aggregated CSV)
├── neural_hybrid.py                 # TabNet + LightGBM architecture
├── optimization_features.py         # Phase 6 features (not called with aggregated data)
├── phase7_features.py              # Phase 7 features (not called with aggregated data)
└── NBA_COLAB_SIMPLE.ipynb          # Training notebook (UPDATED)
```

### Data Files (upload to Colab)
```
aggregated_nba_data.csv.gzip        # Pre-computed 235 features (125K rows)
```

### Output Files (download after training)
```
models/
├── minutes_hybrid_2002_2026.pkl    # Minutes predictor
├── points_hybrid_2002_2026.pkl     # Points predictor
├── rebounds_hybrid_2002_2026.pkl   # Rebounds predictor
├── assists_hybrid_2002_2026.pkl    # Assists predictor
└── threes_hybrid_2002_2026.pkl     # Threes predictor

model_cache/                         # Cached intermediate models (optional)
```

---

## Success Criteria

After training completes, you should have:

✅ **5 trained models** (minutes, points, rebounds, assists, threes)
✅ **Points MAE < 2.5** (target: 2.2-2.4)
✅ **24-dimensional embeddings** (verified via validation cell)
✅ **TabNet + LightGBM hybrid** working correctly
✅ **Uncertainty models** for prediction intervals

If you see these, training was successful! 🎉

---

## What Makes This Different from Previous Training

### Old Way (PlayerStatistics.csv)
- Load raw player stats
- Compute all 235 features from scratch (45 min)
- Train models (3 hours)
- **Total: 3.75 hours**

### New Way (aggregated_nba_data.csv.gzip)
- Load pre-computed 235 features (30 sec)
- Train models (3 hours)
- **Total: 3 hours**

### Advantages
- ✅ 45 minutes saved (no feature engineering)
- ✅ Features match predict_live_FINAL.py exactly
- ✅ Basketball Reference priors already merged
- ✅ Reproducible (same features every time)
- ✅ Less code to maintain

---

## Ready to Go!

**You are now ready to start training.** Just:

1. Upload `aggregated_nba_data.csv.gzip` to Kaggle (or Colab)
2. Open `NBA_COLAB_SIMPLE.ipynb` in Colab
3. Select GPU runtime (A100/L4)
4. Run the training cell
5. Wait 3-4 hours
6. Download models

While training runs in background, you can work on:
- backtest_engine.py completion
- Updating other notebooks
- Testing betting integration locally

**Good luck! 🚀**
