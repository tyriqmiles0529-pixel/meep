# Kaggle Notebook - Complete Usage Guide

## ✅ Notebook is Ready!

**GitHub:** https://github.com/tyriqmiles0529-pixel/meep/blob/main/NBA_COLAB_SIMPLE.ipynb

---

## Quick Start (5 Steps)

### 1. Create Kaggle Notebook
- Go to https://kaggle.com/code
- Click "New Notebook"

### 2. Add Dataset
- Click "Add Data" (right sidebar)
- Search: **"meeper"**
- Click "Add" on your dataset

### 3. Enable GPU
- Settings → Accelerator → **GPU P100 or T4**
- Internet: **On** (required for git clone)

### 4. Import Notebook
- File → Import Notebook
- Paste: `https://github.com/tyriqmiles0529-pixel/meep/blob/main/NBA_COLAB_SIMPLE.ipynb`
- Click "Import"

### 5. Run Cells
- **Run Cell 1** (Setup - 2 min) ← **MUST RUN THIS FIRST!**
- **Run Cell 2** (Training - 7-8 hours)
- Close browser (Kaggle keeps running)
- Come back later
- Run Cells 3-6 (validation, summary, download)

---

## ⚠️ IMPORTANT: Run Order

### ✅ CORRECT Order:
1. **Cell 1** (Setup) - Clones GitHub repo
2. **Cell 2** (Training) - Trains models
3. **Cell 3** (Validation) - Checks embeddings
4. **Cell 4** (Summary) - Lists models
5. **Cell 5** (Download) - Packages models

### ❌ WRONG Order:
- Running Cell 2 before Cell 1 = **FileNotFoundError**
- The training cell expects `/kaggle/working/meep` to exist
- Cell 1 creates this directory by cloning the repo

---

## What Each Cell Does

### Cell 0 (Markdown)
- Header and documentation
- Just informational, doesn't run

### Cell 1 (Setup) - RUN FIRST! ⭐
```python
# Installs packages
!pip install pytorch-tabnet lightgbm scikit-learn pandas numpy tqdm

# Clones your GitHub repo
!git clone https://github.com/tyriqmiles0529-pixel/meep.git
os.chdir('meep')

# Verifies dataset exists
# Checks GPU
```

**Runtime:** ~2 minutes
**Output:** "Setup complete!"

### Cell 2 (Training) - THE MAIN EVENT
```python
!python train_auto.py \
    --aggregated-data /kaggle/input/meeper/aggregated_nba_data.csv.gzip \
    --use-neural \
    --game-neural \
    --neural-epochs 30 \
    --neural-device gpu \
    --verbose \
    --no-window-ensemble
```

**Runtime:** ~7-8 hours
**Output:** 7 trained models (2 game + 5 player)

### Cell 3 (Validation)
- Loads points model
- Checks 24-dim embeddings work
- Validates neural hybrid architecture

**Runtime:** ~1 minute

### Cell 4 (Summary)
- Lists all trained models
- Shows file sizes
- Checks model types

**Runtime:** ~10 seconds

### Cell 5 (Download)
- Packages models into `nba_models_trained.zip`
- Shows download instructions

**Runtime:** ~1 minute

### Cell 6 (Markdown)
- Documentation
- Troubleshooting guide
- Just informational

---

## Expected Output (Cell 1)

```
Installing packages...
  Installing pytorch-tabnet... done
  Installing lightgbm... done
  ...

Downloading training code from GitHub...
Cloning into 'meep'...
remote: Enumerating objects: 1234, done.
remote: Counting objects: 100% (1234/1234), done.
...

Code version:
9fd9c38 ADD: Setup cell to clone GitHub repo

GPU: Tesla P100-PCIE-16GB

Dataset found: 142.3 MB
   Path: /kaggle/input/meeper/aggregated_nba_data.csv.gzip
   Full NBA history: 1947-2026 (80 seasons, 1.6M player-games)
   Training will use: ALL DATA (no cutoff)

Setup complete!
```

---

## Expected Output (Cell 2)

```
======================================================================
NBA NEURAL HYBRID TRAINING - GAME + PLAYER MODELS
======================================================================

Dataset Info:
   Source: /kaggle/input/meeper/aggregated_nba_data.csv.gzip
   Full range: 1947-2026 (80 seasons, 1.6M player-games)
   Training on: ALL DATA (no cutoff)

Expected time: ~7-8 hours total (P100)

Models to train:
   Game: Moneyline (win probability), Spread (margin)
   Player: Minutes, Points, Rebounds, Assists, Threes

======================================================================
STARTING TRAINING...
======================================================================

Loading Pre-Aggregated Dataset
----------------------------------------------------------------------
- Loading from: /kaggle/input/meeper/aggregated_nba_data.csv.gzip
- Loaded 1,632,909 rows
- Reconstructing game-level data from aggregated file...
- Created games_df with 32,451 unique games.

... (7-8 hours of training)

======================================================================
TRAINING COMPLETE!
======================================================================

Models saved to: /kaggle/working/meep/models/
```

---

## Troubleshooting

### Error: "FileNotFoundError: [Errno 2] No such directory: '/kaggle/working/meep'"

**Cause:** You ran Cell 2 before Cell 1

**Solution:**
1. Click Cell 1 (Setup)
2. Click "Run" or press Shift+Enter
3. Wait for "Setup complete!"
4. THEN run Cell 2

---

### Error: "Dataset not found"

**Cause:** You didn't add the "meeper" dataset

**Solution:**
1. Click "Add Data" (right sidebar)
2. Search: "meeper"
3. Find your uploaded dataset
4. Click "Add"
5. Re-run Cell 1

---

### Error: "No GPU available"

**Cause:** GPU not enabled

**Solution:**
1. Settings (gear icon)
2. Accelerator → GPU
3. Choose P100 or T4
4. Save
5. Restart notebook
6. Re-run from Cell 1

---

### Output: "Fetching latest from Kaggle" (WRONG!)

**Cause:** Notebook didn't update properly

**Solution:**
1. Delete notebook
2. Re-import from GitHub URL
3. Make sure you see Cell 1 (Setup) exists
4. Run Cell 1, then Cell 2

---

### Training taking too long?

**Expected times:**
- P100: 7-8 hours
- T4: 8-9 hours

**You can close the browser!**
- Training continues in background
- Come back 8 hours later
- Check progress
- Run remaining cells

---

## Final Checklist

Before starting training:

- [ ] Added "meeper" dataset to notebook
- [ ] Enabled GPU (P100 or T4)
- [ ] Enabled Internet
- [ ] Imported notebook from GitHub
- [ ] See 7 cells total (0-6)
- [ ] Cell 1 contains "git clone"
- [ ] Cell 2 contains "--aggregated-data"
- [ ] Ready to run Cell 1 first

---

## Summary

**Total Runtime:** ~8 hours
**Total Cells:** 7 (Cell 0-6)
**Run Order:** Cell 1 → Cell 2 → Cell 3 → Cell 4 → Cell 5
**Output:** 7 models (~330 MB total)

**Critical:** RUN CELL 1 (SETUP) FIRST!

Then Cell 2 will work without FileNotFoundError.
