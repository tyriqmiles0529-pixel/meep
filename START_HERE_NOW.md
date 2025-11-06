# 🚀 NEXT STEPS - How to Train in Colab

## Quick Start (5 minutes)

### 1️⃣ Get Your Files Ready
You need 2 files from this folder:
- `priors_data.zip` (4.8 MB) ✅ Already here
- `PlayerStatistics.csv` (302.8 MB) ✅ Already here

### 2️⃣ Open Colab Notebook
1. Go to: https://colab.research.google.com
2. Click **File → Upload notebook**
3. Upload `NBA_COLAB_COMPLETE.ipynb` from this folder
4. **Enable GPU**: Runtime → Change runtime type → Hardware accelerator → **GPU** → Save

### 3️⃣ Run the Training
Just run each cell in order:

#### **Cell 1: Upload Your Data**
- Click ▶️ Run
- When prompted: Upload `priors_data.zip` (takes ~30 seconds)
- When prompted again: Upload `PlayerStatistics.csv` (takes ~2-3 minutes)
- Wait for ✅ confirmation messages

#### **Cell 2: Install & Download Code**
- Click ▶️ Run
- Installs packages and downloads latest code from GitHub
- Takes ~1 minute
- Should show: ✅ GPU Available: True

#### **Cell 3: Train Models**
- Click ▶️ Run
- Takes 20-30 minutes with GPU
- ☕ Get coffee!
- Watch the progress in real-time

#### **Cell 4: Download Trained Models**
- Click ▶️ Run
- Downloads `nba_models_trained.zip` to your computer
- Extract it in this folder (will create `models/` and `model_cache/`)

---

## What You'll Get

After training completes, you'll have:

### Game Models (Moneyline & Spread):
- 📊 Trained on 32,521 games (2002-2026)
- 🎯 ~61% accuracy on moneyline
- 📈 5 windowed ensemble models (2002-06, 2007-11, etc.)
- ✅ Ridge + Elo + Four Factors + LightGBM

### Player Models (Props):
- 👤 Trained on 800k+ player-games (2002-2026)
- 🧠 Neural network (TabNet + LightGBM) 
- 📊 Points, Rebounds, Assists, 3PM, Minutes
- 🎲 5 windowed models for each prop

### Features Included:
- ✅ Basketball Reference priors (68 player features, 22 team features)
- ✅ Phase 1-7 features (momentum, consistency, fatigue, etc.)
- ✅ Neural embeddings (deep feature learning)
- ✅ Market signals (if available)

---

## After Downloading Models

### Option A: Use Locally
```powershell
# Extract the downloaded zip to this folder
# Then run predictions:
python player_ensemble_enhanced.py
```

### Option B: Keep Training in Colab
Just re-run the notebook whenever you want to retrain with latest data

---

## Troubleshooting

**Upload fails or times out?**
- File too large for session → Try again
- Split into multiple sessions if needed

**GPU not available?**
- Go to: Runtime → Change runtime type → GPU
- Training will work on CPU (just slower: 1-2 hours)

**Out of memory?**
- Runtime → Restart runtime
- Re-run from Cell 1

**PlayerStatistics not uploading?**
- Make sure the file is exactly 302.8 MB
- Located at: `C:\Users\tmiles11\nba_predictor\PlayerStatistics.csv`

---

## Alternative: Train Locally

If Colab doesn't work or you want to keep it simple:

```powershell
# In this folder:
python train_auto.py --priors-dataset priors_data --player-csv PlayerStatistics.csv --verbose
```

**Pros:** No uploads needed, everything local  
**Cons:** Takes longer without GPU (2-3 hours), uses your computer's resources

---

## Files Overview

📁 **What's already in this folder:**
- ✅ `priors_data.zip` - Upload this to Colab
- ✅ `PlayerStatistics.csv` - Upload this to Colab  
- ✅ `NBA_COLAB_COMPLETE.ipynb` - The training notebook
- ✅ `COLAB_UPLOAD_FILES.md` - Detailed upload guide
- ✅ All Python scripts (for local training if needed)

📥 **What you'll download from Colab:**
- `nba_models_trained.zip` - Your trained models
- Extract → creates `models/` and `model_cache/` folders

---

**Ready?** Open Colab and run the notebook! 🎯
