# Windows PowerShell Setup Guide

## You're running on Windows - Here's how to get the integrated scripts:

### Step 1: Pull the latest code

```powershell
cd C:\Users\tmiles11\nba_predictor

# Fetch the latest changes
git fetch origin

# Switch to the branch with integrated keys
git checkout claude/optimize-prop-analysis-011CUR2rQFGPJgs1KAEaMidk

# Pull the latest code
git pull origin claude/optimize-prop-analysis-011CUR2rQFGPJgs1KAEaMidk
```

### Step 2: Verify you have the files with keys

```powershell
# Check if files exist
dir nba_prop_analyzer_fixed.py
dir train_auto.py

# Check if keys are in the files (should show the key line)
Select-String -Path nba_prop_analyzer_fixed.py -Pattern "4979ac5e1f"
Select-String -Path train_auto.py -Pattern "f005fb2c"
```

### Step 3: Run the scripts

```powershell
# Run the analyzer
python nba_prop_analyzer_fixed.py

# Run the trainer
python train_auto.py
```

## What's Integrated:

✅ **nba_prop_analyzer_fixed.py** - Line 21: API-Sports.io key hardcoded
✅ **train_auto.py** - Line 61: Kaggle key hardcoded

Both scripts work immediately with NO setup, NO environment variables, NO imports.

## Troubleshooting:

**If git checkout fails:**
```powershell
git stash  # Save your local changes
git checkout claude/optimize-prop-analysis-011CUR2rQFGPJgs1KAEaMidk
git pull origin claude/optimize-prop-analysis-011CUR2rQFGPJgs1KAEaMidk
```

**If you don't have the branch:**
```powershell
git fetch origin claude/optimize-prop-analysis-011CUR2rQFGPJgs1KAEaMidk:claude/optimize-prop-analysis-011CUR2rQFGPJgs1KAEaMidk
git checkout claude/optimize-prop-analysis-011CUR2rQFGPJgs1KAEaMidk
```

## Note:

The file is called `train_auto.py` (not `train.py`). Make sure you run the correct file!
