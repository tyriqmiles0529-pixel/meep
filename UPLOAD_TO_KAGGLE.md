# Upload Code Files to Kaggle Dataset

The window model .pkl files contain references to custom Python classes. You need to upload the source code files so pickle can find them.

## Quick Solution: Upload These Files as Kaggle Dataset

### Step 1: Create "nba-code" Dataset on Kaggle

1. Go to: https://www.kaggle.com/datasets
2. Click "New Dataset"
3. Upload these files from your local repo:

```
hybrid_multi_task.py
optimization_features.py
phase7_features.py
rolling_features.py
```

4. Also create a folder called `shared/` and upload:
```
shared/data_loading.py
shared/csv_aggregation.py
```

5. Name the dataset: **"nba-predictor-code"**
6. Make it **Private**
7. Click "Create"

### Step 2: Update Notebook to Use Code Dataset

Add this cell at the beginning of the notebook (after imports):

```python
# Add code files to Python path
import sys
sys.path.insert(0, '/kaggle/input/nba-predictor-code')
```

### Step 3: Add Dataset to Notebook

1. In your Kaggle notebook, click "Add Data"
2. Go to "Your Datasets"
3. Select "nba-predictor-code"
4. Click "Add"

Now the notebook will find all the custom classes when unpickling!

---

## Alternative: Use Pre-Made Dataset

I can create a zip file with all necessary code. Let me know if you want that option.

## Files Needed for Unpickling

The models reference these modules:
- `hybrid_multi_task.HybridMultiTaskPlayer`
- `optimization_features.*` (various feature functions)
- `phase7_features.*` (feature engineering)
- `rolling_features.*` (rolling window calculations)
- `shared.data_loading.*` (data utilities)

All must be importable for pickle.load() to work.
