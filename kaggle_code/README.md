# NBA Predictor Code Package

This package contains Python modules needed to unpickle the NBA window models.

## Files Included

- `hybrid_multi_task.py` - Multi-task model architecture
- `optimization_features.py` - Feature engineering utilities
- `phase7_features.py` - Advanced feature calculations
- `rolling_features.py` - Rolling window features
- `shared/` - Shared utilities (data loading, CSV aggregation)
- `priors_data/` - Prior data for feature engineering

## How to Use on Kaggle

1. Upload this entire folder as a Kaggle dataset
2. Name it: "nba-predictor-code"
3. In your notebook, add:
   ```python
   import sys
   sys.path.insert(0, '/kaggle/input/nba-predictor-code')
   ```
4. Now you can unpickle models without ModuleNotFoundError

## Upload to Kaggle

1. Zip this folder: `kaggle_code.zip`
2. Go to https://www.kaggle.com/datasets
3. Click "New Dataset"
4. Upload the zip file
5. Set title: "nba-predictor-code"
6. Make it Private
7. Click "Create"
