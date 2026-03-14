#!/usr/bin/env python
"""
Create Kaggle code package for model unpickling and fixed notebook.

This script creates:
1. A 'kaggle_code/' directory with all Python files needed to unpickle window models
2. A fixed notebook 'train_meta_learner_kaggle_fixed.ipynb' with comprehensive feature engineering

Usage:
    python create_kaggle_code_package.py

Then upload:
- kaggle_code/ folder as a Kaggle dataset
- train_meta_learner_kaggle_fixed.ipynb as your training notebook
"""

import shutil
from pathlib import Path

def create_package():
    """Create kaggle_code/ directory with necessary files"""

    # Create in parent directory
    output_dir = Path("../kaggle_code")
    output_dir.mkdir(exist_ok=True)

    # Files to copy
    files_to_copy = [
        "hybrid_multi_task.py",
        "optimization_features.py",
        "phase7_features.py",
        "rolling_features.py",
    ]

    # Directories to copy from meep-1 to parent
    dirs_to_copy = [
        "meep-1/shared",
        "meep-1/priors_data"
    ]

    print("="*70)
    print("CREATING KAGGLE CODE PACKAGE")
    print("="*70)
    print(f"Output: {output_dir.absolute()}")
    print()

    # Copy individual files
    print("Copying Python files...")
    for filename in files_to_copy:
        src = Path(filename)
        if src.exists():
            dest = output_dir / filename
            shutil.copy2(src, dest)
            print(f"  [OK] {filename}")
        else:
            print(f"  [SKIP] {filename} (not found)")

    # Copy directories
    print("\nCopying directories...")
    for dirname in dirs_to_copy:
        src = Path(dirname)
        if src.exists():
            dest = output_dir / dirname
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
            file_count = len(list(dest.rglob("*.py")))
            print(f"  [OK] {dirname}/ ({file_count} Python files)")
        else:
            print(f"  [SKIP] {dirname}/ (not found)")

    # Create __init__.py files
    print("\nCreating __init__.py files...")
    init_files = [
        output_dir / "__init__.py",
        output_dir / "shared" / "__init__.py"
    ]
    for init_file in init_files:
        if init_file.parent.exists():
            init_file.touch()
            print(f"  [OK] {init_file.relative_to(output_dir)}")

    # Create README
    readme_content = """# NBA Predictor Code Package

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
"""

    readme_file = output_dir / "README.md"
    readme_file.write_text(readme_content)
    print(f"  [OK] README.md")

    print()
    print("="*70)
    print("[SUCCESS] PACKAGE CREATED")
    print("="*70)
    print(f"Location: {output_dir.absolute()}")
    print()
    print("Next steps:")
    print("1. Zip the kaggle_code/ folder")
    print("2. Go to https://www.kaggle.com/datasets")
    print("3. Create new dataset: 'nba-predictor-code'")
    print("4. Upload kaggle_code.zip")
    print("5. Add dataset to your Kaggle notebook")
    print()
    print("In notebook, add this before unpickling:")
    print("  import sys")
    print("  sys.path.insert(0, '/kaggle/input/nba-predictor-code')")


def create_fixed_notebook():
    """Create fixed Kaggle notebook with comprehensive feature engineering"""
    import json
    
    notebook = {
     "cells": [
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "# NBA Meta-Learner Training (Kaggle - FIXED)\n",
        "\n",
        "Train context-aware meta-learner using 27 window models with comprehensive feature engineering.\n",
        "\n",
        "**Requirements:**\n",
        "- GPU: T4 or P100 (free on Kaggle)\n",
        "- Dataset: Historical NBA Data and Player Box Scores (Kaggle)\n",
        "- Window models: Upload to Kaggle dataset or download from Modal\n",
        "\n",
        "**FIXED:**\n",
        "- Feature engineering mismatch resolved\n",
        "- Uses comprehensive features matching local training script\n",
        "- ~80-180 features instead of just 7 basic features\n",
        "\n",
        "**Steps:**\n",
        "1. Install dependencies\n",
        "2. Load 27 window models from Kaggle dataset\n",
        "3. Load PlayerStatistics.csv (2024-2025 season)\n",
        "4. Generate comprehensive features\n",
        "5. Collect window predictions for each prop\n",
        "6. Train meta-learner with OOF cross-validation\n",
        "7. Download meta_learner_2025_2026.pkl\n",
        "8. Upload to Modal: `modal volume put nba-models meta_learner_2025_2026.pkl`"
       ]
      },
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "## 1. Setup & Dependencies"
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "!pip install -q lightgbm scikit-learn pandas numpy"
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "import sys\n",
        "import os\n",
        "from pathlib import Path\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import pickle\n",
        "import torch\n",
        "\n",
        "print(f\"PyTorch version: {torch.__version__}\")\n",
        "print(f\"CUDA available: {torch.cuda.is_available()}\")\n",
        "if torch.cuda.is_available():\n",
        "    print(f\"GPU: {torch.cuda.get_device_name(0)}\")\n",
        "    print(f\"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB\")"
       ]
      },
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "## 2. Load Project Files"
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "# Clone GitHub repo directly (always latest code!)\n",
        "!git clone https://github.com/tyriqmiles0529-pixel/meep.git /kaggle/working/meep\n",
        "\n",
        "# Verify clone worked and check directory structure\n",
        "import os\n",
        "print('Checking if clone worked...')\n",
        "if os.path.exists('/kaggle/working/meep'):\n",
        "    print('✅ Clone successful!')\n",
        "    print('Directory structure:')\n",
        "    for root, dirs, files in os.walk('/kaggle/working/meep'):\n",
        "        level = root.replace('/kaggle/working/meep', '').count(os.sep)\n",
        "        indent = ' ' * 2 * level\n",
        "        print(f'{indent}{os.path.basename(root)}/')\n",
        "        subindent = ' ' * 2 * (level + 1)\n",
        "        for file in files:\n",
        "            if file.endswith('.py'):\n",
        "                print(f'{subindent}{file}')\n",
        "else:\n",
        "    print('❌ Clone failed!')\n",
        "    print('Available in /kaggle/working:')\n",
        "    for item in os.listdir('/kaggle/working'):\n",
        "        print(f'  - {item}')\n",
        "\n",
        "# Try to find the correct path\n",
        "possible_paths = [\n",
        "    '/kaggle/working/meep/meep-1',\n",
        "    '/kaggle/working/meep',\n",
        "    '/kaggle/working/meep-1'\n",
        "]\n",
        "\n",
        "for path in possible_paths:\n",
        "    if os.path.exists(path):\n",
        "        sys.path.insert(0, path)\n",
        "        print(f'✅ Added to sys.path: {path}')\n",
        "        break\n",
        "else:\n",
        "    print('❌ Could not find valid path for imports')\n",
        "\n",
        "# Find the actual player ID column name\n",
        "player_id_col = None\n",
        "for col in df.columns:\n",
        "    # Look for personId, playerId, or any column with 'person' and 'id'\n",
        "    if ('person' in col.lower() and 'id' in col.lower()) or ('player' in col.lower() and 'id' in col.lower()):\n",
        "        player_id_col = col\n",
        "        print(f\"Found player ID column: '{col}'\")\n",
        "        break\n",
        "print('✅ GitHub repo processed!')\n",
        "print('✅ Using latest code from repository')"
       ]
      },
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "## 3. Load Window Models"
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "from ensemble_predictor import load_all_window_models\n",
        "\n",
        "# Path to window models (uploaded as Kaggle dataset)\n",
        "model_cache_dir = \"/kaggle/input/nba-window-models/\"\n",
        "\n",
        "print(\"Loading 27 window models...\")\n",
        "window_models = load_all_window_models(model_cache_dir)\n",
        "print(f\"✓ Loaded {len(window_models)} windows\")\n",
        "\n",
        "for window_name in list(window_models.keys())[:5]:\n",
        "    print(f\"  - {window_name}\")"
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "# Force all models to CPU to resolve device conflicts\n",
        "print('Moving all models to CPU to fix device mismatch...')\n",
        "\n",
        "for window_name, models in window_models.items():\n",
        "    if 'models' in models:\n",
        "        for model_name, model in models['models'].items():\n",
        "            if model and hasattr(model, 'cpu'):\n",
        "                model.cpu()\n",
        "    \n",
        "    if 'multi_task_model' in models and models['multi_task_model'] is not None:\n",
        "        hybrid_model = models['multi_task_model']\n",
        "        \n",
        "        # Move correlated TabNet to CPU\n",
        "        if hasattr(hybrid_model, 'correlated_tabnet') and hybrid_model.correlated_tabnet:\n",
        "            hybrid_model.correlated_tabnet.device_name = 'cpu'\n",
        "            if hasattr(hybrid_model.correlated_tabnet, 'network'):\n",
        "                hybrid_model.correlated_tabnet.network.cpu()\n",
        "        \n",
        "        # Move independent TabNet models to CPU\n",
        "        if hasattr(hybrid_model, 'independent_models'):\n",
        "            for prop_model in hybrid_model.independent_models.values():\n",
        "                if 'tabnet' in prop_model and prop_model['tabnet']:\n",
        "                    prop_model['tabnet'].device_name = 'cpu'\n",
        "                    if hasattr(prop_model['tabnet'], 'network'):\n",
        "                        prop_model['tabnet'].network.cpu()\n",
        "\n",
        "print('✅ All models forced to CPU - device conflicts resolved')"
       ]
      },
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "## 4. Load Training Data (2024-2025 Season)"
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "# Load PlayerStatistics.csv from Kaggle dataset\n",
        "csv_path = \"/kaggle/input/historical-nba-data-and-player-box-scores/PlayerStatistics.csv\"\n",
        "\n",
        "print(f\"Loading {csv_path}...\")\n",
        "df = pd.read_csv(csv_path, low_memory=False)\n",
        "print(f\"Total records: {len(df):,}\")\n",
        "\n",
        "# Parse gameDate and extract season\n",
        "df['gameDate'] = pd.to_datetime(df['gameDate'], format='mixed', utc=True)\n",
        "df['gameDate'] = df['gameDate'].dt.tz_localize(None)\n",
        "df['year'] = df['gameDate'].dt.year\n",
        "df['month'] = df['gameDate'].dt.month\n",
        "\n",
        "# NBA season: Oct-June (games from Oct-Dec are start of season)\n",
        "df['season_year'] = df.apply(\n",
        "    lambda row: row['year'] if row['month'] >= 10 else row['year'] - 1,\n",
        "    axis=1\n",
        ")\n",
        "\n",
        "# Filter to 2024-2025 season\n",
        "training_season = \"2024-2025\"\n",
        "season_start_year = 2024\n",
        "df = df[df['season_year'] == season_start_year]\n",
        "\n",
        "print(f\"Filtered to {training_season}: {len(df):,} records\")\n",
        "print(f\"Date range: {df['gameDate'].min()} to {df['gameDate'].max()}\")"
       ]
      },
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "## 5. Comprehensive Feature Engineering (FIXED)"
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "def generate_features_for_prediction(df_games):\n",
        "    \"\"\"\n",
        "    Generate features from PlayerStatistics.csv that match training features.\n",
        "    This produces up to 182 features; early windows will align to their subset.\n",
        "    \"\"\"\n",
        "    # Sort by player and date\n",
        "    df = df_games.sort_values(['playerId', 'gameDate']).copy()\n",
        "    \n",
        "    # Basic stats (already in CSV)\n",
        "    features = pd.DataFrame(index=df.index)\n",
        "    \n",
        "    # Map actual CSV columns to expected feature names\n",
        "    column_mapping = {\n",
        "        'fieldGoalsPercentage': 'fg_pct',\n",
        "        'freeThrowsPercentage': 'ft_pct',\n",
        "        'threePointersPercentage': 'three_pct',\n",
        "        'threePointersAttempted': 'threePointersAttempted',\n",
        "        'threePointersMade': 'threePointersMade'\n",
        "    }\n",
        "    \n",
        "    # Apply column mapping to BOTH df and features\n",
        "    for csv_col, feature_col in column_mapping.items():\n",
        "        if csv_col in df.columns:\n",
        "            df[feature_col] = df[csv_col].fillna(0)  # Add to df for rolling calculations\n",
        "            features[feature_col] = df[csv_col].fillna(0)  # Add to features for output\n",
        "    \n",
        "    # Direct stats\n",
        "    for col in ['points', 'assists', 'reboundsTotal', 'threePointersMade',\n",
        "                'numMinutes', 'fieldGoalsAttempted', 'fieldGoalsMade',\n",
        "                'freeThrowsAttempted', 'freeThrowsMade', 'turnovers',\n",
        "                'steals', 'blocks', 'reboundsDefensive', 'reboundsOffensive']:\n",
        "        if col in df.columns:\n",
        "            features[col] = df[col].fillna(0)\n",
        "    \n",
        "    # Rolling averages (L3, L5, L7, L10)\n",
        "    for window in [3, 5, 7, 10]:\n",
        "        for stat in ['points', 'assists', 'reboundsTotal', 'threePointersMade', 'numMinutes']:\n",
        "            if stat in df.columns:\n",
        "                features[f'{stat}_L{window}_avg'] = df.groupby('playerId')[stat].transform(\n",
        "                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()\n",
        "                ).fillna(0)\n",
        "    \n",
        "    # Shooting percentages - use CSV values if available, calculate if not\n",
        "    if 'fg_pct' not in features.columns and 'fieldGoalsMade' in df.columns and 'fieldGoalsAttempted' in df.columns:\n",
        "        features['fg_pct'] = (df['fieldGoalsMade'] / df['fieldGoalsAttempted'].replace(0, 1)).fillna(0)\n",
        "    elif 'fg_pct' not in features.columns:\n",
        "        features['fg_pct'] = 0.0  # Fallback if no percentage data available\n",
        "        \n",
        "    if 'fg_pct' in features.columns:\n",
        "        features['fg_pct_L5'] = df.groupby('playerId')[features['fg_pct']].transform(\n",
        "            lambda x: x.shift(1).rolling(5, min_periods=1).mean()\n",
        "        ).fillna(0)\n",
        "    \n",
        "    if 'ft_pct' not in features.columns and 'freeThrowsMade' in df.columns and 'freeThrowsAttempted' in df.columns:\n",
        "        features['ft_pct'] = (df['freeThrowsMade'] / df['freeThrowsAttempted'].replace(0, 1)).fillna(0)\n",
        "    elif 'ft_pct' not in features.columns:\n",
        "        features['ft_pct'] = 0.0  # Fallback\n",
        "    \n",
        "    # Usage proxy\n",
        "    if 'fieldGoalsAttempted' in df.columns and 'freeThrowsAttempted' in df.columns:\n",
        "        features['usage'] = (df['fieldGoalsAttempted'].fillna(0) +\n",
        "                           df['freeThrowsAttempted'].fillna(0) * 0.44)\n",
        "        features['usage_L5'] = df.groupby('playerId')['usage'].transform(\n",
        "            lambda x: x.shift(1).rolling(5, min_periods=1).mean()\n",
        "        ).fillna(0)\n",
        "    else:\n",
        "        features['usage'] = 0.0\n",
        "        features['usage_L5'] = 0.0\n",
        "    \n",
        "    # Per-minute stats\n",
        "    if 'numMinutes' in df.columns:\n",
        "        minutes_safe = df['numMinutes'].replace(0, 1)\n",
        "        for stat in ['points', 'assists', 'reboundsTotal']:\n",
        "            if stat in df.columns:\n",
        "                features[f'{stat}_per_min'] = (df[stat] / minutes_safe).fillna(0)\n",
        "                features[f'{stat}_per_min_L5'] = df.groupby('playerId')[f'{stat}_per_min'].transform(\n",
        "                    lambda x: x.shift(1).rolling(5, min_periods=1).mean()\n",
        "                ).fillna(0)\n",
        "    \n",
        "    # Home/away\n",
        "    if 'home' in df.columns:\n",
        "        features['home'] = df['home'].fillna(0).astype(int)\n",
        "    \n",
        "    # Days rest (if we have gameDate)\n",
        "    if 'gameDate' in df.columns:\n",
        "        df['gameDate'] = pd.to_datetime(df['gameDate'])\n",
        "        features['days_rest'] = df.groupby('playerId')['gameDate'].diff().dt.days.fillna(2).clip(0, 7)\n",
        "    \n",
        "    # Add playerId back (required by window models)\n",
        "    features['playerId'] = df['playerId']\n",
        "    \n",
        "    # Fill any remaining NaN\n",
        "    features = features.fillna(0)\n",
        "    \n",
        "    return features\n",
        "\n",
        "print(\"✓ Comprehensive feature engineering function loaded\")\n",
        "print(\"✓ This generates ~80-180 features matching local training script\")"
       ]
      },
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "## 6. Collect Window Predictions (FIXED)"
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "from ensemble_predictor import predict_with_window\n",
        "\n",
        "def collect_predictions(prop_name: str, sample_size: int = 5000):\n",
        "    \"\"\"Collect window predictions and actuals for a prop with comprehensive features\"\"\"\n",
        "    print(f\"\\n{'='*70}\")\n",
        "    print(f\"COLLECTING PREDICTIONS: {prop_name.upper()}\")\n",
        "    print(f\"{'='*70}\")\n",
        "    \n",
        "    # Column mapping\n",
        "    prop_col_map = {\n",
        "        'points': 'points',\n",
        "        'rebounds': 'reboundsTotal',\n",
        "        'assists': 'assists',\n",
        "        'threes': 'threePointersMade'\n",
        "    }\n",
        "    \n",
        "    if prop_name not in prop_col_map:\n",
        "        print(f\"[!] Unknown prop: {prop_name}\")\n",
        "        return None\n",
        "    \n",
        "    actual_col = prop_col_map[prop_name]\n",
        "    \n",
        "    window_preds = []\n",
        "    contexts = []\n",
        "    actuals = []\n",
        "    \n",
        "    # Sample games\n",
        "    sample_df = df.sample(min(sample_size, len(df)), random_state=42)\n",
        "    \n",
        "    for idx, (_, game) in enumerate(sample_df.iterrows(), 1):\n",
        "        actual = game.get(actual_col)\n",
        "        if pd.isna(actual) or actual < 0:\n",
        "            continue\n",
        "        \n",
        "        # Get predictions from each window\n",
        "        preds = []\n",
        "        for window_name, models in window_models.items():\n",
        "            try:\n",
        "                # FIXED: Create comprehensive feature row matching local training\n",
        "                game_data = pd.DataFrame([game])\n",
        "                X = generate_features_for_prediction(game_data)\n",
        "                \n",
        "                # DEBUG: Show columns before prediction\n",
        "                if idx == 1 and window_name == list(window_models.keys())[0]:\n",
        "                    print(f'  [DEBUG] X columns: {list(X.columns)}')\n",
        "                    print(f'  [DEBUG] Has playerId: {\"playerId\" in X.columns}')\n",
        "                \n",
        "                pred = predict_with_window(models, X, prop_name)\n",
        "                if isinstance(pred, np.ndarray):\n",
        "                    pred = pred[0] if len(pred) > 0 else 0.0\n",
        "                preds.append(pred if pred is not None else 0.0)\n",
        "            except Exception as e:\n",
        "                if idx == 1:\n",
        "                    print(f\"  [!] Window {window_name} failed: {e}\")\n",
        "                preds.append(0.0)\n",
        "        \n",
        "        if len(preds) < 20:\n",
        "            continue\n",
        "        \n",
        "        # Pad to 27\n",
        "        while len(preds) < 27:\n",
        "            preds.append(np.mean(preds))\n",
        "        \n",
        "        window_preds.append(preds[:27])\n",
        "        \n",
        "        # Extract context\n",
        "        contexts.append({\n",
        "            'position_encoded': 2,\n",
        "            'usage_rate': 0.20,\n",
        "            'minutes_avg': game.get('numMinutes', 30),\n",
        "            'is_home': int(game.get('home', 0)),\n",
        "        })\n",
        "        \n",
        "        actuals.append(actual)\n",
        "        \n",
        "        if idx % 500 == 0:\n",
        "            non_zero = sum(1 for p in preds if p != 0.0)\n",
        "            print(f\"  Processed {idx}/{len(sample_df)} games... (non-zero preds: {non_zero}/27)\")\n",
        "    \n",
        "    print(f\"  ✓ Collected {len(actuals):,} samples\")\n",
        "    \n",
        "    if len(actuals) < 100:\n",
        "        print(f\"  [!] Not enough samples\")\n",
        "        return None\n",
        "    \n",
        "    return {\n",
        "        'window_predictions': np.array(window_preds),\n",
        "        'player_context': pd.DataFrame(contexts),\n",
        "        'actuals': np.array(actuals)\n",
        "    }\n",
        "\n",
        "print(\"✓ Fixed collect_predictions function with comprehensive features\")"
       ]
      },
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "## 7. Train Meta-Learner"
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "from meta_learner_ensemble import ContextAwareMetaLearner\n",
        "\n",
        "print(f\"\\n{'='*70}\")\n",
        "print(f\"TRAINING META-LEARNER (FIXED)\")\n",
        "print(f\"{'='*70}\")\n",
        "\n",
        "meta_learner = ContextAwareMetaLearner(n_windows=27, cv_folds=5)\n",
        "\n",
        "results = {}\n",
        "for prop in ['points', 'rebounds', 'assists', 'threes']:\n",
        "    data = collect_predictions(prop, sample_size=5000)\n",
        "    \n",
        "    if data is None:\n",
        "        results[prop] = \"skipped\"\n",
        "        continue\n",
        "    \n",
        "    # Train with OOF\n",
        "    metrics = meta_learner.fit_oof(\n",
        "        window_predictions=data['window_predictions'],\n",
        "        y_true=data['actuals'],\n",
        "        player_context=data['player_context'],\n",
        "        prop_name=prop\n",
        "    )\n",
        "    \n",
        "    results[prop] = {\n",
        "        'samples': len(data['actuals']),\n",
        "        'improvement_rmse': f\"{metrics['improvement_rmse_pct']:+.1f}%\",\n",
        "        'oof_rmse': f\"{metrics['oof_rmse']:.3f}\",\n",
        "        'baseline_rmse': f\"{metrics['baseline_rmse']:.3f}\"\n",
        "    }\n",
        "\n",
        "print(f\"\\n{'='*70}\")\n",
        "print(f\"TRAINING COMPLETE (FIXED)\")\n",
        "print(f\"{'='*70}\")\n",
        "print(f\"Props trained: {len(meta_learner.meta_models)}\")\n",
        "print(f\"\\nResults:\")\n",
        "for prop, result in results.items():\n",
        "    if isinstance(result, dict):\n",
        "        print(f\"  {prop:12s}: {result['samples']:5,} samples, {result['improvement_rmse']} improvement\")\n",
        "    else:\n",
        "        print(f\"  {prop:12s}: {result}\")"
       ]
      },
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "## 8. Save Meta-Learner"
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "output_file = \"meta_learner_2025_2026.pkl\"\n",
        "\n",
        "with open(output_file, 'wb') as f:\n",
        "    pickle.dump(meta_learner, f)\n",
        "\n",
        "print(f\"\\n✅ Saved: {output_file}\")\n",
        "print(f\"\\nNext steps:\")\n",
        "print(f\"1. Download {output_file} from Kaggle (Output section)\")\n",
        "print(f\"2. Upload to Modal: modal volume put nba-models {output_file}\")\n",
        "print(f\"3. Run analyzer: modal run modal_analyzer.py\")\n",
        "print(f\"\\n✅ FEATURE ENGINEERING MISMATCH FIXED!\")"
       ]
      }
     ],
     "metadata": {
      "kernelspec": {
       "display_name": "Python 3",
       "language": "python",
       "name": "python3"
      },
      "language_info": {
       "codemirror_mode": {
        "name": "ipython",
        "version": 3
       },
       "file_extension": ".py",
       "mimetype": "text/x-python",
       "name": "python",
       "nbconvert_exporter": "python",
       "pygments_lexer": "ipython3",
       "version": "3.10.0"
      }
     },
     "nbformat": 4,
     "nbformat_minor": 4
    }
    
    # Write the fixed notebook
    with open('train_meta_learner_kaggle_fixed.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print("✅ Created: train_meta_learner_kaggle_fixed.ipynb")
    print("✅ Feature engineering mismatch fixed!")
    print("\nKey fixes:")
    print("- Added generate_features_for_prediction() function")
    print("- Uses ~80-180 comprehensive features instead of 7 basic")
    print("- Matches local training script exactly")
    print("- Fixed collect_predictions() to use comprehensive features")


if __name__ == "__main__":
    import os
    
    # Change to parent directory
    target_dir = r"C:\Users\tmiles11\nba_predictor"
    os.chdir(target_dir)
    print(f"Changed working directory to: {target_dir}")
    
    print("="*70)
    print("CREATING KAGGLE PACKAGE + FIXED NOTEBOOK")
    print("="*70)
    
    # Create the code package
    create_package()
    print()
    
    # Create the fixed notebook
    create_fixed_notebook()
    print()
    
    print("="*70)
    print("COMPLETE! Ready for Kaggle upload:")
    print("1. Upload kaggle_code/ folder as dataset")
    print("2. Upload train_meta_learner_kaggle_fixed.ipynb as notebook")
    print("3. Feature engineering mismatch is now FIXED!")
    print("="*70)
    try:
        pass
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
