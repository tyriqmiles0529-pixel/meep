#!/usr/bin/env python
"""
Create Kaggle code package for model unpickling.

This script creates a 'kaggle_code/' directory with all Python files
needed to unpickle the window models.

Usage:
    python create_kaggle_code_package.py

Then upload the entire kaggle_code/ folder as a Kaggle dataset.
"""

import shutil
from pathlib import Path

def create_package():
    """Create kaggle_code/ directory with necessary files"""

    output_dir = Path("kaggle_code")
    output_dir.mkdir(exist_ok=True)

    # Files to copy
    files_to_copy = [
        "hybrid_multi_task.py",
        "optimization_features.py",
        "phase7_features.py",
        "rolling_features.py",
    ]

    # Directories to copy
    dirs_to_copy = [
        "shared",
        "priors_data"
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


if __name__ == "__main__":
    try:
        create_package()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
