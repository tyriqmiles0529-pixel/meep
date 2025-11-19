#!/usr/bin/env python3
"""
Lightning.ai NBA Training Script
=================================

Complete training pipeline for Lightning.ai with GPU acceleration.
Trains game models and player models with neural hybrid (TabNet + LightGBM).

Usage:
    python LIGHTNING_AI_TRAINING.py

This script will:
1. Verify GPU availability
2. Install dependencies (if needed)
3. Download data from Kaggle
4. Train all models with neural hybrid
5. Save trained models to model_cache/

Expected time: 45-90 minutes on T4 GPU
"""

import sys
import subprocess
import os

def check_gpu():
    """Verify GPU is available"""
    print("\n" + "="*70)
    print("GPU Check")
    print("="*70)

    try:
        import torch
        gpu_available = torch.cuda.is_available()

        if gpu_available:
            print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️  No GPU detected. Training will use CPU (slower).")

        return gpu_available
    except ImportError:
        print("❌ PyTorch not installed. Will install now...")
        return False

def install_dependencies():
    """Install required packages"""
    print("\n" + "="*70)
    print("Installing Dependencies")
    print("="*70)

    packages = [
        "kagglehub",
        "pandas",
        "numpy",
        "scikit-learn",
        "lightgbm",
        "torch",
        "pytorch-tabnet",
        "requests",
        "tqdm"
    ]

    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

    print("✅ All dependencies installed")

def verify_installation():
    """Verify critical packages are installed"""
    print("\n" + "="*70)
    print("Verifying Installation")
    print("="*70)

    try:
        import pandas as pd
        print(f"✅ pandas {pd.__version__}")
    except ImportError:
        print("❌ pandas not found")
        return False

    try:
        import lightgbm as lgb
        print(f"✅ lightgbm {lgb.__version__}")
    except ImportError:
        print("❌ lightgbm not found")
        return False

    try:
        import torch
        print(f"✅ torch {torch.__version__}")
    except ImportError:
        print("❌ torch not found")
        return False

    try:
        from pytorch_tabnet.tab_model import TabNetRegressor
        print(f"✅ pytorch-tabnet installed")
    except ImportError:
        print("❌ pytorch-tabnet not found")
        return False

    return True

def setup_kaggle():
    """Setup Kaggle credentials if needed"""
    print("\n" + "="*70)
    print("Kaggle Setup")
    print("="*70)

    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")

    if os.path.exists(kaggle_json):
        print("✅ Kaggle credentials found")
        return True

    print("⚠️  Kaggle credentials not found")
    print("\nTo download data, you need Kaggle API credentials:")
    print("1. Go to https://www.kaggle.com/settings")
    print("2. Click 'Create New Token' under API")
    print("3. Upload kaggle.json to Lightning.ai")
    print("4. Or enter credentials now:")

    username = input("Kaggle Username (or press Enter to skip): ").strip()
    if not username:
        print("Skipping Kaggle setup. You'll need to upload data manually.")
        return False

    api_key = input("Kaggle API Key: ").strip()

    os.makedirs(kaggle_dir, exist_ok=True)
    with open(kaggle_json, 'w') as f:
        f.write(f'{{"username":"{username}","key":"{api_key}"}}')

    os.chmod(kaggle_json, 0o600)
    print("✅ Kaggle credentials saved")
    return True

def train_models(use_gpu=True, epochs=30):
    """Run main training pipeline"""
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)
    print(f"Configuration:")
    print(f"  - Neural Hybrid: Enabled (TabNet + LightGBM)")
    print(f"  - Device: {'GPU' if use_gpu else 'CPU'}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch Size: 4096")
    print("="*70)

    # Build command
    cmd = [
        sys.executable,
        "train_auto.py",
        "--dataset", "eoinamoore/historical-nba-data-and-player-box-scores",
        "--use-neural",
        "--neural-device", "gpu" if use_gpu else "cpu",
        "--neural-epochs", str(epochs),
        "--batch-size", "4096",
        "--verbose",
        "--fresh",
        "--lgb-log-period", "50"
    ]

    print(f"\nRunning command:")
    print(" ".join(cmd))
    print("\n")

    # Run training
    try:
        subprocess.check_call(cmd)
        print("\n" + "="*70)
        print("✅ Training Complete!")
        print("="*70)
        return True
    except subprocess.CalledProcessError as e:
        print("\n" + "="*70)
        print(f"❌ Training failed with error code {e.returncode}")
        print("="*70)
        return False

def show_results():
    """Show trained models"""
    print("\n" + "="*70)
    print("Trained Models")
    print("="*70)

    model_cache = "model_cache"
    if not os.path.exists(model_cache):
        print("⚠️  model_cache/ directory not found")
        return

    files = os.listdir(model_cache)
    if not files:
        print("⚠️  No models found in model_cache/")
        return

    print(f"\nFound {len(files)} files in model_cache/:\n")

    for f in sorted(files):
        size = os.path.getsize(os.path.join(model_cache, f)) / 1e6
        print(f"  - {f:50s} ({size:.1f} MB)")

    print("\n" + "="*70)
    print("Download Instructions")
    print("="*70)
    print("\nOption 1: Zip and Download")
    print("  zip -r nba_models_trained.zip model_cache/")
    print("\nOption 2: Select files in Lightning.ai file browser")
    print("  Right-click → Download")
    print("\nOption 3: Push to GitHub")
    print("  git add model_cache/")
    print("  git commit -m 'Trained models on Lightning.ai'")
    print("  git push origin main")
    print("="*70)

def main():
    """Main execution flow"""
    print("\n" + "#"*70)
    print("# Lightning.ai NBA Model Training")
    print("# Neural Hybrid (TabNet + LightGBM) with All Embeddings")
    print("#"*70)

    # Step 1: Check GPU
    has_gpu = check_gpu()

    # Step 2: Install dependencies
    print("\nInstalling dependencies (this may take 2-3 minutes)...")
    install_dependencies()

    # Step 3: Verify installation
    if not verify_installation():
        print("\n❌ Installation verification failed. Please fix errors above.")
        sys.exit(1)

    # Step 4: Setup Kaggle (optional)
    setup_kaggle()

    # Step 5: Ask user for configuration
    print("\n" + "="*70)
    print("Training Configuration")
    print("="*70)
    print("\nSelect training mode:")
    print("  1. Quick Test (10 epochs, ~15 min)")
    print("  2. Standard Training (30 epochs, ~45 min) [Recommended]")
    print("  3. Full Training (50 epochs, ~90 min)")
    print("  4. Custom")

    choice = input("\nEnter choice [1-4] (default: 2): ").strip() or "2"

    if choice == "1":
        epochs = 10
    elif choice == "2":
        epochs = 30
    elif choice == "3":
        epochs = 50
    elif choice == "4":
        epochs_input = input("Enter number of epochs (10-100): ").strip()
        try:
            epochs = int(epochs_input)
            if epochs < 10 or epochs > 100:
                print("Invalid epochs. Using default 30.")
                epochs = 30
        except ValueError:
            print("Invalid input. Using default 30.")
            epochs = 30
    else:
        print("Invalid choice. Using default 30 epochs.")
        epochs = 30

    # Step 6: Confirm
    print("\n" + "="*70)
    print("Ready to Train")
    print("="*70)
    print(f"Configuration:")
    print(f"  - Neural Hybrid: Enabled")
    print(f"  - Device: {'GPU' if has_gpu else 'CPU'}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Expected time: {epochs * 1.5:.0f}-{epochs * 2:.0f} minutes")
    print("="*70)

    confirm = input("\nProceed with training? [Y/n]: ").strip().lower()
    if confirm and confirm != 'y':
        print("Training cancelled.")
        sys.exit(0)

    # Step 7: Train
    success = train_models(use_gpu=has_gpu, epochs=epochs)

    # Step 8: Show results
    if success:
        show_results()
    else:
        print("\n⚠️  Training failed. Check errors above.")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
