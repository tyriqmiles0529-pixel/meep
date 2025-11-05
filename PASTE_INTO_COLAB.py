# PASTE THIS INTO GOOGLE COLAB - COMPLETE TRAINING SCRIPT
# Just copy everything below and paste into a Colab cell

# ==============================================================================
# 1. INSTALL DEPENDENCIES
# ==============================================================================
print("📦 Installing dependencies...")
!pip install -q kagglehub lightgbm scikit-learn pandas numpy torch pytorch-tabnet nba-api

# ==============================================================================
# 2. SETUP KAGGLE
# ==============================================================================
print("\n📤 Upload your kaggle.json:")
from google.colab import files
uploaded = files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
print("✅ Kaggle configured!")

# ==============================================================================
# 2.5. OPTIONAL: UPLOAD PRIORS DATA (Basketball Reference stats)
# ==============================================================================
print("\n📊 OPTIONAL: Upload priors_data folder")
print("   If you have it, upload the ZIP file now.")
print("   If not, just skip this and press the cancel/X button.")
print("   Training works fine without priors!\n")

priors_path = None
try:
    priors_uploaded = files.upload()
    if priors_uploaded:
        # Extract priors zip to /content (before changing directory)
        priors_file = list(priors_uploaded.keys())[0]
        if priors_file.endswith('.zip'):
            # Use Python's zipfile to avoid bash escaping issues with spaces
            import zipfile
            with zipfile.ZipFile(priors_file, 'r') as zip_ref:
                zip_ref.extractall('/content/priors_data')
            priors_path = "/content/priors_data"
            print(f"✅ Priors data uploaded and extracted to /content/priors_data!")
            
            # Show what was extracted
            import os
            extracted_files = os.listdir('/content/priors_data')
            print(f"   Extracted {len(extracted_files)} items: {extracted_files[:5]}")
        else:
            print("⚠️  Expected a ZIP file. Continuing without priors.")
except Exception as e:
    print(f"ℹ️  No priors uploaded. Training will use defaults (still works!)")
    print(f"   Error: {e}")

# ==============================================================================
# 3. DOWNLOAD CODE
# ==============================================================================
print("\n📥 Downloading latest code from GitHub...")
import os
!rm -rf meep-main main.zip
!wget -q https://github.com/tyriqmiles0529-pixel/meep/archive/refs/heads/main.zip
!unzip -q main.zip

# Change to the downloaded directory
os.chdir('meep-main')
print("✅ Code downloaded!")
print(f"📁 Working directory: {os.getcwd()}")
print(f"📄 Files: {os.listdir('.')[:10]}")

# ==============================================================================
# 4. CHECK GPU
# ==============================================================================
import torch
print(f"\n🎮 GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
else:
    print("   ⚠️  No GPU - Go to Runtime → Change runtime type → GPU")

# ==============================================================================
# 5. TRAIN MODELS (This takes 20-30 minutes)
# ==============================================================================

# Verify we're in the right place
if not os.path.exists('train_auto.py'):
    print("❌ ERROR: train_auto.py not found!")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files: {os.listdir('.')}")
    raise FileNotFoundError("train_auto.py not found. Download may have failed.")

print("\n🚀 Starting training...")
print("⏱️  This will take 20-30 minutes with GPU")
print("   (Loading ALL player data for maximum accuracy)")
print("☕ Get coffee!\n")

# Build command with optional priors
cmd = "python train_auto.py --verbose --fresh --neural-device gpu --neural-epochs 50 --enable-window-ensemble"
if priors_path and os.path.exists(priors_path):
    cmd += f" --priors-dataset {priors_path}"
    print(f"📊 Using uploaded priors data from: {priors_path}")
else:
    print("ℹ️  No priors data - using defaults")

!{cmd}

print("\n✅ TRAINING COMPLETE!")

# ==============================================================================
# 6. SHOW METRICS
# ==============================================================================
print("\n📊 Training Metrics:\n")
!python show_metrics.py

# ==============================================================================
# 7. DOWNLOAD MODELS
# ==============================================================================
print("\n📦 Preparing models for download...")
!zip -r nba_models_trained.zip models/

print("\n💾 Downloading models to your computer...")
files.download('nba_models_trained.zip')

print("\n" + "="*80)
print("✅ ALL DONE!".center(80))
print("="*80)
print("\n📋 Next steps on your local machine:")
print("   1. Extract nba_models_trained.zip")
print("   2. Copy contents to nba_predictor/models/")
print("   3. Run: python riq_analyzer.py")
print("   4. Run: python evaluate.py")
print("\n" + "="*80 + "\n")
