#!/bin/bash
# Google Compute Engine Setup Script for NBA Meta-Learner V4

echo "=== NBA META-LEARNER V4 GCE SETUP ==="
echo "This script sets up the environment for training on GCE"
echo ""

# Update system packages
echo "[*] Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y

# Install Python and pip if not present
echo "[*] Installing Python and pip..."
sudo apt-get install -y python3 python3-pip python3-venv

# Create virtual environment
echo "[*] Creating virtual environment..."
python3 -m venv nba_env
source nba_env/bin/activate

# Upgrade pip
echo "[*] Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "[*] Installing Python dependencies..."
pip install -r gce_requirements.txt

# Create directories
echo "[*] Creating directories..."
mkdir -p data models logs

# Setup Kaggle credentials
echo "[*] Setting up Kaggle credentials..."
echo "Please set your Kaggle credentials:"
echo "export KAGGLE_USERNAME='your_username'"
echo "export KAGGLE_KEY='your_key'"
echo ""

# Check if window models are needed
echo "[*] Checking for window model files..."
if [ ! -f "player_models_*.pkl" ]; then
    echo "⚠️  Window model files not found!"
    echo "You need to upload your player_models_*.pkl files to this directory"
    echo "Use: gcloud compute scp --recurse local_models_dir instance_name:~/"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Set Kaggle credentials: export KAGGLE_USERNAME='...' KAGGLE_KEY='...'"
echo "2. Upload window model files if needed"
echo "3. Run training: python gce_train_meta.py"
echo ""
