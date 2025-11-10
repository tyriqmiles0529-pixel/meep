import os
import subprocess
import torch

def run_command(command):
    """Runs a shell command and prints its output."""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(e.stdout)
        print(e.stderr)
        raise

def main():
    """Main setup script."""
    print("📦 Installing packages...")
    run_command("pip install -q pytorch-tabnet lightgbm scikit-learn pandas numpy tqdm")

    print("\n📥 Downloading training code from GitHub...")
    
    # Navigate to Kaggle working directory if it exists, otherwise use current dir
    if os.path.exists('/kaggle/working'):
        os.chdir('/kaggle/working')

    # Clone your repository if not already cloned
    if not os.path.exists('meep'):
        run_command("git clone https://github.com/tyriqmiles0529-pixel/meep.git")
    
    os.chdir('meep')

    print("\n📍 Code version:")
    run_command("git log -1 --oneline")

    # Check GPU
    print("\n🎮 Checking for GPU...")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"   GPU found: {gpu_name}")
    else:
        print("   GPU not available, will use CPU.")

    # Verify dataset exists
    print("\n🔍 Verifying dataset...")
    dataset_path = '/kaggle/input/meeper/aggregated_nba_data.csv/aggregated_nba_data.csv.gzip'
    if os.path.exists(dataset_path):
        size_mb = os.path.getsize(dataset_path) / 1024 / 1024
        print(f"   ✅ Dataset found: {size_mb:.1f} MB")
        print(f"      Path: {dataset_path}")
        print(f"      Full NBA history: 1947-2026 (80 seasons, 1.6M player-games)")
        print(f"      Training will use: ALL DATA (no cutoff)")
    else:
        print(f"   ❌ Dataset not found at: {dataset_path}")
        print("      Make sure you have added the 'meeper' dataset in your environment.")

    print("\n✅ Setup complete!")

if __name__ == "__main__":
    main()
