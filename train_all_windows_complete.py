#!/usr/bin/env python
"""
Train ALL Windows Fresh - Complete 1947-2024
Trains every window from scratch with CPU-only models.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

# Force CPU mode
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

sys.path.insert(0, ".")


def clear_all_models():
    """Clear all existing player models without confirmation"""
    print("="*60)
    print("CLEARING ALL PLAYER MODELS (GPU + CPU)")
    print("="*60)
    
    player_models_dir = Path("player_models")
    if player_models_dir.exists():
        print(f"[*] Removing existing player_models directory...")
        shutil.rmtree(player_models_dir)
        print("✅ Removed player_models directory")
    else:
        print("✅ No existing player_models directory found")
    
    # Recreate empty directory
    player_models_dir.mkdir(exist_ok=True)
    print("✅ Created fresh player_models directory")
    return True


def get_all_windows():
    """Generate list of ALL windows from 1947-2026"""
    # Define all 3-year windows from 1947-2026
    windows = [
        (1947, 1949), (1950, 1952), (1953, 1955), (1956, 1958), (1959, 1961),
        (1962, 1964), (1965, 1967), (1968, 1970), (1971, 1973), (1974, 1976),
        (1977, 1979), (1980, 1982), (1983, 1985), (1986, 1988), (1989, 1991),
        (1992, 1994), (1995, 1997), (1998, 2000), (2001, 2003), (2004, 2006),
        (2007, 2009), (2010, 2012), (2013, 2015), (2016, 2018), (2019, 2021),
        (2022, 2024), (2025, 2026)
    ]
    
    return windows


def prepare_data():
    """Ensure aggregated dataset exists"""
    print("\n[*] Preparing aggregated dataset...")
    
    data_file = Path("aggregated_nba_data.csv.gz")
    if data_file.exists():
        print(f"✅ Found existing data: {data_file}")
        return str(data_file)
    
    print("[*] Creating aggregated dataset...")
    
    # Check if creation script exists
    if not Path("create_aggregated_dataset.py").exists():
        raise Exception("create_aggregated_dataset.py not found")
    
    # Run data creation
    result = subprocess.run(
        "python create_aggregated_dataset.py",
        shell=True,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"❌ Data creation failed: {result.stderr}")
        raise Exception("Failed to create aggregated dataset")
    
    if data_file.exists():
        print(f"✅ Created aggregated dataset: {data_file}")
        return str(data_file)
    else:
        raise Exception("Data file not created")


def train_window(start_year, end_year, data_file):
    """Train a single window"""
    window_name = f"{start_year}_{end_year}"
    print(f"\n{'='*60}")
    print(f"TRAINING WINDOW: {window_name}")
    print(f"{'='*60}")
    
    cache_dir = f"player_models/{window_name}"
    
    command = f"python train_player_models.py --data {data_file} --min-year {start_year} --max-year {end_year} --shared-epochs 6 --independent-epochs 8 --patience 3 --cache-dir {cache_dir}"
    
    print(f"[*] Command: {command}")
    
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=21600  # 6 hours timeout
    )
    
    if result.returncode == 0:
        # Verify output
        output_dir = Path(cache_dir)
        if output_dir.exists():
            model_files = list(output_dir.glob("*.pkl"))
            print(f"✅ Window {window_name} completed - {len(model_files)} models")
            return True
        else:
            print(f"❌ Window {window_name} failed - no output directory")
            return False
    else:
        print(f"❌ Window {window_name} failed:")
        print(f"Error: {result.stderr}")
        return False


def main():
    """Main training workflow"""
    print("="*80)
    print("TRAIN ALL WINDOWS FRESH - COMPLETE 1947-2024")
    print("="*80)
    print(f"Started at: {datetime.now()}")
    print("This will train ALL windows fresh (no GPU models)")
    print("="*80)
    
    # Step 1: Clear all models
    clear_all_models()
    
    # Step 2: Prepare data
    try:
        data_file = prepare_data()
    except Exception as e:
        print(f"❌ Data preparation failed: {e}")
        return False
    
    # Step 3: Get all windows
    windows = get_all_windows()
    print(f"\n[*] Training ALL {len(windows)} windows:")
    for start, end in windows:
        print(f"  - {start}-{end}")
    
    print(f"\n[*] Estimated time: 80-120 hours (25 windows × 3-5 hours each)")
    print(f"[*] Use 'screen' or 'tmux' to survive disconnections")
    
    # Step 4: Train each window
    successful_windows = []
    failed_windows = []
    
    for i, (start_year, end_year) in enumerate(windows, 1):
        print(f"\n{'='*80}")
        print(f"WINDOW {i}/{len(windows)}: {start_year}-{end_year}")
        print(f"{'='*80}")
        
        try:
            success = train_window(start_year, end_year, data_file)
            if success:
                successful_windows.append(f"{start_year}_{end_year}")
            else:
                failed_windows.append(f"{start_year}_{end_year}")
        except Exception as e:
            print(f"❌ Window {start_year}-{end_year} crashed: {e}")
            failed_windows.append(f"{start_year}_{end_year}")
    
    # Step 5: Summary
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Total windows: {len(windows)}")
    print(f"Successful: {len(successful_windows)}")
    print(f"Failed: {len(failed_windows)}")
    
    if successful_windows:
        print(f"\n✅ Successful windows:")
        for window in successful_windows:
            print(f"  - {window}")
    
    if failed_windows:
        print(f"\n❌ Failed windows:")
        for window in failed_windows:
            print(f"  - {window}")
    
    print(f"\n📁 Output directory: player_models/")
    
    # Check total models
    total_models = 0
    player_models_dir = Path("player_models")
    if player_models_dir.exists():
        for window_dir in player_models_dir.iterdir():
            if window_dir.is_dir():
                models = list(window_dir.glob("*.pkl"))
                total_models += len(models)
        
        print(f"📊 Total model files: {total_models}")
    
    print(f"\n🎯 Next steps:")
    print(f"1. Train meta-learner with all {len(successful_windows)} windows")
    print(f"2. Create production bundle")
    print(f"3. Deploy to production")
    
    print(f"{'='*80}")
    
    return len(failed_windows) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
