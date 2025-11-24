#!/usr/bin/env python
"""
Simple Modal Model Downloader - Windows Compatible
Downloads models without Unicode encoding issues.
"""

import os
import subprocess
import sys
from pathlib import Path

def main():
    """Download models using direct Modal commands"""
    
    # Create player_models directory
    player_models_dir = Path("player_models")
    player_models_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("SIMPLE MODAL MODEL DOWNLOADER")
    print("="*70)
    print(f"Output: {player_models_dir.absolute()}")
    print()
    
    # Set encoding for this session
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    try:
        # Get file list from Modal
        print("[*] Getting file list from Modal...")
        result = subprocess.run(
            ["modal", "volume", "ls", "nba-models"],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        if result.returncode != 0:
            print(f"Error getting file list: {result.stderr}")
            return False
        
        # Parse files
        files = [f.strip() for f in result.stdout.split('\n') if f.strip() and f.endswith('.pkl')]
        print(f"Found {len(files)} .pkl files")
        
        # Download each file using direct command
        success_count = 0
        for i, filename in enumerate(files, 1):
            dest_path = player_models_dir / filename
            
            if dest_path.exists():
                print(f"[{i}/{len(files)}] SKIP {filename} (already exists)")
                success_count += 1
                continue
            
            print(f"[{i}/{len(files)}] Downloading {filename}...")
            
            # Use shell command to avoid subprocess encoding issues
            cmd = f'modal volume get nba-models "{filename}" "{dest_path}"'
            
            try:
                # Run with shell to handle encoding better
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if result.returncode == 0 and dest_path.exists():
                    size_mb = dest_path.stat().st_size / (1024 * 1024)
                    print(f"    OK ({size_mb:.1f} MB)")
                    success_count += 1
                else:
                    print(f"    FAILED")
                    if result.stderr:
                        print(f"    Error: {result.stderr[:200]}")
            except Exception as e:
                print(f"    FAILED: {e}")
        
        print()
        print("="*70)
        print("DOWNLOAD COMPLETE")
        print("="*70)
        print(f"Successfully downloaded: {success_count}/{len(files)} files")
        print(f"Directory: {player_models_dir}")
        
        # List what we got
        if player_models_dir.exists():
            downloaded_files = list(player_models_dir.glob("*.pkl"))
            print(f"Files in directory: {len(downloaded_files)}")
            
            # Show window models specifically
            window_models = [f for f in downloaded_files if f.name.startswith('player_models_')]
            meta_models = [f for f in downloaded_files if f.name.startswith('meta_learner_')]
            
            print(f"  Window models: {len(window_models)}")
            print(f"  Meta-learners: {len(meta_models)}")
            
            if window_models:
                print(f"\nWindow models found:")
                for model in sorted(window_models):
                    print(f"  - {model.name}")
        
        return success_count > 0
        
    except Exception as e:
        print(f"Fatal error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
