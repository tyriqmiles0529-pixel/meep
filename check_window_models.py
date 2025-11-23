#!/usr/bin/env python
"""
Check all window models in Modal volume
"""

import modal

app = modal.App("check-window-models")

# Volumes
model_volume = modal.Volume.from_name("nba-models")

# Simple image
image = modal.Image.debian_slim().pip_install(["pandas"])

@app.function(
    image=image,
    timeout=300,
    volumes={"/models": model_volume}
)
def list_window_models():
    """List all window model files"""
    import os
    from pathlib import Path
    
    models_dir = Path("/models")
    model_files = list(models_dir.glob("player_models_*.pkl"))
    
    print("="*70)
    print("WINDOW MODELS FOUND")
    print("="*70)
    
    windows = []
    for model_file in sorted(model_files):
        file_size = model_file.stat().st_size / (1024*1024)  # MB
        windows.append({
            "file": model_file.name,
            "size_mb": round(file_size, 2)
        })
        print(f"  {model_file.name} ({file_size:.1f} MB)")
    
    print(f"\nTotal windows: {len(windows)}")
    
    # Check for specific windows
    missing_windows = []
    expected_windows = [
        "player_models_2022_2024.pkl",
        "player_models_2025_2026.pkl"
    ]
    
    for expected in expected_windows:
        if not any(expected in w["file"] for w in windows):
            missing_windows.append(expected)
    
    if missing_windows:
        print(f"\n❌ Missing: {missing_windows}")
    else:
        print(f"\n✅ All recent windows present!")
    
    return {
        "total_windows": len(windows),
        "windows": windows,
        "missing": missing_windows
    }

@app.local_entrypoint()
def main():
    result = list_window_models.remote()
    return result

if __name__ == "__main__":
    main()
