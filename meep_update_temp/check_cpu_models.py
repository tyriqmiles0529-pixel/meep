#!/usr/bin/env python
"""
Check what's in the CPU model volume
"""

import modal

app = modal.App("check-cpu-models")

model_volume = modal.Volume.from_name("nba-models-cpu")

image = modal.Image.debian_slim().pip_install(["pandas"])

@app.function(
    image=image,
    gpu=None,
    memory=8192,
    volumes={"/models": model_volume}
)
def list_cpu_models():
    """List and inspect CPU models (without loading pickle)"""
    import os
    import pandas as pd

    print("="*70)
    print("CHECKING CPU MODEL VOLUME")
    print("="*70)

    # List all files
    print("Files in /models:")
    files = os.listdir("/models")
    for f in sorted(files):
        size = os.path.getsize(f"/models/{f}") / (1024*1024)  # MB
        print(f"  {f:30} ({size:.1f} MB)")

    # Check test model exists and get basic info
    test_file = "/models/test_cpu_window_1947_1949.pkl"
    if os.path.exists(test_file):
        print(f"\n✅ Test model found: {test_file}")
        
        # Get file info without loading
        size = os.path.getsize(test_file) / (1024*1024)
        print(f"  Size: {size:.1f} MB")
        
        # Try to peek at pickle header (safer than full load)
        try:
            import pickle
            with open(test_file, 'rb') as f:
                # Move to end to check it's a valid pickle
                f.seek(-10, 2)  # Last 10 bytes
                footer = f.read(10)
                print(f"  Valid pickle file: {b'.' in footer}")
        except Exception as e:
            print(f"  Pickle check failed: {e}")
        
        print(f"  ✅ Model appears to be saved successfully")
    else:
        print(f"❌ Test model not found: {test_file}")

    print(f"\n{'='*70}")
    print(f"✅ CPU model volume inspection complete")
    print(f"Ready to run full training: modal run modal_train_all_cpu.py")

@app.local_entrypoint()
def main():
    """Check CPU models"""
    list_cpu_models.remote()

if __name__ == "__main__":
    main()
