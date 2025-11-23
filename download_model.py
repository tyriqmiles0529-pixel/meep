#!/usr/bin/env python
"""
Download trained meta-learner from Modal volume to local
"""

import modal
import os
from pathlib import Path

app = modal.App("download-model")

# Volume
model_volume = modal.Volume.from_name("nba-models")

# Simple image
image = modal.Image.debian_slim().pip_install(["dill"])

@app.function(
    image=image,
    timeout=600,
    volumes={"/models": model_volume}
)
def get_model_info():
    """Get info about the trained model"""
    import os
    from pathlib import Path
    
    model_file = Path("/models/meta_learner_v4_all_components.pkl")
    
    if model_file.exists():
        size_mb = model_file.stat().st_size / (1024*1024)
        return {
            "exists": True,
            "size_mb": round(size_mb, 2),
            "path": str(model_file)
        }
    else:
        return {"exists": False}

@app.local_entrypoint()
def main():
    """Check if model exists"""
    result = get_model_info.remote()
    
    print("="*50)
    print("MODEL STATUS")
    print("="*50)
    
    if result["exists"]:
        print(f"✅ Model found: {result['path']}")
        print(f"📦 Size: {result['size_mb']} MB")
        print("\n📥 To download: Use Modal CLI or wait for billing reset")
    else:
        print("❌ No trained model found")
    
    return result

if __name__ == "__main__":
    main()
