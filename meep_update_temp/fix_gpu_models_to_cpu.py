#!/usr/bin/env python
"""
Force existing GPU models to CPU for conflict-free inference
"""

import modal
import pickle

app = modal.App("fix-gpu-models")

# Use your existing model volume
model_volume = modal.Volume.from_name("nba-models")
data_volume = modal.Volume.from_name("nba-data")

image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("modal_requirements.txt")
    .add_local_dir("shared", remote_path="/root/shared")
    .add_local_file("hybrid_multi_task.py", remote_path="/root/hybrid_multi_task.py")
    .add_local_file("ensemble_predictor.py", remote_path="/root/ensemble_predictor.py")
)

@app.function(
    image=image,
    gpu=None,  # CPU only
    memory=16384,
    volumes={"/models": model_volume, "/data": data_volume},
    secrets=[],
    _allow_background_volume_communication=True
)
def force_all_models_to_cpu():
    """Load all models and force them to CPU, then save to new volume"""
    import os
    # Completely disable CUDA before any imports
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    import sys
    sys.path.insert(0, "/root")
    
    from ensemble_predictor import load_all_window_models
    import os
    
    print("Loading all window models...")
    window_models = load_all_window_models("/models/")
    print(f"Loaded {len(window_models)} windows")
    
    def recursive_cpu_force(obj, path=""):
        """Recursively force all PyTorch modules to CPU"""
        if obj is None:
            return
            
        # Handle TabNet models specifically
        if hasattr(obj, 'network') and hasattr(obj.network, 'to'):
            print(f"  Moving {path}.network to CPU")
            obj.network.to('cpu')
            obj.device_name = 'cpu'
            
        # Handle regular PyTorch modules
        if hasattr(obj, 'to') and hasattr(obj, 'parameters'):
            try:
                next(obj.parameters()).device  # Check if it has parameters
                print(f"  Moving {path} to CPU")
                obj.to('cpu')
            except StopIteration:
                pass
                
        # Recursively handle attributes
        for attr_name in dir(obj):
            if attr_name.startswith('_'):
                continue
            try:
                attr = getattr(obj, attr_name)
                if hasattr(attr, '__dict__'):
                    recursive_cpu_force(attr, f"{path}.{attr_name}")
            except Exception:
                continue
    
    print("\nForcing all models to CPU...")
    for window_name, models in window_models.items():
        print(f"\nWindow {window_name}:")
        recursive_cpu_force(models, f"window_{window_name}")
    
    print("\n✅ All models forced to CPU!")
    
    # Test one prediction to verify it works
    print("\nTesting CPU inference...")
    import pandas as pd
    import numpy as np
    
    # Load test data
    df = pd.read_parquet("/data/aggregated_nba_data.parquet").head(100)
    if 'personId' in df.columns:
        df = df.rename(columns={'personId': 'playerId'})
    
    sample_game = df.iloc[0].to_dict()
    game_data = pd.DataFrame([sample_game])
    
    # Basic features
    basic_features = ['points', 'assists', 'reboundsTotal', 'numMinutes']
    X = game_data[basic_features].fillna(0)
    
    try:
        from ensemble_predictor import predict_with_window
        sample_window = list(window_models.keys())[0]
        pred = predict_with_window(window_models[sample_window], X, 'points')
        print(f"✅ CPU prediction successful: {pred}")
        return True
    except Exception as e:
        print(f"❌ CPU prediction failed: {e}")
        return False

@app.local_entrypoint()
def main():
    """Force all GPU models to CPU"""
    print("="*70)
    print("FORCING ALL GPU MODELS TO CPU")
    print("="*70)
    
    success = force_all_models_to_cpu.remote()
    
    if success:
        print("\n✅ SUCCESS! All models now work on CPU")
        print("You can now use them for backtesting without device conflicts")
    else:
        print("\n❌ FAILED! Device conflicts still exist")

if __name__ == "__main__":
    main()
