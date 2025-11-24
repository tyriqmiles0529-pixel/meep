#!/usr/bin/env python
"""
Verify Downloaded Models are CPU-Only
Checks if .pkl files contain GPU tensors that would cause issues on CPU-only GCE.
"""

import pickle
import torch
from pathlib import Path

def check_model_for_gpu(model_path):
    """Check if a model file contains GPU tensors"""
    print(f"[*] Checking {model_path.name}...")
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        gpu_tensors_found = []
        
        # Check if it's a dictionary (common format)
        if isinstance(model_data, dict):
            for key, value in model_data.items():
                if hasattr(value, 'is_cuda') and value.is_cuda:
                    gpu_tensors_found.append(f"  - {key}: CUDA tensor")
                elif hasattr(value, 'device') and 'cuda' in str(value.device):
                    gpu_tensors_found.append(f"  - {key}: device={value.device}")
        
        # Check if it's a single model object
        elif hasattr(model_data, 'state_dict'):
            state_dict = model_data.state_dict()
            for key, tensor in state_dict.items():
                if hasattr(tensor, 'is_cuda') and tensor.is_cuda:
                    gpu_tensors_found.append(f"  - {key}: CUDA tensor")
        
        # Generic recursive check for tensors
        elif hasattr(model_data, '__dict__'):
            for attr_name, attr_value in model_data.__dict__.items():
                if hasattr(attr_value, 'is_cuda') and attr_value.is_cuda:
                    gpu_tensors_found.append(f"  - {attr_name}: CUDA tensor")
        
        if gpu_tensors_found:
            print(f"  ❌ GPU TENSORS FOUND:")
            for tensor_info in gpu_tensors_found:
                print(f"    {tensor_info}")
            return False
        else:
            print(f"  ✅ CPU-only model")
            return True
            
    except Exception as e:
        print(f"  ❌ Error loading model: {e}")
        return False

def main():
    """Check all downloaded models"""
    print("="*70)
    print("VERIFYING DOWNLOADED MODELS ARE CPU-ONLY")
    print("="*70)
    
    player_models_dir = Path("player_models")
    if not player_models_dir.exists():
        print("❌ No player_models directory found")
        print("   Download models first: python download_models_simple.py")
        return False
    
    # Find all .pkl files
    model_files = list(player_models_dir.glob("*.pkl"))
    if not model_files:
        print("❌ No .pkl files found in player_models/")
        return False
    
    print(f"[*] Found {len(model_files)} model files to check\n")
    
    cpu_models = []
    gpu_models = []
    
    for model_file in sorted(model_files):
        is_cpu = check_model_for_gpu(model_file)
        if is_cpu:
            cpu_models.append(model_file.name)
        else:
            gpu_models.append(model_file.name)
        print()
    
    # Summary
    print("="*70)
    print("VERIFICATION COMPLETE")
    print("="*70)
    print(f"CPU-only models: {len(cpu_models)}")
    print(f"GPU models: {len(gpu_models)}")
    
    if cpu_models:
        print(f"\n✅ CPU-only models:")
        for model in cpu_models:
            print(f"  - {model}")
    
    if gpu_models:
        print(f"\n❌ GPU models (will fail on CPU-only GCE):")
        for model in gpu_models:
            print(f"  - {model}")
        
        print(f"\n⚠️  RECOMMENDATION:")
        print(f"   - Delete GPU models: rm player_models/{'*'.join(gpu_models)}")
        print(f"   - Train fresh CPU models on GCE: python train_all_windows_gce.py")
    
    if len(gpu_models) == 0:
        print(f"\n✅ All models are CPU-safe for GCE deployment!")
    
    return len(gpu_models) == 0

if __name__ == "__main__":
    success = main()
    if not success:
        print(f"\n💡 SAFEST OPTION: Train fresh CPU models on GCE")
        print(f"   python train_all_windows_gce.py")
