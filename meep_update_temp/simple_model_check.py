#!/usr/bin/env python
"""
Simple Model Architecture Check - No Imports Required

Just loads the pickle file and inspects its structure to determine
whether existing models use hybrid TabNet + LightGBM or LightGBM only.
"""

import modal

app = modal.App("simple-model-check")
image = modal.Image.debian_slim()  # No additional packages needed

nba_models = modal.Volume.from_name("nba-models-cpu")

@app.function(
    image=image,
    volumes={"/models": nba_models},
    timeout=300
)
def check_model_structure():
    """Load model and inspect its structure without importing modules"""
    import pickle
    import os
    
    # Load the last completed model
    model_path = "/models/player_models_2004_2006.pkl"
    
    if not os.path.exists(model_path):
        return {"error": "Model not found"}
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Basic structure inspection
        result = {
            "model_file": "player_models_2004_2006.pkl",
            "file_size_mb": os.path.getsize(model_path) / (1024*1024),
            "top_level_keys": list(model.keys()),
            "architecture": "unknown"
        }
        
        # Check the main structure
        if 'player_models' in model:
            player_models = model['player_models']
            result['player_models_keys'] = list(player_models.keys())
            
            # Look for hybrid indicators
            if 'multi_task_model' in player_models:
                result['architecture'] = "HYBRID TabNet + LightGBM"
                result['has_multi_task'] = True
                
                # Check the multi-task model structure
                multi_task = player_models['multi_task_model']
                result['multi_task_type'] = str(type(multi_task))
                
                # Look for TabNet components
                if hasattr(multi_task, '__dict__'):
                    attrs = list(multi_task.__dict__.keys())
                    result['multi_task_attributes'] = attrs
                    
                    if 'correlated_tabnet' in attrs:
                        result['has_correlated_tabnet'] = True
                    if 'correlated_lgbm' in attrs:
                        result['has_correlated_lgbm'] = True
                    if 'independent_models' in attrs:
                        result['has_independent_models'] = True
            
            # Check if it's just individual models
            elif all(key in ['points', 'assists', 'rebounds', 'threes', 'minutes'] for key in player_models.keys()):
                result['architecture'] = "LightGBM Only"
                result['model_count'] = len(player_models)
                
                # Check model types
                model_types = {}
                for prop, model_obj in player_models.items():
                    if model_obj is not None:
                        model_types[prop] = str(type(model_obj).__name__)
                result['model_types'] = model_types
        
        # Check metadata
        if 'training_metadata' in model:
            metadata = model['training_metadata']
            result['metadata_keys'] = list(metadata.keys())
            if 'method' in metadata:
                result['training_method'] = metadata['method']
            if 'mode' in metadata:
                result['training_mode'] = metadata['mode']
        
        return result
        
    except Exception as e:
        return {"error": str(e)}

@app.local_entrypoint()
def main():
    print("🔍 Checking Model Architecture...")
    print("   Simple inspection without imports")
    print()
    
    result = check_model_structure.remote()
    
    print("="*70)
    print("MODEL ARCHITECTURE RESULTS")
    print("="*70)
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    print(f"📁 Model: {result['model_file']}")
    print(f"💾 Size: {result['file_size_mb']:.1f} MB")
    print(f"🏗️ Architecture: {result['architecture']}")
    
    if result['architecture'] == "HYBRID TabNet + LightGBM":
        print("\n✅ CONFIRMED: HYBRID ARCHITECTURE!")
        print(f"   Multi-task model: {result.get('multi_task_type', 'Unknown')}")
        print(f"   Attributes: {result.get('multi_task_attributes', [])}")
        
        if result.get('has_correlated_tabnet'):
            print("   ✓ Has correlated TabNet encoder")
        if result.get('has_correlated_lgbm'):
            print("   ✓ Has correlated LightGBM models")
        if result.get('has_independent_models'):
            print("   ✓ Has independent models")
        
        print(f"\n🎯 CONCLUSION:")
        print(f"   Your existing 20 models use HYBRID TabNet + LightGBM")
        print(f"   Train remaining 5 models with: modal_hybrid_resume_training.py")
        
    elif result['architecture'] == "LightGBM Only":
        print(f"\n⚠ LIGHTGBM ONLY ARCHITECTURE!")
        print(f"   Model count: {result.get('model_count', 0)}")
        print(f"   Model types: {result.get('model_types', {})}")
        
        print(f"\n🎯 CONCLUSION:")
        print(f"   Your existing 20 models use LightGBM only")
        print(f"   Train remaining 5 models with LightGBM-only approach")
    
    else:
        print(f"\n❓ Unknown structure")
        print(f"   Keys: {result.get('top_level_keys', [])}")
        print(f"   Player models: {result.get('player_models_keys', [])}")
    
    print(f"\n📊 Training method: {result.get('training_method', 'Unknown')}")
    print(f"📊 Training mode: {result.get('training_mode', 'Unknown')}")

if __name__ == "__main__":
    main()
