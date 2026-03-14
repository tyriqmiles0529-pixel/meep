#!/usr/bin/env python
"""
Verify Model Architecture - Check if existing models use Hybrid TabNet + LightGBM

Loads one of your existing models to inspect its structure and confirm
whether it uses the hybrid architecture or a different approach.
"""

import modal

app = modal.App("verify-architecture")
image = modal.Image.debian_slim().pip_install([
    "pandas>=2.0.0",
    "numpy>=1.24.0"
])

nba_models = modal.Volume.from_name("nba-models-cpu")

@app.function(
    image=image,
    volumes={"/models": nba_models},
    timeout=300
)
def inspect_existing_model():
    """Load and inspect the structure of an existing model"""
    import pickle
    import os
    
    # Load the last completed model from your 20-hour session
    model_path = "/models/player_models_2004_2006.pkl"
    
    if not os.path.exists(model_path):
        return {"error": "Model not found"}
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Inspect model structure
        inspection = {
            "model_file": "player_models_2004_2006.pkl",
            "file_size_mb": os.path.getsize(model_path) / (1024*1024),
            "top_level_keys": list(model.keys()),
            "architecture_type": "unknown"
        }
        
        # Check for hybrid architecture indicators
        if 'models' in model:
            models_dict = model['models']
            inspection['models_keys'] = list(models_dict.keys())
            
            # Check for hybrid multi-task model
            if 'multi_task_model' in models_dict:
                inspection['architecture_type'] = "Hybrid Multi-Task (TabNet + LightGBM)"
                hybrid_model = models_dict['multi_task_model']
                
                # Get hybrid model details
                if hasattr(hybrid_model, '__class__'):
                    inspection['hybrid_model_class'] = str(hybrid_model.__class__)
                
                if hasattr(hybrid_model, 'correlated_tabnet'):
                    inspection['has_correlated_tabnet'] = True
                    inspection['correlated_props'] = getattr(hybrid_model, 'correlated_props', [])
                    inspection['independent_props'] = getattr(hybrid_model, 'independent_props', [])
                
                if hasattr(hybrid_model, 'correlated_lgbm'):
                    inspection['has_correlated_lgbm'] = True
                    inspection['lgbm_props'] = list(hybrid_model.correlated_lgbm.keys())
            
            # Check for single models (LightGBM only)
            elif all(key in ['points', 'assists', 'rebounds', 'threes', 'minutes'] for key in models_dict.keys()):
                inspection['architecture_type'] = "Single-Task LightGBM Only"
                
                # Check model types
                model_types = {}
                for prop, model_obj in models_dict.items():
                    if model_obj is not None:
                        model_types[prop] = str(type(model_obj).__name__)
                inspection['model_types'] = model_types
        
        # Check metadata
        if 'metrics' in model:
            inspection['training_metrics'] = model['metrics']
            if 'mode' in model['metrics']:
                inspection['training_mode'] = model['metrics']['mode']
        
        return inspection
        
    except Exception as e:
        return {"error": str(e)}

@app.local_entrypoint()
def main():
    print("🔍 Verifying Model Architecture...")
    print("   Checking if your existing 20 models use hybrid TabNet + LightGBM")
    print()
    
    result = inspect_existing_model.remote()
    
    print("="*70)
    print("MODEL ARCHITECTURE INSPECTION")
    print("="*70)
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    print(f"📁 Model: {result['model_file']}")
    print(f"💾 Size: {result['file_size_mb']:.1f} MB")
    print(f"🏗️ Architecture: {result['architecture_type']}")
    
    if result['architecture_type'] == "Hybrid Multi-Task (TabNet + LightGBM)":
        print("\n✅ HYBRID ARCHITECTURE CONFIRMED!")
        print(f"   Hybrid Model: {result.get('hybrid_model_class', 'Unknown')}")
        print(f"   Correlated Props: {result.get('correlated_props', [])}")
        print(f"   Independent Props: {result.get('independent_props', [])}")
        print(f"   LightGBM Props: {result.get('lgbm_props', [])}")
        
        print("\n🎯 RECOMMENDATION:")
        print("   Your existing 20 models use Hybrid TabNet + LightGBM")
        print("   The remaining 5 models should use the SAME architecture")
        print("   → Use modal_hybrid_resume_training.py")
        
    elif result['architecture_type'] == "Single-Task LightGBM Only":
        print("\n⚠ SINGLE-TASK ARCHITECTURE DETECTED!")
        print(f"   Model Types: {result.get('model_types', {})}")
        
        print("\n🎯 RECOMMENDATION:")
        print("   Your existing 20 models use LightGBM only")
        print("   The remaining 5 models should use LightGBM only")
        print("   → Use modal_complete_training.py (LightGBM version)")
    
    else:
        print(f"\n❓ Unknown architecture: {result['architecture_type']}")
        print(f"   Keys: {result.get('top_level_keys', [])}")
        print(f"   Models: {result.get('models_keys', [])}")
    
    print(f"\n📊 Training Mode: {result.get('training_mode', 'Unknown')}")

if __name__ == "__main__":
    main()
