#!/usr/bin/env python
"""
Quick verification that today's models are valid
"""

import modal
import pickle

app = modal.App("model-verify")
image = modal.Image.debian_slim().pip_install(["pandas", "numpy", "scikit-learn"])
nba_models = modal.Volume.from_name("nba-models-cpu")

@app.function(image=image, volumes={"/models": nba_models})
def verify_latest_model():
    """Load and verify the latest completed model"""
    import os
    
    # Check latest model
    model_path = "/models/player_models_2004_2006.pkl"
    
    if not os.path.exists(model_path):
        return {"status": "error", "message": "Latest model not found"}
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Check structure
        required_keys = ['player_models', 'training_metadata']
        missing_keys = [k for k in required_keys if k not in model]
        
        if missing_keys:
            return {"status": "error", "message": f"Missing keys: {missing_keys}"}
        
        # Check stat models
        player_models = model['player_models']
        expected_stats = ['points', 'assists', 'rebounds', 'threes', 'minutes']
        available_stats = list(player_models.keys())
        
        return {
            "status": "success",
            "model_path": model_path,
            "available_stats": available_stats,
            "expected_stats": expected_stats,
            "metadata": model['training_metadata'].get('window', 'unknown')
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    result = verify_latest_model.remote()
    print("Model Verification Result:")
    print(result)
