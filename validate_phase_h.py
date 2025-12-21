
import pandas as pd
import numpy as np
import torch
import joblib
import os
import sys
from ft_transformer import FTTransformer
from build_features import StrictFeatureEngine # Reuse logic if needed, or manual

# Task: Validate Embedding Generation (FREEZING, DETERMINISM)

def validate_embeddings():
    print("Initializing Phase H Validation...")
    
    # Paths
    MODEL_PATH = "models/ft_transformer_v1.pt"
    ENCODER_PATH = "models/player_id_encoder_v1.joblib"
    SCALER_PATH = "models/scaler_v1.joblib"
    FEATS_PATH = "models/cont_features_v1.joblib"
    
    if not os.path.exists(MODEL_PATH):
        print("Model not found!")
        return
        
    # Load Artifacts
    le = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    cont_features = joblib.load(FEATS_PATH)
    
    # Create Dummy Data (e.g., LeBron James)
    # LeBron ID: 2544
    dummy_input = {
        'PLAYER_ID': 2544,
        'roll_PTS_10': 25.4,
        'roll_AST_10': 8.2,
        'roll_REB_10': 7.5,
        'roll_MIN_10': 34.5,
        'roll_TS_pct_10': 0.58,
        'roll_usage_proxy_10': 28.5,
        'role_trend_min': 32.0,
        'season_PTS_avg': 24.8
    }
    
    print("Dummy Input:", dummy_input)
    
    # Encode Input
    # 1. Player ID
    pid_str = str(dummy_input['PLAYER_ID'])
    # Manual map provided in generate_embeddings
    player_map = {p: i+1 for i, p in enumerate(le.classes_)}
    pid_idx = player_map.get(pid_str, 0)
    print(f"Mapped PID {pid_str} -> {pid_idx}")
    
    # 2. Continuous
    # Construct array in correct order
    vals = [dummy_input.get(f, 0) for f in cont_features]
    vals_arr = np.array(vals).reshape(1, -1)
    vals_scaled = scaler.transform(vals_arr)
    print("Scaled Cont Features:", vals_scaled[0][:3], "...")
    
    # Load Model
    num_players = len(le.classes_) + 1
    model = FTTransformer(num_players, len(cont_features), embed_dim=16)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval() # Mandatory
    
    # Check Freezing
    # Technically eval() affects Dropout/BatchNorm, not gradients.
    # To strictly ensure no gradients, we check require_grad or wrap in no_grad.
    # We will verify no_grad context in execution.
    
    # Run Inference 1
    with torch.no_grad():
        x_cat = torch.LongTensor([pid_idx])
        x_cont = torch.FloatTensor(vals_scaled)
        
        _, emb1 = model(x_cat, x_cont)
        
    # Run Inference 2
    with torch.no_grad():
        _, emb2 = model(x_cat, x_cont)
        
    print("\n--- Validation Results ---")
    
    # 1. Determinism
    diff = torch.abs(emb1 - emb2).sum().item()
    if diff == 0:
        print("✅ Determinism Check Passed (Run 1 == Run 2)")
    else:
        print(f"❌ Determinism Failed! Diff: {diff}")
        
    # 2. Magnitude / Nan
    if torch.isnan(emb1).any():
        print("❌ Nan Detected in Embedding!")
    else:
        print("✅ No NaNs")
        
    norm = torch.norm(emb1).item()
    print(f"Embedding Norm: {norm:.4f}")
    if norm > 0.1 and norm < 100:
        print("✅ Magnitude Reasonable")
    else:
        print("⚠️ Magnitude suspicious")
        
    print("\nGenerated Embedding (First 4 dims):", emb1[0][:4].numpy())
    
    # Finish
    print("\nPhase H Validation Script Completed Successfully.")

if __name__ == "__main__":
    validate_embeddings()
