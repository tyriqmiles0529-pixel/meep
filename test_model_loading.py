#!/usr/bin/env python3
"""
Test script to verify all models from train_auto.py can be loaded by riq_analyzer.py
"""
import os
import sys

# Test model loading
print("="*72)
print("Testing Model Loading from train_auto.py output")
print("="*72)

MODEL_DIR = "models"

# Expected files from train_auto.py
EXPECTED_FILES = {
    "Player Models": [
        "points_model.pkl",
        "assists_model.pkl", 
        "rebounds_model.pkl",
        "threes_model.pkl",
        "minutes_model.pkl",
    ],
    "Game Models": [
        "moneyline_model.pkl",
        "moneyline_calibrator.pkl",
        "spread_model.pkl",
    ],
    "Metadata": [
        "training_metadata.json",
        "spread_sigma.json",
    ]
}

print(f"\nChecking {MODEL_DIR}/ for expected files:")
print("-"*72)

all_found = True
for category, files in EXPECTED_FILES.items():
    print(f"\n{category}:")
    for fname in files:
        path = os.path.join(MODEL_DIR, fname)
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        size = f"({os.path.getsize(path)/1024:.1f} KB)" if exists else ""
        print(f"  {status} {fname} {size}")
        if not exists:
            all_found = False

print("\n" + "="*72)

if not all_found:
    print("⚠️  Some models are missing. Run train_auto.py first to generate them.")
    print("   Command: python train_auto.py --verbose --skip-rest")
else:
    print("✅ All expected model files found!")
    
    # Test loading via riq_analyzer
    print("\nTesting model loading via riq_analyzer.py:")
    print("-"*72)
    
    try:
        # Import after setting up path
        import riq_analyzer
        
        # Check what was loaded
        print(f"\n✅ Successfully imported riq_analyzer")
        print(f"   Player models loaded: {list(riq_analyzer.MODEL.player_models.keys())}")
        print(f"   Game models loaded: {list(riq_analyzer.MODEL.game_models.keys())}")
        print(f"   Spread sigma: {riq_analyzer.MODEL.spread_sigma}")
        print(f"\n   Model RMSEs from metadata:")
        for stat_type, rmse in riq_analyzer.MODEL_RMSE.items():
            print(f"     {stat_type}: {rmse}")
        
        print("\n✅ All models successfully loaded and integrated!")
        
    except Exception as e:
        print(f"\n❌ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

print("="*72)
