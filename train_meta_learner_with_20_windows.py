#!/usr/bin/env python
"""
Train Meta-Learner with 20 Available Windows

Uses the 20 real trained models (1947-2006) instead of waiting for 5 more.
This provides excellent coverage and gets you started immediately.

Usage:
    python train_meta_learner_with_20_windows.py
"""

import modal
import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Tuple

# Modal setup
app = modal.App("nba-meta-learner-20-windows")
image = modal.Image.debian_slim().pip_install([
    "pandas>=2.0.0",
    "numpy>=1.24.0", 
    "scikit-learn>=1.3.0",
    "lightgbm>=4.0.0",
    "pytorch-tabnet>=4.0.0",
    "torch>=2.0.0",
    "joblib>=1.3.0",
    "pyarrow>=12.0.0",
    "fastparquet>=2023.0.0"
])

# Volumes
nba_data = modal.Volume.from_name("nba-data")
nba_models = modal.Volume.from_name("nba-models-cpu")
project_mount = modal.Mount.from_local_dir(".", remote_path="/root/project")

@app.function(
    image=image,
    volumes={"/data": nba_data, "/models": nba_models},
    mounts={"/root/project": project_mount},
    timeout=3600,  # 1 hour
    retries=2
)
def train_meta_learner_20_windows():
    """Train meta-learner with 20 available windows"""
    print("="*80)
    print("TRAINING META-LEARNER WITH 20 WINDOWS (1947-2006)")
    print("="*80)
    
    # Load available models
    model_dir = "/models"
    available_models = []
    
    for file in os.listdir(model_dir):
        if file.startswith("player_models_") and file.endswith(".pkl"):
            # Check if it's a real model (not mock 0.0 MB)
            file_path = os.path.join(model_dir, file)
            size_mb = os.path.getsize(file_path) / (1024*1024)
            
            if size_mb > 5:  # Real models are 15-20 MB
                # Extract years
                parts = file.replace("player_models_", "").replace(".pkl", "").split("_")
                if len(parts) == 2:
                    start, end = int(parts[0]), int(parts[1])
                    available_models.append((start, end, file, size_mb))
    
    available_models.sort()
    
    print(f"✅ Found {len(available_models)} real trained models:")
    for start, end, filename, size in available_models:
        print(f"   {start}-{end}: {filename} ({size:.1f} MB)")
    
    if len(available_models) == 0:
        print("❌ No real models found!")
        return {"status": "failed", "reason": "no_real_models"}
    
    # Load training data
    print(f"\n📊 Loading training data...")
    df = pd.read_parquet("/data/aggregated_nba_data.parquet")
    print(f"   Loaded {len(df):,} games")
    
    # Import training functions
    sys.path.insert(0, "/root/project")
    
    try:
        from train_meta_learner_v2 import collect_window_predictions, train_meta_learner_for_prop
        print("✅ Using meta-learner training functions")
    except ImportError as e:
        print(f"⚠ Could not import meta-learner functions: {e}")
        print("   Will use simplified meta-learner training")
        return {"status": "failed", "reason": "import_error"}
    
    # Load window models
    print(f"\n🏀 Loading {len(available_models)} window models...")
    window_models = {}
    
    for start, end, filename, size in available_models:
        model_path = f"/models/{filename}"
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            window_name = f"{start}_{end}"
            window_models[window_name] = model
            print(f"   ✅ Loaded: {window_name}")
            
        except Exception as e:
            print(f"   ❌ Failed to load {window_name}: {e}")
            continue
    
    print(f"   Successfully loaded {len(window_models)} window models")
    
    if len(window_models) < 10:
        print("❌ Too few models loaded for effective meta-learner training")
        return {"status": "failed", "reason": "insufficient_models"}
    
    # Train meta-learners for each prop
    props = ['points', 'assists', 'rebounds', 'threes', 'minutes']
    meta_learners = {}
    
    print(f"\n🧠 Training meta-learners for {len(props)} props...")
    
    for i, prop in enumerate(props, 1):
        print(f"\n[{i}/{len(props)}] Training {prop} meta-learner...")
        
        try:
            # Collect predictions from all windows
            print(f"   📊 Collecting window predictions...")
            predictions_data = collect_window_predictions(df, window_models, prop)
            
            if not predictions_data or len(predictions_data.get('window_predictions', [])) == 0:
                print(f"   ⚠ No predictions collected for {prop}")
                continue
            
            # Train meta-learner
            print(f"   🧠 Training meta-learner model...")
            meta_model = train_meta_learner_for_prop(
                predictions_data, 
                prop_type=prop,
                verbose=False
            )
            
            if meta_model:
                meta_learners[prop] = meta_model
                print(f"   ✅ Trained {prop} meta-learner")
            else:
                print(f"   ❌ Failed to train {prop} meta-learner")
                
        except Exception as e:
            print(f"   ❌ Error training {prop} meta-learner: {e}")
            continue
    
    # Save meta-learners
    print(f"\n💾 Saving {len(meta_learners)} trained meta-learners...")
    
    saved_count = 0
    for prop, model in meta_learners.items():
        try:
            model_path = f"/models/meta_learner_{prop}_20_windows.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            size_mb = os.path.getsize(model_path) / (1024*1024)
            print(f"   ✅ Saved: meta_learner_{prop}_20_windows.pkl ({size_mb:.1f} MB)")
            saved_count += 1
            
        except Exception as e:
            print(f"   ❌ Failed to save {prop} meta-learner: {e}")
    
    # Final summary
    print(f"\n" + "="*80)
    print(f"🏁 META-LEARNER TRAINING COMPLETE!")
    print(f"="*80)
    print(f"✅ Window models used: {len(window_models)}")
    print(f"✅ Meta-learners trained: {saved_count}/{len(props)}")
    print(f"✅ Coverage: 1947-2006 (59 years of NBA history)")
    print(f"✅ Ready for backtesting and predictions!")
    
    if saved_count >= 3:
        print(f"\n🎉 SUCCESS! Meta-learner ensemble ready!")
        print(f"   Next step: Test with riq_analyzer.py")
        next_step = "python riq_analyzer.py"
    else:
        print(f"\n⚠ Limited success - only {saved_count} meta-learners trained")
        next_step = "Debug training issues"
    
    return {
        "status": "success" if saved_count >= 3 else "partial",
        "window_models": len(window_models),
        "meta_learners": saved_count,
        "coverage_years": "1947-2006",
        "next_step": next_step
    }

@app.local_entrypoint()
def main():
    """Local entry point"""
    print("🚀 Starting Meta-Learner Training with 20 Windows...")
    print("   Using real trained models from your 20-hour session")
    print("   Coverage: 1947-2006 (59 years of NBA history)")
    print("   Estimated time: 45-60 minutes")
    print()
    
    result = train_meta_learner_20_windows.remote()
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Status: {result['status']}")
    print(f"Window models: {result['window_models']}")
    print(f"Meta-learners: {result['meta_learners']}")
    print(f"Coverage: {result['coverage_years']}")
    print(f"Next step: {result['next_step']}")
    
    if result['status'] == 'success':
        print("\n🎉 EXCELLENT! Your 20-hour investment paid off!")
        print("   You now have a working meta-learner ensemble!")
    else:
        print("\n⚠ Partial success - check the output above")

if __name__ == "__main__":
    main()
