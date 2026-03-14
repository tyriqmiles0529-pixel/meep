#!/usr/bin/env python
"""
Modal Hybrid Resume Training - Restores Your TabNet + LightGBM Architecture

Uses the proven approach from your 20-hour training session:
- Copies all dependencies with .add_local_file()
- Imports hybrid_multi_task successfully on Modal
- Preserves your exact TabNet + LightGBM architecture

Usage:
    modal run modal_hybrid_resume_training.py
"""

import modal

app = modal.App("nba-hybrid-resume-training")

# Volumes
data_volume = modal.Volume.from_name("nba-data")
model_volume = modal.Volume.from_name("nba-models-cpu")

# Image with all dependencies copied (proven approach from your 20-hour session)
image = (
    modal.Image.debian_slim()
    .pip_install([
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
    .add_local_dir("shared", remote_path="/root/shared")
    .add_local_file("train_player_models.py", remote_path="/root/train_player_models.py")
    .add_local_file("hybrid_multi_task.py", remote_path="/root/hybrid_multi_task.py")
    .add_local_file("optimization_features.py", remote_path="/root/optimization_features.py")
    .add_local_file("phase7_features.py", remote_path="/root/phase7_features.py")
    .add_local_file("rolling_features.py", remote_path="/root/rolling_features.py")
)

@app.function(
    image=image,
    gpu=None,  # CPU only (like your original session)
    memory=32768,  # 32GB RAM
    timeout=86400,  # 24 hours (maximum practical limit)
    volumes={"/data": data_volume, "/models": model_volume}
)
def resume_hybrid_training():
    """Resume training with your original hybrid TabNet + LightGBM architecture"""
    import sys
    sys.path.insert(0, "/root")
    
    # Suppress warnings for clean output
    import warnings
    warnings.filterwarnings('ignore')
    import os
    os.environ['PYTHONWARNINGS'] = 'ignore'
    
    # Import using the proven approach
    from shared.data_loading import load_player_data, get_year_column
    from train_player_models import create_window_training_data, train_player_window
    from hybrid_multi_task import HybridMultiTaskPlayer
    import pickle
    import pandas as pd
    
    print("="*80)
    print("RESUMING NBA HYBRID TRAINING (TabNet + LightGBM)")
    print("="*80)
    print("✅ Using your original hybrid architecture")
    print("✅ TabNet encoder + LightGBM heads")
    print("✅ Multi-task for correlated props")
    print("✅ Single-task for independent props")
    
    # Check existing models
    model_dir = "/models"
    existing_models = []
    
    for file in os.listdir(model_dir):
        if file.startswith("player_models_") and file.endswith(".pkl"):
            # Check if it's a real model (not mock)
            file_path = os.path.join(model_dir, file)
            size_mb = os.path.getsize(file_path) / (1024*1024)
            
            if size_mb > 5:  # Real models are 15-20 MB
                parts = file.replace("player_models_", "").replace(".pkl", "").split("_")
                if len(parts) == 2:
                    start, end = int(parts[0]), int(parts[1])
                    existing_models.append((start, end, file, size_mb))
    
    existing_models.sort()
    
    print(f"\n✅ Found {len(existing_models)} real trained models:")
    for start, end, filename, size in existing_models:
        print(f"   {start}-{end}: {filename} ({size:.1f} MB)")
    
    # Get missing windows
    all_windows = []
    year = 1947
    
    while year <= 2025 - 2:
        end_year = year + 2
        if not (2022 <= year <= 2024) and year != 2026:
            all_windows.append((year, end_year))
        year += 3
    
    completed = {(w[0], w[1]) for w in existing_models}
    missing = [w for w in all_windows if w not in completed]
    
    print(f"\n📋 Need to complete {len(missing)} windows:")
    for start, end in missing:
        print(f"   ❌ {start}-{end}")
    
    if not missing:
        print("🎉 All windows already completed!")
        return {"status": "complete", "windows": len(existing_models)}
    
    # Load data
    print(f"\n📊 Loading training data...")
    df = pd.read_parquet("/data/aggregated_nba_data.parquet")
    print(f"   Loaded {len(df):,} games")
    
    # Train missing windows with original hybrid architecture
    print(f"\n🏀 Training {len(missing)} windows with hybrid architecture...")
    
    completed_count = 0
    failed_count = 0
    
    for i, (start_year, end_year) in enumerate(missing, 1):
        print(f"\n[{i}/{len(missing)}] Training window {start_year}-{end_year}...")
        
        # Create window data
        year_col = get_year_column(df)
        window_seasons = list(range(start_year, end_year + 1))
        
        try:
            # Use original training function
            window_df = create_window_training_data(df, window_seasons, year_col, verbose=False)
            
            print(f"   📊 Data: {len(window_df):,} rows")
            
            # Train with optimized parameters - 6 epochs, patience 3, no timeout limit
            result = train_player_window(
                window_df, 
                start_year, 
                end_year,
                neural_epochs=6,   # Optimized: 6 epochs with patience 3
                verbose=True,
                use_multi_task=True,  # Hybrid architecture
                use_gpu=False  # CPU like your original session
            )
            
            # Save model
            model_path = f"/models/player_models_{start_year}_{end_year}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(result, f)
            
            size_mb = os.path.getsize(model_path) / (1024*1024)
            print(f"   ✅ Saved: {model_path} ({size_mb:.1f} MB)")
            
            completed_count += 1
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            failed_count += 1
            continue
    
    # Final summary
    print(f"\n" + "="*80)
    print(f"🏁 HYBRID TRAINING COMPLETE!")
    print(f"="*80)
    print(f"✅ Completed windows: {completed_count}")
    print(f"❌ Failed windows: {failed_count}")
    print(f"📊 Total models: {len(existing_models) + completed_count}/25")
    print(f"🧠 Architecture: TabNet + LightGBM Hybrid")
    
    total_models = len(existing_models) + completed_count
    completion_pct = total_models / 25 * 100
    
    if total_models == 25:
        print(f"\n🎉 ALL 25 WINDOWS COMPLETED!")
        print(f"   Ready for meta-learner training with full hybrid ensemble!")
        next_step = "python train_meta_learner_v2.py"
    else:
        print(f"\n⚠ Still missing {25 - total_models} windows")
        next_step = "Debug training issues"
    
    print(f"\n🚀 Next step: {next_step}")
    
    return {
        "status": "completed" if total_models == 25 else "partial",
        "completed_windows": total_models,
        "total_windows": 25,
        "completion_pct": completion_pct,
        "architecture": "TabNet + LightGBM Hybrid",
        "next_step": next_step
    }

@app.local_entrypoint()
def main():
    """Local entry point"""
    print("🚀 Starting NBA Hybrid Resume Training...")
    print("   Restoring your original TabNet + LightGBM architecture")
    print("   Using proven approach from your 20-hour session")
    print("   Estimated time: 2-3 hours")
    print()
    
    result = resume_hybrid_training.remote()
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Status: {result['status']}")
    print(f"Models: {result['completed_windows']}/{result['total_windows']}")
    print(f"Completion: {result['completion_pct']:.1f}%")
    print(f"Architecture: {result['architecture']}")
    print(f"Next step: {result['next_step']}")
    
    if result['status'] == 'completed':
        print("\n🎉 SUCCESS! Your hybrid TabNet + LightGBM ensemble is complete!")
        print("   Your 20+ hour training investment is fully restored!")
    else:
        print("\n⚠ Partial completion - check errors above")

if __name__ == "__main__":
    main()
