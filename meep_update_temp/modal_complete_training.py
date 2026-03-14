#!/usr/bin/env python
"""
Complete Modal Training - All Functions Inline

Eliminates import issues by copying all training functions directly into the script.
This ensures real training instead of mock models.

Usage:
    modal run modal_complete_training.py
"""

import modal
import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import gc
import time
from typing import Dict, List, Tuple, Optional

# Modal setup
app = modal.App("nba-complete-training")
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

# ============================================================================
# COPIED TRAINING FUNCTIONS (Inline to avoid import issues)
# ============================================================================

def generate_features_for_prediction(df_games):
    """Generate features for training - copied from original"""
    # Sort by player and date
    df = df_games.sort_values(['personId', 'gameDate']).copy()
    features = pd.DataFrame(index=df.index)
    
    # Direct stats
    direct_stats = ['points', 'assists', 'reboundsTotal', 'threePointersMade', 'numMinutes']
    for col in direct_stats:
        if col in df.columns:
            features[col] = df[col].fillna(0)
    
    # Rolling averages
    for window in [3, 5, 7, 10]:
        for stat in direct_stats:
            if stat in df.columns:
                feature_name = f'{stat}_L{window}_avg'
                features[feature_name] = df.groupby('personId')[stat].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                ).fillna(0)
    
    # Shooting percentages
    if 'fieldGoalsMade' in df.columns and 'fieldGoalsAttempted' in df.columns:
        features['fg_pct'] = (df['fieldGoalsMade'] / df['fieldGoalsAttempted']).fillna(0)
    if 'threePointersMade' in df.columns and 'threePointersAttempted' in df.columns:
        features['three_pct'] = (df['threePointersMade'] / df['threePointersAttempted']).fillna(0)
    if 'freeThrowsMade' in df.columns and 'freeThrowsAttempted' in df.columns:
        features['ft_pct'] = (df['freeThrowsMade'] / df['freeThrowsAttempted']).fillna(0)
    
    # Per-minute stats
    if 'numMinutes' in df.columns:
        for stat in ['points', 'assists', 'reboundsTotal', 'threePointersMade']:
            if stat in df.columns and stat != 'numMinutes':
                features[f'{stat}_per_min'] = features[stat] / (features['numMinutes'] + 1e-6)
    
    # Game context
    if 'home' in df.columns:
        features['is_home'] = df['home'].astype(int)
    
    # Days rest
    features['days_rest'] = df.groupby('personId')['gameDate'].transform(
        lambda x: x.diff().dt.days.fillna(3)
    ).fillna(3)
    
    # Player ID
    features['personId'] = df['personId']
    
    return features.fillna(0)

def train_player_window_complete(
    window_df: pd.DataFrame,
    start_year: int,
    end_year: int,
    neural_epochs: int = 12,
    verbose: bool = True,
    use_gpu: bool = False
) -> Dict:
    """
    Complete training function - all logic inline to avoid imports
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"TRAINING PLAYER MODELS: {start_year}-{end_year}")
        print(f"MODE: Complete Inline Training")
        print(f"{'='*70}")
        print(f"Training data: {len(window_df):,} rows")
    
    # Generate features
    if verbose:
        print("Generating features...")
    
    features_df = generate_features_for_prediction(window_df)
    
    # Prepare training data
    feature_cols = [c for c in features_df.columns if c not in [
        'personId', 'gameId', 'gameDate', 'firstName', 'lastName'
    ]]
    
    X = features_df[feature_cols].fillna(0)
    
    # Prepare targets
    targets = {
        'points': window_df['points'].fillna(0).values,
        'assists': window_df['assists'].fillna(0).values,
        'rebounds': window_df['reboundsTotal'].fillna(0).values,
        'threes': window_df['threePointersMade'].fillna(0).values,
        'minutes': window_df['numMinutes'].fillna(0).values
    }
    
    # Train/val split
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx]
    X_val = X.iloc[split_idx:]
    
    y_train_dict = {k: v[:split_idx] for k, v in targets.items()}
    y_val_dict = {k: v[split_idx:] for k, v in targets.items()}
    
    if verbose:
        print(f"Train: {len(X_train):,} samples")
        print(f"Val: {len(X_val):,} samples")
        print(f"Features: {len(feature_cols)}")
    
    # Train models for each prop
    trained_models = {}
    prop_metrics = {}
    
    props = ['points', 'assists', 'rebounds', 'threes', 'minutes']
    
    for prop in props:
        if verbose:
            print(f"\n🏀 Training {prop} model...")
        
        try:
            # Use LightGBM for reliability (no GPU issues)
            from lightgbm import LGBMRegressor
            
            model = LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            
            # Train model
            model.fit(X_train, y_train_dict[prop])
            
            # Evaluate
            y_pred = model.predict(X_val)
            mae = np.mean(np.abs(y_pred - y_val_dict[prop]))
            
            trained_models[prop] = model
            prop_metrics[prop] = {'mae': float(mae)}
            
            if verbose:
                print(f"   ✅ {prop}: MAE = {mae:.3f}")
                
        except Exception as e:
            if verbose:
                print(f"   ❌ {prop} failed: {e}")
            trained_models[prop] = None
            prop_metrics[prop] = {'error': str(e)}
    
    # Create result structure
    result = {
        'player_models': trained_models,
        'training_metadata': {
            'window': f"{start_year}-{end_year}",
            'samples': len(window_df),
            'features': len(feature_cols),
            'training_date': '2025-11-20',
            'method': 'inline_lightgbm',
            'prop_metrics': prop_metrics
        }
    }
    
    if verbose:
        print(f"\n✅ Training complete for {start_year}-{end_year}")
        successful_props = sum(1 for m in trained_models.values() if m is not None)
        print(f"   Successfully trained: {successful_props}/{len(props)} props")
    
    return result

# ============================================================================
# MODAL FUNCTIONS
# ============================================================================

def get_existing_windows():
    """Get list of already completed windows"""
    existing = []
    model_dir = "/models"
    
    for file in os.listdir(model_dir):
        if file.startswith("player_models_") and file.endswith(".pkl"):
            # Extract years from filename
            parts = file.replace("player_models_", "").replace(".pkl", "").split("_")
            if len(parts) == 2:
                start, end = int(parts[0]), int(parts[1])
                existing.append((start, end))
    
    return sorted(existing)

def get_missing_windows():
    """Get windows that still need training"""
    # All windows from 1947-2025 (excluding backtesting windows)
    all_windows = []
    year = 1947
    
    while year <= 2025 - 2:
        end_year = year + 2
        
        # Skip excluded windows for proper backtesting
        if not (2022 <= year <= 2024) and year != 2026:
            all_windows.append((year, end_year))
        
        year += 3
    
    existing = get_existing_windows()
    missing = [w for w in all_windows if w not in existing]
    return missing

@app.function(
    image=image,
    volumes={"/data": nba_data, "/models": nba_models},
    timeout=7200,  # 2 hours
    retries=2
)
def complete_training():
    """Complete training with all functions inline - no imports needed"""
    print("="*80)
    print("COMPLETE NBA TRAINING - All Functions Inline")
    print("="*80)
    
    # Check existing progress
    existing = get_existing_windows()
    print(f"✅ Found {len(existing)} completed windows:")
    
    # Show last 5 completed
    for start, end in existing[-5:]:
        print(f"   player_models_{start}_{end}.pkl")
    
    # Get missing windows
    missing = get_missing_windows()
    print(f"\n📋 Need to complete {len(missing)} windows:")
    for start, end in missing:
        print(f"   ❌ {start}-{end}")
    
    if not missing:
        print("🎉 All windows already completed!")
        return {"status": "complete", "windows": len(existing)}
    
    # Load training data
    print(f"\n📊 Loading training data...")
    df = pd.read_parquet("/data/aggregated_nba_data.parquet")
    print(f"   Loaded {len(df):,} games")
    
    # Train missing windows with complete inline function
    print(f"\n🏀 Training {len(missing)} remaining windows...")
    
    completed = 0
    failed = 0
    
    for i, (start_year, end_year) in enumerate(missing, 1):
        print(f"\n[{i}/{len(missing)}] Training window {start_year}-{end_year}...")
        
        # Filter data for this window
        start_date = f"{start_year}-10-01"
        end_date = f"{end_year}-06-30"
        
        window_df = df[
            (df['gameDate'] >= start_date) & 
            (df['gameDate'] <= end_date)
        ].copy()
        
        if len(window_df) == 0:
            print(f"   ⚠ No data for {start_year}-{end_year}, skipping")
            continue
        
        print(f"   📊 Data: {len(window_df):,} games")
        
        try:
            # Use complete inline training function
            model = train_player_window_complete(
                window_df, 
                start_year, 
                end_year,
                neural_epochs=12,
                verbose=True,
                use_gpu=False
            )
            
            # Save model
            model_path = f"/models/player_models_{start_year}_{end_year}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Get file size
            size_mb = os.path.getsize(model_path) / (1024*1024)
            print(f"   ✅ Saved: {model_path} ({size_mb:.1f} MB)")
            
            completed += 1
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            failed += 1
            continue
    
    # Final verification
    final_models = get_existing_windows()
    print(f"\n" + "="*80)
    print(f"🏁 TRAINING COMPLETE!")
    print(f"="*80)
    print(f"✅ Completed windows: {completed}")
    print(f"❌ Failed windows: {failed}")
    print(f"📊 Total models: {len(final_models)}/25")
    print(f"📈 Completion: {len(final_models)/25*100:.1f}%")
    
    if len(final_models) == 25:
        print(f"\n🎉 ALL 25 WINDOWS COMPLETED!")
        print(f"   Ready for meta-learner training with full ensemble!")
        next_step = "python train_meta_learner_v2.py"
    else:
        print(f"\n⚠ Still missing {25 - len(final_models)} windows")
        next_step = "Debug training issues"
    
    print(f"\n🚀 Next step: {next_step}")
    
    return {
        "status": "completed" if len(final_models) == 25 else "partial",
        "completed_windows": len(final_models),
        "total_windows": 25,
        "completion_pct": len(final_models) / 25 * 100,
        "next_step": next_step
    }

@app.local_entrypoint()
def main():
    """Local entry point"""
    print("🚀 Starting Complete NBA Training on Modal...")
    print("   All functions inline - no import issues!")
    print("   Will create REAL trained models (not mock ones)")
    print("   Estimated time: 2-3 hours")
    print()
    
    result = complete_training.remote()
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Status: {result['status']}")
    print(f"Models: {result['completed_windows']}/{result['total_windows']}")
    print(f"Completion: {result['completion_pct']:.1f}%")
    print(f"Next step: {result['next_step']}")
    
    if result['status'] == 'completed':
        print("\n🎉 SUCCESS! All 25 real models ready!")
        print("   Your 20+ hour training investment is complete!")
    else:
        print("\n⚠ Partial completion - check errors above")

if __name__ == "__main__":
    main()
