"""
Backtest Phase 1+2+3 Features on Recent Data
Validates improvements on player stat predictions
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

print("="*80)
print("BACKTEST: Phase 1+2+3 Features Validation")
print("="*80)

# Load the enhanced model (2002-2006 window with Phase 1+2+3)
model_path = Path("model_cache/player_models_2002_2006.pkl")

if not model_path.exists():
    print(f"\n[ERROR] Model not found: {model_path}")
    print("Run training first: python train_auto.py --verbose")
    exit(1)

print(f"\n[OK] Loading enhanced model: {model_path}")
with open(model_path, 'rb') as f:
    models = pickle.load(f)

print("\n" + "="*80)
print("MODEL ANALYSIS")
print("="*80)

# Analyze each stat type
for stat_name in ['points', 'rebounds', 'assists', 'threes']:
    if stat_name not in models:
        print(f"\n[SKIP] {stat_name}: not in models")
        continue

    model = models[stat_name]

    # Check if it's a dict or direct model
    if isinstance(model, dict):
        if 'model' in model:
            lgb_model = model['model']
        else:
            print(f"\n[SKIP] {stat_name}: unexpected dict structure")
            continue
    else:
        lgb_model = model

    # Get features
    if hasattr(lgb_model, 'feature_name_'):
        features = lgb_model.feature_name_
    elif hasattr(lgb_model, 'feature_names_in_'):
        features = lgb_model.feature_names_in_
    else:
        features = []

    print(f"\n{stat_name.upper()} MODEL:")
    print(f"  Total features: {len(features)}")

    # Check for Phase 1 features
    phase1_keywords = ['ts_pct', 'fieldGoalsAttempted', 'threePointersAttempted',
                       'freeThrowsAttempted', 'rate_fga', 'rate_3pa', 'rate_fta',
                       'three_pct', 'ft_pct']
    phase1_features = [f for f in features if any(kw in f for kw in phase1_keywords)]

    # Check for Phase 2 features
    phase2_keywords = ['matchup_pace', 'pace_factor', 'def_matchup_difficulty',
                       'offensive_environment']
    phase2_features = [f for f in features if any(kw in f for kw in phase2_keywords)]

    # Check for Phase 3 features
    phase3_keywords = ['usage_rate', 'rebound_rate', 'assist_rate']
    phase3_features = [f for f in features if any(kw in f for kw in phase3_keywords)]

    print(f"  Phase 1 features: {len(phase1_features)}")
    if phase1_features:
        print(f"    Examples: {phase1_features[:3]}")

    print(f"  Phase 2 features: {len(phase2_features)}")
    if phase2_features:
        print(f"    Examples: {phase2_features[:2]}")

    print(f"  Phase 3 features: {len(phase3_features)}")
    if phase3_features:
        print(f"    Examples: {phase3_features[:2]}")

    # Feature importance (if available)
    if hasattr(lgb_model, 'feature_importances_'):
        importances = lgb_model.feature_importances_

        # Create importance dataframe
        feat_df = pd.DataFrame({
            'feature': features,
            'importance': importances
        }).sort_values('importance', ascending=False)

        print(f"\n  Top 10 Most Important Features:")
        for i, row in feat_df.head(10).iterrows():
            is_phase1 = any(kw in row['feature'] for kw in phase1_keywords)
            is_phase2 = any(kw in row['feature'] for kw in phase2_keywords)
            is_phase3 = any(kw in row['feature'] for kw in phase3_keywords)

            phase_marker = ""
            if is_phase1:
                phase_marker = " [PHASE 1]"
            elif is_phase2:
                phase_marker = " [PHASE 2]"
            elif is_phase3:
                phase_marker = " [PHASE 3]"

            print(f"    {row['feature']:40s}: {row['importance']:6.0f}{phase_marker}")

        # Count phase features in top 20
        phase1_in_top20 = sum(1 for i, row in feat_df.head(20).iterrows()
                             if any(kw in row['feature'] for kw in phase1_keywords))
        phase2_in_top20 = sum(1 for i, row in feat_df.head(20).iterrows()
                             if any(kw in row['feature'] for kw in phase2_keywords))
        phase3_in_top20 = sum(1 for i, row in feat_df.head(20).iterrows()
                             if any(kw in row['feature'] for kw in phase3_keywords))

        print(f"\n  Phase features in top 20:")
        print(f"    Phase 1: {phase1_in_top20}/20")
        print(f"    Phase 2: {phase2_in_top20}/20")
        print(f"    Phase 3: {phase3_in_top20}/20")

print("\n" + "="*80)
print("BACKTEST SUMMARY")
print("="*80)

# Calculate total phase features
total_phase1 = len([f for f in features if any(kw in f for kw in phase1_keywords)])
total_phase2 = len([f for f in features if any(kw in f for kw in phase2_keywords)])
total_phase3 = len([f for f in features if any(kw in f for kw in phase3_keywords)])

print(f"\nTotal Phase Features Across All Models:")
print(f"  Phase 1 (Volume + Efficiency): {total_phase1}")
print(f"  Phase 2 (Matchup + Context):   {total_phase2}")
print(f"  Phase 3 (Advanced Rates):      {total_phase3}")
print(f"  TOTAL NEW FEATURES:            {total_phase1 + total_phase2 + total_phase3}")

if total_phase1 >= 15:
    print("\n[OK] Phase 1 implementation: SUCCESS")
else:
    print("\n[WARNING] Phase 1 implementation: INCOMPLETE")

if total_phase2 >= 3:
    print("[OK] Phase 2 implementation: SUCCESS")
else:
    print("[WARNING] Phase 2 implementation: INCOMPLETE")

if total_phase3 >= 2:
    print("[OK] Phase 3 implementation: SUCCESS")
else:
    print("[WARNING] Phase 3 implementation: INCOMPLETE")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)

print("\n1. If all phases show SUCCESS:")
print("   - Retrain remaining 4 windows (2007-2026)")
print("   - Command: python train_auto.py --verbose")

print("\n2. If any phase shows INCOMPLETE:")
print("   - Check train_auto.py implementation")
print("   - Delete cache and retrain")

print("\n3. After all windows trained:")
print("   - Retrain enhanced selector")
print("   - Command: python train_ensemble_players.py --verbose")

print("\n" + "="*80)
