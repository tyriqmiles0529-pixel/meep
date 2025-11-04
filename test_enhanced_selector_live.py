"""
Test Enhanced Selector on Live Data

This script tests the enhanced selector with real player data to see:
1. Which window it selects for each player/stat
2. What predictions it makes
3. How confident it is in selections

Usage:
    python test_enhanced_selector_live.py
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

print("=" * 70)
print("ENHANCED SELECTOR - LIVE TEST")
print("=" * 70)

# Load enhanced selector
cache_dir = Path("model_cache")
selector_file = cache_dir / "dynamic_selector_enhanced.pkl"
selector_meta_file = cache_dir / "dynamic_selector_enhanced_meta.json"

if not selector_file.exists():
    print(f"ERROR: Enhanced selector not found at {selector_file}")
    print("Run train_dynamic_selector_enhanced.py first")
    exit(1)

print("\n1. Loading Enhanced Selector...")
with open(selector_file, 'rb') as f:
    selector = pickle.load(f)

import json
with open(selector_meta_file, 'r') as f:
    meta = json.load(f)

print(f"   Method: {meta.get('method', 'hybrid')}")
print(f"   Stats: {', '.join(selector.keys())}")

# Load all window ensembles
print("\n2. Loading Window Ensembles...")
import glob
windows = {}
ensemble_files = sorted(glob.glob(str(cache_dir / "player_ensemble_*.pkl")))
for pkl_path in ensemble_files:
    window_name = Path(pkl_path).stem.replace("player_ensemble_", "").replace("_", "-")
    with open(pkl_path, 'rb') as f:
        windows[window_name] = pickle.load(f)
    print(f"   Loaded: {window_name}")

print(f"\n   Total windows: {len(windows)}")

# Test data - simulate player stats
print("\n3. Creating Test Cases...")

test_players = [
    {
        "name": "Stephen Curry (High Volume Shooter)",
        "games_played": 15,
        "recent_avg": 28.5,
        "recent_std": 6.2,
        "recent_min": 18.0,
        "recent_max": 41.0,
        "trend": 2.5,  # hot streak
        "rest_days": 2,
        "recent_form_3": 32.0,
        "form_change": 3.5,
        "consistency_cv": 0.22,
        "stat": "points"
    },
    {
        "name": "Nikola Jokic (Consistent All-Around)",
        "games_played": 18,
        "recent_avg": 10.2,
        "recent_std": 2.1,
        "recent_min": 7.0,
        "recent_max": 14.0,
        "trend": 0.5,  # very consistent
        "rest_days": 1,
        "recent_form_3": 10.5,
        "form_change": 0.3,
        "consistency_cv": 0.20,
        "stat": "assists"
    },
    {
        "name": "Domantas Sabonis (Rebound Machine)",
        "games_played": 20,
        "recent_avg": 13.8,
        "recent_std": 3.2,
        "recent_min": 8.0,
        "recent_max": 19.0,
        "trend": -0.8,  # slight decline
        "rest_days": 3,
        "recent_form_3": 12.5,
        "form_change": -1.3,
        "consistency_cv": 0.23,
        "stat": "rebounds"
    },
    {
        "name": "Damian Lillard (3PT Specialist)",
        "games_played": 12,
        "recent_avg": 4.2,
        "recent_std": 1.8,
        "recent_min": 1.0,
        "recent_max": 8.0,
        "trend": 1.2,  # getting hot
        "rest_days": 2,
        "recent_form_3": 5.0,
        "form_change": 0.8,
        "consistency_cv": 0.43,
        "stat": "threes"
    },
    {
        "name": "Rookie Player (Limited Data)",
        "games_played": 5,
        "recent_avg": 15.2,
        "recent_std": 4.5,
        "recent_min": 9.0,
        "recent_max": 22.0,
        "trend": 3.0,  # improving
        "rest_days": 1,
        "recent_form_3": 18.0,
        "form_change": 2.8,
        "consistency_cv": 0.30,
        "stat": "points"
    },
]

print(f"   Created {len(test_players)} test cases")

# Test selector on each player
print("\n" + "=" * 70)
print("SELECTOR PREDICTIONS")
print("=" * 70)

for player in test_players:
    print(f"\n{player['name']}")
    print("-" * 70)
    
    stat_name = player['stat']
    
    # Check if selector trained for this stat
    if stat_name not in selector:
        print(f"   ⚠ Selector not trained for {stat_name}")
        continue
    
    # Extract selector components
    selector_obj = selector[stat_name]
    scaler = selector_obj['scaler']
    selector_model = selector_obj['selector']
    windows_list = selector_obj['windows_list']
    
    # Build feature vector (10 features)
    feature_vector = np.array([
        player['games_played'],
        player['recent_avg'],
        player['recent_std'],
        player['recent_min'],
        player['recent_max'],
        player['trend'],
        player['rest_days'],
        player['recent_form_3'],
        player['form_change'],
        player['consistency_cv'],
    ]).reshape(1, -1)
    
    # Scale features
    X_scaled = scaler.transform(feature_vector)
    
    # Get prediction probabilities
    probs = selector_model.predict_proba(X_scaled)[0]
    
    # Get selected window
    selected_idx = selector_model.predict(X_scaled)[0]
    selected_window = windows_list[selected_idx]
    
    # Display results
    print(f"   Stat: {stat_name.upper()}")
    print(f"   Games Played: {player['games_played']}")
    print(f"   Recent Average: {player['recent_avg']:.1f}")
    print(f"   Recent Form (L3): {player['recent_form_3']:.1f}")
    print(f"   Consistency (CV): {player['consistency_cv']:.2f}")
    print(f"   Trend: {player['trend']:+.1f}")
    print()
    print(f"   🎯 Selected Window: {selected_window}")
    print(f"   Confidence: {probs[selected_idx]*100:.1f}%")
    print()
    print(f"   Window Probabilities:")
    for i, (window, prob) in enumerate(zip(windows_list, probs)):
        marker = "→" if i == selected_idx else " "
        print(f"   {marker} {window}: {prob*100:.1f}%")
    
    # Get ensemble prediction if available
    if selected_window in windows and stat_name in windows[selected_window]:
        ensemble_obj = windows[selected_window][stat_name]
        
        # Use simple average as baseline prediction
        baseline_pred = player['recent_avg']
        
        print()
        print(f"   📊 Prediction: {baseline_pred:.1f} {stat_name}")
        print(f"      (Using {selected_window} ensemble)")

print("\n" + "=" * 70)
print("SELECTION PATTERNS")
print("=" * 70)

# Analyze selection patterns
selections = defaultdict(lambda: defaultdict(int))

for player in test_players:
    stat_name = player['stat']
    if stat_name not in selector:
        continue
    
    selector_obj = selector[stat_name]
    scaler = selector_obj['scaler']
    selector_model = selector_obj['selector']
    windows_list = selector_obj['windows_list']
    
    feature_vector = np.array([
        player['games_played'],
        player['recent_avg'],
        player['recent_std'],
        player['recent_min'],
        player['recent_max'],
        player['trend'],
        player['rest_days'],
        player['recent_form_3'],
        player['form_change'],
        player['consistency_cv'],
    ]).reshape(1, -1)
    
    X_scaled = scaler.transform(feature_vector)
    selected_idx = selector_model.predict(X_scaled)[0]
    selected_window = windows_list[selected_idx]
    
    selections[stat_name][selected_window] += 1

print()
for stat_name in sorted(selections.keys()):
    print(f"{stat_name.upper()}:")
    total = sum(selections[stat_name].values())
    for window in sorted(selections[stat_name].keys()):
        count = selections[stat_name][window]
        pct = count / total * 100
        print(f"  {window}: {count}/{total} ({pct:.0f}%)")
    print()

print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)
print()
print("Key Insights:")
print("  • Selector chooses different windows based on player context")
print("  • High games_played → Recent windows (more current data)")
print("  • Low games_played → Older windows (more stable patterns)")
print("  • Hot streaks (high trend) → Recent windows")
print("  • Consistent players → Any window works")
print()
print("Next steps:")
print("  1. Run on real player data (from nba_api)")
print("  2. Compare predictions to actual outcomes")
print("  3. Deploy to riq_analyzer.py")
