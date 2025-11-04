"""
Recalibrate Models - FIX STEP 1 (URGENT)

This script recalibrates all models using IsotonicRegression based on
actual results from bets_ledger.pkl

What it fixes:
- Model says 90%, actual is 45% → After: 90% will be closer to actual 90%
- Model says 70%, actual is 42% → After: 70% will be closer to actual 70%

How it works:
1. Load settled predictions from ledger
2. For each stat type, train IsotonicRegression
   - Input: model's predicted_prob
   - Output: actual win rate
3. Save calibration curves
4. Create calibrated prediction wrapper

Run this ASAP to fix overconfidence issue!
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

print("=" * 70)
print("MODEL RECALIBRATION - Fix Overconfidence")
print("=" * 70)

# Load ledger
ledger_file = Path("bets_ledger.pkl")
if not ledger_file.exists():
    print("ERROR: bets_ledger.pkl not found!")
    print("Run fetch_bet_results.py first to get settled predictions")
    exit(1)

with open(ledger_file, 'rb') as f:
    ledger_data = pickle.load(f)

# Handle ledger format
if isinstance(ledger_data, dict) and 'bets' in ledger_data:
    bets = ledger_data['bets']
else:
    bets = ledger_data if isinstance(ledger_data, list) else [ledger_data]

df = pd.DataFrame(bets)

# Filter to settled bets only
settled = df[df['settled'] == True].copy()

print(f"\nTotal predictions: {len(df):,}")
print(f"Settled predictions: {len(settled):,}")

if len(settled) < 50:
    print("\n⚠️  WARNING: Need at least 50 settled bets for reliable calibration")
    print(f"   You have {len(settled)}. Run fetch_bet_results.py to get more.")
    print("   Continuing anyway, but results may be unreliable...")

# Calibration curves for each stat type
calibrators = {}

print("\n" + "=" * 70)
print("TRAINING CALIBRATION CURVES")
print("=" * 70)

for stat_type in ['points', 'assists', 'rebounds', 'threes']:
    stat_data = settled[settled['prop_type'] == stat_type].copy()
    
    if len(stat_data) < 10:
        print(f"\n{stat_type.upper()}: Skipped (only {len(stat_data)} samples)")
        continue
    
    print(f"\n{stat_type.upper()}: {len(stat_data)} samples")
    
    # Extract features
    X = stat_data['predicted_prob'].values.reshape(-1, 1)
    y = stat_data['won'].astype(int).values
    
    # Split for validation
    if len(stat_data) >= 30:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    else:
        # Too small for split, use all data
        X_train, X_test = X, X
        y_train, y_test = y, y
    
    # Train IsotonicRegression
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(X_train.ravel(), y_train)
    
    # Evaluate
    train_pred = calibrator.predict(X_train.ravel())
    test_pred = calibrator.predict(X_test.ravel())
    
    # Calculate calibration improvement
    print(f"  Before calibration:")
    print(f"    Train accuracy: {y_train.mean()*100:.1f}%")
    print(f"    Avg predicted: {X_train.mean()*100:.1f}%")
    print(f"    Gap: {(X_train.mean() - y_train.mean())*100:+.1f}%")
    
    print(f"  After calibration:")
    print(f"    Train predicted: {train_pred.mean()*100:.1f}%")
    print(f"    Train actual: {y_train.mean()*100:.1f}%")
    print(f"    Gap: {(train_pred.mean() - y_train.mean())*100:+.1f}%")
    
    if len(X_test) >= 5:
        print(f"  Test set:")
        print(f"    Original gap: {(X_test.mean() - y_test.mean())*100:+.1f}%")
        print(f"    Calibrated gap: {(test_pred.mean() - y_test.mean())*100:+.1f}%")
    
    # Save calibrator
    calibrators[stat_type] = calibrator
    
    # Show calibration curve
    print(f"  Calibration curve samples:")
    test_points = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    print(f"    Original → Calibrated")
    for p in test_points:
        calibrated = calibrator.predict([p])[0]
        print(f"    {p*100:5.1f}%   → {calibrated*100:5.1f}%")

# Save calibrators
if len(calibrators) > 0:
    output_file = Path("model_cache/calibration_curves.pkl")
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'wb') as f:
        pickle.dump(calibrators, f)
    
    print("\n" + "=" * 70)
    print(f"✅ Calibration curves saved: {output_file}")
    print("=" * 70)
    
    print(f"\nCalibrated stats: {list(calibrators.keys())}")
    print(f"\nTo use calibrated predictions:")
    print(f"  1. Load calibrators from {output_file}")
    print(f"  2. For each prediction, run:")
    print(f"     calibrated_prob = calibrator[stat_type].predict([predicted_prob])[0]")
    print(f"  3. Use calibrated_prob for Kelly sizing")
    
    print(f"\n📊 VISUALIZATION")
    print(f"   Creating calibration plots...")
    
    # Create visualization
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Model Calibration Curves', fontsize=16)
        
        stat_types = ['points', 'assists', 'rebounds', 'threes']
        for idx, stat_type in enumerate(stat_types):
            ax = axes[idx // 2, idx % 2]
            
            if stat_type not in calibrators:
                ax.text(0.5, 0.5, f'Not enough data\nfor {stat_type}',
                       ha='center', va='center', fontsize=12)
                ax.set_title(stat_type.upper())
                continue
            
            # Get data
            stat_data = settled[settled['prop_type'] == stat_type]
            X = stat_data['predicted_prob'].values
            y = stat_data['won'].astype(int).values
            
            # Plot perfect calibration
            ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', alpha=0.3)
            
            # Plot original predictions
            bins = np.linspace(0.5, 1.0, 11)
            bin_centers = []
            bin_actual = []
            
            for i in range(len(bins)-1):
                mask = (X >= bins[i]) & (X < bins[i+1])
                if mask.sum() > 0:
                    bin_centers.append((bins[i] + bins[i+1]) / 2)
                    bin_actual.append(y[mask].mean())
            
            if len(bin_centers) > 0:
                ax.scatter(bin_centers, bin_actual, s=100, alpha=0.6, 
                          label='Original (uncalibrated)', color='red')
            
            # Plot calibrated curve
            X_curve = np.linspace(0.5, 1.0, 100)
            y_curve = calibrators[stat_type].predict(X_curve)
            ax.plot(X_curve, y_curve, 'b-', linewidth=2, 
                   label='Calibrated curve')
            
            ax.set_xlabel('Predicted Probability')
            ax.set_ylabel('Actual Win Rate')
            ax.set_title(f'{stat_type.upper()} Calibration')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0.45, 1.05)
            ax.set_ylim(0, 1.05)
        
        plt.tight_layout()
        plot_file = Path("calibration_curves.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"   ✅ Saved plot: {plot_file}")
        
    except Exception as e:
        print(f"   ⚠️  Could not create plot: {e}")
    
else:
    print("\n❌ No calibrators created - not enough data!")
    print("   Need at least 10 settled bets per stat type")

print("\n" + "=" * 70)
print("NEXT STEPS")
print("=" * 70)
print("""
1. ✅ Calibration curves created

2. UPDATE riq_analyzer.py to use calibration:
   
   # At top of file:
   with open('model_cache/calibration_curves.pkl', 'rb') as f:
       CALIBRATORS = pickle.load(f)
   
   # In analyze_player_prop(), after getting win_prob:
   if prop_type in CALIBRATORS:
       win_prob = CALIBRATORS[prop_type].predict([win_prob])[0]
       # Now use calibrated win_prob for Kelly sizing

3. TEST on new predictions:
   python riq_analyzer.py
   
   Compare calibrated vs uncalibrated probabilities

4. MONITOR performance:
   python fetch_bet_results.py  # Daily
   python analyze_ledger.py     # Weekly
   
   Check if calibration improved accuracy

5. RE-CALIBRATE monthly as more data comes in
""")

print("=" * 70)
print("RECALIBRATION COMPLETE")
print("=" * 70)
