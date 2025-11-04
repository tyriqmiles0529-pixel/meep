"""
Analyze Bets Ledger - Learn from Past Predictions

This script analyzes your bets_ledger.pkl to:
1. Identify which predictions worked and which didn't
2. Calculate model calibration
3. Find patterns in errors
4. Generate recommendations for model improvement
"""

import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path

print("=" * 70)
print("BETS LEDGER ANALYSIS - LEARNING FROM PAST PREDICTIONS")
print("=" * 70)

# Load ledger
ledger_file = Path("bets_ledger.pkl")
if not ledger_file.exists():
    print("❌ bets_ledger.pkl not found!")
    exit(1)

with open(ledger_file, 'rb') as f:
    ledger = pickle.load(f)

# Handle different ledger formats
if isinstance(ledger, dict) and 'bets' in ledger:
    bets = ledger['bets']
    print(f"\n📊 Total entries in ledger: {len(bets):,}")
elif isinstance(ledger, list):
    bets = ledger
    print(f"\n📊 Total entries in ledger: {len(bets):,}")
else:
    print(f"\n📊 Ledger type: {type(ledger)}")
    print("   Attempting to convert...")
    bets = [ledger] if not hasattr(ledger, '__iter__') else list(ledger)
    print(f"   Total entries: {len(bets):,}")

# Convert to DataFrame for analysis
df = pd.DataFrame(bets)

print(f"\n📋 Ledger Structure:")
print(f"   Columns: {list(df.columns)}")
print(f"   Date range: {df['recorded_at'].min()} to {df['recorded_at'].max()}")

# Check what data we have
print(f"\n🔍 Data Availability:")
print(f"   Total predictions: {len(df):,}")
print(f"   Settled bets: {df['settled'].sum():,}")
print(f"   Unsettled bets: {(~df['settled']).sum():,}")
print(f"   With actual results: {df['actual'].notna().sum():,}")
print(f"   With win/loss data: {df['won'].notna().sum():,}")

# ========= ANALYSIS 1: SETTLED BETS =========
if df['settled'].sum() > 0:
    print("\n" + "=" * 70)
    print("ANALYSIS 1: SETTLED BETS PERFORMANCE")
    print("=" * 70)
    
    settled = df[df['settled'] == True].copy()
    
    print(f"\n📊 Settled Bets: {len(settled):,}")
    
    if 'won' in settled.columns and settled['won'].notna().sum() > 0:
        won_bets = settled['won'].sum()
        total_settled = len(settled[settled['won'].notna()])
        accuracy = won_bets / total_settled * 100 if total_settled > 0 else 0
        
        print(f"\n🎯 Overall Performance:")
        print(f"   Won: {won_bets:,}")
        print(f"   Lost: {total_settled - won_bets:,}")
        print(f"   Accuracy: {accuracy:.1f}%")
        print(f"   Expected (breakeven): 52.4% (at -110 odds)")
        print(f"   Difference: {accuracy - 52.4:+.1f}%")
        
        # Performance by stat type
        print(f"\n📊 Performance by Stat Type:")
        for stat_type in settled['prop_type'].unique():
            stat_bets = settled[settled['prop_type'] == stat_type]
            won = stat_bets['won'].sum()
            total = len(stat_bets[stat_bets['won'].notna()])
            acc = won / total * 100 if total > 0 else 0
            
            print(f"   {stat_type.upper():10s}: {won:3d}/{total:3d} = {acc:5.1f}% ", end="")
            if acc > 55:
                print("✅ (Good edge)")
            elif acc > 52:
                print("✓ (Slight edge)")
            elif acc > 48:
                print("~ (Break-even)")
            else:
                print("❌ (Losing)")
        
        # Performance by bookmaker
        if 'bookmaker' in settled.columns:
            print(f"\n📊 Performance by Bookmaker:")
            for bookie in settled['bookmaker'].unique()[:5]:
                bookie_bets = settled[settled['bookmaker'] == bookie]
                won = bookie_bets['won'].sum()
                total = len(bookie_bets[bookie_bets['won'].notna()])
                acc = won / total * 100 if total > 0 else 0
                print(f"   {bookie:15s}: {won:3d}/{total:3d} = {acc:5.1f}%")

# ========= ANALYSIS 2: CALIBRATION =========
print("\n" + "=" * 70)
print("ANALYSIS 2: MODEL CALIBRATION")
print("=" * 70)

print("\n🎯 Calibration Check:")
print("   (Does predicted 70% actually win 70% of the time?)")

# Group predictions by probability buckets
df_with_outcome = df[df['won'].notna()].copy()

if len(df_with_outcome) > 0:
    df_with_outcome['prob_bucket'] = pd.cut(
        df_with_outcome['predicted_prob'] * 100,
        bins=[0, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
        labels=['50-55%', '55-60%', '60-65%', '65-70%', '70-75%', 
                '75-80%', '80-85%', '85-90%', '90-95%', '95-100%']
    )
    
    print(f"\n   Predicted Prob | Actual Win Rate | Count | Calibration")
    print(f"   " + "-" * 60)
    
    for bucket in df_with_outcome['prob_bucket'].cat.categories:
        bucket_data = df_with_outcome[df_with_outcome['prob_bucket'] == bucket]
        if len(bucket_data) > 0:
            actual_rate = bucket_data['won'].mean() * 100
            count = len(bucket_data)
            pred_mid = (float(bucket.split('-')[0].rstrip('%')) + 
                       float(bucket.split('-')[1].rstrip('%'))) / 2
            diff = actual_rate - pred_mid
            
            status = "✅" if abs(diff) < 5 else "⚠️" if abs(diff) < 10 else "❌"
            print(f"   {bucket:11s} | {actual_rate:6.1f}%         | {count:5d} | {diff:+5.1f}% {status}")
else:
    print("   ⚠️  No settled bets with outcomes yet")
    print("   Cannot calculate calibration until bets settle")

# ========= ANALYSIS 3: ERROR PATTERNS =========
if 'actual' in df.columns and df['actual'].notna().sum() > 0:
    print("\n" + "=" * 70)
    print("ANALYSIS 3: PREDICTION ERROR PATTERNS")
    print("=" * 70)
    
    df_with_actual = df[df['actual'].notna()].copy()
    print(f"\n📊 Predictions with actual results: {len(df_with_actual):,}")
    
    # We need to calculate the prediction value
    # From the ledger, we don't have mu_final directly, but we can infer
    # based on predicted_prob and line
    
    print("\n   (Detailed error analysis requires 'predicted_value' field)")
    print("   Current ledger has 'predicted_prob' but not prediction value")
    print("   Recommendation: Update logging to include mu_final")

# ========= ANALYSIS 4: UNSETTLED BETS =========
print("\n" + "=" * 70)
print("ANALYSIS 4: UNSETTLED BETS (OPPORTUNITIES)")
print("=" * 70)

unsettled = df[df['settled'] == False].copy()
print(f"\n📊 Unsettled predictions: {len(unsettled):,}")

if len(unsettled) > 0:
    # Convert game_date to datetime
    unsettled['game_datetime'] = pd.to_datetime(unsettled['game_date'])
    unsettled['days_ago'] = (datetime.now() - unsettled['game_datetime']).dt.days
    
    # Group by age
    print(f"\n   Age of unsettled predictions:")
    print(f"   0-1 days old: {(unsettled['days_ago'] <= 1).sum():,} (can still fetch results)")
    print(f"   2-7 days old: {((unsettled['days_ago'] > 1) & (unsettled['days_ago'] <= 7)).sum():,} (should fetch)")
    print(f"   >7 days old: {(unsettled['days_ago'] > 7).sum():,} (missed)")
    
    # Most common games in unsettled
    recent_unsettled = unsettled[unsettled['days_ago'] <= 7]
    if len(recent_unsettled) > 0:
        print(f"\n   Recent games to fetch results for:")
        games = recent_unsettled.groupby('game').size().sort_values(ascending=False).head(10)
        for game, count in games.items():
            print(f"   {game}: {count} predictions")

# ========= RECOMMENDATIONS =========
print("\n" + "=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)

recommendations = []

# Check if we have enough settled data
settled_count = df['settled'].sum()
if settled_count < 100:
    recommendations.append(
        f"⚠️  Only {settled_count} settled bets - need 100+ for reliable analysis"
    )
    recommendations.append(
        "   → Continue tracking predictions for 2-4 more weeks"
    )

# Check calibration
if len(df_with_outcome) >= 50:
    # Simple calibration check
    high_conf = df_with_outcome[df_with_outcome['predicted_prob'] >= 0.70]
    if len(high_conf) > 10:
        high_conf_acc = high_conf['won'].mean()
        if high_conf_acc < 0.65:
            recommendations.append(
                f"❌ High-confidence predictions (>70%) only hitting {high_conf_acc*100:.1f}%"
            )
            recommendations.append(
                "   → Model is overconfident - consider recalibration"
            )
            recommendations.append(
                "   → Use lower Kelly fractions (0.25x instead of 0.5x)"
            )

# Check for unsettled backlog
old_unsettled = (unsettled['days_ago'] > 2).sum() if len(unsettled) > 0 else 0
if old_unsettled > 50:
    recommendations.append(
        f"⚠️  {old_unsettled} unsettled predictions >2 days old"
    )
    recommendations.append(
        "   → Run result fetcher to update outcomes"
    )
    recommendations.append(
        "   → See: fetch_bet_results.py (to be created)")

# Check edge
if settled_count > 50 and 'won' in df.columns:
    overall_acc = df[df['won'].notna()]['won'].mean()
    if overall_acc > 0.55:
        recommendations.append(
            f"✅ Strong edge: {overall_acc*100:.1f}% accuracy (>55%)"
        )
        recommendations.append(
            "   → Consider increasing bet sizing (0.5x → 1.0x Kelly)"
        )
    elif overall_acc > 0.52:
        recommendations.append(
            f"✓ Positive edge: {overall_acc*100:.1f}% accuracy"
        )
        recommendations.append(
            "   → Continue current strategy"
        )
    else:
        recommendations.append(
            f"❌ Negative edge: {overall_acc*100:.1f}% accuracy (<52%)"
        )
        recommendations.append(
            "   → STOP betting until model is improved"
        )
        recommendations.append(
            "   → Analyze error patterns and retrain"
        )

if not recommendations:
    recommendations.append("✅ No issues detected - model performing well!")
    recommendations.append("   → Continue monitoring and tracking")

print()
for rec in recommendations:
    print(rec)

# ========= ACTIONABLE NEXT STEPS =========
print("\n" + "=" * 70)
print("NEXT STEPS")
print("=" * 70)

print("""
1. FETCH RESULTS FOR UNSETTLED BETS:
   python fetch_bet_results.py
   (Will be created - fetches actuals from nba_api)

2. ANALYZE CALIBRATION:
   python analyze_calibration.py
   (Will be created - detailed calibration analysis)

3. RETRAIN MODELS (if edge is weak):
   python train_auto.py
   python train_dynamic_selector_enhanced.py

4. CONTINUE TRACKING:
   - Keep using riq_analyzer.py daily
   - Bets auto-logged to ledger
   - Run analysis weekly
""")

print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
