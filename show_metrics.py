#!/usr/bin/env python3
"""
Display Training Metrics - Game & Player Performance

Shows comprehensive metrics for:
- Game predictions (moneyline, spread)
- Player props (points, rebounds, assists, threes)
"""

import json
import os
from pathlib import Path


def display_metrics():
    """Display all training metrics in a clean format."""
    
    print("\n" + "="*80)
    print("NBA PREDICTOR - TRAINING METRICS SUMMARY".center(80))
    print("="*80)
    
    # Load metadata
    metadata_path = Path("models/training_metadata.json")
    
    if not metadata_path.exists():
        print("\n❌ No training metadata found!")
        print("   Run: python train_auto.py --verbose --fresh")
        return
    
    with open(metadata_path, 'r') as f:
        meta = json.load(f)
    
    # =========================================================================
    # GAME PREDICTIONS (Moneyline & Spread)
    # =========================================================================
    
    print("\n" + "─"*80)
    print("🏀 GAME PREDICTIONS (Moneyline & Spread)")
    print("─"*80)
    
    game_metrics = meta.get('game_metrics', {})
    
    if game_metrics:
        # Moneyline
        ml_logloss = game_metrics.get('ml_logloss', 'N/A')
        ml_brier = game_metrics.get('ml_brier', 'N/A')
        ml_accuracy = game_metrics.get('ml_accuracy', 'N/A')
        
        print("\n📊 MONEYLINE (Winner Prediction):")
        print(f"   • Log Loss:  {ml_logloss:.4f}" if isinstance(ml_logloss, (int, float)) else f"   • Log Loss:  {ml_logloss}")
        print(f"   • Brier Score: {ml_brier:.4f}" if isinstance(ml_brier, (int, float)) else f"   • Brier Score: {ml_brier}")
        
        if isinstance(ml_accuracy, (int, float)):
            print(f"   • Accuracy:  {ml_accuracy*100:.1f}%")
            
            # Accuracy interpretation
            if ml_accuracy >= 0.57:
                status = "🟢 EXCELLENT"
            elif ml_accuracy >= 0.55:
                status = "🟡 GOOD"
            elif ml_accuracy >= 0.52:
                status = "🟠 PROFITABLE"
            else:
                status = "🔴 NEEDS IMPROVEMENT"
            print(f"   • Status:    {status}")
        
        # Spread
        sp_rmse = game_metrics.get('sp_rmse', 'N/A')
        sp_mae = game_metrics.get('sp_mae', 'N/A')
        spread_sigma = game_metrics.get('spread_sigma', 'N/A')
        sp_accuracy = game_metrics.get('sp_accuracy', 'N/A')
        
        print("\n📊 SPREAD (Margin Prediction):")
        print(f"   • RMSE:      {sp_rmse:.2f} points" if isinstance(sp_rmse, (int, float)) else f"   • RMSE:      {sp_rmse}")
        print(f"   • MAE:       {sp_mae:.2f} points" if isinstance(sp_mae, (int, float)) else f"   • MAE:       {sp_mae}")
        print(f"   • Sigma:     {spread_sigma:.2f}" if isinstance(spread_sigma, (int, float)) else f"   • Sigma:     {spread_sigma}")
        
        if isinstance(sp_accuracy, (int, float)):
            print(f"   • Accuracy:  {sp_accuracy*100:.1f}%")
            
            # Accuracy interpretation
            if sp_accuracy >= 0.55:
                status = "🟢 EXCELLENT"
            elif sp_accuracy >= 0.53:
                status = "🟡 GOOD"
            elif sp_accuracy >= 0.50:
                status = "🟠 BREAK-EVEN"
            else:
                status = "🔴 NEEDS IMPROVEMENT"
            print(f"   • Status:    {status}")
        
        # RMSE interpretation
        if isinstance(sp_rmse, (int, float)):
            if sp_rmse < 10:
                quality = "🟢 Excellent"
            elif sp_rmse < 12:
                quality = "🟡 Good"
            elif sp_rmse < 14:
                quality = "🟠 Acceptable"
            else:
                quality = "🔴 Needs improvement"
            print(f"   • Quality:   {quality}")
    else:
        print("\n⚠️  No game metrics found")
    
    # =========================================================================
    # PLAYER PROPS
    # =========================================================================
    
    print("\n" + "─"*80)
    print("👤 PLAYER PROPS (Points, Rebounds, Assists, Threes)")
    print("─"*80)
    
    player_metrics = meta.get('player_metrics', {})
    
    if player_metrics:
        for prop_name in ['points', 'rebounds', 'assists', 'threes']:
            metrics = player_metrics.get(prop_name, {})
            
            if not metrics:
                continue
            
            rows = metrics.get('rows', 0)
            rmse = metrics.get('rmse', 'N/A')
            mae = metrics.get('mae', 'N/A')
            
            print(f"\n📊 {prop_name.upper()}:")
            print(f"   • Training samples: {rows:,}")
            print(f"   • RMSE:  {rmse:.2f}" if isinstance(rmse, (int, float)) else f"   • RMSE:  {rmse}")
            print(f"   • MAE:   {mae:.2f}" if isinstance(mae, (int, float)) else f"   • MAE:   {mae}")
            
            # RMSE interpretation by prop type
            if isinstance(rmse, (int, float)):
                if prop_name == 'points':
                    if rmse < 4.5:
                        status = "🟢 Excellent"
                    elif rmse < 5.5:
                        status = "🟡 Good"
                    elif rmse < 6.5:
                        status = "🟠 Acceptable"
                    else:
                        status = "🔴 Needs improvement"
                elif prop_name == 'rebounds':
                    if rmse < 1.8:
                        status = "🟢 Excellent"
                    elif rmse < 2.2:
                        status = "🟡 Good"
                    elif rmse < 2.6:
                        status = "🟠 Acceptable"
                    else:
                        status = "🔴 Needs improvement"
                elif prop_name == 'assists':
                    if rmse < 1.5:
                        status = "🟢 Excellent"
                    elif rmse < 1.9:
                        status = "🟡 Good"
                    elif rmse < 2.3:
                        status = "🟠 Acceptable"
                    else:
                        status = "🔴 Needs improvement"
                elif prop_name == 'threes':
                    if rmse < 1.0:
                        status = "🟢 Excellent"
                    elif rmse < 1.3:
                        status = "🟡 Good"
                    elif rmse < 1.6:
                        status = "🟠 Acceptable"
                    else:
                        status = "🔴 Needs improvement"
                else:
                    status = ""
                
                if status:
                    print(f"   • Status: {status}")
    else:
        print("\n⚠️  No player prop metrics found")
    
    # =========================================================================
    # TRAINING CONFIGURATION
    # =========================================================================
    
    print("\n" + "─"*80)
    print("⚙️  TRAINING CONFIGURATION")
    print("─"*80)
    
    era = meta.get('era', {})
    versions = meta.get('versions', {})
    
    print(f"\n🎯 Era Settings:")
    print(f"   • Game season cutoff:   {era.get('game_season_cutoff', 'N/A')}")
    print(f"   • Player season cutoff: {era.get('player_season_cutoff', 'N/A')}")
    print(f"   • Decay factor:         {era.get('decay', 'N/A')}")
    print(f"   • Min weight:           {era.get('min_weight', 'N/A')}")
    
    print(f"\n📦 Library Versions:")
    print(f"   • LightGBM: {versions.get('lightgbm', 'N/A')}")
    print(f"   • Pandas:   {versions.get('pandas', 'N/A')}")
    print(f"   • NumPy:    {versions.get('numpy', 'N/A')}")
    
    # Check for neural network usage
    if os.path.exists("models/points_model_tabnet.zip"):
        print(f"\n🧠 Neural Network: ENABLED (TabNet + LightGBM)")
    else:
        print(f"\n🤖 Model: LightGBM (standard)")
    
    # =========================================================================
    # FEATURE COUNTS
    # =========================================================================
    
    print("\n" + "─"*80)
    print("📊 FEATURES")
    print("─"*80)
    
    game_features = meta.get('game_features', [])
    
    print(f"\n🏀 Game Features: {len(game_features) if game_features else 0}")
    print(f"👤 Player Features: ~80-130 (depending on phases enabled)")
    
    # =========================================================================
    # SUMMARY & RECOMMENDATIONS
    # =========================================================================
    
    print("\n" + "="*80)
    print("📋 SUMMARY")
    print("="*80)
    
    # Calculate overall health
    issues = []
    strengths = []
    
    if game_metrics:
        ml_acc = game_metrics.get('ml_accuracy')
        sp_rmse_val = game_metrics.get('sp_rmse')
        
        if isinstance(ml_acc, (int, float)):
            if ml_acc >= 0.55:
                strengths.append("Strong moneyline predictions")
            elif ml_acc < 0.52:
                issues.append("Moneyline accuracy below break-even")
        
        if isinstance(sp_rmse_val, (int, float)):
            if sp_rmse_val < 11:
                strengths.append("Excellent spread predictions")
            elif sp_rmse_val > 13:
                issues.append("Spread RMSE too high")
    
    if player_metrics:
        for prop in ['points', 'rebounds', 'assists', 'threes']:
            metrics = player_metrics.get(prop, {})
            rmse = metrics.get('rmse')
            if isinstance(rmse, (int, float)):
                thresholds = {'points': 5.5, 'rebounds': 2.2, 'assists': 1.9, 'threes': 1.3}
                if rmse <= thresholds.get(prop, 999):
                    strengths.append(f"Good {prop} predictions")
                elif rmse > thresholds.get(prop, 0) * 1.3:
                    issues.append(f"{prop.capitalize()} RMSE too high")
    
    if strengths:
        print("\n✅ Strengths:")
        for s in strengths:
            print(f"   • {s}")
    
    if issues:
        print("\n⚠️  Issues:")
        for i in issues:
            print(f"   • {i}")
        print("\n💡 Recommendation: Consider retraining with Phase 7 features")
        print("   Run: python train_auto.py --verbose --fresh")
    else:
        print("\n🎉 All metrics look good!")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    display_metrics()
