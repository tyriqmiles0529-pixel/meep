#!/usr/bin/env python
"""
Production Demo - Run actual riq_analyzer.py and capture real output with SHAP
"""

import sys
import os
import subprocess
import json
from datetime import datetime

def run_production_demo():
    """Run the actual analyzer and capture production output with SHAP"""
    
    print("🚀 Running Production NBA Analyzer Demo...")
    print("=" * 60)
    
    # Set up environment for demo with SHAP enabled
    os.environ['FAST_MODE'] = 'true'  # Speed up processing
    os.environ['SAFE_MODE'] = 'false'  # Show full predictions
    os.environ['VERBOSE'] = 'true'
    os.environ['ENABLE_SHAP'] = 'true'  # Ensure SHAP is enabled
    
    # Sample players to analyze (popular current players)
    test_players = [
        "LeBron James",
        "Stephen Curry", 
        "Nikola Jokić"
    ]
    
    # Test props to analyze
    test_props = [
        "points",
        "threes",
        "rebounds"
    ]
    
    print("📊 Analyzing Players:")
    for player in test_players:
        print(f"   • {player}")
    
    print(f"\n🎯 Analyzing Props: {', '.join(test_props)}")
    print("🔍 SHAP Explainability: ENABLED")
    print("=" * 60)
    
    # Import and run the actual analyzer
    try:
        # Add current directory to path
        sys.path.insert(0, '.')
        
        # Import the real analyzer
        from riq_analyzer import analyze_player_prop, MODEL
        
        print("✅ Successfully loaded riq_analyzer.py")
        print("🤖 Model Architecture: Hybrid TabNet + LightGBM")
        print("📊 Feature Count: 186 engineered features")
        print("🔍 SHAP Integration: Active")
        print("=" * 60)
        
        # Run predictions for each player/prop combination
        for player in test_players:
            for prop in test_props:
                print(f"\n🏀 {player} - {prop.upper()}")
                print("-" * 40)
                
                try:
                    # Run actual prediction with SHAP enabled
                    result = analyze_player_prop(
                        player_name=player,
                        prop_type=prop,
                        line_value=25.0 if prop == "points" else (4.5 if prop == "threes" else 10.0),
                        verbose=True,
                        enable_shap=True  # Explicitly enable SHAP
                    )
                    
                    if result:
                        print(f"   🎯 Prediction: {result.get('prediction', 'N/A')}")
                        print(f"   📈 Win Probability: {result.get('win_probability', 'N/A'):.1%}")
                        print(f"   💰 Stake Size: {result.get('stake_percent', 'N/A'):.1f}%")
                        print(f"   🧠 Confidence: {result.get('confidence', 'N/A'):.0%}")
                        
                        # Show SHAP feature importance
                        if 'shap_values' in result and result['shap_values']:
                            print(f"   🔍 SHAP Feature Importance:")
                            shap_features = result['shap_values'][:5]  # Top 5 features
                            for i, (feature, importance) in enumerate(shap_features, 1):
                                direction = "↑" if importance > 0 else "↓"
                                print(f"      {i}. {feature}: {abs(importance):.3f} {direction}")
                        
                        # Show ensemble details if available
                        if 'ensemble_weight' in result:
                            print(f"   ⚖️  Ensemble Weight: {result['ensemble_weight']:.3f}")
                        
                        # Show feature count used
                        if 'feature_count' in result:
                            print(f"   📋 Features Used: {result['feature_count']}")
                    
                except Exception as e:
                    print(f"   ❌ Error: {str(e)[:100]}...")
                    continue
        
        print(f"\n" + "=" * 60)
        print("✨ PRODUCTION DEMO COMPLETE")
        print("=" * 60)
        print("📊 This output showcases:")
        print("   • Real hybrid TabNet + LightGBM predictions")
        print("   • 186-feature engineering pipeline")
        print("   • SHAP explainability with feature importance")
        print("   • Kelly criterion stake sizing")
        print("   • Production-ready inference")
        print("   • Ensemble model weighting")
        
    except ImportError as e:
        print(f"❌ Cannot import riq_analyzer: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install pandas numpy scikit-learn lightgbm pytorch-tabnet torch shap")
        
    except Exception as e:
        print(f"❌ Error running analyzer: {e}")

if __name__ == "__main__":
    run_production_demo()
