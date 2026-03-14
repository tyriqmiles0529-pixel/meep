
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Mock Streamlit to avoid display errors during test
class MockSt:
    def markdown(self, *args, **kwargs): pass
    def divider(self, *args, **kwargs): pass
    def info(self, *args, **kwargs): pass
    def toast(self, *args, **kwargs): pass

# Add app to path
sys.path.append(str(Path(__file__).parent))

from streamlit_app import get_prediction_explanation

def test_explanations():
    print("=== TESTING PREDICTION EXPLANATION ENGINE ===")
    
    cases = [
        {
            "name": "High Confidence Over",
            "row": {"player": "LeBron James", "prop": "points", "win_prob": 65, "side": "OVER", "ev": 0.1, "synergy_impact": 0.5}
        },
        {
            "name": "High Synergy Boost",
            "row": {"player": "Stephen Curry", "prop": "three_pointers", "win_prob": 58, "side": "OVER", "ev": 0.05, "synergy_impact": 4.5}
        },
        {
            "name": "Negative Synergy Conflict",
            "row": {"player": "Luka Doncic", "prop": "assists", "win_prob": 52, "side": "UNDER", "ev": -0.18, "synergy_impact": -3.2}
        },
        {
            "name": "Market Mispricing",
            "row": {"player": "Giannis Antetokounmpo", "prop": "rebounds", "win_prob": 55, "side": "OVER", "ev": 0.25, "synergy_impact": 0.0}
        }
    ]
    
    for case in cases:
        print(f"\nCase: {case['name']}")
        explanation = get_prediction_explanation(case['row'])
        print(explanation)
        
        # Verification
        if case['name'] == "High Synergy Boost" and "Teammate spacing increases" not in explanation:
            print("❌ FAIL: Strong synergy explanation missing")
        elif case['name'] == "Negative Synergy Conflict" and "Potential usage congestion" not in explanation:
            print("❌ FAIL: Negative synergy explanation missing")
        elif case['name'] == "Market Mispricing" and "Significant market mispricing" not in explanation:
            print("❌ FAIL: EV explanation missing")
        else:
            print("✅ PASS")

if __name__ == "__main__":
    test_explanations()
