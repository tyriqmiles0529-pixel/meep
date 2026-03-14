from typing import Dict, Any

def classify_edge(edge: float) -> str:
    """Classifies the value edge into 4 tiers."""
    if edge < 0.03:
        return "ignore"
    elif edge < 0.07:
        return "lean"
    elif edge < 0.12:
        return "strong edge"
    else:
        return "elite edge"

def detect_edge(player: str, prop_type: str, line: float, model_prob: float, market_prob: float) -> Dict[str, Any]:
    """Detects and returns edge structure."""
    edge = model_prob - market_prob
    confidence = classify_edge(edge)
    
    return {
        "player": player,
        "prop_type": prop_type,
        "line": line,
        "model_prob": model_prob,
        "market_prob": market_prob,
        "edge": edge,
        "confidence_level": confidence
    }
