import numpy as np

def american_to_implied(american: int) -> float:
    """Converts American odds to raw implied probability."""
    if american > 0:
        return 100 / (american + 100)
    else:
        return abs(american) / (abs(american) + 100)

def remove_vig(odds_over: int, odds_under: int) -> tuple[float, float, float]:
    """
    Normalizes two-way (Over/Under) market odds by removing the vig.
    Returns: (implied_prob_over, implied_prob_under, fair_line_prob)
    """
    p_over = american_to_implied(odds_over)
    p_under = american_to_implied(odds_under)
    
    overround = p_over + p_under
    
    # Proportional removal
    fair_over = p_over / overround
    fair_under = p_under / overround
    
    return fair_over, fair_under, overround - 1.0

def american_to_decimal(american: int) -> float:
    if american > 0:
        return (american / 100.0) + 1.0
    else:
        return (100.0 / abs(american)) + 1.0
