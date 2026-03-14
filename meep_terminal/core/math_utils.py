import numpy as np
from scipy.stats import norm

class BettingMath:
    @staticmethod
    def calculate_win_prob(mu, sigma, line, pick_side='over'):
        """Calculate win probability using Normal distribution."""
        sigma = max(sigma, 1e-6)
        if pick_side == 'over':
            p = 1.0 - norm.cdf((line - mu) / sigma)
        else:
            p = norm.cdf((line - mu) / sigma)
        return p

    @staticmethod
    def american_to_decimal(odds):
        if odds > 0:
            return (odds / 100.0) + 1.0
        else:
            return (100.0 / abs(odds)) + 1.0

    @staticmethod
    def calculate_ev(win_prob, odds):
        decimal_odds = BettingMath.american_to_decimal(odds)
        return (win_prob * (decimal_odds - 1)) - (1 - win_prob)

    @staticmethod
    def calculate_kelly(win_prob, odds, fraction=0.25):
        decimal_odds = BettingMath.american_to_decimal(odds)
        b = decimal_odds - 1
        f = (b * win_prob - (1 - win_prob)) / b
        return max(0, f * fraction)

    @staticmethod
    def calculate_heat_score(win_prob, ev, std_dev):
        """
        Premium metric combining probability, edge, and volatility.
        Higher score = Better risk-adjusted opportunity.
        """
        # Normalize metrics to common scale
        prob_factor = (win_prob - 0.5) * 10 
        ev_factor = ev * 20
        volatility_penalty = (std_dev / 5.0) # Lower volatility is better
        
        score = (prob_factor + ev_factor) - volatility_penalty
        return round(max(0, score), 2)

    @staticmethod
    def derive_tier(win_prob, ev):
        if win_prob > 0.60 and ev > 0.10: return "A"
        if win_prob > 0.55 and ev > 0.05: return "B"
        return "C"
