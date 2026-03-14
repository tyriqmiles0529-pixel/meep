import numpy as np
import pandas as pd
from scipy.stats import norm, poisson, skewnorm, multivariate_normal
from typing import Dict, List, Optional, Tuple, Any

class MonteCarloEngine:
    """
    V5.5 Advanced Probabilistic Outcome Engine.
    Correlated simulation of players, pace, and team environments.
    """
    
    def __init__(self, iterations: int = 10000):
        self.iterations = iterations

    def simulate_game_environment(self, team_base_pts: float, team_pace: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Phase 1: Simulate core game environment (Pace + Total).
        Returns: (Simulated Pace Array, Simulated Score Array)
        """
        pace_sim = norm.rvs(loc=team_pace, scale=4.5, size=self.iterations)
        score_sim = norm.rvs(loc=team_base_pts, scale=7.0, size=self.iterations)
        return pace_sim, score_sim

    def simulate_correlated_props(self, mu_pts: float, mu_ast: float, rho=0.35) -> tuple[np.ndarray, np.ndarray]:
        """
        Phase 2: Correlated Stat Simulation (e.g. Points ↔ Assists).
        """
        cov = [[4.5**2, rho*4.5*1.8], 
               [rho*4.5*1.8, 1.8**2]] # Placeholder sigmas
        
        # Draw multivariate normally
        out = multivariate_normal.rvs(mean=[mu_pts, mu_ast], cov=cov, size=self.iterations)
        return out[:, 0].clip(0), out[:, 1].clip(0)

    def simulate_outcome(self, mu: float, sigma: float, prop_type: str, 
                         pace_multiplier=1.0) -> np.ndarray:
        """
        Simulates individual stat outcomes with pace-awareness.
        """
        mu = mu * pace_multiplier
        
        if prop_type.lower() in ['three_pointers', 'fg3m', 'threes']:
            return poisson.rvs(mu, size=self.iterations)
        elif prop_type.lower() in ['points', 'pts']:
            return skewnorm.rvs(a=2.0, loc=mu-1, scale=sigma, size=self.iterations).clip(min=0)
        else:
            return norm.rvs(loc=mu, scale=sigma, size=self.iterations).clip(min=0)

    def calculate_probabilities(self, mu: float, sigma: float, line: float, prop_type: str) -> Dict[str, Any]:
        """Final V5.5 Simulation Output."""
        sim_data = self.simulate_outcome(mu, sigma, prop_type)
        
        # Calculate percentages
        p_over = np.mean(sim_data > line)
        p_under = np.mean(sim_data < line)
        
        # Determine side and win probability
        if p_over >= p_under:
             win_prob = p_over
             side = "OVER"
        else:
             win_prob = p_under
             side = "UNDER"
        
        return {
            "prob_over_line": float(p_over),
            "prob_under_line": float(p_under),
            "win_prob": float(win_prob),
            "side": side,
            "median": float(np.median(sim_data)),
            "percentile_10": float(np.percentile(sim_data, 10)),
            "percentile_90": float(np.percentile(sim_data, 90)),
            "expected_value": float(np.mean(sim_data))
        }
