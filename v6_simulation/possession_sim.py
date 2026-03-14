import numpy as np
from typing import Dict, List, Any, Optional

class PossessionLevelSim:
    """
    V6 Evolution: Micro-Stat Modeling.
    Simulates every possession of a game based on lineup GNN outputs.
    """
    def __init__(self, iterations: int = 1000):
        self.iterations = iterations
        
    def simulate_possession(self, offense_orating: float, defense_drating: float, 
                           pace: float, usage_distribution: np.ndarray, 
                           turnover_rate: float = 0.12) -> Dict[str, np.ndarray]:
        """
        Phase 1: Determine possession outcome (Points, Turnover, Rebound).
        """
        # 1. Base Probability (ORtg_off - DRtg_def adjustment)
        # Expected points per 100 possessions
        expected_pts_pp = 1.12 + (offense_orating - 110.0) / 100.0 - (defense_drating - 110.0) / 100.0
        expected_pts_pp = max(0.8, min(1.4, expected_pts_pp))
        
        # 2. Outcome outcomes
        # [iterations]
        outcomes = np.random.choice(
            ['pts', 'to', 'miss'], 
            size=self.iterations, 
            p=[0.45, turnover_rate, 1.0 - 0.45 - turnover_rate]
        )
        
        # 3. Points assignment
        pts_sim = np.zeros(self.iterations)
        pts_mask = outcomes == 'pts'
        three_pt_prob = 0.38
        pts_sim[pts_mask] = np.random.choice([2, 3], size=np.sum(pts_mask), p=[1-three_pt_prob, three_pt_prob])
        
        # 4. Usage distribution: Who takes the shot?
        shooter_indices = np.random.choice(range(5), size=self.iterations, p=usage_distribution)
        
        # 5. Resulting stat mapping
        # [iterations, 5]
        player_points = np.zeros((self.iterations, 5))
        for i in range(5):
             player_points[:, i] = np.where((pts_mask) & (shooter_indices == i), pts_sim, 0)
             
        return {
            "player_points": player_points,
            "outcomes": outcomes,
            "shooter_indices": shooter_indices
        }

    def simulate_full_game(self, offense_lineup_stats: Dict, defense_lineup_stats: Dict, total_possessions: int = 100):
        """
        Phase 2: Full Game Flow Simulation (Possession-by-Posssession aggregation).
        """
        game_stats = {
            "pts": np.zeros((self.iterations, 5)),
            "asts": np.zeros((self.iterations, 5)),
            "rebs": np.zeros((self.iterations, 5))
        }
        
        # We loop through average possessions (e.g., 100 per game)
        for _ in range(total_possessions):
            # Simulation per possession
            p_res = self.simulate_possession(
                offense_lineup_stats['orating'],
                defense_lineup_stats['drating'],
                offense_lineup_stats['pace'],
                offense_lineup_stats['usages']
            )
            
            game_stats["pts"] += p_res["player_points"]
            
            # Simple Assist logic: if pts, 60% chance of assist from teammate
            pts_mask = np.sum(p_res["player_points"], axis=1) > 0
            assistors = np.random.choice(range(5), size=self.iterations)
            # Ensure assistor != shooter (simplified)
            for i in range(5):
                 is_ast = (pts_mask) & (assistors == i) & (p_res["shooter_indices"] != i)
                 game_stats["asts"][:, i] += np.random.choice([0, 1], size=self.iterations, p=[0.4, 0.6]) * is_ast

        return game_stats
