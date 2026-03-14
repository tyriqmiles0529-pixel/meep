import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

# Local imports
from v6_graph.lineup_gnn import V6LineupNeuralFlow
from v6_simulation.possession_sim import PossessionLevelSim
from v6_engine.game_flow import GameFlowDynamics

class V6PossessionFlowService:
    """
    V6 Core Architecture: Possession & Lineup Flow.
    Coordinates the GNN synergy, possession sim, and game-flow logic.
    """
    def __init__(self, dna_dim: int = 16):
        # 1. Neural GNN
        self.gnn = V6LineupNeuralFlow(dna_dim=dna_dim)
        
        # 2. Possession-Level Simulator
        self.possession_engine = PossessionLevelSim(iterations=2000)
        
        # 3. Game-Flow Orchestrator
        self.game_flow = GameFlowDynamics()

    def generate_v6_projections(self, offense_dna_list: torch.Tensor, 
                                defense_dna_list: torch.Tensor, 
                                synergy_matrix: torch.Tensor,
                                target_pace: float = 100.0) -> Dict:
        """
        V6 "Master Flow":
        Offense DNA + Synergy -> GNN -> Ratings/Usage -> Possession Sim -> Game Stats.
        """
        # 1. GNN: Refine DNA based on teammates
        # [1, 5, 16], [1, 5, 1], [1, 5, 16] (simplified inputs)
        ratings, usages, refined_dna = self.gnn(offense_dna_list, synergy_matrix)
        
        # 2. Derive Lineup Stats
        # ratings: [ORtg_boost, DRtg_boost]
        orating = 112.0 + float(ratings[0, 0].detach())
        usages_np = usages[0].detach().numpy() # [5] softmax
        
        # 3. Possession-Level Simulation (Full Game)
        # Total possessions = pace (standardized to 48 mins)
        total_poss = int(target_pace) 
        
        offense_lineup_stats = {
            "orating": orating,
            "drating": 112.0, # Baseline def (simplified)
            "pace": target_pace,
            "usages": usages_np
        }
        
        # 4. Simulation Results
        # Simulates 48 mins of possession-level flow
        sim_stats = self.possession_engine.simulate_full_game(
            offense_lineup_stats, 
            defense_lineup_stats={"drating": 112.0}, 
            total_possessions=total_poss
        )
        
        # 5. Extract Probabilistic Metrics
        v6_results = []
        for i in range(5):
             p_pts = sim_stats["pts"][:, i]
             v6_results.append({
                  "player_idx": i,
                  "expected_pts": float(np.mean(p_pts)),
                  "median_pts": float(np.median(p_pts)),
                  "p80_pts": float(np.percentile(p_pts, 80)),
                  "p20_pts": float(np.percentile(p_pts, 20)),
                  "win_prob": float(np.mean(p_pts > 24.5)) # Placeholder for line
             })
             
        return {
            "orating": orating,
            "lineup_synergy": float(torch.mean(refined_dna)), # Proxy for synergy
            "player_projections": v6_results,
            "sim_iterations": self.possession_engine.iterations
        }

    def augment_v5_service(self, v5_stats: Dict, synergy_score: float) -> Dict:
        """
        V6 Strategy: Use GNN outputs to adjust V5 Monte Carlo probabilities.
        """
        # 1. Synergy Delta: High coordination (GNN > 0.05) boosts probabilities
        synergy_delta = 1.0 + (synergy_score * 0.5) # Scale synergy impact
        
        # 2. Adjust win probability
        raw_prob = v5_stats.get('win_prob', 50.0)
        adjusted_prob = min(99.0, max(1.0, raw_prob * synergy_delta))
        
        return {
            "raw_win_prob": raw_prob,
            "synergy_adjusted_prob": round(adjusted_prob, 1),
            "synergy_score": round(synergy_score, 4),
            "synergy_impact": round((adjusted_prob - raw_prob), 1)
        }
