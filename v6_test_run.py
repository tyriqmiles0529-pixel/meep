import torch
import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from v6_engine.possession_flow_service import V6PossessionFlowService

def run_v6_possession_flow_test():
    print("INITIALIZING V6 POSSESSION FLOW TEST...")
    start_time = time.time()
    
    # 1. LINEUP SELECTION: Boston Celtics (High Synergy Template)
    # Positions: PG, SG, SF, PF, C
    lineup_names = ["Jrue Holiday", "Derrick White", "Jaylen Brown", "Jayson Tatum", "Al Horford"]
    print(f"Target Lineup: {', '.join(lineup_names)}")
    
    # 2. MOCK DNA & SYNERGY DATA
    # Node features: 16-dim FT-Transformer latents
    # We simulate a "Modern Spacing" DNA profile for BOS
    offense_dna = torch.randn(1, 5, 16) # [Batch, Players, DNA_Dim]
    defense_dna = torch.randn(1, 5, 16) # Opponent Defensive Profile
    
    # Edge Matrix: [1, 5, 5, 4] (Passing, Spacing, Switchability, Gravity)
    # Boosting synergy between Tatum (3) and Brown (2)
    synergy_matrix = torch.rand(1, 5, 5, 4) * 0.5
    synergy_matrix[0, 3, 2, :] += 0.4 # Tatum -> Brown synergy
    synergy_matrix[0, 2, 3, :] += 0.4 # Brown -> Tatum synergy

    # 3. INITIALIZE V6 SERVICE
    v6_service = V6PossessionFlowService(dna_dim=16)
    
    # 4. EXECUTE V6 MASTER FLOW
    print("\nExecuting Lineup GNN & Possession Simulator...")
    results = v6_service.generate_v6_projections(
        offense_dna_list=offense_dna,
        defense_dna_list=defense_dna,
        synergy_matrix=synergy_matrix,
        target_pace=100.5 # High-pace context
    )
    
    # 5. INTEGRATE MARKET INTELLIGENCE (V5.5 MOCK)
    # We simulate bookmaker lines to detect V6 edges
    mock_lines = {
        "Jayson Tatum": 26.5,
        "Jaylen Brown": 22.5,
        "Derrick White": 15.5,
        "Jrue Holiday": 12.5,
        "Al Horford": 9.5
    }
    
    final_output = []
    print("\nCalculating Market Edges (V6 Fair Line vs. Sportsbook)...")
    
    for i, p_res in enumerate(results['player_projections']):
        name = lineup_names[i]
        line = mock_lines.get(name, 20.0)
        
        # Calculate Edge based on V6 Possession Probabilities
        win_prob = p_res['win_prob']
        implied_market_prob = 0.5238 # -110 standard
        edge = win_prob - implied_market_prob
        
        edge_label = "ignore"
        if edge > 0.12: edge_label = "ELITE EDGE"
        elif edge > 0.07: edge_label = "STRONG EDGE"
        elif edge > 0.03: edge_label = "LEAN"
        
        row = {
            "player": name,
            "synergy_score": round(results['lineup_synergy'], 3),
            "expected_pts": round(p_res['expected_pts'], 2),
            "median_pts": round(p_res['median_pts'], 2),
            "p80_pts": round(p_res['p80_pts'], 2),
            "market_line": line,
            "v6_win_prob": round(win_prob * 100, 1),
            "v6_edge": round(edge * 100, 1),
            "confidence": edge_label
        }
        final_output.append(row)

    # 6. PERSIST RESULTS
    df = pd.DataFrame(final_output)
    df.to_csv("predictions/v6_test_results.csv", index=False)
    with open("predictions/v6_test_metadata.json", "w") as f:
        json.dump(results, f, indent=4)
    
    # 7. PERFORMANCE & DIAGNOSTICS
    duration = time.time() - start_time
    print(f"\nTEST COMPLETE in {duration:.2f}s")
    print("-" * 50)
    print(df[['player', 'expected_pts', 'market_line', 'v6_win_prob', 'v6_edge', 'confidence']])
    print("-" * 50)
    print(f"Lineup Synergy: {results['lineup_synergy']:.4f}")
    print(f"Possession Loops: {results['sim_iterations']} iterations")
    print(f"Results saved to predictions/v6_test_results.csv")

if __name__ == "__main__":
    run_v6_possession_flow_test()
