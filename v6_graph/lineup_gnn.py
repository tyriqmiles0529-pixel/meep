import torch
import torch.nn as nn
import torch.nn.functional as F

class LineupGNNLayer(nn.Module):
    """
    V6 Core: Neural Lineup Coordination.
    A Graph-style layer to model how 5 players on court interact.
    """
    def __init__(self, node_dim=16, edge_dim=4, out_dim=16):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        
        # Message Function: M = f(Ni, Nj, Eij)
        self.msg_fc = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, out_dim),
            nn.LeakyReLU(),
            nn.Linear(out_dim, out_dim)
        )
        
        # Update Function: Ni' = g(Ni, ΣM)
        self.update_fc = nn.Sequential(
            nn.Linear(node_dim + out_dim, out_dim),
            nn.LeakyReLU(),
            nn.Linear(out_dim, node_dim)
        )

    def forward(self, nodes, edges):
        """
        nodes: [batch, 5, node_dim] (5 players on court)
        edges: [batch, 5, 5, edge_dim] (inter-player synergy: passing, spacing)
        """
        batch_size = nodes.size(0)
        
        # 1. Message Passing (Fully connected graph for 5-man lineup)
        # We compute messages between all pairs on court
        # Expanding nodes for pairwise interaction
        node_i = nodes.unsqueeze(2).expand(-1, -1, 5, -1) # [B, 5, 5, D]
        node_j = nodes.unsqueeze(1).expand(-1, 5, -1, -1) # [B, 5, 5, D]
        
        # [B, 5, 5, 2*D + E]
        combined = torch.cat([node_i, node_j, edges], dim=-1)
        messages = self.msg_fc(combined) # [B, 5, 5, out_dim]
        
        # 2. Aggregation: sum messages from teammates
        # Masking self-loops for pure teammate interaction
        mask = (1.0 - torch.eye(5, device=nodes.device)).view(1, 5, 5, 1)
        agg_messages = (messages * mask).sum(dim=2) # [B, 5, out_dim]
        
        # 3. Update node states
        update_input = torch.cat([nodes, agg_messages], dim=-1)
        new_nodes = self.update_fc(update_input) # [B, 5, node_dim]
        
        return new_nodes

class V6LineupNeuralFlow(nn.Module):
    """
    V6 Evolution: Lineup-Aware Synergy Engine.
    Combines player DNA with GNN court dynamics.
    """
    def __init__(self, dna_dim=16, edge_dim=4):
        super().__init__()
        self.gnn = LineupGNNLayer(node_dim=dna_dim, edge_dim=edge_dim)
        
        # Lineup Impact Head: Predicts Offensive/Defensive Rating Boost
        self.rating_head = nn.Sequential(
            nn.Linear(dna_dim * 5, 32),
            nn.ReLU(),
            nn.Linear(32, 2) # [ORtg_boost, DRtg_boost]
        )
        
        # Statistical Usage Sharing Head
        self.usage_head = nn.Sequential(
            nn.Linear(dna_dim * 5, 32),
            nn.ReLU(),
            nn.Linear(32, 5), # 5 usage adjustments
            nn.Softmax(dim=-1)
        )

    def forward(self, player_dna_list, synergy_matrix):
        """
        player_dna_list: [batch, 5, 16]
        synergy_matrix: [batch, 5, 5, 4] (passing, spacing, switchability, gravity)
        """
        # 1. GNN Multi-player Interaction
        refined_dna = self.gnn(player_dna_list, synergy_matrix)
        
        # 2. Lineup Context
        lineup_flat = refined_dna.view(refined_dna.size(0), -1)
        ratings = self.rating_head(lineup_flat)
        usages = self.usage_head(lineup_flat)
        
        return ratings, usages, refined_dna
