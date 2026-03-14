import torch
import torch.nn as nn
import torch.nn.functional as F

class MatchupDNACrossAttention(nn.Module):
    """
    V5 Element: Contextual DNA Matching.
    Learns how an offensive DNA strand interacts with a defensive DNA strand.
    """
    def __init__(self, embed_dim=16, heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        # Multi-Head Attention to capture multiple facets of the matchup
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=heads, batch_first=True)
        
        # Matchup encoding head
        self.matchup_fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, offense_dna, defense_dna):
        """
        offense_dna: [batch, embed_dim]
        defense_dna: [batch, embed_dim]
        """
        # Reformat for attention: [batch, 1, embed_dim]
        q = offense_dna.unsqueeze(1)
        k = defense_dna.unsqueeze(1)
        v = defense_dna.unsqueeze(1)
        
        # Query = Offense, Key/Value = Defense
        # "How does this offense look against this defense?"
        attn_output, _ = self.attention(q, k, v)
        
        # Residual connection + Projection
        matchup_latent = self.matchup_fc(attn_output.squeeze(1))
        
        return matchup_latent

class V5DynamicFTTransformer(nn.Module):
    """
    V5 Evolution: Sequence-Aware FT-Transformer.
    Uses a temporal window (L10) to generate dynamic embeddings.
    """
    def __init__(self, num_players, num_teams, num_cont, embed_dim=16, seq_len=10):
        super().__init__()
        self.player_emb = nn.Embedding(num_players, embed_dim)
        self.team_emb = nn.Embedding(num_teams, embed_dim)
        
        # Continuous feature encoder (per timestep)
        self.cont_encoder = nn.Linear(num_cont, embed_dim)
        
        # Temporal Encoder (Transformer)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True)
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # Matchup Engine
        self.matchup_attention = MatchupDNACrossAttention(embed_dim)
        
        # Final Point Estimate Head
        self.regressor = nn.Linear(embed_dim, 1)

    def forward(self, player_idx, team_idx, opp_team_idx, seq_cont, lineup_context=None):
        """
        seq_cont: [batch, seq_len, num_cont]
        lineup_context: [batch, 1] (Usage Impact Factor, e.g. 1.15)
        """
        # 1. Temporal Encoding of Player Form
        # [batch, seq_len, embed_dim]
        player_static = self.player_emb(player_idx).unsqueeze(1).expand(-1, seq_cont.size(1), -1)
        cont_latent = self.cont_encoder(seq_cont)
        
        # Fuse static identity + temporal stats
        x = self.temporal_transformer(player_static + cont_latent)
        
        # Latent DNA for the player (Mean of sequence)
        offense_dna = x.mean(dim=1)
        
        # PHASE V.5: Lineup Context Modification
        # Boost player DNA latent space if usage impact is > 1.0 (Star Out)
        if lineup_context is not None:
             offense_dna = offense_dna * lineup_context
             
        # 2. Opponent Defensive DNA
        defense_dna = self.team_emb(opp_team_idx)
        
        # 3. Contextual Matchup DNA (The V5 "Secret Sauce")
        matchup_dna = self.matchup_attention(offense_dna, defense_dna)
        
        # 4. Final Prediction
        mu = self.regressor(matchup_dna)
        
        return mu, matchup_dna
