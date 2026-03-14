
import torch
import torch.nn as nn

class FTTransformer(nn.Module):
    def __init__(self, num_players, num_continuous, embed_dim=16, depth=2, heads=4):
        super().__init__()
        
        # Categorical Embedding (Player ID)
        self.player_embedding = nn.Embedding(num_players, embed_dim)
        
        # Continuous Feature Encoding (Linear project to embed_dim)
        self.cont_encoder = nn.Linear(num_continuous, embed_dim)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=heads, dim_feedforward=embed_dim*2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Output Heads (Weak Supervision)
        # Predict Next Game PTS Bin (Low, Med, High, Explosion)
        self.head = nn.Linear(embed_dim, 4) 
        
    def forward(self, x_cat, x_cont):
        # x_cat: [batch, 1] (PlayerID)
        # x_cont: [batch, num_cont]
        
        # 1. Embed Categorical
        # [batch, 1, dim]
        # x_cat might be [batch] or [batch, 1]. Ensure [batch].
        if x_cat.dim() == 2:
            x_cat = x_cat.squeeze(1)
            
        emb_cat = self.player_embedding(x_cat).unsqueeze(1)
        
        # 2. Embed Continuous
        # [batch, 1, dim]
        emb_cont = self.cont_encoder(x_cont).unsqueeze(1)
        
        # 3. Stack as Sequence: [Player, Stats]
        # x: [batch, 2, dim]
        x = torch.cat([emb_cat, emb_cont], dim=1)
        
        # 4. Transform
        x = self.transformer(x)
        
        # 5. Pooling (Mean of sequence) -> [batch, dim]
        representation = x.mean(dim=1)
        
        # 6. Prediction
        logits = self.head(representation)
        
        return logits, representation

import numpy as np

class FTTransformerFeatureExtractor:
    def __init__(self, cardinalities, embed_dim=16, device='cpu'):
        self.num_players = cardinalities[0] if cardinalities else 0
        self.embed_dim = embed_dim
        self.device = device
        self.embeddings = None

    def load(self, path):
        try:
            state = torch.load(path, map_location=self.device)
            # Try to extract player embeddings
            if 'player_embedding.weight' in state:
                self.embeddings = state['player_embedding.weight']
            elif hasattr(state, 'keys'): 
                 # Maybe it IS the weight tensor directly? Unlikely.
                 pass
        except Exception as e:
            print(f"Error loading embedding state: {e}")

    def transform(self, X_cat):
        # X_cat: numpy array of shape [batch, 1] or [batch]
        if self.embeddings is None:
            return np.zeros((len(X_cat), self.embed_dim))
            
        try:
            indices = torch.tensor(X_cat, dtype=torch.long).squeeze()
            # Handle out of bounds
            indices = torch.clamp(indices, 0, self.embeddings.shape[0]-1)
            
            emb = self.embeddings[indices]
            if isinstance(emb, torch.Tensor):
                return emb.detach().cpu().numpy()
            return emb
        except Exception as e:
            print(f"Error transforming embeddings: {e}")
            return np.zeros((len(X_cat), self.embed_dim))
