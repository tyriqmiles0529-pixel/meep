import torch
import torch.nn as nn
import numpy as np

class FTTransformerFeatureExtractor(nn.Module):
    def __init__(self, cardinalities, embed_dim=16, device='cpu'):
        super().__init__()
        self.device = device
        self.embeddings = nn.ModuleList([
            nn.Embedding(c, embed_dim) for c in cardinalities
        ])
        
        # Simple attention-based aggregation or concatenation
        self.output_dim = len(cardinalities) * embed_dim
        
    def forward(self, x_cat):
        # x_cat: [batch_size, num_cat_features]
        embeddings = []
        for i, emb_layer in enumerate(self.embeddings):
            # Ensure correct type and device
            col_data = x_cat[:, i].long().to(self.device)
            embeddings.append(emb_layer(col_data))
            
        # Stack: [batch_size, num_cat, embed_dim]
        # Flatten for now: [batch_size, num_cat * embed_dim]
        x_emb = torch.cat(embeddings, dim=1)
        return x_emb

    def fit(self, X_train, y_train, epochs=10, batch_size=512):
        # Placeholder for pretraining logic (e.g. reconstruction or contrastive)
        # For this simplified version, we just initialize random embeddings 
        # or valid embedding layers ready for use.
        print("FT-Transformer initialized (Pretraining skipped in lite version)")
        pass

    def transform(self, X_cat):
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_cat, dtype=torch.float32)
            # We strictly need LongTensor for embedding lookups
            # But x_cat passed in might be floats if from numpy/pandas
            out = self.forward(X_tensor)
            return out.cpu().numpy()

    def save(self, path):
        torch.save(self.state_dict(), path)
