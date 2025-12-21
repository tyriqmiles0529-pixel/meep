
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# Configuration
INPUT_PATH = "data/pro_training_set.csv" # Training Data (1997-2024)
OUTPUT_MODEL = "models/ft_transformer_v1.pt"
OUTPUT_EMBEDDINGS = "features/player_embedding_v1.parquet"
OUTPUT_ENCODER = "models/player_id_encoder_v1.joblib"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 1024
EPOCHS = 5 # Quick train, we generate static embeddings
EMBED_DIM = 16

from ft_transformer import FTTransformer

class NBADataset(Dataset):
    def __init__(self, X_cat, X_cont, y=None):
        self.X_cat = torch.LongTensor(X_cat)
        self.X_cont = torch.FloatTensor(X_cont)
        self.y = torch.LongTensor(y) if y is not None else None
        
    def __len__(self):
        return len(self.X_cat)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X_cat[idx], self.X_cont[idx], self.y[idx]
        return self.X_cat[idx], self.X_cont[idx]

def train_embeddings():
    print(f"Loading data from {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    
    # 1. Preprocessing
    # Features for the encoder
    # Static-ish: PlayerID
    # Dynamic: Rolling Stats (Capture "Form")
    
    cont_features = [
        'roll_PTS_10', 'roll_AST_10', 'roll_REB_10', 'roll_MIN_10',
        'roll_TS_pct_10', 'roll_usage_proxy_10', 'role_trend_min',
        'season_PTS_avg'
    ]
    
    # Fill NaN
    df[cont_features] = df[cont_features].fillna(0)
    
    # Label Encode PlayerID
    # We need to handle "New Players" in production.
    # We reserve ID 0 for "Unknown".
    le = LabelEncoder()
    # Fit on all available players in TRAINING set
    df['PLAYER_ID_str'] = df['PLAYER_ID'].astype(str)
    
    # Create vocabulary
    unique_players = df['PLAYER_ID_str'].unique()
    le.fit(unique_players)
    
    # Transform
    df['player_idx'] = le.transform(df['PLAYER_ID_str']) + 1 # +1 to reserve 0
    num_players = len(unique_players) + 1
    
    # Scale Continuous
    scaler = StandardScaler()
    X_cont = scaler.fit_transform(df[cont_features])
    
    # Target: Bin PTS
    # 0-10, 10-20, 20-30, 30+ ??
    # Let's use quartiles or fixed bins.
    # Fixed: <10, 10-18, 18-25, 25+
    def bin_pts(pts):
        if pts < 10: return 0
        elif pts < 18: return 1
        elif pts < 25: return 2
        else: return 3
        
    y = df['PTS'].apply(bin_pts).values
    X_cat = df['player_idx'].values
    
    # Dataset
    dataset = NBADataset(X_cat, X_cont, y)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Model
    print(f"Initializing FT-Transformer (Players: {num_players}, Cont: {len(cont_features)})...")
    model = FTTransformer(num_players, len(cont_features), embed_dim=EMBED_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Train Loop
    print("Training Encoder...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for b_cat, b_cont, b_y in dataloader:
            b_cat, b_cont, b_y = b_cat.to(DEVICE), b_cont.to(DEVICE), b_y.to(DEVICE)
            
            optimizer.zero_grad()
            logits, _ = model(b_cat, b_cont)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")
        
    # Generate Embeddings for ALL rows
    # We want the embedding for every game_date, player.
    print("Generating Embeddings...")
    model.eval()
    
    all_cat = torch.LongTensor(X_cat).to(DEVICE)
    all_cont = torch.FloatTensor(X_cont).to(DEVICE)
    
    # Process in batches to avoid OOM
    embeddings = []
    
    dl_eval = DataLoader(NBADataset(X_cat, X_cont), batch_size=4096)
    
    with torch.no_grad():
        for b_cat, b_cont in dl_eval:
            b_cat, b_cont = b_cat.to(DEVICE), b_cont.to(DEVICE)
            _, rep = model(b_cat, b_cont)
            embeddings.append(rep.cpu().numpy())
            
    embeddings = np.vstack(embeddings)
    
    # Add to DF
    embed_cols = [f'emb_{i}' for i in range(EMBED_DIM)]
    df_emb = pd.DataFrame(embeddings, columns=embed_cols)
    
    # We need to join this back to keys: GAME_ID, PLAYER_ID
    result = pd.concat([df[['GAME_ID', 'PLAYER_ID', 'GAME_DATE']], df_emb], axis=1)
    
    print(f"Saving embeddings to {OUTPUT_EMBEDDINGS}...")
    result.to_parquet(OUTPUT_EMBEDDINGS)
    
    print("Saving artifacts...")
    torch.save(model.state_dict(), OUTPUT_MODEL)
    joblib.dump(le, OUTPUT_ENCODER)
    joblib.dump(scaler, "models/scaler_v1.joblib")
    print("Feature List saved.")
    joblib.dump(cont_features, "models/cont_features_v1.joblib")
    
    print("Done.")

if __name__ == "__main__":
    train_embeddings()
