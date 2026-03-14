import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

def generate_archetypes(data_path, n_clusters=16):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, low_memory=False)
    
    # Get embedding columns
    emb_cols = [c for c in df.columns if c.startswith('emb_')]
    if not emb_cols:
        print("No embeddings found in dataset.")
        return
    
    # Get latest embedding for each player
    print("Extracting latest embeddings for each player...")
    date_col = 'GAME_DATE' if 'GAME_DATE' in df.columns else 'date' if 'date' in df.columns else 'gameDate'
    id_col = 'PLAYER_ID' if 'PLAYER_ID' in df.columns else 'player_id'
    
    # Sort and get tail
    df_sorted = df.sort_values(date_col)
    latest_rows = df_sorted.groupby(id_col).tail(1)
    
    # Filter for players with valid embeddings
    player_embs = latest_rows.dropna(subset=emb_cols).copy()
    X = player_embs[emb_cols].values
    
    # Clustering
    print(f"Clustering into {n_clusters} archetypes...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    player_embs.loc[:, 'archetype_id'] = clusters
    
    # Characterize clusters by average stats
    base_stats = ['points', 'assists', 'reboundsTotal', 'minutes', 'three_pointers', 'PTS', 'AST', 'REB', 'MIN']
    available_stats = [s for s in base_stats if s in player_embs.columns]
    
    cluster_profiles = player_embs.groupby('archetype_id')[available_stats].mean()
    print("\nCluster Profiles (Averages):")
    print(cluster_profiles)
    
    # Save the model and the mapping
    output_dir = Path("models/archetypes")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(kmeans, output_dir / "kmeans_model.joblib")
    joblib.dump(scaler, output_dir / "scaler.joblib")
    
    # Save player mappings
    name_col = 'player_name' if 'player_name' in player_embs.columns else 'PLAYER_NAME'
    mapping = player_embs[[id_col, name_col, 'archetype_id']].copy()
    
    # Normalize column names for the interface
    mapping = mapping.rename(columns={id_col: 'player_id', name_col: 'player_name'})
    
    # Cast player_id to string and clean up
    mapping['player_id'] = mapping['player_id'].astype(float).astype(int).astype(str)
    mapping.to_csv(output_dir / "player_archetypes.csv", index=False)
    print(f"\nSaved archetype mapping to {output_dir / 'player_archetypes.csv'}")
    
    return cluster_profiles

if __name__ == "__main__":
    generate_archetypes("final_feature_matrix_with_per_min_1997_onward.csv", n_clusters=16)
