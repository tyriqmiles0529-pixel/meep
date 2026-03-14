#!/usr/bin/env python3
"""
Quick Modal test to debug embeddings in seconds
"""

import modal
import pandas as pd
import numpy as np
import sys
import os

app = modal.App("test-embeddings")

# Minimal image for quick testing
image = modal.Image.debian_slim().pip_install(
    "pandas",
    "numpy",
    "scikit-learn",
    "pyyaml",
    "kaggle"
).add_local_file("train_meta_learner_v4.py", remote_path="/root/train_meta_learner_v4.py")

@app.function(
    image=image,
    cpu=2.0,
    memory=4096,
    timeout=300,  # 5 minutes max
    secrets=[
        modal.Secret.from_name("KAGGLE_USERNAME"),
        modal.Secret.from_name("KAGGLE_KEY")
    ]
)
def test_embeddings_quick():
    """Quick test of embeddings with minimal data"""
    import sys
    import os
    from pathlib import Path
    
    sys.path.insert(0, "/root")
    os.chdir("/root")
    
    from train_meta_learner_v4 import PlayerIdentityEmbeddings
    
    print("="*60)
    print("QUICK EMBEDDINGS TEST ON MODAL")
    print("="*60)
    
    # Download tiny sample from Kaggle
    try:
        kaggle_username = os.getenv("username")
        kaggle_key = os.getenv("key")
        
        if kaggle_username and kaggle_key:
            os.environ['KAGGLE_USERNAME'] = kaggle_username
            os.environ['KAGGLE_KEY'] = kaggle_key
        
        import kaggle
        
        print("[*] Downloading sample data...")
        kaggle.api.dataset_download_files(
            'eoinamoore/historical-nba-data-and-player-box-scores',
            path='/root/',
            unzip=True
        )
        
        csv_path = Path("/root/PlayerStatistics.csv")
        if not csv_path.exists():
            return {"status": "error", "message": "CSV not found after download"}
            
        print(f"[✓] Downloaded: {csv_path}")
        
    except Exception as e:
        return {"status": "error", "message": f"Download failed: {e}"}
    
    # Load small sample
    print("[*] Loading sample...")
    df = pd.read_csv(csv_path, low_memory=False)
    sample_df = df.head(500)  # Tiny sample for speed
    print(f"    Sample size: {len(sample_df)} rows")
    
    # Create playerName column
    if 'playerName' not in sample_df.columns:
        if 'firstName' in sample_df.columns and 'lastName' in sample_df.columns:
            sample_df['playerName'] = sample_df['firstName'] + ' ' + sample_df['lastName']
            print("[✓] Created playerName column")
    
    print(f"\n[*] Columns available: {list(sample_df.columns)[:10]}")
    print(f"[*] Sample playerName values: {sample_df['playerName'].head().tolist()}")
    
    # Test embeddings
    config = {
        'embedding_dim': 4,
        'min_games_for_embedding': 2,  # Very low for testing
        'player_id_col': 'playerName'
    }
    
    print(f"\n[*] Testing embeddings with config: {config}")
    embeddings = PlayerIdentityEmbeddings(config)
    
    # Debug player counts
    if 'playerName' in sample_df.columns:
        player_counts = sample_df['playerName'].value_counts()
        print(f"[*] Player counts: {player_counts.head()}")
        eligible = player_counts[player_counts >= config['min_games_for_embedding']]
        print(f"[*] Eligible players: {len(eligible)}")
    
    try:
        embeddings.fit(sample_df)
        result = {
            "status": "success",
            "fitted": embeddings.fitted,
            "embeddings_count": len(embeddings.player_embeddings),
            "message": f"Successfully learned {len(embeddings.player_embeddings)} player embeddings"
        }
        print(f"[✓] {result['message']}")
        return result
        
    except Exception as e:
        print(f"[!] Embeddings failed: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

@app.local_entrypoint()
def main():
    """Run quick test"""
    result = test_embeddings_quick.remote()
    print(f"\n{'='*60}")
    print(f"TEST RESULT: {result['status']}")
    if result['status'] == 'success':
        print(f"✅ {result['message']}")
    else:
        print(f"❌ {result['message']}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
