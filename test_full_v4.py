#!/usr/bin/env python3
"""
Comprehensive test of all V4 components on Modal
"""

import modal
import pandas as pd
import numpy as np
import sys
import os

app = modal.App("test-full-v4")

# Full image with all dependencies
image = modal.Image.debian_slim().pip_install(
    "pandas",
    "numpy",
    "scikit-learn",
    "lightgbm",
    "pytorch-tabnet",
    "torch",
    "pyyaml",
    "scipy",
    "kaggle"
).add_local_file("train_meta_learner_v4.py", remote_path="/root/train_meta_learner_v4.py")

@app.function(
    image=image,
    cpu=4.0,
    memory=8192,
    timeout=600,  # 10 minutes max
    secrets=[
        modal.Secret.from_name("KAGGLE_USERNAME"),
        modal.Secret.from_name("KAGGLE_KEY")
    ]
)
def test_all_v4_components():
    """Test all V4 components with minimal data"""
    import sys
    import os
    from pathlib import Path
    
    sys.path.insert(0, "/root")
    os.chdir("/root")
    
    from train_meta_learner_v4 import (
        MetaLearnerV4, 
        PlayerIdentityEmbeddings,
        CrossWindowResidualCorrection,
        TemporalMemoryOverWindows,
        ExperimentConfig
    )
    
    print("="*70)
    print("COMPREHENSIVE V4 COMPONENTS TEST")
    print("="*70)
    
    results = {}
    
    # 1. Download and prepare data
    try:
        print("[*] Downloading sample data...")
        kaggle_username = os.getenv("username")
        kaggle_key = os.getenv("key")
        
        if kaggle_username and kaggle_key:
            os.environ['KAGGLE_USERNAME'] = kaggle_username
            os.environ['KAGGLE_KEY'] = kaggle_key
        
        import kaggle
        
        kaggle.api.dataset_download_files(
            'eoinamoore/historical-nba-data-and-player-box-scores',
            path='/root/',
            unzip=True
        )
        
        csv_path = Path("/root/PlayerStatistics.csv")
        df = pd.read_csv(csv_path, low_memory=False)
        
        # Create playerName and process dates
        if 'playerName' not in df.columns:
            if 'firstName' in df.columns and 'lastName' in df.columns:
                df['playerName'] = df['firstName'] + ' ' + df['lastName']
        
        if 'gameDate' in df.columns:
            df['gameDate'] = pd.to_datetime(df['gameDate'], format='mixed', utc=True)
            df['gameDate'] = df['gameDate'].dt.tz_localize(None)
            df['year'] = df['gameDate'].dt.year
            df['month'] = df['gameDate'].dt.month
            df['season_year'] = df.apply(
                lambda row: row['year'] if row['month'] >= 10 else row['year'] - 1,
                axis=1
            )
        
        # Filter to recent seasons and small sample
        sample_df = df[df['season_year'].isin([2019, 2020])].head(1000)
        print(f"[✓] Loaded {len(sample_df)} sample records")
        results['data_prep'] = "success"
        
    except Exception as e:
        print(f"[!] Data prep failed: {e}")
        results['data_prep'] = f"failed: {e}"
        return results
    
    # 2. Test PlayerIdentityEmbeddings
    try:
        print("\n[*] Testing PlayerIdentityEmbeddings...")
        
        config = {
            'embedding_dim': 8,
            'min_games_for_embedding': 3,  # Low threshold
            'player_id_col': 'playerName'
        }
        
        embeddings = PlayerIdentityEmbeddings(config)
        print(f"    Config: {config}")
        
        # Debug player counts
        if 'playerName' in sample_df.columns:
            player_counts = sample_df['playerName'].value_counts()
            eligible = player_counts[player_counts >= config['min_games_for_embedding']]
            print(f"    Eligible players: {len(eligible)}")
        
        embeddings.fit(sample_df)
        print(f"[✓] Embeddings: {len(embeddings.player_embeddings)} players")
        results['embeddings'] = "success"
        
    except Exception as e:
        print(f"[!] Embeddings failed: {e}")
        results['embeddings'] = f"failed: {e}"
    
    # 3. Test CrossWindowResidualCorrection
    try:
        print("\n[*] Testing CrossWindowResidualCorrection...")
        
        # Create dummy window predictions (2D: samples x windows)
        window_preds = {
            'points': np.random.normal(15, 5, (100, 27)),  # 100 samples, 27 windows
            'rebounds': np.random.normal(7, 3, (100, 27)),
            'assists': np.random.normal(4, 2, (100, 27)),
            'threes': np.random.normal(2, 1.5, (100, 27))
        }
        
        actuals = {
            'points': np.random.normal(15, 5, 100),
            'rebounds': np.random.normal(7, 3, 100),
            'assists': np.random.normal(4, 2, 100),
            'threes': np.random.normal(2, 1.5, 100)
        }
        
        config = {'method': 'gradient_boosting', 'n_estimators': 10}
        residual = CrossWindowResidualCorrection(config)
        residual.fit(window_preds, actuals)
        print("[✓] Residual correction fitted")
        results['residual'] = "success"
        
    except Exception as e:
        print(f"[!] Residual correction failed: {e}")
        results['residual'] = f"failed: {e}"
    
    # 4. Test TemporalMemoryOverWindows
    try:
        print("\n[*] Testing TemporalMemoryOverWindows...")
        
        config = {'method': 'transformer', 'sequence_length': 27, 'hidden_dim': 16}
        temporal = TemporalMemoryOverWindows(config)
        temporal.fit(window_preds, actuals)
        print("[✓] Temporal memory fitted")
        results['temporal'] = "success"
        
    except Exception as e:
        print(f"[!] Temporal memory failed: {e}")
        results['temporal'] = f"failed: {e}"
    
    # 5. Test MetaLearnerV4 integration
    try:
        print("\n[*] Testing MetaLearnerV4 integration...")
        
        # Create minimal config
        config_dict = {
            'experiment': {'name': 'test', 'run_id': 'test123'},
            'feature_flags': {
                'residual_correction': True,
                'player_embeddings': True,
                'temporal_memory': True
            },
            'components': {
                'residual_correction': {'method': 'gradient_boosting', 'n_estimators': 10},
                'player_embeddings': {'embedding_dim': 8, 'min_games_for_embedding': 3, 'player_id_col': 'playerName'},
                'temporal_memory': {'method': 'transformer', 'sequence_length': 27, 'hidden_dim': 16}
            }
        }
        
        config = ExperimentConfig(config_dict)
        meta_learner = MetaLearnerV4(config)
        print("[✓] MetaLearnerV4 initialized")
        
        # Test the fit_v4 method with minimal data
        v4_results = meta_learner.fit_v4(window_preds, actuals, sample_df, sample_df)
        print("[✓] MetaLearnerV4 fit_v4 completed")
        results['meta_learner_v4'] = "success"
        
    except Exception as e:
        print(f"[!] MetaLearnerV4 failed: {e}")
        import traceback
        traceback.print_exc()
        results['meta_learner_v4'] = f"failed: {e}"
    
    print(f"\n{'='*70}")
    print("FINAL TEST RESULTS:")
    for component, result in results.items():
        status = "✅" if result == "success" else "❌"
        print(f"  {status} {component}: {result}")
    print(f"{'='*70}")
    
    return results

@app.local_entrypoint()
def main():
    """Run comprehensive test"""
    result = test_all_v4_components.remote()
    print(f"\nOverall: {'SUCCESS' if all(r == 'success' for r in result.values()) else 'FAILED'}")

if __name__ == "__main__":
    main()
