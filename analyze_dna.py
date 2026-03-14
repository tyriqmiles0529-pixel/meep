import pandas as pd
import numpy as np
from pathlib import Path

def analyze_dna():
    print("Analyzing Latent DNA Strands (Embs 0-15)...")
    master_path = "final_feature_matrix_with_per_min_1997_onward.csv"
    
    # Load a sample to find what they correlate with
    df = pd.read_csv(master_path, nrows=100000)
    
    emb_cols = [f'emb_{i}' for i in range(16)]
    raw_stats = ['PTS', 'AST', 'REB', 'MIN', 'three_pointers', 'STL', 'BLK', 'TOV']
    for s in ['PTS', 'AST', 'REB', 'three_pointers']:
        df[f'{s}_per_MIN'] = df[s] / (df['MIN'] + 0.1)
    
    stats = raw_stats + [f'{s}_per_MIN' for s in ['PTS', 'AST', 'REB', 'three_pointers']]
    correlations = df[emb_cols + stats].corr()
    
    results = {}
    for i in range(16):
        row = correlations.loc[f'emb_{i}', stats]
        # Find the max correlation that ISN'T PTS or MIN if possible
        others = row.drop(['PTS', 'MIN'], errors='ignore')
        best_other = others.abs().idxmax()
        val_other = others[best_other]
        
        results[i] = {
            'strand': i,
            'primary': row.abs().idxmax(),
            'distinctive': best_other,
            'distinctive_val': val_other
        }
        print(f"DNA Strand {i:02d}: Distinctive = {best_other:<20} ({val_other:>5.2f})")

if __name__ == "__main__":
    analyze_dna()
