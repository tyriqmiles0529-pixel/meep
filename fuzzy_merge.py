import pandas as pd
import os
from difflib import get_close_matches

# Paths
RAW_EOIN = 'raw_data/eoinamoore'
RAW_SUMIT = 'raw_data/sumitrodatta'

def clean_name(name):
    if pd.isna(name): return ""
    return str(name).lower().replace('.', '').strip()

def main():
    print("Loading datasets for fuzzy matching...")
    # Load Eoin's players (from Games/PlayerStats)
    # We can just load the unique names from the reconstructed dataset or PlayerStatistics
    p_stats = pd.read_csv(os.path.join(RAW_EOIN, 'PlayerStatistics.csv'), usecols=['firstName', 'lastName'])
    p_stats['clean_name'] = (p_stats['firstName'] + ' ' + p_stats['lastName']).apply(clean_name)
    eoin_names = p_stats['clean_name'].unique()
    print(f"Unique players in Eoin's dataset: {len(eoin_names)}")
    
    # Load Sumit's players (Season Stats)
    season_stats = pd.read_csv(os.path.join(RAW_SUMIT, 'Player Per Game.csv'), usecols=['player'])
    season_stats['clean_name'] = season_stats['player'].apply(clean_name)
    sumit_names = season_stats['clean_name'].unique()
    print(f"Unique players in Sumit's dataset: {len(sumit_names)}")
    
    # Find mismatches
    common = set(eoin_names).intersection(set(sumit_names))
    missing_in_sumit = set(eoin_names) - set(sumit_names)
    print(f"Exact matches: {len(common)}")
    print(f"Missing in Sumit: {len(missing_in_sumit)}")
    
    # Fuzzy Match
    mapping = {}
    print("Fuzzy matching...")
    for name in missing_in_sumit:
        matches = get_close_matches(name, sumit_names, n=1, cutoff=0.85)
        if matches:
            mapping[name] = matches[0]
            # print(f"Mapped '{name}' -> '{matches[0]}'")
            
    print(f"Fuzzy matched {len(mapping)} additional players.")
    
    # Save mapping
    mapping_df = pd.DataFrame(list(mapping.items()), columns=['eoin_name', 'sumit_name'])
    mapping_df.to_csv('player_name_mapping.csv', index=False)
    print("Saved mapping to player_name_mapping.csv")

if __name__ == "__main__":
    main()
