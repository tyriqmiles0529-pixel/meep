import pandas as pd
import sys
try:
    df = pd.read_csv('final_feature_matrix_with_per_min_1997_onward.csv', nrows=1)
    
    # Check for ID
    id_candidates = ['player_id', 'PLAYER_ID', 'personId', 'PersonId', 'person_id', 'PERSON_ID', 'personid']
    found_id = next((c for c in id_candidates if c in df.columns), None)
    
    if found_id:
        print(f"Confirmed: File has valid ID column: '{found_id}'")
        sys.exit(0)
    else:
        # Check partials
        partials = [c for c in df.columns if 'player' in c.lower() and 'id' in c.lower()]
        if partials:
             print(f"found partial matches: {partials}")
             sys.exit(0)
             
        print("CRITICAL: File MISSING ID column!")
        print(f"First 10 columns: {df.columns[:10].tolist()}")
        sys.exit(1)
        
except Exception as e:
    print(f"Error reading file: {e}")
    sys.exit(1)
