
import joblib
import pandas as pd

schema_path = "models/production_v4/features.joblib"
data_path = "final_feature_matrix_with_per_min_1997_onward.csv"

try:
    schema = joblib.load(schema_path)
    df = pd.read_csv(data_path, nrows=1)
    
    missing = [c for c in schema if c not in df.columns]
    found = [c for c in schema if c in df.columns]
    
    print(f"Total schema features: {len(schema)}")
    print(f"Missing from data: {len(missing)}")
    if missing:
        print("First 20 missing:", missing[:20])
    print(f"Found in data: {len(found)}")
    
    # Check for embeddings
    emb_cols = [c for c in df.columns if c.startswith('emb_')]
    print(f"Embeddings in data: {len(emb_cols)}")

except Exception as e:
    print(f"Error: {e}")
