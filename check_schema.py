
import joblib
import pandas as pd

schema_path = "models/production_v4/features.joblib"
try:
    schema = joblib.load(schema_path)
    print("Schema features:", schema)
except Exception as e:
    print(f"Error: {e}")
