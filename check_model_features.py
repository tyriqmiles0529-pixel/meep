import joblib
import os

def check_features():
    path = os.path.join("models", "points", "xgb_model_2025.pkl")
    if os.path.exists(path):
        model = joblib.load(path)
        if hasattr(model, 'feature_names_in_'):
            print(f"Feature count: {len(model.feature_names_in_)}")
            print(f"First 20 features: {model.feature_names_in_[:20]}")
        elif hasattr(model, 'get_booster'):
            # XGB Specific
            fnames = model.get_booster().feature_names
            print(f"Feature count: {len(fnames)}")
            print(f"First 20 features: {fnames[:20]}")
    else:
        print("Model not found")

if __name__ == "__main__":
    check_features()
