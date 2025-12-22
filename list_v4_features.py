import joblib
import os

def list_all_features():
    path = os.path.join("models", "points", "xgb_model_2025.pkl")
    if os.path.exists(path):
        model = joblib.load(path)
        fnames = model.get_booster().feature_names
        print("\n".join(fnames))
    else:
        print("Model not found")

if __name__ == "__main__":
    list_all_features()
