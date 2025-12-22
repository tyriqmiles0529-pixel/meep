import joblib
import os

def audit_stacker():
    for target in ['points', 'rebounds', 'assists', 'three_pointers']:
        path = os.path.join("models", target, "ridge_model_2025.pkl")
        if os.path.exists(path):
            model = joblib.load(path)
            print(f"\nTarget: {target}")
            print(f"Coefs: {getattr(model, 'coef_', 'N/A')}")
            print(f"Intercept: {getattr(model, 'intercept_', 'N/A')}")
        else:
            print(f"No stacker for {target}")

if __name__ == "__main__":
    audit_stacker()
