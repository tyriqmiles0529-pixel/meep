import joblib
import os

def check_lgb():
    path = os.path.join("models", "points", "lgb_model_2025.pkl")
    if os.path.exists(path):
        model = joblib.load(path)
        print(f"LGB Objective: {model.objective_}")
        print(f"LGB Classes: {getattr(model, 'classes_', 'None (Regression)')}")
    else:
        print("LGB model not found")

if __name__ == "__main__":
    check_lgb()
