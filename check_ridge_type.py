import joblib
import os

def check_ridge():
    path = os.path.join("models", "points", "ridge_model_2025.pkl")
    if os.path.exists(path):
        model = joblib.load(path)
        # Check if it has coef_
        if hasattr(model, 'coef_'):
            print(f"Ridge Coef Count: {len(model.coef_)}")
            print(f"Intercept: {model.intercept_}")
        elif hasattr(model, 'ridge'):
             # Maybe it's the BaselineModels object
             print(f"It is likely a BaselineModels object: {type(model)}")
             if hasattr(model, 'ridge'):
                  print(f"Internal Ridge Coef Count: {len(model.ridge.coef_)}")
    else:
        print("Model not found")

if __name__ == "__main__":
    check_ridge()
