import joblib
import xgboost as xgb
import os

def check_params():
    path = os.path.join("models", "points", "xgb_model_2025.pkl")
    if os.path.exists(path):
        model = joblib.load(path)
        print(f"XGB Type: {type(model)}")
        if isinstance(model, xgb.XGBRegressor):
            print(f"Objective: {model.get_params().get('objective')}")
            print(f"n_estimators: {model.get_params().get('n_estimators')}")
        elif isinstance(model, xgb.XGBClassifier):
            print("ALERT: This is an XGBClassifier!")
            print(f"Objective: {model.get_params().get('objective')}")
        else:
            print(f"Params: {model.get_params()}")
    else:
        print("Model not found")

if __name__ == "__main__":
    check_params()
