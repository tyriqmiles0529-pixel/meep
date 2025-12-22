from catboost import CatBoostRegressor
import os

def check_cat():
    path = os.path.join("models", "points", "cat_model_2025.cbm")
    if os.path.exists(path):
        model = CatBoostRegressor()
        model.load_model(path)
        print(f"CatBoost Loss: {model.get_all_params().get('loss_function')}")
    else:
        print("CatBoost model not found")

if __name__ == "__main__":
    check_cat()
