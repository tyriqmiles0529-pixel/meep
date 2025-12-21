
import pandas as pd
import shap
import catboost as cb
import matplotlib.pyplot as plt
import os

MODEL_PATH = "models/base_v1/cat_PTS.joblib"
DATA_PATH = "data/pro_training_set.csv"
OUTPUT_DIR = "analysis"

def explain_model():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load Model (CatBoost)
    # joblib load might return the object.
    import joblib
    model = joblib.load(MODEL_PATH)
    
    # Load Data (Sample 2024 val)
    print("Loading data...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    
    drop_cols = [
        'SEASON_ID', 'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 
        'TEAM_NAME', 'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL', 
        'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 
        'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'PLUS_MINUS', 
        'FANTASY_PTS', 'VIDEO_AVAILABLE', 'season_type', 'season_start_year',
        'prev_date', 'TEAM_ID_OPP'
    ]
    
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    # Filter for 2024 season
    val_df = df[df['season_start_year'] == 2024].copy()
    X = val_df[feature_cols]
    
    # Sample 500 rows for speed
    X_sample = X.sample(500, random_state=42)
    
    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Summary Plot
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary_pts.png"))
    print("Saved SHAP summary plot.")

if __name__ == "__main__":
    explain_model()
