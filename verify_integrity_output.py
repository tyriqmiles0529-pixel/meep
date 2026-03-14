import pandas as pd
import sys
# Add current directory to path
sys.path.append('.')
from predict_live_FINAL import LivePredictionEngine

print("Initializing engine...")
engine = LivePredictionEngine(
    models_dir="models",
    aggregated_data_path="final_feature_matrix_with_per_min_1997_onward.csv"
)

# Get valid test case from data
print("Checking data...")
if engine.aggregated_data.empty:
    print("Error: Dataset empty")
    sys.exit(1)

cols = list(engine.aggregated_data.columns)
print(f"Total Columns: {len(cols)}")
print(f"First 20 Columns: {cols[:20]}")
# Find ID-like columns
id_cols = [c for c in cols if 'id' in c.lower()]
print(f"ID Columns: {id_cols}")

name_cols = [c for c in cols if 'name' in c.lower() or 'player' in c.lower()]
print(f"Name/Player Columns: {name_cols}")

sys.exit(0)

# Run prediction
print("Running prediction...")
try:
    # Ensure date is string or datetime as expected
    game_date = row['gameDate'] if 'gameDate' in row else row['date']
    
    preds = engine.predict_player_props(
        player_id=str(p_id),
        player_name=p_name,
        team_curr=p_team,
        opponent=p_opp,
        is_home=True, # Dummy value
        game_date=game_date
    )
    
    print("\nPrediction Output Structure:")
    has_penalty = False
    has_integrity = False
    
    for prop in ['points', 'assists', 'rebounds']:
        if prop in preds:
            print(f"\n[{prop.upper()}]")
            val = preds[prop]
            print(val)
            
            if 'confidence_penalty' in val:
                has_penalty = True
                print(f"  -> FAIL: confidence_penalty found (Value: {val['confidence_penalty']})")
                print(f"  -> SUCCESS: confidence_penalty found (Value: {val['confidence_penalty']})")
            else:
                print("  -> FAIL: confidence_penalty MISSING")
                
            if 'integrity_concerns' in val:
                has_integrity = True
                print(f"  -> Integity Concerns field present: {val['integrity_concerns']}")
    
    if has_penalty:
        print("\n\n[PASS] Integrity safeguards are correctly attached to output.")
    else:
        print("\n\n[FAIL] Confidence penalty missing from output.")
        
except Exception as e:
    print(f"\n[ERROR] Prediction failed: {e}")
    import traceback
    traceback.print_exc()
