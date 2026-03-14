import pandas as pd
import sys
sys.path.append('.')
from predict_live_FINAL import LivePredictionEngine

# Initialize engine
engine = LivePredictionEngine(
    models_dir="models",
    aggregated_data_path="final_feature_matrix_with_per_min_1997_onward.csv"
)

# Generate predictions for today
predictions = engine.predict_all_games(date="2026-01-28", explain=False)

print(f"Predictions DataFrame shape: {predictions.shape}")
print(f"Columns: {predictions.columns.tolist()}")
print(f"\nFirst row:")
if not predictions.empty:
    first_row = predictions.iloc[0]
    print(f"Type: {type(first_row)}")
    for key in first_row.index:
        val = first_row[key]
        print(f"  {key}: {type(val).__name__} = {val if not isinstance(val, dict) else '{...}'}")
        
    # Test flattening
    print("\n\nTesting flattening logic:")
    flattened_rows = []
    for _, row in predictions.head(2).iterrows():
        base_info = {
            'player_id': row.get('player_id'),
            'player_name': row.get('player_name'),
            'team': row.get('team'),
            'opponent': row.get('opponent'),
            'is_home': row.get('is_home'),
            'game_date': row.get('game_date')
        }
        
        # Extract each prop with its metrics
        for prop in ['points', 'assists', 'rebounds', 'threes', 'minutes']:
            if prop in row and isinstance(row[prop], dict):
                prop_data = row[prop]
                flattened_rows.append({
                    **base_info,
                    'prop_type': prop,
                    'prediction': prop_data.get('prediction'),
                    'uncertainty': prop_data.get('uncertainty'),
                    'confidence_penalty': prop_data.get('confidence_penalty'),
                    'integrity_concerns': str(prop_data.get('integrity_concerns')) if prop_data.get('integrity_concerns') else None,
                    'lower_80': prop_data.get('lower_80'),
                    'upper_80': prop_data.get('upper_80'),
                    'lower_95': prop_data.get('lower_95'),
                    'upper_95': prop_data.get('upper_95')
                })
    
    print(f"Flattened {len(flattened_rows)} rows")
    if flattened_rows:
        df_test = pd.DataFrame(flattened_rows)
        print(df_test.head())
        df_test.to_csv("test_flattened.csv", index=False)
        print("\nSaved to test_flattened.csv")
