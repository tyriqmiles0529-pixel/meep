from predict_live_FINAL import LivePredictionEngine
import pandas as pd

engine = LivePredictionEngine(
    models_dir="models",
    aggregated_data_path="final_feature_matrix_with_per_min_1997_onward.csv"
)

today = '2026-02-02'
# Note: Engine renames 'date' -> 'gameDate' in __init__
df_today = engine.aggregated_data[engine.aggregated_data['gameDate'].astype(str).str.startswith(today)]

if df_today.empty:
    print("No rows for today.")
else:
    row = df_today.iloc[0]
    name = row.get('player_name', row.get('PLAYER_NAME', 'Unknown'))
    print(f"Testing {name} (ID: {row['player_id']})...")
    try:
        # Standardize inputs
        pred = engine.predict_player_props(
            player_id=str(row['player_id']),
            player_name=name,
            team_abbr=str(row.get('team', 'UNK')),
            opponent_abbr='UNK',
            is_home=True,
            game_date=today
        )
        print("Prediction Keys:", pred.keys())
        if 'proj_PTS' in pred:
            print(f"PTS Proj: {pred['proj_PTS']}")
        if 'confidence_penalty' in pred:
             print(f"Conf Penalty: {pred['confidence_penalty']}")
    except Exception as e:
        import traceback
        traceback.print_exc()
