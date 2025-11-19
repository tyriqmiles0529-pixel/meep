def get_prediction_from_window(window_model, game_row, prop_name):
    try:
        features = pd.DataFrame([{
            'points': float(game_row.get('points', 0)),
            'assists': float(game_row.get('assists', 0)),
            'reboundsTotal': float(game_row.get('reboundsTotal', 0)),
            'threePointersMade': float(game_row.get('threePointersMade', 0)),
            'numMinutes': float(game_row.get('numMinutes', 0)),
            'fieldGoalsAttempted': float(game_row.get('fieldGoalsAttempted', 0)),
            'freeThrowsAttempted': float(game_row.get('freeThrowsAttempted', 0)),
        }])
        if isinstance(window_model, dict) and 'multi_task_model' in window_model:
            model = window_model['multi_task_model']
            if hasattr(model, 'predict'):
                preds = model.predict(features)
                if isinstance(preds, dict) and prop_name in preds:
                    return float(preds[prop_name][0])
        if isinstance(window_model, dict) and prop_name in window_model:
            model = window_model[prop_name]
            if model is not None and hasattr(model, 'predict'):
                pred = model.predict(features)
                return float(pred[0])
        return 0.0
    except:
        return 0.0

def collect_predictions_simple(prop_name, actual_col_name):
    print(f"COLLECTING: {prop_name}")
    test_game = df_train.iloc[0]
    test_pred = get_prediction_from_window(window_models[0], test_game, prop_name)
    print(f"Test: {test_pred}")
    window_preds = []
    actuals = []
    total = min(500, len(df_train))
    for idx in range(total):
        if idx % 100 == 0:
            print(f"{idx}/{total}")
        game = df_train.iloc[idx]
        actual = game.get(actual_col_name)
        if pd.isna(actual) or actual < 0:
            continue
        preds = []
        for window in window_models:
            preds.append(get_prediction_from_window(window, game, prop_name))
        if sum(1 for p in preds if p != 0.0) < 10:
            continue
        while len(preds) < 27:
            preds.append(np.mean([p for p in preds if p != 0.0]))
        window_preds.append(preds[:27])
        actuals.append(actual)
    print(f"Done: {len(actuals)} samples")
    return {'window_preds': np.array(window_preds), 'actuals': np.array(actuals)}

prop_data = {}
prop_data['points'] = collect_predictions_simple('points', 'points')
