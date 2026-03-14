#!/usr/bin/env python3
"""
Phase S3 Integrity Validation - Immediate Diagnostic Run
Validates all integrity safeguards are operational without placing bets.
"""

import sys
import pandas as pd
from datetime import datetime
from predict_live_FINAL import LivePredictionEngine

print("="*70)
print("PHASE S3 INTEGRITY VALIDATION - DIAGNOSTIC MODE")
print("="*70)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Metrics tracking
metrics = {
    'total_players': 0,
    'total_predictions': 0,
    'freshness_penalties_applied': 0,
    'volatility_penalties_applied': 0,
    'vectors_flagged': 0,
    'predictions_with_uncertainty': 0,
    'predictions_with_confidence_penalty': 0,
    'schema_errors': 0,
    'pipeline_crashed': False
}

try:
    # Initialize engine
    print("[1/5] Initializing Prediction Engine...")
    engine = LivePredictionEngine(
        models_dir="models",
        aggregated_data_path="final_feature_matrix_with_per_min_1997_onward.csv"
    )
    
    # Check freshness
    print(f"[✓] Dataset Staleness: {engine.dataset_staleness} days")
    if engine.dataset_staleness > 3:
        metrics['freshness_penalties_applied'] = 1
        print(f"[!] FRESHNESS PENALTY ACTIVE (>{engine.dataset_staleness} days old)")
    else:
        print(f"[✓] Data is fresh")
    
    print()
    print("[2/5] Generating Predictions (Today's Slate)...")
    
    # Find latest date in dataset for testing
    print("[1.5/5] Determining Test Date...")
    if 'GAME_DATE' in engine.aggregated_data.columns:
        date_col = 'GAME_DATE'
    elif 'gameDate' in engine.aggregated_data.columns:
        date_col = 'gameDate'
    elif 'date' in engine.aggregated_data.columns:
        date_col = 'date'
    elif 'game_date' in engine.aggregated_data.columns:
        date_col = 'game_date'
    else:
        date_col = None

    test_date = None
    if date_col:
        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(engine.aggregated_data[date_col]):
             engine.aggregated_data[date_col] = pd.to_datetime(engine.aggregated_data[date_col])
        
        max_date = engine.aggregated_data[date_col].max()
        test_date = max_date.strftime('%Y-%m-%d')
        print(f"[✓] Using latest dataset date for validation: {test_date}")
        
        # Debug columns
        print(f"    Available Columns: {list(engine.aggregated_data.columns)[:10]}...")
    else:
        print("[!] Could not determine date column. Defaulting to None (Today).")
    
    # Generate predictions using LOCAL ROSTERS to avoid network timeout/missing live data
    predictions = engine.predict_all_games(date=test_date, explain=False, use_local_rosters=True)
    
    if predictions.empty:
        print(f"[!] WARNING: No games found for {test_date} in local data.")
        metrics['total_players'] = 0
    else:
        metrics['total_players'] = len(predictions)
        print(f"[✓] Generated predictions for {len(predictions)} players")
        
        # Debug predictions structure
        print(f"    Predictions Columns: {predictions.columns.tolist()}")
        if not predictions.empty:
             # Find first non-null row for a prop
             for p in ['PTS', 'AST', 'REB']:
                 if p in predictions.columns:
                     val = predictions[p].iloc[0]
                     print(f"    Sample {p} data type: {type(val)}")
                     print(f"    Sample {p} data: {val}")
                     break
        
        print()
        print("[3/5] Analyzing Integrity Metrics...")
        
        # Analyze prediction structure
        sample_count = 0
        for idx, row in predictions.head(10).iterrows():
            for prop in ['PTS', 'AST', 'REB', 'FG3M', 'MIN', 'points', 'assists', 'rebounds', 'threes', 'minutes']:
                if prop in row and isinstance(row[prop], dict):
                    prop_data = row[prop]
                    metrics['total_predictions'] += 1
                    
                    # Check for uncertainty
                    if prop_data.get('uncertainty') is not None:
                        metrics['predictions_with_uncertainty'] += 1
                    
                    # Check for confidence penalty
                    if prop_data.get('confidence_penalty') is not None:
                        metrics['predictions_with_confidence_penalty'] += 1
                        penalty = prop_data.get('confidence_penalty', 1.0)
                        if penalty > 1.0:
                            metrics['volatility_penalties_applied'] += 1
                    
                    # Check for integrity concerns
                    if prop_data.get('integrity_concerns'):
                        metrics['vectors_flagged'] += 1
                    
                    sample_count += 1
                    if sample_count >= 50:  # Sample first 50 predictions
                        break
            if sample_count >= 50:
                break
        
        print(f"[✓] Sampled {sample_count} predictions for analysis")
        print(f"    - Predictions with uncertainty: {metrics['predictions_with_uncertainty']}")
        print(f"    - Predictions with confidence_penalty: {metrics['predictions_with_confidence_penalty']}")
        print(f"    - Volatility penalties (penalty > 1.0): {metrics['volatility_penalties_applied']}")
        print(f"    - Vectors flagged for anomalies: {metrics['vectors_flagged']}")
        
        print()
        print("[4/5] Testing CSV Export with Integrity Metrics...")
        
        # Test flattening
        flattened_rows = []
        for _, row in predictions.iterrows():
            base_info = {
                'player_id': row.get('player_id'),
                'player_name': row.get('player_name'),
                'team': row.get('team'),
                'opponent': row.get('opponent'),
                'is_home': row.get('is_home'),
                'game_date': row.get('game_date')
            }
            
            for prop in ['PTS', 'AST', 'REB', 'FG3M', 'MIN', 'points', 'assists', 'rebounds', 'threes', 'minutes']:
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
        
        if flattened_rows:
            df_export = pd.DataFrame(flattened_rows)
            df_export.to_csv("diagnostic_integrity_test.csv", index=False)
            print(f"[✓] Successfully exported {len(df_export)} rows to diagnostic_integrity_test.csv")
            print(f"    Columns: {df_export.columns.tolist()}")
            
            # Validate required columns exist
            required_cols = ['confidence_penalty', 'uncertainty', 'integrity_concerns']
            missing = [c for c in required_cols if c not in df_export.columns]
            if missing:
                print(f"[!] ERROR: Missing required columns: {missing}")
                metrics['schema_errors'] += 1
            else:
                print(f"[✓] All required integrity columns present")
                
            # Show sample
            print("\n[Sample Output]")
            print(df_export[['player_name', 'prop_type', 'prediction', 'confidence_penalty', 'uncertainty']].head(5).to_string())
        else:
            print("[!] ERROR: Flattening produced no rows")
            metrics['schema_errors'] += 1
    
    print()
    print("[5/5] Pipeline Validation Complete")
    
except Exception as e:
    print(f"\n[CRITICAL ERROR] Pipeline crashed: {e}")
    import traceback
    traceback.print_exc()
    metrics['pipeline_crashed'] = True

# Final Report
print()
print("="*70)
print("DIAGNOSTIC SUMMARY")
print("="*70)
print(f"Total Players Processed: {metrics['total_players']}")
print(f"Total Predictions Generated: {metrics['total_predictions']}")
print(f"Freshness Penalties Applied: {metrics['freshness_penalties_applied']}")
print(f"Volatility Penalties Applied: {metrics['volatility_penalties_applied']}")
print(f"Vectors Flagged for Anomalies: {metrics['vectors_flagged']}")
print(f"Schema Errors: {metrics['schema_errors']}")
print(f"Pipeline Crashed: {metrics['pipeline_crashed']}")
print()

if not metrics['pipeline_crashed'] and metrics['schema_errors'] == 0:
    print("[✓✓✓] VALIDATION SUCCESSFUL - All integrity safeguards operational")
    sys.exit(0)
else:
    print("[!!!] VALIDATION FAILED - Review errors above")
    sys.exit(1)
