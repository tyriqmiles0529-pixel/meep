import pandas as pd
from predict_live_FINAL import LivePredictionEngine
import os

def check():
    engine = LivePredictionEngine()
    df = engine.aggregated_data
    if df is None:
        print("No data found")
        return
    
    date_col = next((c for c in ['gameDate', 'GAME_DATE', 'date', 'game_date'] if c in df.columns), None)
    print(f"Date column: {date_col}")
    if date_col:
        print(f"Max date: {df[date_col].max()}")
        print(f"Total rows: {len(df)}")
    else:
        print(f"Columns: {df.columns.tolist()}")

if __name__ == "__main__":
    check()
