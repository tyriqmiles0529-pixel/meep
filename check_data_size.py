#!/usr/bin/env python
"""
Check the size and row count of the training data file
"""

import modal

app = modal.App("check-data-size")

# Volume
data_volume = modal.Volume.from_name("nba-data")

# Simple image with pandas
image = (
    modal.Image.debian_slim()
    .pip_install([
        "pandas>=2.0.0",
        "pyarrow>=12.0.0",
        "fastparquet>=2023.0.0"
    ])
)

@app.function(
    image=image,
    volumes={"/data": data_volume},
    timeout=300
)
def check_training_data():
    """Check the size and row count of training data"""
    import pandas as pd
    import os
    
    # Load the aggregated data
    data_path = "/data/aggregated_nba_data.parquet"
    
    if not os.path.exists(data_path):
        return {"error": "Training data file not found"}
    
    try:
        # Get file size
        file_size_mb = os.path.getsize(data_path) / (1024 * 1024)
        
        # Load and count rows
        print("Loading training data...")
        df = pd.read_parquet(data_path)
        
        # Get basic info
        total_rows = len(df)
        total_cols = len(df.columns)
        
        # Get date range
        if 'gameDate' in df.columns:
            date_range = f"{df['gameDate'].min()} to {df['gameDate'].max()}"
        elif 'date' in df.columns:
            date_range = f"{df['date'].min()} to {df['date'].max()}"
        else:
            date_range = "Unknown"
        
        # Get unique players and games
        if 'personId' in df.columns:
            unique_players = df['personId'].nunique()
        else:
            unique_players = "Unknown"
            
        if 'gameId' in df.columns:
            unique_games = df['gameId'].nunique()
        else:
            unique_games = "Unknown"
        
        return {
            "file": "aggregated_nba_data.parquet",
            "file_size_mb": round(file_size_mb, 1),
            "total_rows": total_rows,
            "total_columns": total_cols,
            "unique_players": unique_players,
            "unique_games": unique_games,
            "date_range": date_range,
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 1)
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.local_entrypoint()
def main():
    print("🔍 Checking NBA Training Data Size...")
    print()
    
    result = check_training_data.remote()
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    print("="*70)
    print("NBA TRAINING DATA SUMMARY")
    print("="*70)
    print(f"📁 File: {result['file']}")
    print(f"💾 Size: {result['file_size_mb']:,} MB")
    print(f"📊 Rows: {result['total_rows']:,}")
    print(f"📋 Columns: {result['total_columns']}")
    print(f"👥 Players: {result['unique_players']:,}")
    print(f"🏀 Games: {result['unique_games']:,}")
    print(f"📅 Date Range: {result['date_range']}")
    print(f"🧠 Memory Usage: {result['memory_usage_mb']:,} MB")
    
    print(f"\n📈 Data Scale:")
    if result['total_rows'] > 10_000_000:
        print(f"   🚀 Massive dataset ({result['total_rows']/1_000_000:.1f}M rows)")
    elif result['total_rows'] > 5_000_000:
        print(f"   ✅ Large dataset ({result['total_rows']/1_000_000:.1f}M rows)")
    elif result['total_rows'] > 1_000_000:
        print(f"   📊 Medium dataset ({result['total_rows']/1_000_000:.1f}M rows)")
    else:
        print(f"   📉 Small dataset ({result['total_rows']:,} rows)")

if __name__ == "__main__":
    main()
