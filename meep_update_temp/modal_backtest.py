#!/usr/bin/env python
"""
Run Backtest on Modal (2024-2025 Season)

Runs backtest in cloud with enough RAM to load full dataset.

Usage:
    modal run modal_backtest.py
"""

import modal

app = modal.App("nba-backtest")

# Volumes
data_volume = modal.Volume.from_name("nba-data")
model_volume = modal.Volume.from_name("nba-models")

# Image with dependencies
# UPDATED: Set use_gpu=False on models after loading
image = (
    modal.Image.debian_slim()
    .pip_install("pandas", "numpy", "pyarrow", "scikit-learn", "lightgbm", "pytorch-tabnet", "torch")
    .add_local_dir("shared", remote_path="/root/shared")
    .add_local_file("backtest_2024_2025.py", remote_path="/root/backtest_2024_2025.py")  # v11: use_gpu=False
    .add_local_file("hybrid_multi_task.py", remote_path="/root/hybrid_multi_task.py")
)


@app.function(
    image=image,
    cpu=8.0,  # Use CPU instead (avoid GPU device mismatch issues)
    memory=32768,  # 32GB RAM
    timeout=3600,  # 1 hour
    volumes={
        "/data": data_volume,
        "/models": model_volume
    }
)
def run_backtest():
    """Run backtest with plenty of RAM"""
    import sys
    sys.path.insert(0, "/root")

    # Download models from volume to local cache
    import os
    import shutil
    from pathlib import Path

    print("Setting up model cache...")
    cache_dir = Path("/root/model_cache")
    cache_dir.mkdir(exist_ok=True)

    # Copy models from volume
    model_source = Path("/models")
    if model_source.exists():
        for model_file in model_source.glob("player_models_*.pkl"):
            shutil.copy(model_file, cache_dir / model_file.name)
        for meta_file in model_source.glob("player_models_*_meta.json"):
            shutil.copy(meta_file, cache_dir / meta_file.name)

    print(f"Copied {len(list(cache_dir.glob('*.pkl')))} models to cache")

    # Copy data file
    print("Copying data file...")
    data_source = Path("/data/aggregated_nba_data.parquet")
    data_dest = Path("/root/aggregated_nba_data.parquet")
    if data_source.exists():
        shutil.copy(data_source, data_dest)
        print(f"Data file ready: {data_dest.stat().st_size / 1e9:.2f} GB")

    # Change to root directory
    os.chdir("/root")

    # Run backtest
    print("\n" + "="*70)
    print("STARTING BACKTEST")
    print("="*70)

    import backtest_2024_2025
    backtest_2024_2025.main()

    # Return results file
    results_path = Path("/root/backtest_results_2024_2025.json")
    if results_path.exists():
        with open(results_path) as f:
            results = f.read()
        print("\n[OK] Backtest complete!")
        print(f"Results saved to: {results_path}")
        return results
    else:
        print("\n[!] No results file generated")
        return None


@app.local_entrypoint()
def main():
    """Run backtest on Modal"""
    print("\n[*] Launching backtest on Modal...")
    print("   RAM: 32GB")
    print("   Timeout: 1 hour")

    result = run_backtest.remote()

    if result:
        # Save results locally
        with open("backtest_results_2024_2025.json", "w") as f:
            f.write(result)
        print("\n[OK] Results downloaded to: backtest_results_2024_2025.json")
