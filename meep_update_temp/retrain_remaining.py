#!/usr/bin/env python
"""
Retrain REMAINING Windows (2004-2026)

2001-2003 is already trained, so skip it.
Train: 2004-2006, 2007-2009, ..., 2025-2026

Run: modal run retrain_remaining.py
"""

import modal

app = modal.App("retrain-remaining")

# Volumes
data_volume = modal.Volume.from_name("nba-data")
model_volume = modal.Volume.from_name("nba-models")

# Image with all dependencies
image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("modal_requirements.txt")
    .add_local_dir("shared", remote_path="/root/shared")
    .add_local_file("train_player_models.py", remote_path="/root/train_player_models.py")
    .add_local_file("hybrid_multi_task.py", remote_path="/root/hybrid_multi_task.py")
    .add_local_file("rolling_features.py", remote_path="/root/rolling_features.py")
)


@app.function(
    image=image,
    gpu="A10G",  # $1.10/hour
    memory=65536,  # 64GB RAM
    timeout=14400,  # 4 hours per window
    volumes={
        "/data": data_volume,
        "/models": model_volume
    }
)
def train_one_window(window_start: int, window_end: int):
    """Train one window with rolling features"""
    import sys
    import os
    from pathlib import Path

    sys.path.insert(0, "/root")
    os.chdir("/root")

    print("="*70)
    print(f"TRAINING WINDOW: {window_start}-{window_end}")
    print("="*70)

    # Load data
    print("\n[*] Loading aggregated data...")
    from shared.data_loading import load_aggregated_player_data, get_year_column

    agg_df = load_aggregated_player_data(
        "/data/aggregated_nba_data.parquet",
        verbose=True
    )

    year_col = get_year_column(agg_df)
    print(f"[*] Year column: {year_col}")

    # Create window data WITH rolling features
    print(f"\n[*] Creating window training data...")
    print("[*] ADDING ROLLING FEATURES (L5, L10, L20 averages + trends + variance)")

    from train_player_models import create_window_training_data

    window_seasons = list(range(window_start, window_end + 1))
    window_df = create_window_training_data(
        agg_df,
        window_seasons,
        year_col,
        verbose=True
    )

    print(f"\n[*] Window data shape: {window_df.shape}")
    print(f"[*] Features: {len([c for c in window_df.columns if c not in ['points', 'reboundsTotal', 'assists', 'threePointersMade', 'numMinutes']])}")

    # Train with TabNet embeddings
    print(f"\n[*] Training with TabNet + LightGBM hybrid...")
    from train_player_models import train_player_window

    result = train_player_window(
        window_df,
        window_start,
        window_end,
        neural_epochs=20,
        verbose=True
    )

    # Save to Modal volume
    print(f"\n[*] Saving to Modal volume...")
    import pickle
    import json

    model_path = f"/models/player_models_{window_start}_{window_end}.pkl"
    meta_path = f"/models/player_models_{window_start}_{window_end}_meta.json"

    with open(model_path, 'wb') as f:
        pickle.dump(result['models'], f)

    with open(meta_path, 'w') as f:
        json.dump(result['metrics'], f, indent=2)

    # Commit to volume
    model_volume.commit()

    print(f"[OK] Saved {window_start}-{window_end}")

    # Print metrics (handle different structures)
    if result['metrics'] and 'points' in result['metrics']:
        print(f"     MAE: Points={result['metrics']['points']['mae']:.2f}, "
              f"Rebounds={result['metrics']['rebounds']['mae']:.2f}, "
              f"Assists={result['metrics']['assists']['mae']:.2f}")
    else:
        print(f"     Metrics: {list(result['metrics'].keys()) if result['metrics'] else 'N/A'}")

    return {
        'window': f"{window_start}-{window_end}",
        'metrics': result['metrics']
    }


@app.function(
    image=image,
    timeout=600,
    volumes={"/models": model_volume}
)
def delete_old_windows():
    """Delete old windows EXCEPT 2001-2003 (already trained)"""
    from pathlib import Path

    print("="*70)
    print("DELETING OLD WINDOWS (KEEPING 2001-2003)")
    print("="*70)

    # Delete 2004-2026, but NOT 2001-2003
    windows_to_delete = [
        "player_models_2004_2006",
        "player_models_2007_2009",
        "player_models_2010_2012",
        "player_models_2013_2015",
        "player_models_2016_2018",
        "player_models_2019_2021",
        "player_models_2022_2024",
        "player_models_2025_2026",
    ]

    for window in windows_to_delete:
        pkl_file = Path(f"/models/{window}.pkl")
        meta_file = Path(f"/models/{window}_meta.json")

        deleted = False
        if pkl_file.exists():
            pkl_file.unlink()
            print(f"[OK] Deleted {window}.pkl")
            deleted = True

        if meta_file.exists():
            meta_file.unlink()
            print(f"[OK] Deleted {window}_meta.json")
            deleted = True

        if not deleted:
            print(f"[SKIP] {window} (not found)")

    model_volume.commit()
    print("\n[OK] Old windows deleted (kept 2001-2003)")


@app.local_entrypoint()
def main():
    """Retrain remaining windows (2004-2026) in parallel"""

    print("="*70)
    print("RETRAINING REMAINING WINDOWS (2004-2026)")
    print("="*70)
    print("\nSkipping 2001-2003 (already trained)")
    print("\nThis will:")
    print("  1. Delete old 2004-2026 windows from Modal")
    print("  2. Retrain 8 windows in parallel:")
    print("     - 2004-2006, 2007-2009, 2010-2012, 2013-2015")
    print("     - 2016-2018, 2019-2021, 2022-2024, 2025-2026")
    print("  3. Features:")
    print("     - Rolling features (L5, L10, L20 averages + trends + variance)")
    print("     - TabNet embeddings (24-dim)")
    print("     - Feature interactions")
    print("     - Hybrid multi-task architecture")
    print("\n  Cost: ~$18 (8 windows × 2 hours × $1.10/hour)")
    print("  Time: ~2 hours (parallel execution)")
    print("="*70)

    # Step 1: Delete old windows (except 2001-2003)
    print("\n[*] Step 1: Deleting old windows (keeping 2001-2003)...")
    delete_old_windows.remote()

    # Step 2: Train remaining windows in parallel
    print("\n[*] Step 2: Training 8 windows in parallel...")

    windows = [
        (2004, 2006),
        (2007, 2009),
        (2010, 2012),
        (2013, 2015),
        (2016, 2018),
        (2019, 2021),
        (2022, 2024),
        (2025, 2026),
    ]

    # Launch all windows in parallel
    results = []
    for start, end in windows:
        result = train_one_window.spawn(start, end)
        results.append(result)
        print(f"[*] Launched training: {start}-{end}")

    # Wait for all to complete
    print("\n[*] Waiting for all windows to complete...")
    print("    (This will take ~2 hours with A10G GPU)")

    all_results = []
    for i, result_future in enumerate(results, 1):
        result = result_future.get()
        all_results.append(result)
        print(f"\n[{i}/8] Completed: {result['window']}")
        if result['metrics']:
            print(f"      Metrics: {result['metrics']}")

    print("\n" + "="*70)
    print("RETRAINING COMPLETE!")
    print("="*70)
    print(f"Trained {len(all_results)} windows")
    print("\nAll 25 windows now ready:")
    print("  - 2001-2003: Already trained (kept)")
    print("  - 2004-2026: Just trained (8 windows)")
    print("  - 1947-2000: Already on Modal (18 windows)")
    print("\nNext steps:")
    print("  1. Download all models: python download_all_models.py")
    print("  2. Use ensemble: python riq_analyzer.py --use-ensemble")
