#!/usr/bin/env python
"""
Run RIQ Analyzer on Modal

Benefits:
- More RAM (no memory issues)
- Faster (more CPU cores)
- Can use ensemble without local resource limits

Run: modal run modal_analyzer.py

VERSION: 2025-11-19-v7 - PLAYER PROPS ONLY + SHAP + working ensemble
"""

import modal

app = modal.App("nba-analyzer-v7")  # Changed app name to force rebuild

# Volumes
model_volume = modal.Volume.from_name("nba-models")

# Image with all dependencies
# FORCE REBUILD: 2025-11-19-v7 - Player props + SHAP + ensemble working
image = (
    modal.Image.debian_slim()
    .pip_install(
        "pandas",
        "numpy",
        "scikit-learn",
        "lightgbm",
        "pytorch-tabnet",
        "torch",
        "requests",
        "nba_api",
        "scipy",
        "shap",  # For explainability
    )
    .run_commands("echo 'Image built: 2025-11-19-v7'")  # Force cache bust
    .add_local_file("riq_analyzer.py", remote_path="/root/riq_analyzer.py")
    .add_local_file("ensemble_predictor.py", remote_path="/root/ensemble_predictor.py")
    .add_local_file("meta_learner_ensemble.py", remote_path="/root/meta_learner_ensemble.py")
    .add_local_file("explainability.py", remote_path="/root/explainability.py")
    .add_local_dir("shared", remote_path="/root/shared")
    .add_local_file("hybrid_multi_task.py", remote_path="/root/hybrid_multi_task.py")
    .add_local_file("optimization_features.py", remote_path="/root/optimization_features.py")
    .add_local_file("phase7_features.py", remote_path="/root/phase7_features.py")
    .add_local_file("rolling_features.py", remote_path="/root/rolling_features.py")
    .add_local_dir("priors_data", remote_path="/root/priors_data")
)


@app.function(
    image=image,
    cpu=8.0,  # 8 CPU cores
    memory=16384,  # 16GB RAM
    timeout=3600,  # 1 hour
    volumes={"/models": model_volume},
    secrets=[
        modal.Secret.from_name("api-sports-key"),  # API-Sports key
        modal.Secret.from_name("theodds-api-key"),  # TheOdds API key for player props
    ]
)
def run_analyzer(use_ensemble: bool = True, use_minutes_first: bool = False):
    """
    Run RIQ analyzer with ensemble mode and optional minutes-first pipeline.

    Args:
        use_ensemble: Use 27-window ensemble + meta-learner (default: True)
        use_minutes_first: Use minutes-first pipeline (default: False, +5-10% accuracy)
    """
    import sys
    import os
    from pathlib import Path

    # CRITICAL: Force CPU mode BEFORE any imports that might load torch
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

    sys.path.insert(0, "/root")
    os.chdir("/root")

    # Copy models from volume to local cache
    print("="*70)
    print("SETTING UP MODEL CACHE")
    print("="*70)

    cache_dir = Path("/root/model_cache")
    cache_dir.mkdir(exist_ok=True)

    model_source = Path("/models")
    if model_source.exists():
        import shutil

        # Copy all window models
        for model_file in model_source.glob("player_models_*.pkl"):
            dest = cache_dir / model_file.name
            if not dest.exists():
                shutil.copy(model_file, dest)
                print(f"[OK] Copied {model_file.name}")

        # Copy meta files
        for meta_file in model_source.glob("player_models_*_meta.json"):
            dest = cache_dir / meta_file.name
            if not dest.exists():
                shutil.copy(meta_file, dest)

        # Copy meta-learner if available (try 2025-2026 first, then 2024-2025)
        meta_learner_found = False
        for meta_name in ["meta_learner_2025_2026.pkl", "meta_learner_2024_2025.pkl"]:
            meta_learner = model_source / meta_name
            if meta_learner.exists():
                shutil.copy(meta_learner, cache_dir / meta_name)
                print(f"[OK] Copied meta-learner: {meta_name}")
                meta_learner_found = True
                break

        if not meta_learner_found:
            print(f"[!] Meta-learner not found in {model_source}")
            print(f"[!] Available meta files: {list(model_source.glob('meta_*'))}")
            print(f"[!] Train meta-learner with: python train_meta_learner.py")

    models_count = len(list(cache_dir.glob("player_models_*.pkl")))
    print(f"[OK] {models_count} window models ready")

    # Get API keys from Modal secrets (automatically set as env vars)
    api_key = os.getenv("API_SPORTS_KEY", "")
    theodds_key = os.getenv("THEODDS_API_KEY", "")

    if api_key:
        print(f"[OK] API-Sports key configured: {api_key[:8]}...{api_key[-4:]}")
    else:
        print("[!] WARNING: API_SPORTS_KEY not found!")

    if theodds_key:
        print(f"[OK] TheOdds key configured: {theodds_key[:8]}...{theodds_key[-4:]}")
    else:
        print("[!] WARNING: THEODDS_API_KEY not found - player props may not be fetched!")

    # Test API-Sports connection
    if api_key:
        print("\n[*] Testing API-Sports connection...")
        import requests
        import datetime
        test_url = "https://v2.nba.api-sports.io/games"
        headers = {
            'x-rapidapi-host': 'v2.nba.api-sports.io',
            'x-rapidapi-key': api_key
        }
        try:
            today = datetime.date.today()
            date_str = today.strftime("%Y-%m-%d")
            print(f"[*] Checking {date_str}...")

            # Use method 4: Just date parameter (no filters)
            response = requests.get(test_url, headers=headers, params={'date': date_str}, timeout=10)

            if response.status_code == 200:
                data = response.json()
                results = data.get('results', 0)
                print(f"[OK] Found {results} games")
                if results > 0:
                    try:
                        for game in data['response'][:3]:
                            home = game.get('teams', {}).get('home', {}).get('name', 'Unknown')
                            away = game.get('teams', {}).get('visitors', {}).get('name', 'Unknown')
                            print(f"    - {home} vs {away}")
                    except:
                        pass
            else:
                print(f"[!] API Error: {response.status_code}")
        except Exception as e:
            print(f"[!] API Test Failed: {e}")

    # Import and run analyzer
    print("\n" + "="*70)
    print(f"RUNNING ANALYZER (Ensemble: {use_ensemble})")
    print("="*70)

    # Modify sys.argv to pass flags
    sys.argv = ["riq_analyzer.py"]
    if use_ensemble:
        sys.argv.append("--use-ensemble")
    if use_minutes_first:
        sys.argv.append("--minutes-first")

    # Import riq_analyzer module but DON'T run it via import
    # Instead, call the run_analysis function directly
    import riq_analyzer

    # Enable DEBUG_MODE for detailed logging
    riq_analyzer.DEBUG_MODE = True

    # Disable SHAP temporarily (causes NBA API timeouts)
    riq_analyzer.SHAP_AVAILABLE = False

    # Initialize the MODEL with ensemble and minutes-first flags
    riq_analyzer.MODEL = riq_analyzer.ModelPredictor(
        use_ensemble=use_ensemble,
        use_minutes_first=use_minutes_first
    )

    # Run analysis directly (ensures output is captured)
    riq_analyzer.run_analysis()

    print("\n" + "="*70)
    print("ANALYZER COMPLETE")
    print("="*70)

    # Return results
    results_file = Path("/root/bets_ledger.pkl")
    if results_file.exists():
        import pickle
        with open(results_file, 'rb') as f:
            ledger = pickle.load(f)

        print(f"\n[OK] Found {len(ledger)} bets in ledger")
        return {
            "status": "success",
            "bets_count": len(ledger),
            "message": "Analyzer completed successfully"
        }
    else:
        return {
            "status": "no_bets",
            "message": "No bets generated (no games today?)"
        }


@app.local_entrypoint()
def main(use_ensemble: bool = True, minutes_first: bool = False):
    """
    Run analyzer on Modal

    Args:
        use_ensemble: Use 27-window ensemble + meta-learner (default: True)
        minutes_first: Use minutes-first pipeline (default: False, +5-10% accuracy)
    """
    print("="*70)
    print("NBA ANALYZER ON MODAL")
    print("="*70)
    print(f"\nEnsemble mode: {use_ensemble}")
    if minutes_first:
        print(f"Minutes-first mode: {minutes_first} (+5-10% accuracy expected)")
    print("\nResources:")
    print("  - CPU: 8 cores")
    print("  - RAM: 16GB")
    print("  - Timeout: 1 hour")
    print("="*70)

    result = run_analyzer.remote(use_ensemble=use_ensemble, use_minutes_first=minutes_first)

    print("\n" + "="*70)
    print("RESULT")
    print("="*70)
    print(f"Status: {result['status']}")
    print(f"Message: {result['message']}")
    if 'bets_count' in result:
        print(f"Bets generated: {result['bets_count']}")
