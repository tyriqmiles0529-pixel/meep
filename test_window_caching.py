#!/usr/bin/env python3
"""
Test script to demonstrate the new 5-year window caching logic.
Shows how the system identifies which windows need training.
"""

import os
import json
from pathlib import Path

def simulate_window_caching():
    """Simulates the new caching logic to show how it works."""

    cache_dir = "model_cache"
    os.makedirs(cache_dir, exist_ok=True)

    # Simulate data from 2002 to 2025
    all_seasons = list(range(2002, 2026))
    window_size = 5
    current_season_year = 2025

    print("=" * 70)
    print("5-YEAR WINDOW CACHING SIMULATION")
    print("=" * 70)
    print(f"Data range: {min(all_seasons)}-{max(all_seasons)}")
    print(f"Current season: {current_season_year}")
    print(f"Window size: {window_size} years")
    print()

    windows_to_process = []

    for i in range(0, len(all_seasons), window_size):
        window_seasons = all_seasons[i:i+window_size]
        start_year = int(window_seasons[0])
        end_year = int(window_seasons[-1])
        cache_path = f"{cache_dir}/ensemble_{start_year}_{end_year}.pkl"
        cache_meta_path = f"{cache_dir}/ensemble_{start_year}_{end_year}_meta.json"

        is_current_window = current_season_year in window_seasons
        cache_exists = os.path.exists(cache_path) and os.path.getsize(cache_path) > 0
        cache_valid = False

        # Validate cache with metadata
        if cache_exists and os.path.exists(cache_meta_path):
            try:
                with open(cache_meta_path, 'r') as f:
                    meta = json.load(f)
                    cached_seasons = set(meta.get('seasons', []))
                    expected_seasons = set(map(int, window_seasons))
                    cache_valid = cached_seasons == expected_seasons
                    if cache_valid:
                        print(f"[OK] Window {start_year}-{end_year}: Valid cache found")
            except Exception as e:
                print(f"[WARN] Window {start_year}-{end_year}: Cache metadata invalid ({e})")
                cache_valid = False

        # Decide whether to process this window
        if is_current_window:
            print(f"[TRAIN] Window {start_year}-{end_year}: Current season - will train")
            windows_to_process.append({
                'seasons': window_seasons,
                'start_year': start_year,
                'end_year': end_year,
                'reason': 'Contains current season'
            })
        elif not cache_valid:
            status = "missing" if not cache_exists else "invalid"
            print(f"[TRAIN] Window {start_year}-{end_year}: Cache {status} - will train")
            windows_to_process.append({
                'seasons': window_seasons,
                'start_year': start_year,
                'end_year': end_year,
                'reason': f'Cache {status}'
            })

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if not windows_to_process:
        print("[OK] All windows cached and up-to-date. No training needed!")
    else:
        print(f"Will process {len(windows_to_process)} window(s) sequentially:")
        for idx, window in enumerate(windows_to_process, 1):
            print(f"  {idx}. {window['start_year']}-{window['end_year']}: {window['reason']}")
        print()
        print(f"RAM Savings: Only loading {len(windows_to_process)} windows instead of all data")
        print(f"Time Savings: Skipping {len(all_seasons)//window_size - len(windows_to_process)} cached windows")

    print("=" * 70)

if __name__ == "__main__":
    simulate_window_caching()
