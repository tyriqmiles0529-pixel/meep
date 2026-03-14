#!/usr/bin/env python3
"""
Quick test to verify player priors merging and data preparation.
Run this before full training to catch issues early.
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 70)
print("PLAYER PRIORS MERGE TEST")
print("=" * 70)

# Simulate the key data loading steps
print("\n1. Loading sample player data...")
player_csv = Path("PlayerStatistics.csv")
if not player_csv.exists():
    print("ERROR: PlayerStatistics.csv not found!")
    exit(1)

# Load small sample
ps = pd.read_csv(player_csv, nrows=1000)
print(f"   Loaded {len(ps):,} sample rows")
print(f"   Columns: {list(ps.columns)[:10]}...")

# Check critical columns
print("\n2. Checking critical columns...")
has_teamid = 'teamId' in ps.columns
has_home = 'home' in ps.columns
has_personid = 'personId' in ps.columns
has_date = any('date' in c.lower() for c in ps.columns)

print(f"   teamId: {'✓' if has_teamid else '✗ MISSING (will use home flag)'}")
print(f"   home: {'✓' if has_home else '✗ MISSING (CRITICAL!)'}")
print(f"   personId: {'✓' if has_personid else '✗ MISSING (CRITICAL!)'}")
print(f"   date column: {'✓' if has_date else '✗ MISSING (CRITICAL!)'}")

if has_home:
    print(f"   home values: {ps['home'].value_counts().to_dict()}")
    print(f"   home dtype: {ps['home'].dtype}")

# Test season_end_year calculation
if has_date:
    date_col = [c for c in ps.columns if 'date' in c.lower()][0]
    print(f"\n3. Testing season_end_year calculation from {date_col}...")
    
    ps[date_col] = pd.to_datetime(ps[date_col], errors='coerce')
    
    def _season_from_date(dt):
        if pd.api.types.is_datetime64_any_dtype(dt):
            d = dt
        else:
            d = pd.to_datetime(dt, errors="coerce", utc=False)
        y = d.dt.year
        m = d.dt.month
        return np.where(m >= 8, y + 1, y)
    
    ps['season_end_year'] = _season_from_date(ps[date_col])
    
    non_null = ps['season_end_year'].notna().sum()
    print(f"   season_end_year populated: {non_null} / {len(ps)} ({non_null/len(ps)*100:.1f}%)")
    print(f"   Season range: {ps['season_end_year'].min():.0f} - {ps['season_end_year'].max():.0f}")
    
    # Check unique seasons
    seasons = sorted(ps['season_end_year'].dropna().unique())
    print(f"   Unique seasons (sample): {seasons[:10]}")

# Check if priors exist
print("\n4. Checking Basketball Reference priors...")
priors_dir = Path("priors_data")
if priors_dir.exists():
    priors_files = list(priors_dir.glob("*.csv"))
    print(f"   Found {len(priors_files)} prior files:")
    for f in priors_files:
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"     • {f.name} ({size_mb:.1f} MB)")
    
    # Try loading one prior file
    if priors_files:
        sample_prior = priors_files[0]
        print(f"\n   Loading sample: {sample_prior.name}...")
        prior_df = pd.read_csv(sample_prior, nrows=100)
        print(f"     Rows (sample): {len(prior_df):,}")
        print(f"     Columns: {list(prior_df.columns)[:15]}...")
        
        # Check for season column
        season_cols = [c for c in prior_df.columns if 'season' in c.lower()]
        print(f"     Season columns: {season_cols}")
else:
    print("   ✗ priors_data directory not found")
    print("   → Priors will not be merged (models will still work)")

# Test home flag merge simulation
if has_home:
    print("\n5. Testing home flag conversion...")
    h = ps['home']
    hnum = pd.to_numeric(h, errors="coerce")
    is_home = np.where(
        hnum.notna(),
        (hnum.fillna(0) != 0).astype(int),
        h.astype(str).str.strip().str.lower().isin(["1", "true", "t", "home", "h", "yes", "y"]).astype(int)
    )
    ps['is_home'] = is_home
    
    print(f"   is_home value counts: {pd.Series(is_home).value_counts().to_dict()}")
    print(f"   Non-null is_home: {pd.Series(is_home).notna().sum()} / {len(ps)}")

print("\n" + "=" * 70)
print("VERDICT:")
print("=" * 70)

issues = []
warnings = []

if not has_personid:
    issues.append("✗ CRITICAL: personId column missing - training will fail")
if not has_date:
    issues.append("✗ CRITICAL: date column missing - season features unavailable")
if not has_home and not has_teamid:
    issues.append("✗ CRITICAL: Both home and teamId missing - cannot merge game context")
elif not has_teamid:
    warnings.append("⚠ teamId missing but home exists - will use home flag (OK)")

if has_date:
    null_pct = (1 - ps['season_end_year'].notna().sum() / len(ps)) * 100
    if null_pct > 50:
        warnings.append(f"⚠ {null_pct:.1f}% of dates are null - check date format")

if not priors_dir.exists():
    warnings.append("⚠ No Basketball Reference priors - accuracy will be lower")

if issues:
    print("\nCRITICAL ISSUES (must fix):")
    for issue in issues:
        print(f"  {issue}")
elif warnings:
    print("\nWARNINGS (can proceed but review):")
    for warning in warnings:
        print(f"  {warning}")
else:
    print("\n✅ All checks passed! Ready to train player models.")

print("=" * 70)

# Quick recommendation
print("\nNEXT STEPS:")
if issues:
    print("  1. Fix critical issues above")
    print("  2. Re-run this test")
    print("  3. Then run full training")
else:
    print("  ✓ Run full training with:")
    print("    python train_auto.py --dataset <your-dataset> --verbose")
    if warnings:
        print("\n  Note: Warnings above are informational - training should work")

print("=" * 70)
