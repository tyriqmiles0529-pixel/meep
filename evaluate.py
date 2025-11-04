#!/usr/bin/env python3
"""
NBA Predictor - Post-Game Evaluation Pipeline

Consolidates 3 steps into 1 command:
1. Fetch actual results (fetch_bet_results_incremental.py)
2. Recalibrate models (recalibrate_models.py)
3. Analyze performance (analyze_ledger.py)

Usage:
    python evaluate.py                     # Run all steps (auto-retry enabled)
    python evaluate.py --no-retry          # Disable auto-retry (single fetch run)
    python evaluate.py --fetch-only        # Just fetch results (auto-retry enabled)
    python evaluate.py --analyze-only      # Just analyze performance
    python evaluate.py --recalibrate       # Fetch + recalibrate (skip analysis)
    
Customization:
    python evaluate.py --retry-delay 30    # 30s delay between retries (default: 60)
    python evaluate.py --max-fetches 50    # 50 API calls per batch (default: 100)
    python evaluate.py --no-retry          # Disable auto-retry completely
"""

import os
import sys
import pickle
import json
import argparse
import time
from datetime import datetime, timezone, date
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression

# ============================================================================
# STEP 1: FETCH ACTUAL RESULTS
# ============================================================================

def fetch_actual_results(verbose: bool = True, max_fetches: int = 100) -> int:
    """
    Fetch actual player stats for unsettled bets.
    Returns number of bets settled.
    
    Args:
        verbose: Print progress
        max_fetches: Max API calls per run (rate limit protection)
    """
    if verbose:
        print("\n" + "="*70)
        print("STEP 1: FETCHING ACTUAL RESULTS")
        print("="*70)
    
    try:
        from nba_api.stats.endpoints import playergamelog
        from nba_api.stats.static import players as nba_players
    except ImportError:
        print("❌ nba_api not installed. Run: pip install nba_api")
        return 0
    
    # Load ledger
    ledger_file = "bets_ledger.pkl"
    if not os.path.exists(ledger_file):
        if verbose:
            print("⚠ No bets ledger found. Run riq_analyzer.py first.")
        return 0
    
    with open(ledger_file, "rb") as f:
        ledger = pickle.load(f)
    
    bets = ledger.get("bets", [])
    if not bets:
        if verbose:
            print("⚠ No bets in ledger")
        return 0
    
    # Find unsettled bets
    unsettled = [b for b in bets if not b.get("settled", False)]
    if not unsettled:
        if verbose:
            print("✅ All bets already settled!")
        return 0
    
    if verbose:
        print(f"📊 Total bets: {len(bets):,}")
        print(f"📋 Unsettled: {len(unsettled):,}")
        print(f"✅ Settled: {len(bets) - len(unsettled):,}")
    
    settled_count = 0
    api_calls = 0
    
    for bet in unsettled:
        # Rate limit protection
        if api_calls >= max_fetches:
            if verbose:
                print(f"\n⚠️ Reached API call limit ({max_fetches})")
                print(f"   Settled {settled_count} bets this run")
                print(f"   Remaining unsettled: {len(unsettled) - settled_count}")
            break
        # Only settle past games
        game_date = bet.get("game_date")
        if not game_date:
            continue
        
        try:
            dt_val = pd.to_datetime(game_date, errors='coerce')
            if dt_val is None or pd.isna(dt_val):
                continue
            
            # Skip future games
            if dt_val.tz_localize(None) > pd.Timestamp.now():
                continue
        except:
            continue
        
        # Fetch actual stat
        player_name = bet.get("player", "")
        prop_type = bet.get("prop_type", "")
        
        if not player_name or not prop_type:
            continue
        
        actual = _fetch_player_stat(player_name, prop_type, str(game_date))
        api_calls += 1  # Count API call
        
        if actual is None:
            continue
        
        # Determine win/loss
        line = float(bet.get("line", 0.0))
        pick = str(bet.get("pick", ""))
        
        if actual == line:
            # Push
            bet.update({"settled": True, "actual": actual, "won": None})
            settled_count += 1
            continue
        
        if pick == "over":
            won = actual > line
        else:
            won = actual < line
        
        bet.update({"settled": True, "actual": actual, "won": bool(won)})
        settled_count += 1
        
        if verbose and settled_count % 10 == 0:
            print(f"   Settled {settled_count} bets...")
    
    # Save updated ledger
    with open(ledger_file, "wb") as f:
        pickle.dump(ledger, f)
    
    if verbose:
        print(f"\n✅ Settled {settled_count} bets")
        print(f"📁 Updated: {ledger_file}")
    
    return settled_count


def _fetch_player_stat(player_name: str, prop_type: str, date_str: str) -> Optional[float]:
    """Fetch actual player stat for the given date using nba_api."""
    try:
        from nba_api.stats.endpoints import playergamelog
        from nba_api.stats.static import players as nba_players
        
        # Resolve player ID
        plist = nba_players.get_players()
        pid = None
        for p in plist:
            if str(p.get('full_name', '')).lower() == str(player_name).lower():
                pid = p.get('id')
                break
        
        if pid is None:
            return None
        
        # Parse date and derive season
        dt_val = pd.to_datetime(date_str, errors='coerce')
        if dt_val is None or pd.isna(dt_val):
            return None
        
        season = _season_str_from_date(dt_val.date())
        
        # Fetch game log
        gl = playergamelog.PlayerGameLog(player_id=pid, season=season, season_type_all_star='Regular Season')
        df = gl.get_data_frames()[0]
        
        if df is None or df.empty:
            return None
        
        # Match by date
        target = pd.to_datetime(df['GAME_DATE'], errors='coerce').dt.date
        mask = target == dt_val.date()
        row = df.loc[mask]
        
        if row.empty:
            return None
        
        # Get stat column
        stat_col_map = {"points": "PTS", "assists": "AST", "rebounds": "REB", "threes": "FG3M", "minutes": "MIN"}
        col = stat_col_map.get(prop_type)
        
        if col is None or col not in row.columns:
            return None
        
        val = row.iloc[0][col]
        
        # Handle minutes format
        if col == "MIN":
            s = str(val)
            if ":" in s:
                mm, ss = s.split(":")
                return float(mm) + float(ss)/60.0
            return float(val)
        
        return float(val)
    
    except Exception:
        return None


def _season_str_from_date(d: date) -> str:
    """Convert date to NBA season string (e.g., '2024-25')."""
    end_year = d.year + (1 if d.month >= 8 else 0)
    return f"{end_year-1}-{str(end_year)[-2:]}"


# ============================================================================
# STEP 2: RECALIBRATE MODELS
# ============================================================================

def recalibrate_models(min_samples: int = 200, verbose: bool = True) -> bool:
    """
    Recalibrate prediction probabilities using isotonic regression.
    Returns True if recalibration succeeded.
    """
    if verbose:
        print("\n" + "="*70)
        print("STEP 2: RECALIBRATING MODELS")
        print("="*70)
    
    ledger_file = "bets_ledger.pkl"
    if not os.path.exists(ledger_file):
        if verbose:
            print("⚠ No ledger found")
        return False
    
    with open(ledger_file, "rb") as f:
        ledger = pickle.load(f)
    
    bets = ledger.get("bets", [])
    df = pd.DataFrame(bets)
    
    if df.empty:
        if verbose:
            print("⚠ No bets to calibrate")
        return False
    
    # Filter to settled bets only (exclude pushes)
    df = df[df["won"].notna()].copy()
    
    if len(df) < min_samples:
        if verbose:
            print(f"⚠ Not enough settled bets: {len(df)} < {min_samples}")
            print(f"   Need {min_samples - len(df)} more settled predictions")
        return False
    
    if verbose:
        print(f"📊 Calibrating with {len(df):,} settled predictions")
    
    # Train isotonic regression per prop type
    calibration = {}
    
    for prop_type, grp in df.groupby("prop_type"):
        if len(grp) < 50:  # Need minimum samples per type
            continue
        
        p_pred = pd.to_numeric(grp["predicted_prob"], errors="coerce").clip(0.01, 0.99)
        y = grp["won"].astype(int)
        
        # Sort for isotonic regression
        sorted_idx = np.argsort(p_pred)
        p_sorted = p_pred.iloc[sorted_idx].values
        y_sorted = y.iloc[sorted_idx].values
        
        # Bin into deciles for smoother calibration
        bins = np.linspace(0.05, 0.95, 11)
        inds = np.digitize(p_sorted, bins, right=True)
        
        xs, ys = [], []
        for i in range(1, len(bins)):
            mask = inds == i
            if mask.any():
                xs.append(float(bins[i-1]))
                ys.append(float(y_sorted[mask].mean()))
        
        if len(xs) >= 2:
            calibration[prop_type] = {"bins": xs, "vals": ys}
            if verbose:
                print(f"   ✅ {prop_type}: {len(grp):,} samples → {len(xs)} bins")
        else:
            if verbose:
                print(f"   ⚠ {prop_type}: Not enough bins ({len(xs)})")
    
    if not calibration:
        if verbose:
            print("⚠ No calibration curves created")
        return False
    
    # Save calibration
    calibration_file = "calibration.pkl"
    with open(calibration_file, "wb") as f:
        pickle.dump(calibration, f)
    
    if verbose:
        print(f"\n✅ Calibrated {len(calibration)} prop types")
        print(f"📁 Saved to: {calibration_file}")
    
    return True


# ============================================================================
# STEP 3: ANALYZE PERFORMANCE
# ============================================================================

def analyze_performance(verbose: bool = True) -> Dict[str, Any]:
    """
    Analyze betting performance and model accuracy.
    Returns metrics dictionary.
    """
    if verbose:
        print("\n" + "="*70)
        print("STEP 3: ANALYZING PERFORMANCE")
        print("="*70)
    
    ledger_file = "bets_ledger.pkl"
    if not os.path.exists(ledger_file):
        if verbose:
            print("⚠ No ledger found")
        return {}
    
    with open(ledger_file, "rb") as f:
        ledger = pickle.load(f)
    
    bets = ledger.get("bets", [])
    if not bets:
        if verbose:
            print("⚠ No bets in ledger")
        return {}
    
    df = pd.DataFrame(bets)
    
    # Overall stats
    total = len(df)
    settled = df["settled"].sum()
    unsettled = total - settled
    
    if verbose:
        print(f"\n📊 OVERALL STATISTICS")
        print(f"{'─'*70}")
        print(f"   Total predictions: {total:,}")
        print(f"   Settled: {settled:,} ({settled/total*100:.1f}%)")
        print(f"   Unsettled: {unsettled:,} ({unsettled/total*100:.1f}%)")
    
    # Settled bets only (exclude pushes)
    df_settled = df[df["won"].notna()].copy()
    
    if df_settled.empty:
        if verbose:
            print("\n⚠ No settled bets to analyze")
        return {"total": total, "settled": settled, "win_rate": 0.0}
    
    # Win rate
    wins = df_settled["won"].sum()
    losses = len(df_settled) - wins
    win_rate = wins / len(df_settled) if len(df_settled) > 0 else 0.0
    
    if verbose:
        print(f"\n🎯 WIN RATE")
        print(f"{'─'*70}")
        print(f"   Wins: {wins:,}")
        print(f"   Losses: {losses:,}")
        print(f"   Win Rate: {win_rate*100:.1f}%")
        
        if win_rate >= 0.52:
            print(f"   Status: ✅ PROFITABLE (>52%)")
        elif win_rate >= 0.50:
            print(f"   Status: 🟡 BREAKEVEN (50-52%)")
        else:
            print(f"   Status: 🔴 UNPROFITABLE (<50%)")
    
    # By prop type
    if verbose:
        print(f"\n📈 BY PROP TYPE")
        print(f"{'─'*70}")
        print(f"   {'Type':<12} {'Count':<8} {'Win Rate':<12} {'Status'}")
        print(f"   {'─'*12} {'─'*8} {'─'*12} {'─'*10}")
    
    type_stats = {}
    for prop_type, grp in df_settled.groupby("prop_type"):
        type_wins = grp["won"].sum()
        type_count = len(grp)
        type_win_rate = type_wins / type_count if type_count > 0 else 0.0
        
        status = "✅" if type_win_rate >= 0.52 else ("🟡" if type_win_rate >= 0.50 else "🔴")
        
        type_stats[prop_type] = {
            "count": type_count,
            "wins": type_wins,
            "win_rate": type_win_rate
        }
        
        if verbose:
            print(f"   {prop_type:<12} {type_count:<8} {type_win_rate*100:>6.1f}%      {status}")
    
    # Confidence buckets
    if "win_prob" in df_settled.columns:
        if verbose:
            print(f"\n🎲 BY CONFIDENCE BUCKET")
            print(f"{'─'*70}")
            print(f"   {'Confidence':<15} {'Count':<8} {'Actual Win %':<15} {'Calibration'}")
            print(f"   {'─'*15} {'─'*8} {'─'*15} {'─'*12}")
        
        buckets = [
            (0, 55, "50-55%"),
            (55, 60, "55-60%"),
            (60, 65, "60-65%"),
            (65, 70, "65-70%"),
            (70, 100, "70%+")
        ]
        
        for low, high, label in buckets:
            mask = (df_settled["win_prob"] >= low) & (df_settled["win_prob"] < high)
            bucket_df = df_settled[mask]
            
            if len(bucket_df) > 0:
                bucket_wins = bucket_df["won"].sum()
                bucket_win_rate = bucket_wins / len(bucket_df)
                expected = (low + high) / 2 / 100
                diff = bucket_win_rate - expected
                
                if abs(diff) < 0.05:
                    calib = "✅ Good"
                elif diff > 0:
                    calib = "⚠ Overperform"
                else:
                    calib = "🔴 Underperform"
                
                if verbose:
                    print(f"   {label:<15} {len(bucket_df):<8} {bucket_win_rate*100:>6.1f}%          {calib}")
    
    metrics = {
        "total": total,
        "settled": settled,
        "unsettled": unsettled,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "by_type": type_stats
    }
    
    if verbose:
        print(f"\n{'─'*70}")
        print(f"✅ Analysis complete")
    
    return metrics


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="NBA Predictor - Post-Game Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py                       # Run all (auto-retry ON by default)
  python evaluate.py --fetch-only          # Just fetch (auto-retry ON)
  python evaluate.py --analyze-only        # Just analyze
  python evaluate.py --no-retry            # Disable auto-retry (single run)
  
Customization:
  python evaluate.py --retry-delay 30      # 30s delay between retries
  python evaluate.py --max-fetches 50      # 50 API calls per batch
        """
    )
    
    parser.add_argument("--fetch-only", action="store_true",
                        help="Only fetch results (skip recalibrate and analyze)")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only analyze performance (skip fetch and recalibrate)")
    parser.add_argument("--recalibrate", action="store_true",
                        help="Fetch and recalibrate (skip analysis)")
    parser.add_argument("--min-samples", type=int, default=200,
                        help="Minimum samples needed for recalibration (default: 200)")
    parser.add_argument("--no-retry", action="store_true",
                        help="Disable auto-retry (single fetch run only)")
    parser.add_argument("--retry-delay", type=int, default=60,
                        help="Seconds to wait between retry attempts (default: 60)")
    parser.add_argument("--max-fetches", type=int, default=100,
                        help="Max API calls per batch (default: 100)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress output")
    
    args = parser.parse_args()
    verbose = not args.quiet
    auto_retry = not args.no_retry  # Auto-retry enabled by default
    
    if verbose:
        print("=" * 70)
        print("NBA PREDICTOR - POST-GAME EVALUATION PIPELINE")
        print("=" * 70)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if auto_retry:
            print(f"Auto-retry: ENABLED (delay: {args.retry_delay}s, batch: {args.max_fetches})")
        else:
            print(f"Auto-retry: DISABLED (single run)")
    
    # Execute requested steps
    if args.fetch_only:
        if auto_retry:
            # Auto-retry until all settled
            total_settled = 0
            retry_count = 0
            
            while True:
                settled = fetch_actual_results(verbose=verbose, max_fetches=args.max_fetches)
                total_settled += settled
                
                if settled == 0:
                    if verbose:
                        print(f"\n✅ All bets settled! Total: {total_settled}")
                    break
                
                retry_count += 1
                if verbose:
                    print(f"\n⏳ Waiting {args.retry_delay} seconds before retry #{retry_count}...")
                time.sleep(args.retry_delay)
        else:
            fetch_actual_results(verbose=verbose, max_fetches=args.max_fetches)
    
    elif args.analyze_only:
        analyze_performance(verbose=verbose)
    
    elif args.recalibrate:
        if auto_retry:
            # Auto-retry fetch
            total_settled = 0
            retry_count = 0
            
            while True:
                settled = fetch_actual_results(verbose=verbose, max_fetches=args.max_fetches)
                total_settled += settled
                
                if settled == 0:
                    break
                
                retry_count += 1
                if verbose:
                    print(f"\n⏳ Waiting {args.retry_delay} seconds before retry #{retry_count}...")
                time.sleep(args.retry_delay)
            
            if total_settled > 0:
                recalibrate_models(min_samples=args.min_samples, verbose=verbose)
        else:
            settled = fetch_actual_results(verbose=verbose, max_fetches=args.max_fetches)
            if settled > 0:
                recalibrate_models(min_samples=args.min_samples, verbose=verbose)
    
    else:
        # Run all steps
        if auto_retry:
            # Auto-retry fetch
            total_settled = 0
            retry_count = 0
            
            while True:
                settled = fetch_actual_results(verbose=verbose, max_fetches=args.max_fetches)
                total_settled += settled
                
                if settled == 0:
                    break
                
                retry_count += 1
                if verbose:
                    print(f"\n⏳ Waiting {args.retry_delay} seconds before retry #{retry_count}...")
                time.sleep(args.retry_delay)
            
            if total_settled > 0:
                recalibrate_models(min_samples=args.min_samples, verbose=verbose)
        else:
            settled = fetch_actual_results(verbose=verbose, max_fetches=args.max_fetches)
            
            if settled > 0:
                recalibrate_models(min_samples=args.min_samples, verbose=verbose)
        
        analyze_performance(verbose=verbose)
    
    if verbose:
        print("\n" + "=" * 70)
        print("✅ EVALUATION COMPLETE")
        print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
