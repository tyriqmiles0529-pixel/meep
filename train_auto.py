#!/usr/bin/env python3
"""
End-to-end NBA training pipeline (always fetches from Kaggle).

What it trains
1) Game-level models (TeamStatistics)
   - Moneyline classifier: P(home wins), with isotonic probability calibration.
   - Spread regressor: expected margin (home - away), plus residual sigma for cover probabilities.
   - Time-safe OOF predictions to feed into player models (learn-from-each-other without leakage).

2) Player-level models (PlayerStatistics)
   - Minutes model (regression).
   - Points, Rebounds, Assists, 3PM models (regression).
   - Every player row includes BOTH team and opponent context, matchup edges, OOF game signals, and player rolling rates/minutes trends.
   - Works even if teamId/opponentTeamId are missing — uses 'home' flag to pick the correct side.

Key safety and robustness
- Leakage-safe team context: all rolling stats are shifted by 1 (pre-game only).
- Missing dates: never dropped; per-team order falls back to numeric gameId.
- Clean console output with grouped sections and rounded metrics.
- LightGBM noise reduced (force_col_wise=True, verbosity=-1); sklearn fallback available.

Practical era handling
- Season cutoffs for data inclusion (player and game).
- Season features in models (season_end_year, season_decade).
- Time-decay sample weights with lockout downweight (1999, 2012).

Optional data sources (independent of each other)
- Betting odds dataset: Market lines, spreads, totals, implied probabilities
- Basketball Reference statistical priors (7 CSVs):
  Team priors (2 CSVs): Team Summaries, Team Abbrev
  Player priors (4 CSVs): Per 100 Poss, Advanced, Player Shooting, Player Play By Play
  Note: "Priors" refers to STATISTICAL CONTEXT from past seasons, NOT betting odds

Run (PowerShell)
- pip install kagglehub
- python .\\train_auto.py --dataset "eoinamoore/historical-nba-data-and-player-box-scores" --verbose --skip-rest --fresh --lgb-log-period 50
"""

from __future__ import annotations

import sys
import os
import re
import json
import math
import argparse
import warnings
import requests
import gc
from train_ensemble_enhanced import train_all_ensemble_components
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from optimization_features import (
    MomentumAnalyzer,
    MetaWindowSelector,
    MarketSignalAnalyzer,
    EnsembleStacker,
    add_all_optimization_features,
    add_variance_features,
    add_ceiling_floor_features,
    add_context_weighted_averages,
    add_opponent_strength_features,
    add_fatigue_features
)
from neural_hybrid import NeuralHybridPredictor, TABNET_AVAILABLE, TORCH_AVAILABLE
from phase7_features import add_phase7_features

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, log_loss, brier_score_loss
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.calibration import CalibratedClassifierCV

# Global threads limit for learners (can be overridden via --n-jobs)
N_JOBS: int = -1

# Suppress pandas FutureWarnings for deprecated errors='ignore' parameter
warnings.filterwarnings('ignore', category=FutureWarning, message=".*errors='ignore' is deprecated.*")

# Prefer LightGBM; fallback to sklearn HistGBM
_HAS_LGB = False
try:
    import lightgbm as lgb  # type: ignore
    _HAS_LGB = True
    try:
        lgb.set_config(verbosity=-1)
    except Exception:
        pass
except Exception:
    pass

try:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
except Exception:  # pragma: no cover
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

# Kaggle (required)
try:
    import kagglehub
except Exception:
    kagglehub = None

# NBA API for real-time stats (optional - graceful fallback if unavailable)
try:
    from nba_api.stats.endpoints import LeagueDashTeamStats
    from nba_api.stats.static import teams as nba_teams
    _HAS_NBA_API = True
except Exception:
    _HAS_NBA_API = False

# Silence upcoming sklearn FutureWarning for CalibratedClassifierCV(cv='prefit')
warnings.filterwarnings(
    "ignore",
    message="The `cv='prefit'` option is deprecated",
    category=FutureWarning,
)

# ---------------- Feature version (invalidate cache when features change) ----------------
# Increment this whenever you add new features to force cache rebuild
FEATURE_VERSION = "5.0"  # Phase 5 features: position, starter status, injury tracking

# ---------------- Kaggle credentials (hardcoded for venv compatibility) ----------------
KAGGLE_KEY = "bcb440122af5ae76181e68d48ca728e6"

# ---------------- The Odds API credentials ----------------
THEODDS_API_KEY = os.getenv("THEODDS_API_KEY") or ""  # Set in keys.py or environment
KAGGLE_USERNAME = "tyriqmiles"

if KAGGLE_KEY and KAGGLE_KEY != "YOUR_KEY_HERE":
    os.environ['KAGGLE_KEY'] = KAGGLE_KEY
    if KAGGLE_USERNAME:
        os.environ['KAGGLE_USERNAME'] = KAGGLE_USERNAME

# ---------------- Pretty printing helpers ----------------

def _line(char: str = "─", n: int = 60) -> str:
    return char * n

def _sec(title: str) -> str:
    return f"\\n{_line()} \\n{title}\\n{_line()}"

def _fmt(n: float, nd: int = 3) -> str:
    try:
        return f"{n:.{nd}f}"
    except Exception:
        return str(n)

def log(msg: str, verbose: bool):
    if verbose:
        print(msg, flush=True)

# ---------------- Era helpers ----------------

LOCKOUT_SEASONS = {1999, 2012}

def _season_from_date(dt: pd.Series) -> pd.Series:
    """
    Convert a UTC-naive datetime to NBA season end-year.
    Season end-year = year if month <= 7; else year+1 (Aug..Dec map to next year's season).
    """
    # Don't re-parse if already datetime (causes NaT when format='mixed' not specified)
    if pd.api.types.is_datetime64_any_dtype(dt):
        d = dt
    else:
        d = pd.to_datetime(dt, errors="coerce", utc=False)

    y = d.dt.year
    m = d.dt.month
    return np.where(m >= 8, y + 1, y)

def _decade_from_season(season_end_year: pd.Series) -> pd.Series:
    s = pd.to_numeric(season_end_year, errors="coerce")
    return (s // 10) * 10

def _parse_season_cutoff(arg_val: str, kind: str) -> int:
    """
    kind='game' presets: 'classic'->1997, 'balanced'->2002
    kind='player' presets: 'min'->1998, 'balanced'->2002, 'modern'->2005
    Also accepts integer strings, e.g., '2001', '2005'.
    """
    v = str(arg_val).strip().lower()
    if kind == "game":
        presets = {"classic": 1997, "balanced": 2002}
    else:
        presets = {"min": 1998, "balanced": 2002, "modern": 2005}
    if v in presets:
        return presets[v]
    try:
        return int(v)
    except Exception:
        # Fallback to balanced defaults
        return presets["balanced"]

def _compute_sample_weights(seasons: np.ndarray, decay: float, min_weight: float, lockout_weight: float) -> np.ndarray:
    seasons = seasons.astype("float64")
    valid = np.isfinite(seasons)
    if not valid.any():
        return np.ones_like(seasons, dtype="float64")
    max_season = int(np.nanmax(seasons[valid]))
    base = np.ones_like(seasons, dtype="float64")
    base[valid] = decay ** (max_season - seasons[valid])
    base = np.maximum(base, min_weight)
    # Lockouts
    for lo in LOCKOUT_SEASONS:
        base = np.where(seasons == lo, base * lockout_weight, base)
    # For NaN seasons, keep weight at 1.0
    base = np.where(valid, base, 1.0)
    return base.astype("float64")

# ---------------- Game feature schema ----------------

GAME_FEATURES: List[str] = [
    "home_advantage", "neutral_site",
    "home_recent_pace", "away_recent_pace",
    "home_off_strength", "home_def_strength",
    "away_off_strength", "away_def_strength",
    "home_recent_winrate", "away_recent_winrate",
    # matchup features
    "match_off_edge", "match_def_edge", "match_pace_sum", "winrate_diff",
    # schedule/injury (often constants if --skip-rest)
    "home_days_rest", "away_days_rest",
    "home_b2b", "away_b2b",
    "home_injury_impact", "away_injury_impact",
    # era features
    "season_end_year", "season_decade",
    # betting market features (optional - from --odds-dataset)
    "market_implied_home", "market_implied_away",
    "market_spread", "spread_move",
    "market_total", "total_move",
    # Basketball Reference statistical priors (optional - from --priors-dataset)
    # These provide historical performance context, NOT betting odds
    "home_o_rtg_prior", "home_d_rtg_prior", "home_pace_prior",
    "away_o_rtg_prior", "away_d_rtg_prior", "away_pace_prior",
    "home_srs_prior", "away_srs_prior",
    # Four Factors (Dean Oliver's key team efficiency metrics)
    "home_efg_prior", "home_tov_pct_prior", "home_orb_pct_prior", "home_ftr_prior",
    "away_efg_prior", "away_tov_pct_prior", "away_orb_pct_prior", "away_ftr_prior",
    # Opponent Four Factors
    "home_opp_efg_prior", "home_opp_tov_pct_prior", "home_drb_pct_prior", "home_opp_ftr_prior",
    "away_opp_efg_prior", "away_opp_tov_pct_prior", "away_drb_pct_prior", "away_opp_ftr_prior",
    # Additional team stats
    "home_ts_pct_prior", "home_3par_prior", "home_mov_prior",
    "away_ts_pct_prior", "away_3par_prior", "away_mov_prior",
]

GAME_DEFAULTS: Dict[str, float] = {
    "home_advantage": 1.0,
    "neutral_site": 0.0,
    "home_recent_pace": 1.0,
    "away_recent_pace": 1.0,
    "home_off_strength": 1.0,
    "home_def_strength": 1.0,
    "away_off_strength": 1.0,
    "away_def_strength": 1.0,
    "home_recent_winrate": 0.5,
    "away_recent_winrate": 0.5,
    "match_off_edge": 0.0,
    "match_def_edge": 0.0,
    "match_pace_sum": 2.0,
    "winrate_diff": 0.0,
    "home_days_rest": 2.0,
    "away_days_rest": 2.0,
    "home_b2b": 0.0,
    "away_b2b": 0.0,
    "home_injury_impact": 0.0,
    "away_injury_impact": 0.0,
    # era defaults (will be filled from actual dates when present)
    "season_end_year": 2002.0,
    "season_decade": 2000.0,
    # betting market defaults (neutral/typical values when odds unavailable)
    "market_implied_home": 0.5,
    "market_implied_away": 0.5,
    "market_spread": 0.0,
    "spread_move": 0.0,
    "market_total": 210.0,  # typical NBA total
    "total_move": 0.0,
    # Basketball Reference statistical priors defaults (league-average baseline when unavailable)
    "home_o_rtg_prior": 110.0,  # ~league average offensive rating
    "home_d_rtg_prior": 110.0,  # ~league average defensive rating
    "home_pace_prior": 100.0,   # ~league average pace
    "away_o_rtg_prior": 110.0,
    "away_d_rtg_prior": 110.0,
    "away_pace_prior": 100.0,
    "home_srs_prior": 0.0,      # Simple Rating System (0 = average team)
    "away_srs_prior": 0.0,
    # Four Factors defaults (league averages)
    "home_efg_prior": 0.52,     # effective FG% ~52%
    "home_tov_pct_prior": 0.14, # turnover % ~14%
    "home_orb_pct_prior": 0.23, # offensive rebound % ~23%
    "home_ftr_prior": 0.24,     # free throw rate (FTA/FGA) ~24%
    "away_efg_prior": 0.52,
    "away_tov_pct_prior": 0.14,
    "away_orb_pct_prior": 0.23,
    "away_ftr_prior": 0.24,
    # Opponent Four Factors defaults
    "home_opp_efg_prior": 0.52,
    "home_opp_tov_pct_prior": 0.14,
    "home_drb_pct_prior": 0.77,  # defensive rebound % ~77%
    "home_opp_ftr_prior": 0.24,
    "away_opp_efg_prior": 0.52,
    "away_opp_tov_pct_prior": 0.14,
    "away_drb_pct_prior": 0.77,
    "away_opp_ftr_prior": 0.24,
    # Additional team stats defaults
    "home_ts_pct_prior": 0.56,   # true shooting % ~56%
    "home_3par_prior": 0.38,     # 3-point attempt rate ~38%
    "home_mov_prior": 0.0,       # margin of victory ~0 for avg team
    "away_ts_pct_prior": 0.56,
    "away_3par_prior": 0.38,
    "away_mov_prior": 0.0,
}

# ---------------- Utilities ----------------

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def _id_to_str(s: pd.Series) -> pd.Series:
    s_num = pd.to_numeric(s, errors="coerce")
    out = pd.Series(index=s.index, dtype="object")
    num_mask = s_num.notna()
    out.loc[num_mask] = s_num.loc[num_mask].astype("Int64").astype(str)
    out.loc[~num_mask] = s.astype(str)
    return out.astype(str).str.strip()

def _find_dataset_files(ds_root: Path) -> Tuple[Optional[Path], Optional[Path]]:
    teams_path: Optional[Path] = None
    players_path: Optional[Path] = None
    for p in ds_root.glob("*.csv"):
        n = p.name.lower()
        if "teamstatistics" in n or "team_statistics" in n:
            teams_path = p
        if "playerstatistics" in n or "player_stats" in n or "playerstats" in n:
            players_path = p
    # Fallback: scan names that start with Team/Player
    if teams_path is None:
        cand = [p for p in ds_root.glob("*.csv") if "team" in p.name.lower()]
        if cand:
            teams_path = cand[0]
    if players_path is None:
        cand = [p for p in ds_root.glob("*.csv") if "player" in p.name.lower()]
        if cand:
            players_path = cand[0]
    return teams_path, players_path

def _fresh_run_dir(base: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def _copy_if_exists(src: Optional[Path], dst_dir: Path) -> Optional[Path]:
    if src and src.exists():
        out = dst_dir / src.name
        out.write_bytes(src.read_bytes())
        return out
    return None

def _team_name_to_abbrev(team_name: str) -> str:
    """Convert team name to standard 3-letter abbreviation."""
    if pd.isna(team_name):
        return "UNK"
    
    name = str(team_name).strip().upper()
    
    # Simple mapping for current NBA teams
    team_map = {
        # Full names
        "ATLANTA HAWKS": "ATL", "BOSTON CELTICS": "BOS", "BROOKLYN NETS": "BRK",
        "CHARLOTTE HORNETS": "CHO", "CHICAGO BULLS": "CHI", "CLEVELAND CAVALIERS": "CLE",
        "DALLAS MAVERICKS": "DAL", "DENVER NUGGETS": "DEN", "DETROIT PISTONS": "DET",
        "GOLDEN STATE WARRIORS": "GSW", "HOUSTON ROCKETS": "HOU", "INDIANA PACERS": "IND",
        "LA CLIPPERS": "LAC", "LOS ANGELES CLIPPERS": "LAC", "LA LAKERS": "LAL",
        "LOS ANGELES LAKERS": "LAL", "MEMPHIS GRIZZLIES": "MEM", "MIAMI HEAT": "MIA",
        "MILWAUKEE BUCKS": "MIL", "MINNESOTA TIMBERWOLVES": "MIN", "NEW ORLEANS PELICANS": "NOP",
        "NEW YORK KNICKS": "NYK", "OKLAHOMA CITY THUNDER": "OKC", "ORLANDO MAGIC": "ORL",
        "PHILADELPHIA 76ERS": "PHI", "PHOENIX SUNS": "PHO", "PORTLAND TRAIL BLAZERS": "POR",
        "SACRAMENTO KINGS": "SAC", "SAN ANTONIO SPURS": "SAS", "TORONTO RAPTORS": "TOR",
        "UTAH JAZZ": "UTA", "WASHINGTON WIZARDS": "WAS",
        # Just mascot/city
        "HAWKS": "ATL", "ATLANTA": "ATL", "CELTICS": "BOS", "BOSTON": "BOS",
        "NETS": "BRK", "BROOKLYN": "BRK", "HORNETS": "CHO", "CHARLOTTE": "CHO",
        "BULLS": "CHI", "CHICAGO": "CHI", "CAVALIERS": "CLE", "CLEVELAND": "CLE",
        "MAVERICKS": "DAL", "DALLAS": "DAL", "NUGGETS": "DEN", "DENVER": "DEN",
        "PISTONS": "DET", "DETROIT": "DET", "WARRIORS": "GSW", "GOLDEN STATE": "GSW",
        "ROCKETS": "HOU", "HOUSTON": "HOU", "PACERS": "IND", "INDIANA": "IND",
        "CLIPPERS": "LAC", "LAKERS": "LAL", "GRIZZLIES": "MEM", "MEMPHIS": "MEM",
        "HEAT": "MIA", "MIAMI": "MIA", "BUCKS": "MIL", "MILWAUKEE": "MIL",
        "TIMBERWOLVES": "MIN", "MINNESOTA": "MIN", "PELICANS": "NOP", "NEW ORLEANS": "NOP",
        "KNICKS": "NYK", "NEW YORK": "NYK", "THUNDER": "OKC", "OKLAHOMA CITY": "OKC",
        "MAGIC": "ORL", "ORLANDO": "ORL", "76ERS": "PHI", "PHILADELPHIA": "PHI",
        "SUNS": "PHO", "PHOENIX": "PHO", "TRAIL BLAZERS": "POR", "PORTLAND": "POR",
        "KINGS": "SAC", "SACRAMENTO": "SAC", "SPURS": "SAS", "SAN ANTONIO": "SAS",
        "RAPTORS": "TOR", "TORONTO": "TOR", "JAZZ": "UTA", "UTAH": "UTA",
        "WIZARDS": "WAS", "WASHINGTON": "WAS",
        # Historical teams
        "NEW JERSEY NETS": "NJN", "SEATTLE SUPERSONICS": "SEA", "VANCOUVER GRIZZLIES": "VAN",
        "CHARLOTTE BOBCATS": "CHA", "NEW ORLEANS HORNETS": "NOH", "NEW ORLEANS/OKLAHOMA CITY HORNETS": "NOK",
    }
    
    # Try exact match first
    if name in team_map:
        return team_map[name]
    
    # Try to match by checking if any key is in the name
    for key, abbrev in team_map.items():
        if key in name:
            return abbrev
    
    # Fallback: try to extract 3-letter code if it looks like one
    parts = name.split()
    if len(parts) > 0:
        last = parts[-1]
        if len(last) <= 3 and last.isalpha():
            return last.upper()
    
    return "UNK"

# ---------------- Fetch current season completed games from NBA API ----------------

def fetch_current_season_games(season: str = "2025-26", verbose: bool = False) -> Optional[pd.DataFrame]:
    """
    Fetch completed games from the current NBA season using nba_api.
    Returns DataFrame in the same format as historical TeamStatistics CSV.

    Args:
        season: NBA season string (e.g., "2025-26")
        verbose: Print debug info

    Returns:
        DataFrame with columns: gameId, date, teamId, opponentTeamId, home, teamScore, opponentScore, teamTricode
        Returns None if nba_api is unavailable or no games found.
    """
    if not _HAS_NBA_API:
        log("- NBA API not available, skipping current season data", verbose)
        return None

    try:
        import time
        from nba_api.stats.endpoints import LeagueGameLog

        log(f"- Fetching completed games from {season} season via nba_api...", verbose)

        # Fetch game log for current season (both teams per game)
        time.sleep(0.6)  # Rate limiting
        game_log = LeagueGameLog(
            season=season,
            season_type_all_star='Regular Season',
            player_or_team_abbreviation='T'  # Team
        )
        df = game_log.get_data_frames()[0]

        if df.empty:
            log(f"- No completed games found for {season} season", verbose)
            return None

        log(f"- Fetched {len(df)} team-game records from nba_api", verbose)

        # Convert to our format
        # nba_api columns: TEAM_ID, TEAM_ABBREVIATION, GAME_ID, GAME_DATE, MATCHUP, WL, PTS, etc.
        result = pd.DataFrame()

        # Extract game ID (format: "0022500123")
        result['gameId'] = df['GAME_ID'].astype(str)

        # Parse date
        result['date'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')

        # Team ID
        result['teamId'] = df['TEAM_ID'].astype(str)

        # Extract opponent team ID from MATCHUP (e.g., "LAL @ GSW" or "LAL vs. GSW")
        # We'll need to look up opponent by matching games
        result['teamTricode'] = df['TEAM_ABBREVIATION']

        # Home flag: "vs." means home, "@" means away
        result['home'] = df['MATCHUP'].str.contains('vs.', na=False).astype(int)

        # Team score
        result['teamScore'] = pd.to_numeric(df['PTS'], errors='coerce')

        # For opponent info, we need to join games with themselves
        # Group by GAME_ID and pair home/away teams
        games_grouped = result.groupby('gameId')

        final_rows = []
        for gid, group in games_grouped:
            if len(group) != 2:
                continue  # Skip if not exactly 2 teams (shouldn't happen)

            home_row = group[group['home'] == 1]
            away_row = group[group['home'] == 0]

            if home_row.empty or away_row.empty:
                continue

            # Create home team row
            home_dict = home_row.iloc[0].to_dict()
            home_dict['opponentTeamId'] = away_row.iloc[0]['teamId']
            home_dict['opponentScore'] = away_row.iloc[0]['teamScore']
            final_rows.append(home_dict)

            # Create away team row
            away_dict = away_row.iloc[0].to_dict()
            away_dict['opponentTeamId'] = home_row.iloc[0]['teamId']
            away_dict['opponentScore'] = home_row.iloc[0]['teamScore']
            final_rows.append(away_dict)

        if not final_rows:
            log(f"- No valid game pairs found in {season} data", verbose)
            return None

        result_df = pd.DataFrame(final_rows)

        # Clean up columns to match historical format
        result_df = result_df[['gameId', 'date', 'teamId', 'opponentTeamId', 'home',
                                'teamScore', 'opponentScore', 'teamTricode']]

        log(f"- Successfully formatted {len(result_df)} team-game records from {season}", verbose)
        return result_df

    except Exception as e:
        log(f"- Error fetching current season games: {e}", verbose)
        import traceback
        if verbose:
            traceback.print_exc()
        return None


def load_or_fetch_historical_odds(games_df: pd.DataFrame, api_key: str, cache_path: Path, verbose: bool = False, max_requests: int = 50) -> pd.DataFrame:
    """
    Load historical odds from cache or fetch missing dates from The Odds API.
    Uses intelligent caching to minimize API costs (10 credits per date!).

    Args:
        games_df: DataFrame with games (must have 'date' column)
        api_key: The Odds API key
        cache_path: Path to CSV cache file
        verbose: Print debug info
        max_requests: Maximum number of API requests to make (default 50 = 500 credits)

    Returns:
        DataFrame with historical odds merged by date + teams
    """
    # Load existing cache if available
    if cache_path.exists():
        log(f"- Loading historical odds cache from {cache_path}", verbose)
        cached_odds = pd.read_csv(cache_path, parse_dates=['date'])
        log(f"- Loaded {len(cached_odds)} cached odds records", verbose)
    else:
        cached_odds = pd.DataFrame()
        log("- No odds cache found, will fetch from API", verbose)

    # Get unique game dates from games_df
    games_df['date'] = pd.to_datetime(games_df['date'], errors='coerce')

    # Filter to only dates from June 2020 onwards (when historical odds API data starts)
    odds_start_date = pd.Timestamp('2020-06-01')
    games_df_filtered = games_df[games_df['date'] >= odds_start_date]
    unique_dates = games_df_filtered['date'].dt.date.dropna().unique()
    unique_dates_sorted = sorted(unique_dates)

    log(f"- Found {len(unique_dates_sorted)} unique game dates from June 2020 onwards (historical odds available)", verbose)

    # Determine which dates need fetching
    if not cached_odds.empty:
        cached_odds['date'] = pd.to_datetime(cached_odds['date'], errors='coerce')
        cached_dates = set(cached_odds['date'].dt.date.unique())
        missing_dates = [d for d in unique_dates_sorted if d not in cached_dates]
    else:
        missing_dates = unique_dates_sorted

    log(f"- Need to fetch odds for {len(missing_dates)} dates", verbose)

    # Fetch missing dates (with limit to control costs)
    if missing_dates and len(missing_dates) > max_requests:
        log(f"- WARNING: {len(missing_dates)} dates missing, but max_requests={max_requests}", verbose)
        log(f"- Fetching most recent {max_requests} dates only", verbose)
        missing_dates = missing_dates[-max_requests:]  # Fetch most recent dates

    new_odds_list = []
    for i, game_date in enumerate(missing_dates):
        date_str = game_date.strftime("%Y-%m-%d")
        log(f"- Fetching {i+1}/{len(missing_dates)}: {date_str}", verbose)

        odds_df = fetch_historical_odds_for_date(date_str, api_key, verbose=verbose)
        if odds_df is not None and not odds_df.empty:
            new_odds_list.append(odds_df)

    # Combine cached + new odds
    all_odds_frames = [cached_odds] if not cached_odds.empty else []
    if new_odds_list:
        all_odds_frames.extend(new_odds_list)

    if not all_odds_frames:
        log("- No historical odds available", verbose)
        return pd.DataFrame()

    combined_odds = pd.concat(all_odds_frames, ignore_index=True)

    # Save updated cache
    # Ensure parent directory exists before saving
    import os
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    combined_odds.to_csv(cache_path, index=False)
    log(f"- Saved {len(combined_odds)} odds records to cache: {cache_path}", verbose)

    return combined_odds


def fetch_historical_odds_for_date(game_date: str, api_key: str, verbose: bool = False) -> Optional[pd.DataFrame]:
    """
    Fetch historical odds from The Odds API for a specific date.
    Returns DataFrame with closing lines (spreads, totals, moneylines) for that date's games.

    Args:
        game_date: Date string in YYYY-MM-DD format
        api_key: The Odds API key
        verbose: Print debug info

    Returns:
        DataFrame with odds data, or None if unavailable/error

    Note: Historical odds cost 10 credits per request (expensive!)
    """
    try:
        import requests
        import time

        # Convert date to ISO 8601 timestamp (11 PM ET on game date for closing lines)
        # NBA games typically start 7-10 PM ET, so 11 PM captures most closing lines
        timestamp = f"{game_date}T23:00:00Z"

        url = f"https://api.the-odds-api.com/v4/historical/sports/basketball_nba/odds"
        params = {
            "apiKey": api_key,
            "regions": "us",  # US bookmakers only to save credits
            "markets": "h2h,spreads,totals",  # Game markets only (no player props)
            "oddsFormat": "american",
            "date": timestamp
        }

        log(f"- Fetching historical odds for {game_date}...", verbose)
        time.sleep(0.3)  # Rate limiting

        resp = requests.get(url, params=params, timeout=120)

        if resp.status_code != 200:
            log(f"- Historical odds fetch failed: {resp.status_code} - {resp.text[:200]}", verbose)
            return None

        data = resp.json()
        events = data.get("data", [])

        if not events:
            log(f"- No historical odds found for {game_date}", verbose)
            return None

        log(f"- Fetched {len(events)} games with odds for {game_date}", verbose)

        # Parse odds into DataFrame
        rows = []
        for event in events:
            home_team = event.get("home_team", "")
            away_team = event.get("away_team", "")
            commence_time = event.get("commence_time", "")

            # Parse date from commence_time
            event_date = pd.to_datetime(commence_time, errors='coerce')

            bookmakers = event.get("bookmakers", [])
            if not bookmakers:
                continue

            # Aggregate odds across bookmakers (use consensus/average)
            spreads_home = []
            spreads_away = []
            totals_over = []
            totals_under = []
            ml_home = []
            ml_away = []

            for book in bookmakers:
                markets = book.get("markets", [])
                for market in markets:
                    market_key = market.get("key", "")
                    outcomes = market.get("outcomes", [])

                    if market_key == "spreads":
                        for outcome in outcomes:
                            if outcome.get("name") == home_team:
                                spreads_home.append(outcome.get("point", 0))
                            elif outcome.get("name") == away_team:
                                spreads_away.append(outcome.get("point", 0))

                    elif market_key == "totals":
                        for outcome in outcomes:
                            point = outcome.get("point")
                            if outcome.get("name", "").lower() == "over":
                                totals_over.append(point)
                            elif outcome.get("name", "").lower() == "under":
                                totals_under.append(point)

                    elif market_key == "h2h":
                        for outcome in outcomes:
                            if outcome.get("name") == home_team:
                                ml_home.append(outcome.get("price", 0))
                            elif outcome.get("name") == away_team:
                                ml_away.append(outcome.get("price", 0))

            # Calculate consensus values (median to avoid outliers)
            row = {
                "date": event_date,
                "home_team": home_team,
                "away_team": away_team,
                "market_spread": np.median(spreads_home) if spreads_home else 0.0,
                "market_total": np.median(totals_over) if totals_over else 0.0,
                "market_ml_home": int(np.median(ml_home)) if ml_home else 0,
                "market_ml_away": int(np.median(ml_away)) if ml_away else 0,
            }

            # Calculate implied probabilities from moneyline
            if row["market_ml_home"] != 0:
                if row["market_ml_home"] > 0:
                    row["market_implied_home"] = 100 / (row["market_ml_home"] + 100)
                else:
                    row["market_implied_home"] = abs(row["market_ml_home"]) / (abs(row["market_ml_home"]) + 100)
            else:
                row["market_implied_home"] = 0.5

            row["market_implied_away"] = 1.0 - row["market_implied_home"]

            rows.append(row)

        if not rows:
            return None

        df = pd.DataFrame(rows)
        log(f"- Parsed {len(df)} games with odds", verbose)
        return df

    except Exception as e:
        log(f"- Error fetching historical odds: {e}", verbose)
        import traceback
        if verbose:
            traceback.print_exc()
        return None


def fetch_historical_player_props_for_date(game_date: str, api_key: str, verbose: bool = False) -> Optional[pd.DataFrame]:
    """
    Fetch historical player prop odds from The Odds API for a specific date.
    Returns DataFrame with player prop lines (points, rebounds, assists, threes).

    Args:
        game_date: Date in YYYY-MM-DD format
        api_key: The Odds API key
        verbose: Whether to print debug info

    Returns:
        DataFrame with columns: date, player_name, team, opponent, prop_type, line, over_odds, under_odds
    """
    try:
        import time

        # Use 11 PM ET for closing lines (same as game odds)
        timestamp = f"{game_date}T23:00:00Z"

        url = f"https://api.the-odds-api.com/v4/historical/sports/basketball_nba/odds"
        params = {
            "apiKey": api_key,
            "regions": "us",
            "markets": "player_points,player_rebounds,player_assists,player_threes",
            "oddsFormat": "american",
            "date": timestamp
        }

        log(f"- Fetching player props for {game_date}...", verbose)
        time.sleep(0.6)  # Rate limiting

        response = requests.get(url, params=params, timeout=120)

        if response.status_code != 200:
            log(f"- API returned status {response.status_code} for {game_date}", verbose)
            return None

        data = response.json()

        if not data or 'data' not in data:
            log(f"- No player prop data for {game_date}", verbose)
            return None

        games = data.get('data', [])

        # Parse player props
        rows = []
        for game in games:
            home_team = game.get('home_team', '')
            away_team = game.get('away_team', '')
            commence_time = game.get('commence_time', game_date)

            bookmakers = game.get('bookmakers', [])

            for bookmaker in bookmakers:
                markets = bookmaker.get('markets', [])

                for market in markets:
                    market_key = market.get('key', '')

                    # Only process player prop markets
                    if not market_key.startswith('player_'):
                        continue

                    # Map market key to prop type
                    prop_type_map = {
                        'player_points': 'points',
                        'player_rebounds': 'rebounds',
                        'player_assists': 'assists',
                        'player_threes': 'threes'
                    }
                    prop_type = prop_type_map.get(market_key, market_key)

                    outcomes = market.get('outcomes', [])

                    # Group by player (each player has Over and Under outcomes)
                    player_props = {}
                    for outcome in outcomes:
                        player_name = outcome.get('description', '')
                        line = outcome.get('point')
                        odds = outcome.get('price')
                        outcome_type = outcome.get('name', '')  # 'Over' or 'Under'

                        if player_name not in player_props:
                            player_props[player_name] = {'line': line}

                        if outcome_type == 'Over':
                            player_props[player_name]['over_odds'] = odds
                        elif outcome_type == 'Under':
                            player_props[player_name]['under_odds'] = odds

                    # Create rows for each player
                    for player_name, prop_data in player_props.items():
                        if 'line' in prop_data:
                            rows.append({
                                'date': game_date,
                                'commence_time': commence_time,
                                'home_team': home_team,
                                'away_team': away_team,
                                'player_name': player_name,
                                'prop_type': prop_type,
                                'line': prop_data.get('line'),
                                'over_odds': prop_data.get('over_odds'),
                                'under_odds': prop_data.get('under_odds'),
                                'bookmaker': bookmaker.get('title', '')
                            })

        if not rows:
            return None

        df = pd.DataFrame(rows)

        # Calculate consensus (median) lines across bookmakers for each player+prop
        consensus_rows = []
        for (player, prop_type), group in df.groupby(['player_name', 'prop_type']):
            consensus_rows.append({
                'date': game_date,
                'player_name': player,
                'prop_type': prop_type,
                'market_line': group['line'].median(),
                'market_over_odds': group['over_odds'].median(),
                'market_under_odds': group['under_odds'].median(),
                'num_books': len(group)
            })

        consensus_df = pd.DataFrame(consensus_rows)
        log(f"- Parsed {len(consensus_df)} player props ({len(consensus_df['player_name'].unique())} players)", verbose)
        return consensus_df

    except Exception as e:
        log(f"- Error fetching player props: {e}", verbose)
        import traceback
        if verbose:
            traceback.print_exc()
        return None


def load_or_fetch_historical_player_props(
    players_df: pd.DataFrame,
    api_key: str,
    cache_path: Path,
    verbose: bool = False,
    max_requests: int = 100
) -> pd.DataFrame:
    """
    Load historical player props from cache or fetch missing dates from The Odds API.

    Args:
        players_df: Player stats DataFrame with 'date' column
        api_key: The Odds API key
        cache_path: Path to cache CSV file
        verbose: Whether to print debug info
        max_requests: Maximum number of API requests (10 credits each)

    Returns:
        DataFrame with historical player prop lines
    """
    log(f"Loading/fetching historical player props (max {max_requests} requests)...", verbose)

    # Load existing cache if available
    if cache_path.exists():
        cached_props = pd.read_csv(cache_path)
        log(f"- Loaded {len(cached_props):,} cached player props from {cache_path.name}", verbose)
    else:
        print("No season_end_year column found; cannot automate 5-year window training.")
        games_df = pd.DataFrame()  # Ensure games_df is defined as an empty DataFrame if not already
        print(f"- Games: {len(games_df):,} rows")
        cached_props = pd.DataFrame()
        log(f"- No cache found, will create {cache_path.name}", verbose)

    # Get unique dates from players_df
    if 'date' not in players_df.columns:
        log("Warning: No 'date' column in players_df", verbose)
        return cached_props

    players_df['date'] = pd.to_datetime(players_df['date'], errors='coerce')

    # Filter to only dates from June 2020 onwards (when historical odds API data starts)
    odds_start_date = pd.Timestamp('2020-06-01')
    players_df_filtered = players_df[players_df['date'] >= odds_start_date]
    unique_dates = players_df_filtered['date'].dt.strftime('%Y-%m-%d').dropna().unique()
    unique_dates = sorted(unique_dates, reverse=True)  # Most recent first

    log(f"- Player data spans {len(unique_dates)} unique dates from June 2020 onwards (historical odds available)", verbose)

    # Determine which dates are missing from cache
    if not cached_props.empty and 'date' in cached_props.columns:
        cached_dates = set(cached_props['date'].astype(str).unique())
    else:
        cached_dates = set()

    missing_dates = [d for d in unique_dates if d not in cached_dates]
    log(f"- {len(missing_dates)} dates missing from cache", verbose)

    # Limit to max_requests
    if len(missing_dates) > max_requests:
        log(f"- Limiting to {max_requests} most recent dates (cost control)", verbose)
        missing_dates = missing_dates[:max_requests]

    # Fetch missing dates
    new_props = []
    for i, date in enumerate(missing_dates, 1):
        log(f"- Fetching {i}/{len(missing_dates)}: {date}", verbose)
        date_props = fetch_historical_player_props_for_date(date, api_key, verbose)
        if date_props is not None and not date_props.empty:
            new_props.append(date_props)

    # Combine new and cached props
    if new_props:
        new_props_df = pd.concat(new_props, ignore_index=True)
        log(f"- Fetched {len(new_props_df):,} new player props", verbose)

        # Combine with cache
        if not cached_props.empty:
            combined_props = pd.concat([cached_props, new_props_df], ignore_index=True)
        else:
            combined_props = new_props_df

        # Remove duplicates
        combined_props = combined_props.drop_duplicates(subset=['date', 'player_name', 'prop_type'], keep='last')

        # Save updated cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        combined_props.to_csv(cache_path, index=False)
        log(f"- Saved {len(combined_props):,} total props to cache", verbose)

        return combined_props
    else:
        log("- No new props fetched", verbose)
        return cached_props


def fetch_current_season_player_stats(season: str = "2025-26", verbose: bool = False) -> Optional[pd.DataFrame]:
    """
    Fetch current season player box scores from nba_api.
    Returns DataFrame in same format as historical PlayerStatistics CSV.

    Args:
        season: NBA season string (e.g., "2025-26")
        verbose: Print debug info

    Returns:
        DataFrame with player game logs, or None if unavailable
    """
    if not _HAS_NBA_API:
        log("- NBA API not available, skipping current season player data", verbose)
        return None

    try:
        import time
        from nba_api.stats.endpoints import LeagueGameLog

        log(f"- Fetching player stats from {season} season via nba_api...", verbose)

        # Fetch player game log for current season
        time.sleep(0.6)  # Rate limiting
        player_log = LeagueGameLog(
            season=season,
            season_type_all_star='Regular Season',
            player_or_team_abbreviation='P'  # Player
        )
        df = player_log.get_data_frames()[0]

        if df.empty:
            log(f"- No player stats found for {season} season", verbose)
            return None

        log(f"- Fetched {len(df)} player-game records from nba_api", verbose)

        # Convert to our format
        # nba_api columns: PLAYER_ID, PLAYER_NAME, TEAM_ID, TEAM_ABBREVIATION, GAME_ID, GAME_DATE,
        # MATCHUP, WL, MIN, PTS, REB, AST, FG3M, etc.
        result = pd.DataFrame()

        result['gameId'] = df['GAME_ID'].astype(str)
        result['date'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')
        result['playerId'] = df['PLAYER_ID'].astype(str)
        result['playerName'] = df['PLAYER_NAME']
        result['teamId'] = df['TEAM_ID'].astype(str)
        result['teamTricode'] = df['TEAM_ABBREVIATION']

        # Parse minutes (format: "MM:SS" or just number)
        def parse_minutes(min_str):
            if pd.isna(min_str):
                return 0.0
            min_str = str(min_str).strip()
            if ':' in min_str:
                parts = min_str.split(':')
                return float(parts[0]) + float(parts[1]) / 60.0
            return float(min_str)

        result['minutes'] = df['MIN'].apply(parse_minutes)
        result['points'] = pd.to_numeric(df['PTS'], errors='coerce').fillna(0)
        result['rebounds'] = pd.to_numeric(df['REB'], errors='coerce').fillna(0)
        result['assists'] = pd.to_numeric(df['AST'], errors='coerce').fillna(0)
        result['threes'] = pd.to_numeric(df['FG3M'], errors='coerce').fillna(0)

        # Add home flag
        result['home'] = df['MATCHUP'].str.contains('vs.', na=False).astype(int)

        log(f"- Successfully formatted {len(result)} player-game records from {season}", verbose)
        return result

    except Exception as e:
        log(f"- Error fetching current season player stats: {e}", verbose)
        import traceback
        if verbose:
            traceback.print_exc()
        return None


# ---------------- Build games from TeamStatistics ----------------

def build_games_from_teamstats(teams_path: Path, verbose: bool, skip_rest: bool) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    header_cols = list(pd.read_csv(teams_path, nrows=0).columns)
    header_norm = {_norm(c): c for c in header_cols}

    # Flexible resolver for common column variants (case-insensitive)
    def resolve(names: list[str]) -> Optional[str]:
        for name in names:
            key = _norm(name)
            if key in header_norm:
                return header_norm[key]
        return None

    gid_c   = resolve(["gameId", "GAME_ID", "game_id", "gid"]) or "gameId"
    date_c  = resolve(["gameDate", "date", "GAME_DATE", "game_date"]) or "gameDate"
    tid_c   = resolve(["teamId", "TEAM_ID", "team_id", "tid"]) or "teamId"
    opp_c   = resolve(["opponentTeamId", "opponent_id", "oppTeamId", "opp_id"]) or "opponentTeamId"
    home_c  = resolve(["home", "isHome", "HOME"]) or "home"
    tscore_c= resolve(["teamScore", "score", "TEAM_SCORE"]) or "teamScore"
    oscore_c= resolve(["opponentScore", "oppScore", "OPP_SCORE"]) or "opponentScore"
    tri_c   = resolve([
        "teamTricode", "teamTriCode", "tricode", "triCode", "team_code", "tri_code", "TEAM_TRICODE",
        "teamAbbreviation", "team_abbreviation", "abbreviation", "abbr"
    ])  # optional
    # Try multiple ways to get a usable team name
    tname_c = (
        resolve(["teamFullName", "team_full_name", "TEAM_FULL_NAME"]) or
        resolve(["teamName", "team_name", "TEAM_NAME", "team", "Team"]) or
        resolve(["nickname", "teamNickname", "team_nickname"])
    )  # optional
    tcity_c = resolve(["teamCity", "city", "TEAM_CITY"])  # optional city to combine with nickname

    # Diagnostics: show resolved columns (helps when abbrevs are missing later)
    log_cols = {
        "gid": gid_c, "date": date_c, "teamId": tid_c, "oppTeamId": opp_c, "home": home_c,
        "teamScore": tscore_c, "oppScore": oscore_c, "teamTricode": tri_c, "teamName": tname_c,
    }
    log(f"Resolved TeamStatistics columns: {log_cols}", verbose)

    # Build usecols from resolved names that actually exist
    wanted = [gid_c, date_c, tid_c, opp_c, home_c, tscore_c, oscore_c]
    if tri_c:
        wanted.append(tri_c)
    if tname_c:
        wanted.append(tname_c)
    usecols = [c for c in wanted if c in header_cols]

    # Read with dtype specification to reduce memory usage
    dtype_spec = {
        gid_c: str,
        tid_c: str,
        opp_c: str,
        home_c: str,
        tscore_c: 'float32',
        oscore_c: 'float32'
    }
    if tri_c:
        dtype_spec[tri_c] = str
    if tname_c:
        dtype_spec[tname_c] = str

    ts = pd.read_csv(teams_path, usecols=usecols, dtype=dtype_spec, parse_dates=[date_c])

    for c in ["gameId", "teamId", "opponentTeamId"]:
        if c in ts.columns:
            ts[c] = _id_to_str(ts[c])

    # home flag to 0/1
    if "home" in ts.columns:
        hv = ts["home"]
        hvn = pd.to_numeric(hv, errors="coerce")
        ts["home_flag"] = np.where(hvn.notna(), (hvn.fillna(0) != 0).astype(int),
                                   hv.astype(str).str.strip().str.lower().isin(["1","true","t","home","h","yes","y"]).astype(int))
    else:
        ts["home_flag"] = 1
    # Downcast to smallest int to save RAM
    ts["home_flag"] = ts["home_flag"].astype("int8")

    # date (keep NaT) - use format='mixed' to handle both ISO8601 and simple datetime
    ts[date_c] = pd.to_datetime(ts[date_c], errors="coerce", format='mixed', utc=True).dt.tz_convert(None)

    # MEMORY OPTIMIZATION: Filter to seasons >= 2002 immediately after loading
    # This reduces TeamStatistics from ~144k rows (1946-2026) to ~65k rows (2002-2026)
    if "season" in ts.columns:
        orig_len = len(ts)
        ts = ts[ts["season"] >= 2002].copy()
        if verbose:
            log(f"  Filtered TeamStatistics by season: {orig_len:,} → {len(ts):,} rows (2002+, saved ~{(orig_len - len(ts)) * 0.3 / 1024:.1f} MB)", True)
    elif date_c in ts.columns:
        # Fallback: filter by date if no season column
        orig_len = len(ts)
        ts = ts[ts[date_c] >= "2002-01-01"].copy()
        if verbose and len(ts) < orig_len:
            log(f"  Filtered TeamStatistics by date: {orig_len:,} → {len(ts):,} rows (2002+, saved ~{(orig_len - len(ts)) * 0.3 / 1024:.1f} MB)", True)

    # scores
    ts[tscore_c] = pd.to_numeric(ts[tscore_c], errors="coerce")
    ts[oscore_c] = pd.to_numeric(ts[oscore_c], errors="coerce")
    ts = ts.dropna(subset=[gid_c, tid_c, opp_c, tscore_c, oscore_c]).copy()

    # pair home/away
    ts_sorted = ts.sort_values([gid_c, "home_flag"], ascending=[True, False])
    home_rows = ts_sorted[ts_sorted["home_flag"] == 1].drop_duplicates(gid_c, keep="first")
    away_rows = ts_sorted[ts_sorted["home_flag"] == 0].drop_duplicates(gid_c, keep="first")

    # Build base game frame with optional team names
    right_cols = [gid_c, tid_c, tscore_c, date_c]
    if tname_c:
        right_cols.append(tname_c)
    g = home_rows.merge(
        away_rows[right_cols].rename(columns={
            tid_c: "away_tid", tscore_c: "away_score", date_c: "date_check",
            (tname_c if tname_c else "__none__"): ("away_name" if tname_c else "__none__")
        }),
        left_on=gid_c, right_on=gid_c, how="left"
    )

    # fill away from opponent fields if missing
    need_away = g["away_tid"].isna()
    if need_away.any():
        tss = ts.set_index(gid_c)
        idx = g.loc[need_away, gid_c]
        g.loc[need_away, "away_tid"] = _id_to_str(tss.loc[idx, opp_c])
        g.loc[need_away, "away_score"] = pd.to_numeric(tss.loc[idx, oscore_c], errors="coerce")

    rename_map = {gid_c: "gid", tid_c: "home_tid", tscore_c: "home_score", date_c: "date"}
    if tname_c and tname_c in g.columns:
        rename_map[tname_c] = "home_name"
    g = g.rename(columns=rename_map)
    keep_cols = ["gid", "date", "home_tid", "away_tid", "home_score", "away_score"]
    # Derive home/away names if possible
    if tname_c and tname_c in g.columns and "home_name" not in g.columns:
        rename_map = {tname_c: "home_name"}
        g = g.rename(columns=rename_map)
    # If we have city and nickname pieces, build full names
    if ("home_name" not in g.columns) and tcity_c and tcity_c in home_rows.columns and tname_c and tname_c in home_rows.columns:
        g["home_name"] = (home_rows[tcity_c].astype(str).fillna("") + " " + home_rows[tname_c].astype(str).fillna("")).str.strip()
    if ("away_name" not in g.columns) and tcity_c and tcity_c in away_rows.columns and tname_c and tname_c in away_rows.columns:
        g["away_name"] = (away_rows[tcity_c].astype(str).fillna("") + " " + away_rows[tname_c].astype(str).fillna("")).str.strip()
    if "home_name" in g.columns:
        keep_cols.append("home_name")
    if "away_name" in g.columns:
        keep_cols.append("away_name")
    g = g[keep_cols]

    g = g.dropna(subset=["home_tid", "away_tid", "home_score", "away_score"]).copy()
    for c in ["gid", "home_tid", "away_tid"]:
        g[c] = _id_to_str(g[c])

    # Parse dates - handle both ISO8601 (with Z) and simple datetime formats
    # Use format='mixed' to automatically handle multiple date formats
    g["date"] = pd.to_datetime(g["date"], errors='coerce', format='mixed', utc=True).dt.tz_convert(None)

    # Count how many dates failed to parse
    failed_dates = g["date"].isna().sum()
    if failed_dates > 0:
        print(f"Warning: {failed_dates} / {len(g)} games have unparseable dates - dropping these games")
        g = g.dropna(subset=["date"]).copy()

    print(f"Successfully parsed dates for {len(g):,} games (date range: {g['date'].min()} to {g['date'].max()})")

    # season features
    g["season_end_year"] = _season_from_date(g["date"]).astype("float32")
    g["season_decade"]   = _decade_from_season(g["season_end_year"]).astype("float32")
    
    # Create team abbreviations from team names for priors merging
    if "home_name" in g.columns and "away_name" in g.columns:
        g["home_abbrev"] = g["home_name"].apply(_team_name_to_abbrev)
        g["away_abbrev"] = g["away_name"].apply(_team_name_to_abbrev)

    # long view for rolling context
    long_home = g[["gid", "date", "season_end_year", "home_tid", "home_score", "away_score"]].rename(
        columns={"home_tid": "tid", "home_score": "team_pts", "away_score": "opp_pts"}
    )
    long_away = g[["gid", "date", "season_end_year", "away_tid", "away_score", "home_score"]].rename(
        columns={"away_tid": "tid", "away_score": "team_pts", "home_score": "opp_pts"}
    )
    teams_long = pd.concat(
        [long_home[["gid", "date", "season_end_year", "tid", "team_pts", "opp_pts"]],
         long_away[["gid", "date", "season_end_year", "tid", "team_pts", "opp_pts"]]],
        ignore_index=True
    )

    teams_long["gid_num"] = pd.to_numeric(teams_long["gid"], errors="coerce")
    teams_long = teams_long.sort_values(["tid", "date", "gid_num"], ascending=[True, True, True], na_position="last")

    # leakage-safe rolling with min_periods=1
    teams_long["off_pts_10"] = teams_long.groupby("tid")["team_pts"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    teams_long["def_pts_10"] = teams_long.groupby("tid")["opp_pts"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    teams_long["tot_pts"]    = teams_long["team_pts"] + teams_long["opp_pts"]
    teams_long["pace_10"]    = teams_long.groupby("tid")["tot_pts"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())

    teams_long["win"] = (teams_long["team_pts"] > teams_long["opp_pts"]).astype(int)
    teams_long["wins_prev"]  = teams_long.groupby("tid")["win"].transform(lambda x: x.shift(1).cumsum())
    teams_long["games_prev"] = teams_long.groupby("tid").cumcount()
    teams_long["winrate_prev"] = np.where(
        teams_long["games_prev"] > 0,
        teams_long["wins_prev"] / teams_long["games_prev"].clip(lower=1),
        0.5
    )

    # per-season medians for normalization (era-aware)
    def _per_season_median(col: str, fallback: float) -> pd.Series:
        s = teams_long.groupby("season_end_year")[col].transform("median")
        s = s.fillna(fallback)
        z = s.replace(0, np.nan)
        z = z.fillna(fallback if fallback != 0 else 1.0)
        return z

    off_med_g  = np.nanmedian(teams_long["off_pts_10"].values)
    def_med_g  = np.nanmedian(teams_long["def_pts_10"].values)
    pace_med_g = np.nanmedian(teams_long["pace_10"].values)
    off_med_s  = _per_season_median("off_pts_10", off_med_g if np.isfinite(off_med_g) and off_med_g > 0 else 1.0)
    def_med_s  = _per_season_median("def_pts_10", def_med_g if np.isfinite(def_med_g) and def_med_g > 0 else 1.0)
    pace_med_s = _per_season_median("pace_10", pace_med_g if np.isfinite(pace_med_g) and pace_med_g > 0 else 1.0)

    teams_long["off_norm"]  = teams_long["off_pts_10"] / off_med_s
    teams_long["def_norm"]  = def_med_s / teams_long["def_pts_10"]
    teams_long["pace_norm"] = teams_long["pace_10"] / pace_med_s

    # split to home/away and merge
    hstats = teams_long.rename(columns={
        "tid": "home_tid",
        "off_norm": "home_off_strength",
        "def_norm": "home_def_strength",
        "pace_norm": "home_recent_pace",
        "winrate_prev": "home_recent_winrate"
    })[["gid", "season_end_year", "home_tid", "home_off_strength", "home_def_strength", "home_recent_pace", "home_recent_winrate"]].drop_duplicates(["gid", "home_tid"])
    astats = teams_long.rename(columns={
        "tid": "away_tid",
        "off_norm": "away_off_strength",
        "def_norm": "away_def_strength",
        "pace_norm": "away_recent_pace",
        "winrate_prev": "away_recent_winrate"
    })[["gid", "season_end_year", "away_tid", "away_off_strength", "away_def_strength", "away_recent_pace", "away_recent_winrate"]].drop_duplicates(["gid", "away_tid"])

    for c in ["gid", "home_tid"]:
        hstats[c] = _id_to_str(hstats[c])
    for c in ["gid", "away_tid"]:
        astats[c] = _id_to_str(astats[c])

    g = g.merge(hstats, on=["gid", "home_tid", "season_end_year"], how="left").merge(astats, on=["gid", "away_tid", "season_end_year"], how="left")

    # matchup features
    g["match_off_edge"] = g["home_off_strength"] - g["away_def_strength"]
    g["match_def_edge"] = g["home_def_strength"] - g["away_off_strength"]
    g["match_pace_sum"] = g["home_recent_pace"] + g["away_recent_pace"]
    g["winrate_diff"]   = g["home_recent_winrate"] - g["away_recent_winrate"]

    # optional rest/b2b
    if not skip_rest:
        sched_home = g[["home_tid", "date"]].rename(columns={"home_tid": "tid"})
        sched_away = g[["away_tid", "date"]].rename(columns={"away_tid": "tid"})
        sched = pd.concat([sched_home, sched_away], ignore_index=True)
        sched["tid"]  = _id_to_str(sched["tid"])
        sched["date"] = pd.to_datetime(sched["date"], errors="coerce", utc=True).dt.tz_convert(None)
        sched = sched.dropna(subset=["tid", "date"]).drop_duplicates(["tid", "date"]).sort_values(["tid", "date"])

        sched["prev_date"] = sched.groupby("tid")["date"].shift(1)
        delta = sched["date"] - sched["prev_date"]
        sched["days_rest"] = pd.to_numeric(delta.dt.days, errors="coerce").fillna(2.0).astype("float32")
        sched["b2b"] = (sched["days_rest"] <= 1).astype("int8")

        hr = sched.rename(columns={"tid": "home_tid"})[["home_tid", "date", "days_rest", "b2b"]].rename(
            columns={"days_rest": "home_days_rest", "b2b": "home_b2b"}
        )
        ar = sched.rename(columns={"tid": "away_tid"})[["away_tid", "date", "days_rest", "b2b"]].rename(
            columns={"days_rest": "away_days_rest", "b2b": "away_b2b"}
        )

        for df_, tid_col in ((hr, "home_tid"), (ar, "away_tid")):
            df_[tid_col] = _id_to_str(df_[tid_col])
            df_["date"]  = pd.to_datetime(df_["date"], errors="coerce", utc=True).dt.tz_convert(None)
            df_.sort_values([tid_col, "date"], inplace=True)
            df_.drop_duplicates([tid_col, "date"], keep="last", inplace=True)

        g.sort_values(["gid", "date"], inplace=True)
        g.drop_duplicates(subset=["gid", "home_tid", "away_tid", "date"], keep="last", inplace=True)
        g = g.merge(hr, on=["home_tid", "date"], how="left")
        g = g.merge(ar, on=["away_tid", "date"], how="left")
    else:
        g["home_days_rest"] = GAME_DEFAULTS["home_days_rest"]
        g["away_days_rest"] = GAME_DEFAULTS["away_days_rest"]
        g["home_b2b"] = GAME_DEFAULTS["home_b2b"]
        g["away_b2b"] = GAME_DEFAULTS["away_b2b"]

    # era features (decade again in case of merges)
    g["season_decade"] = _decade_from_season(g["season_end_year"]).astype("float32")

    # fill defaults
    for col, val in GAME_DEFAULTS.items():
        if col not in g.columns:
            g[col] = val
        g[col] = pd.to_numeric(g[col], errors="coerce").fillna(val).astype("float32")

    g["home_advantage"] = np.float32(1.0)
    g["neutral_site"]   = np.float32(0.0)

    # FIX: Don't duplicate season_end_year/season_decade when building games_df
    base_cols = ["gid", "date", "season_end_year", "season_decade", "home_tid", "away_tid", "home_score", "away_score"]
    # Add team abbreviations if they exist (for priors merging)
    if "home_abbrev" in g.columns:
        base_cols.append("home_abbrev")
    if "away_abbrev" in g.columns:
        base_cols.append("away_abbrev")
    feature_cols = [f for f in GAME_FEATURES if f not in base_cols]
    games_df = g[base_cols + feature_cols].copy()

    games_df["date"] = pd.to_datetime(games_df["date"], errors="coerce", utc=True).dt.tz_convert(None)
    games_df = games_df.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)

    # context for players (both sides)
    context_cols = [
        "gid", "date", "season_end_year", "season_decade", "home_tid", "away_tid",
        "home_recent_pace", "home_off_strength", "home_def_strength", "home_recent_winrate",
        "away_recent_pace", "away_off_strength", "away_def_strength", "away_recent_winrate",
        "match_off_edge", "match_def_edge", "match_pace_sum", "winrate_diff"
    ]
    context_map = g[context_cols].drop_duplicates(["gid", "home_tid", "away_tid"]).copy()

    # Build team ID → abbreviation mapping for priors integration
    team_id_to_abbrev: Dict[str, str] = {}
    if tri_c and (tri_c in ts.columns) and (tid_c in ts.columns):
        ts_clean = ts[[tid_c, tri_c]].drop_duplicates()
        ts_clean[tri_c] = ts_clean[tri_c].astype(str).str.strip().str.upper()
        # For each team ID, take the most common abbreviation (handles rebrands)
        team_id_to_abbrev = ts_clean.groupby(tid_c)[tri_c].agg(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
        ).to_dict()
        log(f"Built team ID → abbreviation mapping for {len(team_id_to_abbrev)} teams", verbose)

    log(f"Built TeamStatistics games frame: {len(games_df):,} rows", verbose)
    return games_df, context_map, team_id_to_abbrev

# ---------------- Train game models + OOF ----------------

def _fit_game_models(
    games_df: pd.DataFrame,
    seed: int,
    verbose: bool,
    folds: int = 5,
    lgb_log_period: int = 0,
    sample_weights: Optional[np.ndarray] = None,
) -> Tuple[object, Optional[CalibratedClassifierCV], object, float, pd.DataFrame, Dict[str, float]]:
    # Ensure all required GAME_FEATURES are present; add missing with sensible defaults
    missing_cols = [c for c in GAME_FEATURES if c not in games_df.columns]
    if missing_cols:
        for c in missing_cols:
            games_df[c] = GAME_DEFAULTS.get(c, np.nan)
    
    X_full = games_df[GAME_FEATURES].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
    y_ml = (games_df["home_score"].values > games_df["away_score"].values).astype(int)
    y_sp = (games_df["home_score"].values - games_df["away_score"].values).astype(float)

    # chronological order (NaT last)
    order = pd.to_datetime(games_df["date"], errors="coerce")
    idx_sorted = order.sort_values(na_position="last", kind="mergesort").index
    X_sorted = X_full.loc[idx_sorted]
    y_ml_sorted = y_ml[idx_sorted]
    y_sp_sorted = y_sp[idx_sorted]
    gid_sorted = games_df.loc[idx_sorted, "gid"].astype(str).values
    if sample_weights is None or len(sample_weights) != len(games_df):
        w_sorted = np.ones(len(X_sorted), dtype=np.float64)
    else:
        w_sorted = sample_weights[idx_sorted]

    n = len(X_sorted)
    if n <= 1:
        ml_dummy = DummyClassifier(strategy="prior").fit(X_sorted.iloc[:1], y_ml_sorted[:1])
        sp_dummy = DummyRegressor(strategy="mean").fit(X_sorted.iloc[:1], y_sp_sorted[:1])
        return ml_dummy, None, sp_dummy, float("nan"), pd.DataFrame(), {
            "train_size": int(n), "val_size": 0,
            "ml_logloss": float("nan"), "ml_brier": float("nan"),
            "sp_rmse": float("nan"), "sp_mae": float("nan"),
        }

    # Time-based folds (equal chunks in sorted order)
    fold_sizes = [n // folds] * folds
    for i in range(n % folds):
        fold_sizes[i] += 1
    splits = []
    start = 0
    for fs in fold_sizes:
        end = start + fs
        splits.append((start, end))
        start = end

    oof_ml = np.full(n, np.nan, dtype=np.float32)
    oof_sp = np.full(n, np.nan, dtype=np.float32)

    # Common LGB params (quiet + col-wise)
    clf_params = dict(
        objective="binary", learning_rate=0.05, num_leaves=31, max_depth=-1,
        colsample_bytree=0.9, subsample=0.8, subsample_freq=5,
        n_estimators=800, random_state=seed, n_jobs=N_JOBS,
        force_col_wise=True, verbosity=-1
    )
    reg_params = dict(
        objective="regression", learning_rate=0.05, num_leaves=31, max_depth=-1,
        colsample_bytree=0.9, subsample=0.8, subsample_freq=5,
        n_estimators=800, random_state=seed, n_jobs=N_JOBS,
        force_col_wise=True, verbosity=-1
    )

    # Train per fold: train on [0:val_start], predict on [val_start:val_end]
    for k, (val_start, val_end) in enumerate(splits):
        if val_start == 0:
            continue
        tr_slice = slice(0, val_start)
        vl_slice = slice(val_start, val_end)

        X_tr, y_ml_tr, y_sp_tr = X_sorted.iloc[tr_slice, :], y_ml_sorted[tr_slice], y_sp_sorted[tr_slice]
        X_vl = X_sorted.iloc[vl_slice, :]
        w_tr = w_sorted[tr_slice]

        if _HAS_LGB:
            clf_k = lgb.LGBMClassifier(**{**clf_params, "random_state": seed + k})
            clf_k.fit(X_tr, y_ml_tr, sample_weight=w_tr)
            oof_ml[val_start:val_end] = clf_k.predict_proba(X_vl)[:, 1]

            reg_k = lgb.LGBMRegressor(**{**reg_params, "random_state": seed + k})
            reg_k.fit(X_tr, y_sp_tr, sample_weight=w_tr)
            oof_sp[val_start:val_end] = reg_k.predict(X_vl)
        else:
            clf_k = HistGradientBoostingClassifier(
                learning_rate=0.06, max_depth=None, max_iter=400, random_state=seed + k,
                validation_fraction=0.2, early_stopping=True
            )
            clf_k.fit(X_tr, y_ml_tr, sample_weight=w_tr)
            oof_ml[val_start:val_end] = clf_k.predict_proba(X_vl)[:, 1]

            reg_k = HistGradientBoostingRegressor(
                learning_rate=0.06, max_depth=None, max_iter=400, random_state=seed + k,
                validation_fraction=0.2, early_stopping=True
            )
            reg_k.fit(X_tr, y_sp_tr, sample_weight=w_tr)
            oof_sp[val_start:val_end] = reg_k.predict(X_vl)

    # Final train/val split (last 20%) for metrics/calibration
    split = max(1, min(int(n * 0.8), n - 1))
    X_tr, X_val = X_sorted.iloc[:split, :], X_sorted.iloc[split:, :]
    y_ml_tr, y_ml_val = y_ml_sorted[:split], y_ml_sorted[split:]
    y_sp_tr, y_sp_val = y_sp_sorted[:split], y_sp_sorted[split:]
    w_tr, w_val = w_sorted[:split], w_sorted[split:]

    # LightGBM callbacks for iteration logging (if requested)
    lgb_callbacks = []
    if _HAS_LGB and lgb_log_period and lgb_log_period > 0:
        lgb_callbacks.append(lgb.log_evaluation(period=int(lgb_log_period)))

    if _HAS_LGB:
        clf_final = lgb.LGBMClassifier(**{**clf_params, "n_estimators": 1000, "random_state": seed})
        clf_final.fit(
            X_tr, y_ml_tr, sample_weight=w_tr,
            eval_set=[(X_val, y_ml_val)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(80, verbose=bool(lgb_callbacks))] + lgb_callbacks
        )
        p_val_raw = clf_final.predict_proba(X_val)[:, 1]

        calibrator = CalibratedClassifierCV(clf_final, cv="prefit", method="isotonic")
        # Isotonic supports sample_weight
        calibrator.fit(X_val, y_ml_val, sample_weight=w_val)
        p_val_cal = calibrator.predict_proba(X_val)[:, 1]

        reg_final = lgb.LGBMRegressor(**{**reg_params, "n_estimators": 1000, "random_state": seed})
        reg_final.fit(
            X_tr, y_sp_tr, sample_weight=w_tr,
            eval_set=[(X_val, y_sp_val)],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(80, verbose=bool(lgb_callbacks))] + lgb_callbacks
        )
        y_pred_val = reg_final.predict(X_val)
    else:
        clf_final = HistGradientBoostingClassifier(
            learning_rate=0.06, max_depth=None, max_iter=400, random_state=seed,
            validation_fraction=0.2, early_stopping=True
        )
        clf_final.fit(X_tr, y_ml_tr, sample_weight=w_tr)
        p_val_raw = clf_final.predict_proba(X_val)[:, 1]
        calibrator = None
        p_val_cal = p_val_raw

        reg_final = HistGradientBoostingRegressor(
            learning_rate=0.06, max_depth=None, max_iter=400, random_state=seed,
            validation_fraction=0.2, early_stopping=True
        )
        reg_final.fit(X_tr, y_sp_tr, sample_weight=w_tr)
        y_pred_val = reg_final.predict(X_val)

    # Metrics
    ml_logloss = float(log_loss(y_ml_val, p_val_cal)) if len(np.unique(y_ml_val)) == 2 else float("nan")
    ml_brier = float(brier_score_loss(y_ml_val, p_val_cal))
    sp_rmse = float(math.sqrt(mean_squared_error(y_sp_val, y_pred_val)))
    sp_mae = float(mean_absolute_error(y_sp_val, y_pred_val))

    # Spread sigma from validation residuals
    spread_sigma = float(np.std(y_sp_val - y_pred_val, ddof=1))

    # Compose OOF frame
    oof_df = pd.DataFrame({
        "gid": gid_sorted,
        "oof_ml_prob": oof_ml,
        "oof_spread_pred": oof_sp,
    })

    # Pretty summary
    print(_sec("Game model metrics (validation)"))
    print(f"- Moneyline: logloss={_fmt(ml_logloss)}, Brier={_fmt(ml_brier)}")
    print(f"- Spread:    RMSE={_fmt(sp_rmse)}, MAE={_fmt(sp_mae)}, sigma={_fmt(spread_sigma)}")

    metrics = {
        "train_size": int(len(X_tr)),
        "val_size": int(len(X_val)),
        "ml_logloss": ml_logloss,
        "ml_brier": ml_brier,
        "sp_rmse": sp_rmse,
        "sp_mae": sp_mae,
        "spread_sigma": spread_sigma,
    }
    return clf_final, calibrator, reg_final, spread_sigma, oof_df, metrics

# ---------------- Player dataset (with opponent context) ----------------

def build_players_from_playerstats(
    player_path: Path,
    games_context: pd.DataFrame,
    oof_games: pd.DataFrame,
    verbose: bool,
    priors_players: Optional[pd.DataFrame] = None,
    window_seasons: Optional[Set[int]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Build training frames for player models from PlayerStatistics.csv.
    Supports files without teamId/opponentTeamId by using the 'home' flag to select side context.
    Includes opponent context columns (opp_*).
    Merges Basketball Reference player priors if provided.

    Args:
        window_seasons: If provided, filters data to only these seasons (±1 for context) to save memory.
                       This enables per-window processing with 80%+ memory reduction.
    """
    # Read header first (no usecols) so we can detect what exists
    hdr = list(pd.read_csv(player_path, nrows=0).columns)

    # Helpers to resolve columns (case-insensitive, many aliases)
    def resolve_any(groups: list[list[str]]) -> Optional[str]:
        norm_map = {_norm(c): c for c in hdr}
        for group in groups:
            for cand in group:
                nc = _norm(cand)
                if nc in norm_map:
                    return norm_map[nc]
        return None

    # Column groups (include common names + your exact names)
    gid_col   = resolve_any([["gameId", "GAME_ID", "game_id", "gid"]])
    date_col  = resolve_any([["gameDate", "date", "GAME_DATE", "game_date"]])
    pid_col   = resolve_any([["personId", "playerId", "PLAYER_ID", "player_id", "person_id"]])
    name_full_col  = resolve_any([["playerName", "PLAYER_NAME", "player_name"]])  # optional full name
    fname_col      = resolve_any([["firstName", "FIRST_NAME", "first_name"]])    # optional first name
    lname_col      = resolve_any([["lastName", "LAST_NAME", "last_name"]])        # optional last name
    # Keep legacy name_col for backward compatibility
    name_col       = name_full_col or fname_col or lname_col
    tid_col   = resolve_any([["teamId", "TEAM_ID", "team_id", "tid"]])  # may be None
    home_col  = resolve_any([["home", "isHome", "HOME"]])  # present in your file

    min_col   = resolve_any([["numMinutes", "minutes", "mins", "min", "MIN"]])  # numMinutes in your file
    pts_col   = resolve_any([["points", "pts", "POINTS", "PTS"]])
    reb_col   = resolve_any([["reboundsTotal", "rebounds", "REB", "REBOUNDS"]])  # reboundsTotal in your file
    ast_col   = resolve_any([["assists", "ast", "ASSISTS", "AST"]])
    tpm_col   = resolve_any([["threePointersMade", "3pm", "FG3M", "threes", "three_pm"]])
    starter_col = resolve_any([["starter", "isStarter", "started", "STARTER", "IS_STARTER"]])  # likely missing

    # PHASE 1 FEATURES: Shot volume + efficiency
    fga_col = resolve_any([["fieldGoalsAttempted", "fga", "FGA", "field_goals_attempted"]])
    three_pa_col = resolve_any([["threePointersAttempted", "3pa", "FG3A", "three_pa", "three_point_attempts"]])
    fta_col = resolve_any([["freeThrowsAttempted", "fta", "FTA", "free_throw_attempts"]])
    fg_pct_col = resolve_any([["fieldGoalsPercentage", "fg_pct", "FG_PCT"]])
    three_pct_col = resolve_any([["threePointersPercentage", "fg3_pct", "FG3_PCT", "three_pct"]])
    ft_pct_col = resolve_any([["freeThrowsPercentage", "ft_pct", "FT_PCT"]])

    # Read all detected columns (avoid usecols mismatch)
    # FIXED: Include both fname_col and lname_col separately, not just name_col
    want_cols = [gid_col, date_col, pid_col, name_full_col, fname_col, lname_col, tid_col, home_col,
                 min_col, pts_col, reb_col, ast_col, tpm_col, starter_col,
                 fga_col, three_pa_col, fta_col, fg_pct_col, three_pct_col, ft_pct_col]  # PHASE 1
    usecols = [c for c in want_cols if c is not None]
    ps = pd.read_csv(player_path, low_memory=False, usecols=sorted(set(usecols)) if usecols else None)

    # MEMORY OPTIMIZATION 1: Filter to seasons >= 2002 immediately after loading
    # This reduces PlayerStatistics from ~1.6M rows (1946-2026) to ~833k rows (2002-2026)
    # NOTE: season_end_year will be added from games merge, so check date instead
    if date_col and date_col in ps.columns:
        ps[date_col] = pd.to_datetime(ps[date_col], errors="coerce", format='mixed', utc=True).dt.tz_convert(None)
        orig_len = len(ps)
        ps = ps[ps[date_col] >= "2002-01-01"].copy()
        if verbose and len(ps) < orig_len:
            memory_saved = (orig_len - len(ps)) * 0.19  # ~0.19 KB per row for PlayerStatistics
            log(f"  Filtered PlayerStatistics by date: {orig_len:,} → {len(ps):,} rows (2002+, saved ~{memory_saved / 1024:.1f} MB)", True)

    # MEMORY OPTIMIZATION 2: Filter to window seasons if provided (per-window processing)
    # This reduces from ~833k rows (all seasons) to ~150k per 5-year window (82% reduction!)
    if window_seasons is not None and date_col and date_col in ps.columns:
        # Compute season_end_year early for filtering
        ps["_temp_season"] = _season_from_date(ps[date_col])

        # Add ±1 padding for rolling features that need prior season context
        padded_seasons = set(window_seasons)
        if len(padded_seasons) > 0:
            min_season = min(padded_seasons)
            max_season = max(padded_seasons)
            padded_seasons = padded_seasons | {min_season - 1, max_season + 1}

        orig_len = len(ps)
        ps = ps[ps["_temp_season"].isin(padded_seasons)].copy()
        ps = ps.drop(columns=["_temp_season"])

        if verbose and len(ps) < orig_len:
            memory_saved = (orig_len - len(ps)) * 0.19
            log(f"  Filtered to window seasons {sorted(window_seasons)}: {orig_len:,} → {len(ps):,} rows (saved ~{memory_saved / 1024:.1f} MB)", True)

    # Show what we detected (easier to debug)
    print(_sec("Detected player columns"))
    print(f"- gid: {gid_col}  date: {date_col}  pid: {pid_col}  name_full: {name_full_col}  first: {fname_col}  last: {lname_col}")
    print(f"- teamId: {tid_col}  home_flag: {home_col}")
    print(f"- minutes: {min_col}  points: {pts_col}  rebounds: {reb_col}  assists: {ast_col}  threes: {tpm_col}")
    print(f"  DEBUG: Loaded {len(ps):,} player-game rows from CSV")

    # IDs to string (where present)
    for c in [gid_col, pid_col, tid_col]:
        if c and c in ps.columns:
            ps[c] = _id_to_str(ps[c])
    # Convert identifier/text columns to categorical to reduce memory
    for c in [gid_col, pid_col, tid_col, name_full_col, fname_col, lname_col]:
        if c and c in ps.columns:
            try:
                ps[c] = ps[c].astype("category")
            except Exception:
                pass

    # Ensure player identifier
    if not pid_col or pid_col not in ps.columns:
        if name_col and name_col in ps.columns:
            keys = ps[name_col].astype(str).fillna("unknown")
            pid_codes, _ = pd.factorize(keys, sort=True)
            ps["__player_id__"] = pd.Series(pid_codes, index=ps.index).astype("Int64").astype(str)
            pid_col = "__player_id__"
        else:
            raise KeyError("No player identifier found (tried personId/playerId/player_name).")

    # Build a full-name join column when possible (used for priors name fallback)
    # Check if name_full_col has actual data (not all nan)
    if name_full_col and name_full_col in ps.columns and ps[name_full_col].notna().any():
        ps["__name_join__"] = ps[name_full_col].astype(str)
        if verbose:
            log(f"  Using full name column: {name_full_col}", True)
    elif fname_col or lname_col:
        # Ensure fn/ln are pandas Series so .fillna() is always available.
        if fname_col and fname_col in ps.columns:
            fn = ps[fname_col].astype(str)
        else:
            fn = pd.Series([""] * len(ps), index=ps.index, dtype="object")
        if lname_col and lname_col in ps.columns:
            ln = ps[lname_col].astype(str)
        else:
            ln = pd.Series([""] * len(ps), index=ps.index, dtype="object")
        ps["__name_join__"] = (fn.fillna("") + " " + ln.fillna("")).str.strip()
        if verbose:
            sample_names = ps["__name_join__"].dropna().head(5).tolist()
            log(f"  Constructed names from firstName + lastName: {sample_names}", True)
    elif name_col and name_col in ps.columns:
        ps["__name_join__"] = ps[name_col].astype(str)
        if verbose:
            log(f"  Using name column: {name_col}", True)
    else:
        ps["__name_join__"] = None
        if verbose:
            log(f"  WARNING: No name columns available for player matching!", True)

    # Date parse (CRITICAL: format='mixed' needed for multiple date formats)
    # NOTE: Date may have been parsed already at line 1623 during early filtering - don't re-parse if already datetime
    if date_col and date_col in ps.columns:
        if not pd.api.types.is_datetime64_any_dtype(ps[date_col]):
            ps[date_col] = pd.to_datetime(ps[date_col], errors="coerce", format='mixed', utc=True).dt.tz_convert(None)
    else:
        ps["__no_date__"] = pd.NaT
        date_col = "__no_date__"

    # Era features from date
    ps["season_end_year"] = _season_from_date(ps[date_col]).astype("float32")

    # DEBUG: Show season_end_year population
    if verbose:
        non_null_seasons = ps["season_end_year"].notna().sum()
        log(f"  season_end_year populated: {non_null_seasons:,} / {len(ps):,} rows ({(non_null_seasons/len(ps)*100 if len(ps) else 0):.1f}%)", True)
    ps["season_decade"]   = _decade_from_season(ps["season_end_year"]).astype("float32")

    # Numeric conversions
    for stat_col in [min_col, pts_col, reb_col, ast_col, tpm_col, fga_col, three_pa_col, fta_col, fg_pct_col, three_pct_col, ft_pct_col]:
        if stat_col and stat_col in ps.columns:
            ps[stat_col] = pd.to_numeric(ps[stat_col], errors="coerce")

    # Starter flag (optional)
    if starter_col and starter_col in ps.columns:
        s = ps[starter_col]
        num = pd.to_numeric(s, errors="coerce")
        ps["starter_flag"] = np.where(num.notna(), (num.fillna(0) != 0).astype(int),
                                      s.astype(str).str.strip().str.lower().isin(["1", "true", "t", "yes", "y", "starter", "start"]).astype(int))
    else:
        ps["starter_flag"] = 0
    # Downcast flags
    ps["starter_flag"] = ps["starter_flag"].astype("int8")

    # Home/away flag from player CSV (strong signal if teamId is missing)
    if home_col and home_col in ps.columns:
        h = ps[home_col]
        hnum = pd.to_numeric(h, errors="coerce")
        ps["is_home"] = np.where(
            hnum.notna(),
            (hnum.fillna(0) != 0).astype(int),
            h.astype(str).str.strip().str.lower().isin(["1", "true", "t", "home", "h", "yes", "y"]).astype(int)
        )
    else:
        ps["is_home"] = np.nan  # will fallback later

    # Sort per player by date then gid numeric
    ps["gid_num"] = pd.to_numeric(ps[gid_col], errors="coerce")
    ps = ps.sort_values([pid_col, date_col, "gid_num"], ascending=[True, True, True], na_position="last")

    # Rolling minutes trend
    if min_col and min_col in ps.columns:
        ps["min_prev_mean5"]  = ps.groupby(pid_col)[min_col].transform(lambda x: x.shift(1).rolling(5,  min_periods=1).mean())
        ps["min_prev_mean10"] = ps.groupby(pid_col)[min_col].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        ps["min_prev_last1"]  = ps.groupby(pid_col)[min_col].transform(lambda x: x.shift(1))
    else:
        ps["min_prev_mean5"] = 20.0
        ps["min_prev_mean10"] = 20.0
        ps["min_prev_last1"] = 20.0

    # Player rest days and back-to-back indicator
    if date_col and date_col in ps.columns:
        ps[date_col] = pd.to_datetime(ps[date_col], errors="coerce")
        ps["days_rest"] = ps.groupby(pid_col)[date_col].transform(lambda x: (x - x.shift(1)).dt.days.fillna(3.0))
        ps["days_rest"] = ps["days_rest"].clip(0, 10).fillna(3.0).astype("float32")  # Cap at 10 days, default 3
        ps["player_b2b"] = (ps["days_rest"] <= 1.0).astype("float32")  # Back-to-back if 1 day or less
    else:
        ps["days_rest"] = 3.0
        ps["player_b2b"] = 0.0

    # Enhanced rolling trends (last 3, 5, 10 games) for all key stats
    def rolling_stats(stat_col: str) -> None:
        if stat_col and stat_col in ps.columns:
            ps[f"{stat_col}_L3"]  = ps.groupby(pid_col)[stat_col].transform(lambda x: x.shift(1).rolling(3,  min_periods=1).mean())
            ps[f"{stat_col}_L5"]  = ps.groupby(pid_col)[stat_col].transform(lambda x: x.shift(1).rolling(5,  min_periods=1).mean())
            ps[f"{stat_col}_L10"] = ps.groupby(pid_col)[stat_col].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        else:
            ps[f"{stat_col}_L3"] = 0.0
            ps[f"{stat_col}_L5"] = 0.0
            ps[f"{stat_col}_L10"] = 0.0

    rolling_stats(pts_col)
    rolling_stats(reb_col)
    rolling_stats(ast_col)
    rolling_stats(tpm_col)

    # PHASE 1.1: Shot volume rolling stats
    rolling_stats(fga_col)
    rolling_stats(three_pa_col)
    rolling_stats(fta_col)

    # PHASE 1.2: True Shooting % calculation
    if pts_col and fga_col and fta_col:
        if pts_col in ps.columns and fga_col in ps.columns and fta_col in ps.columns:
            def calc_ts_pct_row(row):
                """Calculate True Shooting % = PTS / (2 * (FGA + 0.44 * FTA))"""
                pts = row[pts_col] if pd.notna(row[pts_col]) else 0
                fga = row[fga_col] if pd.notna(row[fga_col]) else 0
                fta = row[fta_col] if pd.notna(row[fta_col]) else 0
                denominator = 2 * (fga + 0.44 * fta)
                return pts / denominator if denominator > 0 else 0.56  # league average

            ps['ts_pct'] = ps.apply(calc_ts_pct_row, axis=1)

            # Rolling TS% (last 5, 10, season average)
            ps['ts_pct_L5'] = ps.groupby(pid_col)['ts_pct'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
            ps['ts_pct_L10'] = ps.groupby(pid_col)['ts_pct'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
            ps['ts_pct_season'] = ps.groupby(pid_col)['ts_pct'].transform(lambda x: x.shift(1).expanding(min_periods=1).mean())

    # PHASE 1.2: Shooting percentage rolling averages
    if three_pct_col and three_pct_col in ps.columns:
        ps['three_pct_L5'] = ps.groupby(pid_col)[three_pct_col].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())

    if ft_pct_col and ft_pct_col in ps.columns:
        ps['ft_pct_L5'] = ps.groupby(pid_col)[ft_pct_col].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())

    # DEBUG: Verify Phase 1 features created
    if verbose:
        phase1_expected = ["fieldGoalsAttempted_L5", "threePointersAttempted_L5", "freeThrowsAttempted_L5",
                          "rate_fga", "rate_3pa", "rate_fta", "ts_pct_L5", "ts_pct_L10", "three_pct_L5"]
        phase1_created = [f for f in phase1_expected if f in ps.columns]
        phase1_missing = [f for f in phase1_expected if f not in ps.columns]
        log(f"  [DEBUG] Phase 1 feature engineering: {len(phase1_created)}/{len(phase1_expected)} features created", True)
        if phase1_missing:
            log(f"    Missing: {phase1_missing}", True)

    # Home/Away performance splits (rolling averages split by location)
    if "is_home" in ps.columns:
        for stat_col in [pts_col, reb_col, ast_col, tpm_col]:
            if stat_col and stat_col in ps.columns:
                # Home performance
                home_mask = ps["is_home"] == 1
                ps.loc[home_mask, f"{stat_col}_home_avg"] = ps.loc[home_mask].groupby(pid_col)[stat_col].transform(
                    lambda x: x.shift(1).rolling(10, min_periods=1).mean()
                )
                # Away performance
                away_mask = ps["is_home"] == 0
                ps.loc[away_mask, f"{stat_col}_away_avg"] = ps.loc[away_mask].groupby(pid_col)[stat_col].transform(
                    lambda x: x.shift(1).rolling(10, min_periods=1).mean()
                )
                # Fill NaNs with overall average
                ps[f"{stat_col}_home_avg"] = ps[f"{stat_col}_home_avg"].fillna(ps[stat_col].median())
                ps[f"{stat_col}_away_avg"] = ps[f"{stat_col}_away_avg"].fillna(ps[stat_col].median())

    # Per-minute rates (shifted)
    def rate(prev_count_col: Optional[str]) -> pd.Series:
        if not (min_col and prev_count_col and prev_count_col in ps.columns and min_col in ps.columns):
            return pd.Series(index=ps.index, dtype="float32")
        minutes_shift = ps.groupby(pid_col)[min_col].shift(1)
        stat_shift = ps.groupby(pid_col)[prev_count_col].shift(1)
        r = stat_shift / minutes_shift.replace(0, np.nan)
        return r.fillna(r.median(skipna=True)).astype("float32")

    ps["rate_pts"] = rate(pts_col).fillna(0.5).astype("float32")
    ps["rate_reb"] = rate(reb_col).fillna(0.2).astype("float32")
    ps["rate_ast"] = rate(ast_col).fillna(0.2).astype("float32")
    ps["rate_3pm"] = rate(tpm_col).fillna(0.1).astype("float32")

    # PHASE 1.1: Shot volume per-minute rates
    ps["rate_fga"] = rate(fga_col).fillna(0.3).astype("float32")
    ps["rate_3pa"] = rate(three_pa_col).fillna(0.1).astype("float32")
    ps["rate_fta"] = rate(fta_col).fillna(0.1).astype("float32")

    # PHASE 3: Advanced rate stats (Usage, Rebound %, Assist %)
    # These require team totals, calculated after merge with game context
    # For now, create placeholders - will be filled in later
    ps["usage_rate_L5"] = np.nan
    ps["rebound_rate_L5"] = np.nan
    ps["assist_rate_L5"] = np.nan

    # Build side-aware context keyed by (gid, is_home), including opponent context
    ctx = games_context.copy()

    # Home-side row: team_* from home_*, opp_* from away_*
    home_side = ctx[[
        "gid", "date", "season_end_year", "season_decade", "home_tid", "away_tid",
        "home_recent_pace", "home_off_strength", "home_def_strength", "home_recent_winrate",
        "away_recent_pace", "away_off_strength", "away_def_strength", "away_recent_winrate",
        "match_off_edge", "match_def_edge", "match_pace_sum", "winrate_diff"
    ]].rename(columns={
        "home_tid": "tid", "away_tid": "opp_tid",
        "home_recent_pace": "team_recent_pace",
        "home_off_strength": "team_off_strength",
        "home_def_strength": "team_def_strength",
        "home_recent_winrate": "team_recent_winrate",
        "away_recent_pace": "opp_recent_pace",
        "away_off_strength": "opp_off_strength",
        "away_def_strength": "opp_def_strength",
        "away_recent_winrate": "opp_recent_winrate",
    })
    home_side["is_home"] = 1

    # Away-side row: team_* from away_*, opp_* from home_*
    away_side = ctx[[
        "gid", "date", "season_end_year", "season_decade", "home_tid", "away_tid",
        "away_recent_pace", "away_off_strength", "away_def_strength", "away_recent_winrate",
        "home_recent_pace", "home_off_strength", "home_def_strength", "home_recent_winrate",
        "match_off_edge", "match_def_edge", "match_pace_sum", "winrate_diff"
    ]].rename(columns={
        "away_tid": "tid", "home_tid": "opp_tid",
        "away_recent_pace": "team_recent_pace",
        "away_off_strength": "team_off_strength",
        "away_def_strength": "team_def_strength",
        "away_recent_winrate": "team_recent_winrate",
        "home_recent_pace": "opp_recent_pace",
        "home_off_strength": "opp_off_strength",
        "home_def_strength": "opp_def_strength",
        "home_recent_winrate": "opp_recent_winrate",
    })
    away_side["is_home"] = 0

    side_ctx_tid = pd.concat([home_side, away_side], ignore_index=True)
    side_ctx_flag = side_ctx_tid.drop(columns=["tid", "opp_tid"]).drop_duplicates(["gid", "is_home"])

    # Drop season columns from ps before merge (they come from game context to avoid conflicts)
    ps = ps.drop(columns=["season_end_year", "season_decade"], errors="ignore")

    # DEBUG: Log merge inputs
    if verbose:
        log(f"  DEBUG - BEFORE MERGE:", True)
        log(f"    ps shape: {ps.shape}, columns: {list(ps.columns)[:20]}", True)
        log(f"    side_ctx_tid shape: {side_ctx_tid.shape}, columns: {list(side_ctx_tid.columns)}", True)
        log(f"    side_ctx_flag shape: {side_ctx_flag.shape}, columns: {list(side_ctx_flag.columns)}", True)

        # Check merge keys
        if tid_col and tid_col in ps.columns:
            log(f"    Merge path: tid (gameId + teamId)", True)
            log(f"    ps[{gid_col}] sample: {ps[gid_col].head(10).tolist()}", True)
            log(f"    ps[{tid_col}] sample: {ps[tid_col].head(10).tolist()}", True)
            log(f"    side_ctx_tid['gid'] sample: {side_ctx_tid['gid'].head(10).tolist()}", True)
            log(f"    side_ctx_tid['tid'] sample: {side_ctx_tid['tid'].head(10).tolist()}", True)
            log(f"    ps[{gid_col}] dtype: {ps[gid_col].dtype}, unique count: {ps[gid_col].nunique()}", True)
            log(f"    ps[{tid_col}] dtype: {ps[tid_col].dtype}, unique count: {ps[tid_col].nunique()}", True)
            log(f"    side_ctx_tid['gid'] dtype: {side_ctx_tid['gid'].dtype}, unique count: {side_ctx_tid['gid'].nunique()}", True)
            log(f"    side_ctx_tid['tid'] dtype: {side_ctx_tid['tid'].dtype}, unique count: {side_ctx_tid['tid'].nunique()}", True)
        elif "is_home" in ps.columns and ps["is_home"].notna().any():
            log(f"    Merge path: is_home flag", True)
            log(f"    ps['is_home'] value counts: {ps['is_home'].value_counts().to_dict()}", True)
            log(f"    side_ctx_flag['is_home'] value counts: {side_ctx_flag['is_home'].value_counts().to_dict()}", True)
        else:
            log(f"    Merge path: fallback (per-game average)", True)

    # Join context to players:
    # 1) If teamId exists AND has valid data: join by (gid, tid)
    # 2) Else if home flag exists: join by (gid, is_home)
    # 3) Else: fallback to per-game average context (team_* and opp_*), plus matchup
    # Check if teamId has valid data (not NaN and not string 'nan')
    has_valid_tid = False
    if tid_col and tid_col in ps.columns:
        tid_values = ps[tid_col].astype(str)
        has_valid_tid = (ps[tid_col].notna().any() and
                        (tid_values != 'nan').any() and
                        (tid_values != '').any())

    if has_valid_tid:
        ps_join = ps.merge(
            side_ctx_tid,
            left_on=[gid_col, tid_col],
            right_on=["gid", "tid"],
            how="left",
            validate="many_to_one"
        )
    elif "is_home" in ps.columns and ps["is_home"].notna().any():
        ps_join = ps.merge(
            side_ctx_flag,
            left_on=[gid_col, "is_home"],
            right_on=["gid", "is_home"],
            how="left",
            validate="many_to_one"
        )
    else:
        avg_cols = [
            "season_end_year", "season_decade",
            "team_recent_pace", "team_off_strength", "team_def_strength", "team_recent_winrate",
            "opp_recent_pace", "opp_off_strength", "opp_def_strength", "opp_recent_winrate",
            "match_off_edge", "match_def_edge", "match_pace_sum", "winrate_diff"
        ]
        avg_ctx = side_ctx_tid.groupby("gid", as_index=False)[avg_cols].mean()
        ps_join = ps.merge(avg_ctx, left_on=gid_col, right_on="gid", how="left")
        # Add is_home as NaN since we don't have team/home info
        ps_join["is_home"] = np.nan

    # Ensure is_home column always exists (safety check for all paths)
    if "is_home" not in ps_join.columns:
        ps_join["is_home"] = np.nan

    # ============================================================================
    # PHASE 2: Matchup & Context Features
    # ============================================================================
    # Now that we have team/opponent context from merge, add matchup features

    # Pace-adjusted opportunities (normalize stats by game pace)
    if "team_recent_pace" in ps_join.columns and "opp_recent_pace" in ps_join.columns:
        # Average pace of this matchup (both teams combined)
        ps_join["matchup_pace"] = ((ps_join["team_recent_pace"] + ps_join["opp_recent_pace"]) / 2).fillna(100.0)
        # Pace adjustment factor (>1.0 = faster pace = more opportunities)
        ps_join["pace_factor"] = (ps_join["matchup_pace"] / 100.0).fillna(1.0)
    else:
        ps_join["matchup_pace"] = 100.0
        ps_join["pace_factor"] = 1.0

    # Defensive matchup quality (opponent's defensive strength)
    if "opp_def_strength" in ps_join.columns:
        # Higher opp_def_strength = tougher defense = harder to score
        ps_join["def_matchup_difficulty"] = ps_join["opp_def_strength"].fillna(0.0)
    else:
        ps_join["def_matchup_difficulty"] = 0.0

    # Offensive environment (team's offensive strength + opponent's defensive weakness)
    if "team_off_strength" in ps_join.columns and "opp_def_strength" in ps_join.columns:
        ps_join["offensive_environment"] = (ps_join["team_off_strength"] - ps_join["opp_def_strength"]).fillna(0.0)
    else:
        ps_join["offensive_environment"] = 0.0

    # ============================================================================
    # PHASE 3: Advanced Rate Stats (Usage, Rebound %, Assist %)
    # ============================================================================
    # These require team-level totals, which we'll approximate from player priors
    # NOTE: Full implementation would need team totals from game-level data
    # For now, use Basketball Reference priors if available

    # Check if we have prior columns with usage/rebound/assist data
    prior_usage_cols = [c for c in ps_join.columns if 'usg' in c.lower() or 'usage' in c.lower()]
    prior_rebpct_cols = [c for c in ps_join.columns if 'trb%' in c.lower() or 'reb_pct' in c.lower() or 'rebounding_pct' in c.lower()]
    prior_astpct_cols = [c for c in ps_join.columns if 'ast%' in c.lower() or 'ast_pct' in c.lower() or 'assist_pct' in c.lower()]

    # If priors have usage rate, use it; otherwise estimate from FGA + FTA
    if prior_usage_cols:
        # Use first matching prior column
        ps_join["usage_rate_prior"] = ps_join[prior_usage_cols[0]].fillna(15.0)  # league avg ~20%
    else:
        # Estimate usage rate from shot volume per minute
        # Simple approximation: higher rate_fga + rate_fta = higher usage
        if "rate_fga" in ps_join.columns and "rate_fta" in ps_join.columns:
            ps_join["usage_rate_prior"] = ((ps_join["rate_fga"] + 0.44 * ps_join["rate_fta"]) * 5.0).fillna(15.0).clip(0, 40)
        else:
            ps_join["usage_rate_prior"] = 15.0

    # If priors have rebound %, use it; otherwise keep placeholder
    if prior_rebpct_cols:
        ps_join["rebound_rate_prior"] = ps_join[prior_rebpct_cols[0]].fillna(10.0)
    else:
        # Keep placeholder NaN - will be filled with default in model training
        pass

    # If priors have assist %, use it; otherwise keep placeholder
    if prior_astpct_cols:
        ps_join["assist_rate_prior"] = ps_join[prior_astpct_cols[0]].fillna(10.0)
    else:
        # Keep placeholder NaN - will be filled with default in model training
        pass

    # Rolling averages of usage/rebound/assist rates (if we have them)
    # Group by player and calculate L5 averages
    if "usage_rate_prior" in ps_join.columns:
        ps_join["usage_rate_L5"] = ps_join.groupby(pid_col)["usage_rate_prior"].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        ).fillna(15.0)

    if "rebound_rate_prior" in ps_join.columns:
        ps_join["rebound_rate_L5"] = ps_join.groupby(pid_col)["rebound_rate_prior"].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        ).fillna(10.0)
    else:
        # Keep existing NaN placeholder - will be filled later
        if "rebound_rate_L5" not in ps_join.columns:
            ps_join["rebound_rate_L5"] = 10.0

    if "assist_rate_prior" in ps_join.columns:
        ps_join["assist_rate_L5"] = ps_join.groupby(pid_col)["assist_rate_prior"].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        ).fillna(10.0)
    else:
        # Keep existing NaN placeholder - will be filled later
        if "assist_rate_L5" not in ps_join.columns:
            ps_join["assist_rate_L5"] = 10.0

    # DEBUG: Log merge results and Phase 2+3 features
    if verbose:
        log(f"  DEBUG - AFTER MERGE:", True)
        log(f"    ps_join shape: {ps_join.shape}", True)
        log(f"    'season_end_year' in columns: {'season_end_year' in ps_join.columns}", True)
        if "season_end_year" in ps_join.columns:
            non_null_seasons = ps_join["season_end_year"].notna().sum()
            log(f"    season_end_year non-null: {non_null_seasons:,} / {len(ps_join):,} ({(non_null_seasons/len(ps_join)*100 if len(ps_join) else 0):.1f}%)", True)
            if non_null_seasons > 0:
                log(f"    season_end_year sample values: {ps_join['season_end_year'].dropna().head(10).tolist()}", True)
            else:
                log(f"    WARNING: season_end_year is all NaN - merge failed!", True)
        else:
            log(f"    WARNING: season_end_year column missing after merge!", True)

        # Verify Phase 2+3 features created
        phase2_expected = ["matchup_pace", "pace_factor", "def_matchup_difficulty", "offensive_environment"]
        phase3_expected = ["usage_rate_L5", "rebound_rate_L5", "assist_rate_L5"]
        phase2_created = [f for f in phase2_expected if f in ps_join.columns]
        phase3_created = [f for f in phase3_expected if f in ps_join.columns]
        log(f"  [DEBUG] Phase 2 features: {len(phase2_created)}/{len(phase2_expected)} created", True)
        if len(phase2_created) < len(phase2_expected):
            phase2_missing = [f for f in phase2_expected if f not in ps_join.columns]
            log(f"    Missing: {phase2_missing}", True)
        log(f"  [DEBUG] Phase 3 features: {len(phase3_created)}/{len(phase3_expected)} created", True)
        if len(phase3_created) < len(phase3_expected):
            phase3_missing = [f for f in phase3_expected if f not in ps_join.columns]
            log(f"    Missing: {phase3_missing}", True)

    # ========================================================================
    # PHASE 4: ADVANCED CONTEXT FEATURES (Opponent Defense + Player Context)
    # ========================================================================
    # Expected improvement: +2-3% accuracy across all prop types
    
    # 4A. OPPONENT DEFENSIVE STRENGTH (by stat type)
    # Calculate how opponent team performs defensively against each stat
    # Use rolling stats of opponent's allowed points/assists/rebounds per game
    
    # Create opponent defensive metrics if opponent columns exist
    if "opp_def_strength" in ps_join.columns:
        # Opponent defense baseline (already have general def_strength)
        ps_join["opp_def_vs_points"] = ps_join["opp_def_strength"].fillna(1.0)
        ps_join["opp_def_vs_assists"] = ps_join["opp_def_strength"].fillna(1.0)
        ps_join["opp_def_vs_rebounds"] = ps_join["opp_def_strength"].fillna(1.0)
        ps_join["opp_def_vs_threes"] = ps_join["opp_def_strength"].fillna(1.0)
    else:
        # Defaults if not available
        ps_join["opp_def_vs_points"] = 1.0
        ps_join["opp_def_vs_assists"] = 1.0
        ps_join["opp_def_vs_rebounds"] = 1.0
        ps_join["opp_def_vs_threes"] = 1.0
    
    # 4B. PLAYER CONTEXT FEATURES
    
    # Rest days impact (back-to-back games typically reduce performance)
    # Calculate days since last game for each player
    if "date" in ps_join.columns and pid_col in ps_join.columns:
        ps_join["date_dt"] = pd.to_datetime(ps_join["date"], errors="coerce")
        ps_join["prev_game_date"] = ps_join.groupby(pid_col)["date_dt"].shift(1)
        ps_join["rest_days"] = (ps_join["date_dt"] - ps_join["prev_game_date"]).dt.days
        ps_join["rest_days"] = ps_join["rest_days"].fillna(3.0).clip(0, 14).astype("float32")
        
        # Back-to-back indicator (0 or 1 rest day)
        ps_join["is_b2b"] = (ps_join["rest_days"] <= 1).astype("float32")
        
        # Well-rested indicator (3+ days rest)
        ps_join["is_rested"] = (ps_join["rest_days"] >= 3).astype("float32")
        
        # Cleanup temporary columns
        ps_join = ps_join.drop(columns=["date_dt", "prev_game_date"], errors="ignore")
    else:
        ps_join["rest_days"] = 2.0  # Average
        ps_join["is_b2b"] = 0.0
        ps_join["is_rested"] = 0.5
    
    # Minutes trend (is player's role expanding or shrinking?)
    # Positive = getting more minutes, negative = getting less
    if min_col and min_col in ps_join.columns and pid_col in ps_join.columns:
        # L5 average vs L10 average
        ps_join["mins_L5"] = ps_join.groupby(pid_col)[min_col].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )
        ps_join["mins_L10"] = ps_join.groupby(pid_col)[min_col].transform(
            lambda x: x.shift(1).rolling(10, min_periods=1).mean()
        )
        ps_join["mins_trend"] = (ps_join["mins_L5"] - ps_join["mins_L10"]).fillna(0.0).astype("float32")
        
        # Normalize trend (-1 to +1 scale, roughly)
        ps_join["mins_trend"] = ps_join["mins_trend"].clip(-10, 10) / 10.0
        
        # Role expansion/reduction indicators
        ps_join["role_expanding"] = (ps_join["mins_trend"] > 0.2).astype("float32")
        ps_join["role_shrinking"] = (ps_join["mins_trend"] < -0.2).astype("float32")
        
        # Cleanup
        ps_join = ps_join.drop(columns=["mins_L5", "mins_L10"], errors="ignore")
    else:
        ps_join["mins_trend"] = 0.0
        ps_join["role_expanding"] = 0.0
        ps_join["role_shrinking"] = 0.0
    
    # 4C. GAME SCRIPT FACTORS
    
    # Expected game closeness (affects garbage time / stat accumulation)
    # Close games = more playing time, blowouts = less
    if "oof_spread_pred" in ps_join.columns:
        ps_join["expected_margin"] = ps_join["oof_spread_pred"].abs().fillna(5.0).astype("float32")
        ps_join["likely_close_game"] = (ps_join["expected_margin"] <= 6.0).astype("float32")
        ps_join["likely_blowout"] = (ps_join["expected_margin"] >= 12.0).astype("float32")
    else:
        ps_join["expected_margin"] = 5.0
        ps_join["likely_close_game"] = 0.5
        ps_join["likely_blowout"] = 0.2
    
    # Game pace impact on stats (faster pace = more possessions = more stats)
    if "matchup_pace" in ps_join.columns:
        # Already have pace_factor, enhance with interaction
        ps_join["pace_x_minutes"] = (ps_join.get("pace_factor", 1.0) * 
                                      ps_join[min_col].shift(1).rolling(5, min_periods=1).mean().fillna(25.0) / 25.0
                                      if min_col and min_col in ps_join.columns else 1.0)
        ps_join["pace_x_minutes"] = ps_join["pace_x_minutes"].fillna(1.0).clip(0.5, 2.0).astype("float32")
    else:
        ps_join["pace_x_minutes"] = 1.0
    
    # Home court advantage factor (varies by player)
    if "is_home" in ps_join.columns:
        # Some players perform better at home, some on road
        # Calculate player's home/away performance ratio
        if pts_col and pts_col in ps_join.columns:
            home_pts = ps_join[ps_join["is_home"] == 1].groupby(pid_col)[pts_col].mean()
            away_pts = ps_join[ps_join["is_home"] == 0].groupby(pid_col)[pts_col].mean()
            home_adv_ratio = (home_pts / away_pts.replace(0, np.nan)).fillna(1.0).clip(0.5, 2.0)
            # Convert to float first to avoid categorical issues
            ps_join["player_home_advantage"] = ps_join[pid_col].map(home_adv_ratio).astype("float32", errors='ignore')
            ps_join["player_home_advantage"] = ps_join["player_home_advantage"].fillna(1.0).astype("float32")
        else:
            ps_join["player_home_advantage"] = 1.0
    else:
        ps_join["player_home_advantage"] = 1.0
    
    # DEBUG: Log Phase 4 features created
    if verbose:
        phase4_expected = [
            "opp_def_vs_points", "opp_def_vs_assists", "opp_def_vs_rebounds", "opp_def_vs_threes",
            "rest_days", "is_b2b", "is_rested",
            "mins_trend", "role_expanding", "role_shrinking",
            "expected_margin", "likely_close_game", "likely_blowout",
            "pace_x_minutes", "player_home_advantage"
        ]
        phase4_created = [f for f in phase4_expected if f in ps_join.columns]
        phase4_missing = [f for f in phase4_expected if f not in ps_join.columns]
        log(f"  [DEBUG] Phase 4 features: {len(phase4_created)}/{len(phase4_expected)} created", True)
        if phase4_missing:
            log(f"    Missing: {phase4_missing}", True)
        else:
            log(f"    ✓ All Phase 4 features successfully created!", True)
    
    # ========================================================================
    # PHASE 5: POSITION-SPECIFIC FEATURES & ADVANCED OPTIMIZATIONS
    # ========================================================================
    # Expected improvement: +3-5% accuracy (especially for rebounds)
    
    # 5A. POSITION DETECTION AND CLASSIFICATION
    # Infer player position from stat patterns (guards = assists, centers = rebounds)
    if pts_col and ast_col and reb_col and pts_col in ps_join.columns and ast_col in ps_join.columns and reb_col in ps_join.columns:
        # Calculate player averages for classification
        player_avg_stats = ps_join.groupby(pid_col).agg({
            ast_col: 'mean',
            reb_col: 'mean',
            pts_col: 'mean',
            tpm_col: 'mean' if tpm_col and tpm_col in ps_join.columns else lambda x: 0
        }).reset_index()
        
        # Position classification based on stat ratios
        # Guards: high assists/rebounds ratio
        # Forwards: balanced
        # Centers: high rebounds/assists ratio
        player_avg_stats['ast_to_reb_ratio'] = (player_avg_stats[ast_col] / 
                                                 player_avg_stats[reb_col].replace(0, np.nan)).fillna(1.0)
        
        # Classify positions (simplified)
        def classify_position(row):
            ratio = row['ast_to_reb_ratio']
            if ratio > 1.5:
                return 'guard'
            elif ratio < 0.5:
                return 'center'
            else:
                return 'forward'
        
        player_avg_stats['position_inferred'] = player_avg_stats.apply(classify_position, axis=1)
        
        # Map back to ps_join
        position_map = dict(zip(player_avg_stats[pid_col], player_avg_stats['position_inferred']))
        # Map position (handle categorical dtype)
        ps_join['position'] = ps_join[pid_col].map(position_map)
        ps_join['position'] = ps_join['position'].astype(str).fillna('forward')
        
        # One-hot encode position
        ps_join['is_guard'] = (ps_join['position'] == 'guard').astype('float32')
        ps_join['is_forward'] = (ps_join['position'] == 'forward').astype('float32')
        ps_join['is_center'] = (ps_join['position'] == 'center').astype('float32')
        
    else:
        # Defaults if stats not available
        ps_join['position'] = 'forward'
        ps_join['is_guard'] = 0.33
        ps_join['is_forward'] = 0.34
        ps_join['is_center'] = 0.33
    
    # 5B. POSITION-SPECIFIC OPPONENT DEFENSE
    # Centers face different defense than guards
    if 'opp_def_vs_rebounds' in ps_join.columns and 'position' in ps_join.columns:
        # Amplify defensive difficulty for centers on rebounds
        ps_join['opp_def_vs_rebounds_adj'] = ps_join['opp_def_vs_rebounds'] * (1 + 0.2 * ps_join['is_center'])
        # Amplify defensive difficulty for guards on assists
        ps_join['opp_def_vs_assists_adj'] = ps_join['opp_def_vs_assists'] * (1 + 0.15 * ps_join['is_guard'])
    else:
        ps_join['opp_def_vs_rebounds_adj'] = ps_join.get('opp_def_vs_rebounds', 1.0)
        ps_join['opp_def_vs_assists_adj'] = ps_join.get('opp_def_vs_assists', 1.0)
    
    # 5C. STARTER STATUS (inferred from minutes played)
    # Starters typically play 28+ minutes, bench 10-25
    if min_col and min_col in ps_join.columns:
        # Calculate average minutes per player
        ps_join['avg_minutes'] = ps_join.groupby(pid_col)[min_col].transform(
            lambda x: x.shift(1).rolling(10, min_periods=3).mean()
        ).fillna(20.0)
        
        # Starter probability (smooth transition, not binary)
        ps_join['starter_prob'] = ((ps_join['avg_minutes'] - 15) / 15).clip(0, 1).astype('float32')
        
        # Minutes ceiling (starters can play 40+, bench rarely >30)
        ps_join['minutes_ceiling'] = (25 + 15 * ps_join['starter_prob']).astype('float32')
    else:
        ps_join['starter_prob'] = 0.5
        ps_join['minutes_ceiling'] = 32.0
    
    # 5D. INJURY RECENCY (games missed detection)
    # Players with gaps in their game log may be returning from injury
    if "date" in ps_join.columns and pid_col in ps_join.columns:
        ps_join = ps_join.sort_values([pid_col, "date"])
        
        # Games since last appearance (gap detection)
        ps_join['days_since_last_game'] = ps_join.groupby(pid_col)['rest_days'].transform(
            lambda x: x.fillna(3.0)
        ) if 'rest_days' in ps_join.columns else 3.0
        
        # Injury return flag (7+ days missed = likely injury)
        ps_join['likely_injury_return'] = (ps_join.get('days_since_last_game', 3) >= 7).astype('float32')
        
        # Games since return (performance typically improves after 2-3 games back)
        ps_join['games_since_injury'] = 0.0
        for pid in ps_join[pid_col].unique():
            mask = ps_join[pid_col] == pid
            injury_flags = ps_join.loc[mask, 'likely_injury_return'].values
            games_since = np.zeros(len(injury_flags))
            counter = 10  # Start high (not injured)
            for i in range(len(injury_flags)):
                if injury_flags[i] == 1:
                    counter = 0  # Reset on injury return
                games_since[i] = counter
                counter = min(counter + 1, 10)
            ps_join.loc[mask, 'games_since_injury'] = games_since
        
        ps_join['games_since_injury'] = ps_join['games_since_injury'].clip(0, 10).astype('float32')
    else:
        ps_join['likely_injury_return'] = 0.0
        ps_join['games_since_injury'] = 10.0
    
    # DEBUG: Log Phase 5 features
    if verbose:
        phase5_expected = [
            "position", "is_guard", "is_forward", "is_center",
            "opp_def_vs_rebounds_adj", "opp_def_vs_assists_adj",
            "starter_prob", "minutes_ceiling",
            "likely_injury_return", "games_since_injury"
        ]
        phase5_created = [f for f in phase5_expected if f in ps_join.columns]
        phase5_missing = [f for f in phase5_expected if f not in ps_join.columns]
        log(f"  [DEBUG] Phase 5 features: {len(phase5_created)}/{len(phase5_expected)} created", True)
        if phase5_missing:
            log(f"    Missing: {phase5_missing}", True)
        else:
            log(f"    ✓ All Phase 5 features successfully created!", True)
    
    # ========================================================================
    # PHASE 6: MOMENTUM & OPTIMIZATION FEATURES
    # ========================================================================
    # Expected improvement: +5-8% accuracy (trend detection, market signals)
    
    try:
        from optimization_features import (
            MomentumAnalyzer, MarketSignalAnalyzer,
            add_variance_features, add_ceiling_floor_features,
            add_context_weighted_averages, add_opponent_strength_features,
            add_fatigue_features
        )
        
        # 6A. MOMENTUM FEATURES (trend detection across timeframes)
        if verbose:
            log("  Adding momentum features (short/medium/long term trends)...", True)
        
        momentum_analyzer = MomentumAnalyzer(short_window=3, med_window=7, long_window=15)
        
        # Add momentum for each stat type
        stat_momentum_cols = []
        if pts_col and pts_col in ps_join.columns:
            ps_join = momentum_analyzer.add_momentum_features(ps_join, pts_col, pid_col)
            stat_momentum_cols.extend([f'{pts_col}_momentum_short', f'{pts_col}_momentum_med', 
                                      f'{pts_col}_momentum_long', f'{pts_col}_acceleration',
                                      f'{pts_col}_hot_streak', f'{pts_col}_cold_streak'])
        
        if reb_col and reb_col in ps_join.columns:
            ps_join = momentum_analyzer.add_momentum_features(ps_join, reb_col, pid_col)
            stat_momentum_cols.extend([f'{reb_col}_momentum_short', f'{reb_col}_momentum_med', 
                                      f'{reb_col}_momentum_long', f'{reb_col}_acceleration'])
        
        if ast_col and ast_col in ps_join.columns:
            ps_join = momentum_analyzer.add_momentum_features(ps_join, ast_col, pid_col)
            stat_momentum_cols.extend([f'{ast_col}_momentum_short', f'{ast_col}_momentum_med', 
                                      f'{ast_col}_momentum_long', f'{ast_col}_acceleration'])
        
        if min_col and min_col in ps_join.columns:
            ps_join = momentum_analyzer.add_momentum_features(ps_join, min_col, pid_col)
            stat_momentum_cols.extend([f'{min_col}_momentum_short', f'{min_col}_momentum_med', 
                                      f'{min_col}_momentum_long', f'{min_col}_acceleration'])
        
        # 6B. VARIANCE/CONSISTENCY FEATURES (player reliability)
        if verbose:
            log("  Adding variance/consistency features...", True)
        
        stat_cols_for_variance = [c for c in [pts_col, reb_col, ast_col, min_col] if c and c in ps_join.columns]
        ps_join = add_variance_features(ps_join, stat_cols_for_variance, pid_col, windows=[5, 10, 20])
        
        # 6C. CEILING/FLOOR FEATURES (upside/downside potential)
        if verbose:
            log("  Adding ceiling/floor (risk) features...", True)
        
        ps_join = add_ceiling_floor_features(ps_join, stat_cols_for_variance, pid_col, window=20)
        
        # 6D. CONTEXT-WEIGHTED AVERAGES (home/away splits)
        if verbose:
            log("  Adding context-weighted averages...", True)
        
        if home_col and home_col in ps_join.columns:
            ps_join = add_context_weighted_averages(ps_join, stat_cols_for_variance, pid_col, home_col)
        
        # 6E. OPPONENT STRENGTH FEATURES
        if verbose:
            log("  Adding opponent strength normalization...", True)
        
        opp_def_cols = [c for c in ps_join.columns if 'opp_def' in c.lower()]
        if opp_def_cols:
            ps_join = add_opponent_strength_features(ps_join, opp_def_cols)
        
        # 6F. FATIGUE FEATURES (workload, schedule density)
        if verbose:
            log("  Adding fatigue/workload features...", True)
        
        if min_col and min_col in ps_join.columns:
            ps_join = add_fatigue_features(ps_join, pid_col, min_col)
        
        # 6G. MARKET SIGNAL FEATURES (if betting data available)
        if 'spread_move' in ps_join.columns or 'total_move' in ps_join.columns:
            if verbose:
                log("  Adding market signal features (line movement, steam moves)...", True)
            
            market_analyzer = MarketSignalAnalyzer()
            # Market signals already partially added, enhance with steam detection
            if 'spread_move' in ps_join.columns:
                ps_join['is_steam_spread'] = (ps_join['spread_move'].abs() > 1.5).astype('float32')
            if 'total_move' in ps_join.columns:
                ps_join['is_steam_total'] = (ps_join['total_move'].abs() > 2.0).astype('float32')
        
        if verbose:
            phase6_created = [c for c in stat_momentum_cols if c in ps_join.columns]
            log(f"  [DEBUG] Phase 6 features: {len(phase6_created)}+ optimization features created", True)
            log(f"    ✓ Momentum tracking for points, rebounds, assists, minutes", True)
            log(f"    ✓ Variance/consistency + ceiling/floor analysis", True)
            log(f"    ✓ Context-weighted averages + opponent strength normalization", True)
            log(f"    ✓ Fatigue/workload tracking", True)
    
    except ImportError as e:
        if verbose:
            log(f"  [WARNING] Could not import optimization_features: {e}", True)
            log(f"    Continuing without momentum features...", True)

    # Add OOF game predictions
    oof = oof_games.copy()
    ps_join = ps_join.merge(oof[["gid", "oof_ml_prob", "oof_spread_pred"]], on="gid", how="left")

    # Merge Basketball Reference player priors (if provided)
    if priors_players is not None and not priors_players.empty:
        # MEMORY OPTIMIZATION: Filter priors to only seasons present in ps_join
        # This prevents loading priors for 1950-2001 when we only need 2002-2026
        if "season_end_year" in ps_join.columns and "season_for_game" in priors_players.columns:
            ps_seasons = set(ps_join["season_end_year"].dropna().unique())
            if len(ps_seasons) > 0:  # Only filter if we have valid seasons
                orig_priors_len = len(priors_players)
                priors_players = priors_players[priors_players["season_for_game"].isin(ps_seasons)].copy()
                if verbose and orig_priors_len > len(priors_players):
                    log(f"  Filtered priors from {orig_priors_len:,} to {len(priors_players):,} rows (seasons {min(ps_seasons):.0f}-{max(ps_seasons):.0f})", True)

        log(f"Merging Basketball Reference player priors ({len(priors_players):,} player-seasons, {len(priors_players.columns)} features)", verbose)

        # Ensure comparable dtypes for IDs
        try:
            if pid_col in ps_join.columns:
                ps_join[pid_col] = ps_join[pid_col].astype(str)
            if "player_id" in priors_players.columns:
                priors_players["player_id"] = priors_players["player_id"].astype(str)
        except Exception:
            pass

        # Merge on (player_id, season_end_year)
        # priors_players has: player_id, season_for_game (the season these priors apply to)
        # ps_join has: pid_col (player ID), season_end_year (the season of the game)
        merge_cols = [c for c in priors_players.columns if c not in ["player_id", "season_for_game", "player"]]

        ps_join = ps_join.merge(
            priors_players[["player_id", "season_for_game"] + merge_cols],
            left_on=[pid_col, "season_end_year"],
            right_on=["player_id", "season_for_game"],
            how="left",
            suffixes=("", "_prior")
        )

        # Drop duplicate ID columns from merge
        ps_join = ps_join.drop(columns=["player_id", "season_for_game"], errors="ignore")

        # Count how many rows have priors via ID merge (any merged prior column)
        prior_cols_present = [c for c in merge_cols if c in ps_join.columns]
        non_null_priors = ps_join[prior_cols_present].notna().any(axis=1).sum() if prior_cols_present else 0
        log(f"  ID-merge matched: {non_null_priors:,} / {len(ps_join):,} player-game rows ({(non_null_priors/len(ps_join)*100 if len(ps_join) else 0):.1f}%)", verbose)

        # Fallback: If 0 matched, try name-based merge (handles different ID systems BR vs NBA)
        if non_null_priors == 0:
            pri_name_col = "player" if "player" in priors_players.columns else None
            # Use the constructed full name join column when available
            join_name_col = "__name_join__" if "__name_join__" in ps_join.columns else name_full_col or name_col
            if pri_name_col and join_name_col and join_name_col in ps_join.columns:
                def _name_key(s: pd.Series) -> pd.Series:
                    """
                    Normalize names for fuzzy matching.
                    Handles: unicode, suffixes, punctuation, capitalization.
                    """
                    return (
                        s.astype(str)
                         .str.normalize('NFKD')  # Normalize unicode (handles Dončić → Doncic)
                         .str.encode('ascii', errors='ignore')
                         .str.decode('ascii')
                         .str.lower()
                         .str.replace(r"[^a-z]+", " ", regex=True)  # Remove all non-letters
                         .str.strip()
                         .str.replace(r"\s+", " ", regex=True)  # Collapse multiple spaces
                         .str.replace(r"\s+(jr|sr|ii|iii|iv|v)$", "", regex=True)  # Remove suffixes (Jr., Sr., II, III, IV, V)
                    )

                # Build normalized name keys
                ps_join["__name_key__"] = _name_key(ps_join[join_name_col])
                priors_players["__name_key__"] = _name_key(priors_players[pri_name_col])

                # Quick diagnostic: name overlap and season overlap
                if verbose:
                    # DEBUG: Show raw names before normalization
                    try:
                        raw_kaggle = ps_join[join_name_col].dropna().unique()[:10].tolist()
                        raw_priors = priors_players[pri_name_col].dropna().unique()[:10].tolist()
                        log(f"  DEBUG - Raw Kaggle names: {raw_kaggle}", True)
                        log(f"  DEBUG - Raw Priors names: {raw_priors}", True)
                    except Exception as e:
                        log(f"  DEBUG - Could not show raw names: {e}", True)

                    p_names = set(ps_join["__name_key__"].dropna().unique()[:5000])
                    r_names = set(priors_players["__name_key__"].dropna().unique()[:5000])
                    name_intersection = p_names & r_names
                    log(f"  Name overlap (sample up to 5k): {len(name_intersection)} common normalized names", True)
                    if len(name_intersection) > 0:
                        log(f"  Sample common names: {list(name_intersection)[:10]}", True)
                    else:
                        # Show samples from each to debug
                        log(f"  Sample Kaggle names (normalized): {list(p_names)[:10]}", True)
                        log(f"  Sample Priors names (normalized): {list(r_names)[:10]}", True)
                    
                    # Check season overlap
                    p_seasons = set(ps_join["season_end_year"].dropna().unique())
                    r_seasons = set(priors_players["season_for_game"].dropna().unique())
                    season_intersection = p_seasons & r_seasons
                    log(f"  Season overlap: {len(season_intersection)} common seasons", True)
                    if len(season_intersection) > 0:
                        log(f"  Common seasons: {sorted(season_intersection)[:10]}", True)
                    else:
                        log(f"  Kaggle seasons: {sorted(p_seasons)[:10]}", True)
                        log(f"  Priors seasons: {sorted(r_seasons)[:10]}", True)

                # Drop any previously merged prior columns to avoid duplicates
                prior_cols_present = [c for c in merge_cols if c in ps_join.columns]
                if prior_cols_present:
                    ps_join = ps_join.drop(columns=prior_cols_present, errors="ignore")

                # Initialize prior columns
                for col in merge_cols:
                    if col not in ps_join.columns:
                        ps_join[col] = np.nan

                # Merge by (name_key, season) - strict match first (BATCHED to avoid memory explosion)
                batch_size = 5000  # Process in smaller chunks
                if verbose:
                    log(f"  Merging priors in batches of {batch_size:,} rows to avoid memory issues", True)

                for start_idx in range(0, len(ps_join), batch_size):
                    end_idx = min(start_idx + batch_size, len(ps_join))
                    batch = ps_join.iloc[start_idx:end_idx][["__name_key__", "season_end_year"]].copy()

                    merged_batch = batch.merge(
                        priors_players[["__name_key__", "season_for_game"] + merge_cols],
                        left_on=["__name_key__", "season_end_year"],
                        right_on=["__name_key__", "season_for_game"],
                        how="left",
                        suffixes=("", "_prior")
                    )

                    # Deduplicate if merge created duplicates (keep first)
                    if len(merged_batch) > len(batch):
                        merged_batch = merged_batch.drop_duplicates(subset=["__name_key__", "season_end_year"], keep="first")

                    # Safety check: ensure merged_batch length matches batch length
                    if len(merged_batch) != len(batch):
                        # If lengths don't match, skip this batch
                        continue

                    # Update ps_join with merged columns
                    for col in merge_cols:
                        src_col = col if col in merged_batch.columns else (col + "_prior" if col + "_prior" in merged_batch.columns else None)
                        if src_col and src_col in merged_batch.columns:
                            # Ensure we're assigning the correct length
                            values_to_assign = merged_batch[src_col].values
                            if len(values_to_assign) == (end_idx - start_idx):
                                ps_join.iloc[start_idx:end_idx, ps_join.columns.get_loc(col)] = values_to_assign
                
                # For unmatched rows, try +/- 1 season (handles off-by-one issues) — chunked to save memory
                prior_cols_present = [c for c in merge_cols if c in ps_join.columns]
                matched_mask = ps_join[prior_cols_present].notna().any(axis=1) if prior_cols_present else pd.Series([False] * len(ps_join))
                
                # Chunked fuzzy matching (Option A fallback)
                batch_size = 1000
                for season_offset in [-1, 1]:
                    # Recompute unmatched indices each pass
                    if prior_cols_present:
                        matched_mask = ps_join[prior_cols_present].notna().any(axis=1)
                    else:
                        matched_mask = pd.Series([False] * len(ps_join), index=ps_join.index)
                    unmatched_idx = np.flatnonzero(~matched_mask.values)
                    if len(unmatched_idx) == 0:
                        break
                    if verbose:
                        log(f"  Fuzzy season match (offset {season_offset}): processing {len(unmatched_idx):,} rows in batches of {batch_size}", True)
                    for start_idx in range(0, len(unmatched_idx), batch_size):
                        idx_batch = unmatched_idx[start_idx:start_idx + batch_size]
                        batch = ps_join.iloc[idx_batch][["__name_key__", "season_end_year"]].copy()
                        batch["_batch_idx"] = range(len(batch))  # Track original position
                        batch["_temp_season"] = pd.to_numeric(batch["season_end_year"], errors="coerce") + season_offset
                        fuzzy = batch.merge(
                            priors_players[["__name_key__", "season_for_game"] + merge_cols],
                            left_on=["__name_key__", "_temp_season"],
                            right_on=["__name_key__", "season_for_game"],
                            how="left",
                            suffixes=("", "_fuzz")
                        )

                        # Deduplicate: keep first match per batch index
                        if "_batch_idx" in fuzzy.columns and len(fuzzy) > len(batch):
                            fuzzy = fuzzy.drop_duplicates("_batch_idx", keep="first")
                            fuzzy = fuzzy.sort_values("_batch_idx")

                        # Safety check: ensure length matches
                        if len(fuzzy) != len(idx_batch):
                            continue  # Skip this batch if length mismatch
                        # Update only where ps_join is missing
                        for col in merge_cols:
                            src_col = col if col in fuzzy.columns else (col + "_fuzz" if col + "_fuzz" in fuzzy.columns else None)
                            if src_col is None:
                                continue
                            vals = pd.to_numeric(fuzzy[src_col], errors="coerce") if fuzzy[src_col].dtype.kind in "OUSV" else fuzzy[src_col]
                            to_update = ps_join.iloc[idx_batch][col].isna() if col in ps_join.columns else pd.Series([True]*len(idx_batch))
                            if col not in ps_join.columns:
                                ps_join[col] = np.nan
                                to_update = pd.Series([True]*len(idx_batch))
                            if to_update.any():
                                ps_join.iloc[idx_batch, ps_join.columns.get_loc(col)] = np.where(to_update.values, vals.values, ps_join.iloc[idx_batch][col].values)
                
                ps_join = ps_join.drop(columns=["__name_key__"], errors="ignore")

                prior_cols_present = [c for c in merge_cols if c in ps_join.columns]
                name_matched = ps_join[prior_cols_present].notna().any(axis=1).sum() if prior_cols_present else 0
                log(f"  Name-merge matched: {name_matched:,} / {len(ps_join):,} player-game rows ({(name_matched/len(ps_join)*100 if len(ps_join) else 0):.1f}%)", verbose)

                # Report combined match rate (ID + name merging)
                if verbose:
                    log(f"  TOTAL matched (ID + name): {name_matched:,} / {len(ps_join):,} player-game rows ({(name_matched/len(ps_join)*100 if len(ps_join) else 0):.1f}%)", True)

                if name_matched == 0 and verbose:
                    # Diagnostics to help identify mismatch
                    try:
                        samp_ps = ps_join[[pid_col, name_col, "season_end_year"]].drop_duplicates().head(5)
                        samp_pr = priors_players[["player_id", pri_name_col, "season_for_game"]].drop_duplicates().head(5)
                        log(f"  Sample player IDs from players file: {samp_ps.to_dict(orient='records')}", True)
                        log(f"  Sample player IDs from priors: {samp_pr.to_dict(orient='records')}", True)
                    except Exception:
                        pass

    # Final safety check: Ensure is_home column exists after all merges
    if "is_home" not in ps_join.columns:
        ps_join["is_home"] = np.nan

    # Build frames
    frames: Dict[str, pd.DataFrame] = {}

    # Build base context columns - start with core columns
    base_ctx_cols = [
        "is_home",
        "season_end_year", "season_decade",
        "team_recent_pace", "team_off_strength", "team_def_strength", "team_recent_winrate",
        "opp_recent_pace",  "opp_off_strength",  "opp_def_strength",  "opp_recent_winrate",
        "match_off_edge", "match_def_edge", "match_pace_sum", "winrate_diff",
        "oof_ml_prob", "oof_spread_pred", "starter_flag",
        # Player fatigue features
        "days_rest", "player_b2b",
    ]

    # Add dynamic rolling stats columns (use actual column names that were resolved)
    for col in [pts_col, reb_col, ast_col, tpm_col]:
        if col:
            base_ctx_cols.extend([f"{col}_L3", f"{col}_L5", f"{col}_L10"])
            base_ctx_cols.extend([f"{col}_home_avg", f"{col}_away_avg"])

    # Add PHASE 1: Shot volume rolling stats (use actual column names)
    for col in [fga_col, three_pa_col, fta_col]:
        if col:
            base_ctx_cols.extend([f"{col}_L3", f"{col}_L5", f"{col}_L10"])

    # Add PHASE 1: Shot volume per-minute rates
    base_ctx_cols.extend(["rate_fga", "rate_3pa", "rate_fta"])

    # Add PHASE 1: Efficiency features
    base_ctx_cols.extend(["ts_pct_L5", "ts_pct_L10", "ts_pct_season", "three_pct_L5", "ft_pct_L5"])

    # Add PHASE 2: Matchup & context features
    base_ctx_cols.extend(["matchup_pace", "pace_factor", "def_matchup_difficulty", "offensive_environment"])

    # Add PHASE 3: Advanced rate stats
    base_ctx_cols.extend(["usage_rate_L5", "rebound_rate_L5", "assist_rate_L5"])
    
    # Add PHASE 4: Advanced context features
    base_ctx_cols.extend([
        "opp_def_vs_points", "opp_def_vs_assists", "opp_def_vs_rebounds", "opp_def_vs_threes",
        "rest_days", "is_b2b", "is_rested",
        "mins_trend", "role_expanding", "role_shrinking",
        "expected_margin", "likely_close_game", "likely_blowout", "pace_x_minutes", "player_home_advantage"
    ])
    
    # Add PHASE 5: Position-specific and status features
    base_ctx_cols.extend([
        "position", "is_guard", "is_forward", "is_center",
        "opp_def_vs_rebounds_adj", "opp_def_vs_assists_adj",
        "starter_prob", "minutes_ceiling", "avg_minutes",
        "likely_injury_return", "games_since_injury", "days_since_last_game"
    ])
    
    # Add PHASE 6: Momentum features (if available)
    momentum_features = []
    for stat in [pts_col, reb_col, ast_col, min_col]:
        if stat:
            momentum_features.extend([
                f"{stat}_momentum_short", f"{stat}_momentum_med", f"{stat}_momentum_long",
                f"{stat}_acceleration", f"{stat}_hot_streak", f"{stat}_cold_streak"
            ])
    base_ctx_cols.extend(momentum_features)
    
    # Add PHASE 6: Variance/consistency features
    variance_features = []
    for stat in [pts_col, reb_col, ast_col, min_col]:
        if stat:
            for window in [5, 10, 20]:
                variance_features.extend([f"{stat}_cv_{window}", f"{stat}_stability_{window}"])
    base_ctx_cols.extend(variance_features)
    
    # Add PHASE 6: Ceiling/floor features
    ceiling_floor_features = []
    for stat in [pts_col, reb_col, ast_col, min_col]:
        if stat:
            ceiling_floor_features.extend([f"{stat}_ceiling", f"{stat}_floor", f"{stat}_range"])
    base_ctx_cols.extend(ceiling_floor_features)
    
    # Add PHASE 6: Context-weighted averages (home/away specific)
    context_features = []
    for stat in [pts_col, reb_col, ast_col, min_col]:
        if stat and home_col:
            # These will be dynamically named based on home_col values
            context_features.extend([f"{stat}_avg_{home_col}_0", f"{stat}_avg_{home_col}_1"])
    base_ctx_cols.extend(context_features)
    
    # Add PHASE 6: Fatigue features
    fatigue_features = [
        "cumulative_mins_3", "cumulative_mins_7", "cumulative_mins_14",
        "avg_mins_last_7", "workload_spike",
        "games_last_7d", "games_last_14d", "games_last_30d"
    ]
    base_ctx_cols.extend(fatigue_features)
    
    # Add market signal features (if available)
    base_ctx_cols.extend(["is_steam_spread", "is_steam_total"])
    
    # ===========================================================================
    # ADD PHASE 7 FEATURES: Situational Context & Adaptive Temporal
    # ===========================================================================
    print("\n🚀 Adding Phase 7 features (situational context, opponent history, adaptive weights)...")
    try:
        # Build stat_cols list safely
        stat_cols = []
        if pts_col and pts_col in ps_join.columns:
            stat_cols.append(pts_col)
        if reb_col and reb_col in ps_join.columns:
            stat_cols.append(reb_col)
        if ast_col and ast_col in ps_join.columns:
            stat_cols.append(ast_col)
        if threes_col and threes_col in ps_join.columns:
            stat_cols.append(threes_col)
        
        ps_join = add_phase7_features(
            ps_join,
            stat_cols=stat_cols,
            season_col='season_end_year',
            date_col=date_col
        )
        
        # Add Phase 7 feature names to base_ctx_cols
        phase7_features = [
            # Season context
            'games_into_season', 'games_remaining_in_season', 'is_early_season',
            'is_late_season', 'is_mid_season', 'season_fatigue_factor',
            # Schedule density
            'days_since_last_game', 'games_in_last_7_days', 'avg_rest_days_L5', 'is_compressed_schedule',
            # Revenge games
            'is_revenge_game'
        ]
        
        # Opponent history features (per stat)
        for stat in [pts_col, reb_col, ast_col, three_col]:
            if stat:
                phase7_features.extend([
                    f'{stat}_vs_opponent_career',
                    f'{stat}_vs_opponent_L3',
                    f'{stat}_vs_opponent_trend'
                ])
        
        # Adaptive temporal features (per stat, per window)
        for stat in [pts_col, reb_col, ast_col, three_col]:
            if stat:
                for window in [5, 10, 15]:
                    phase7_features.extend([
                        f'{stat}_adaptive_L{window}',
                        f'{stat}_consistency_L{window}'
                    ])
        
        base_ctx_cols.extend(phase7_features)
        print(f"✅ Phase 7 features added! Total new features: {len(phase7_features)}")
        
    except Exception as e:
        print(f"⚠️ Phase 7 feature addition failed: {e}")
        print("   Continuing without Phase 7 features...")

    # Filter to only columns that actually exist in ps_join
    base_ctx_cols = [c for c in base_ctx_cols if c in ps_join.columns]

    # Minutes
    if min_col and min_col in ps_join.columns:
        minutes_df = ps_join[[gid_col, pid_col] + base_ctx_cols + [
            "min_prev_mean5", "min_prev_mean10", "min_prev_last1", min_col
        ]].copy()
        # carry ordering date for time-safe OOF
        if date_col and date_col in ps_join.columns:
            minutes_df["__order_date__"] = ps_join[date_col]
        minutes_df = minutes_df.dropna(subset=[min_col]).reset_index(drop=True)
        # standardized keys for later merges
        minutes_df["gid_key"] = minutes_df[gid_col]
        minutes_df["pid_key"] = minutes_df[pid_col]
        frames["minutes"] = minutes_df
    else:
        frames["minutes"] = pd.DataFrame()

    # Helper builder
    def build_stat_frame(stat_col: Optional[str], rate_col: str) -> pd.DataFrame:
        if not stat_col or stat_col not in ps_join.columns:
            return pd.DataFrame()
        cols = [gid_col, pid_col] + base_ctx_cols + [rate_col]
        df = ps_join[cols].copy()
        if min_col and min_col in ps_join.columns:
            df["minutes"] = ps_join[min_col]
        # standardized keys for later merges
        df["gid_key"] = df[gid_col]
        df["pid_key"] = df[pid_col]
        df["label"] = pd.to_numeric(ps_join[stat_col], errors="coerce")
        df = df.dropna(subset=["label"]).reset_index(drop=True)
        return df

    frames["points"]   = build_stat_frame(pts_col, "rate_pts")
    frames["rebounds"] = build_stat_frame(reb_col, "rate_reb")
    frames["assists"]  = build_stat_frame(ast_col, "rate_ast")
    frames["threes"]   = build_stat_frame(tpm_col, "rate_3pm")

    # Counts
    print(_sec("Player training frames"))
    for k, v in frames.items():
        print(f"- {k}: {len(v):,} rows")

    return frames

# ---------------- Fit player models ----------------

def _fit_minutes_model(df: pd.DataFrame, seed: int, verbose: bool) -> Tuple[object, Dict[str, float]]:
    if df.empty:
        mdl = DummyRegressor(strategy="mean").fit([[0]], [24.0])
        return mdl, {"rows": 0, "rmse": float("nan"), "mae": float("nan")}
    features = [
        # team + opponent context
        "is_home",
        "season_end_year", "season_decade",
        "team_recent_pace", "team_off_strength", "team_def_strength", "team_recent_winrate",
        "opp_recent_pace",  "opp_off_strength",  "opp_def_strength",  "opp_recent_winrate",
        # matchup + game signals
        "match_off_edge", "match_def_edge", "match_pace_sum", "winrate_diff",
        "oof_ml_prob", "oof_spread_pred", "starter_flag",
        # minutes trends
        "min_prev_mean5", "min_prev_mean10", "min_prev_last1",
        # NEW: player fatigue features
        "days_rest", "player_b2b"
    ]
    features = [f for f in features if f in df.columns]
    X = df[features].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32")

    # Find minutes column (could be numMinutes, minutes, mins, min, or MIN)
    min_col_candidates = ["numMinutes", "minutes", "mins", "min", "MIN"]
    min_col = None
    for col in min_col_candidates:
        if col in df.columns:
            min_col = col
            break

    if min_col is None:
        raise ValueError(f"Minutes column not found. Available columns: {list(df.columns)}")

    y = pd.to_numeric(df[min_col], errors="coerce").astype(float).values
    w = pd.to_numeric(df.get("sample_weight", pd.Series(np.ones(len(df)))), errors="coerce").fillna(1.0).values

    n = len(X)
    split = max(1, min(int(n * 0.8), n - 1))
    X_tr, X_val = X.iloc[:split, :], X.iloc[split:, :]
    y_tr, y_val = y[:split], y[split:]
    w_tr, w_val = w[:split], w[split:]

    if _HAS_LGB:
        reg = lgb.LGBMRegressor(
            objective="regression",
            learning_rate=0.05, num_leaves=31, max_depth=-1,
            colsample_bytree=0.9, subsample=0.8, subsample_freq=5,
            n_estimators=800, random_state=seed, n_jobs=N_JOBS,
            force_col_wise=True, verbosity=-1
        )
    else:
        reg = HistGradientBoostingRegressor(
            learning_rate=0.06, max_depth=None, max_iter=400, random_state=seed,
            validation_fraction=0.2, early_stopping=True
        )
    reg.fit(X_tr, y_tr, sample_weight=w_tr)
    y_pred = reg.predict(X_val) if len(X_val) else np.array([])
    rmse = float(math.sqrt(mean_squared_error(y_val, y_pred))) if len(y_pred) else float("nan")
    mae = float(mean_absolute_error(y_val, y_pred)) if len(y_pred) else float("nan")

    print(_sec("Minutes model metrics (validation)"))
    print(f"- RMSE={_fmt(rmse)}, MAE={_fmt(mae)}")
    return reg, {"rows": int(n), "rmse": rmse, "mae": mae}

def _fit_stat_model(df: pd.DataFrame, seed: int, verbose: bool, name: str, use_neural: bool = False, neural_epochs: int = 100, use_gpu: bool = False) -> Tuple[object, Optional[object], Dict[str, float]]:
    if df.empty:
        mdl = DummyRegressor(strategy="mean").fit([[0]], [0.0])
        return mdl, None, {"rows": 0, "rmse": float("nan"), "mae": float("nan")}
    features = [
        # team + opponent context
        "is_home",
        "season_end_year", "season_decade",
        "team_recent_pace", "team_off_strength", "team_def_strength", "team_recent_winrate",
        "opp_recent_pace",  "opp_off_strength",  "opp_def_strength",  "opp_recent_winrate",
        # matchup + game signals
        "match_off_edge", "match_def_edge", "match_pace_sum", "winrate_diff",
        "oof_ml_prob", "oof_spread_pred", "starter_flag",
        # player rates
        "rate_pts", "rate_reb", "rate_ast", "rate_3pm",
        # NEW: player fatigue features
        "days_rest", "player_b2b",
        # NEW: enhanced rolling trends (last 3, 5, 10 games)
        "points_L3", "points_L5", "points_L10",
        "rebounds_L3", "rebounds_L5", "rebounds_L10",
        "assists_L3", "assists_L5", "assists_L10",
        "threepoint_goals_L3", "threepoint_goals_L5", "threepoint_goals_L10",
        # PHASE 1.1: shot volume rolling trends
        "fieldGoalsAttempted_L3", "fieldGoalsAttempted_L5", "fieldGoalsAttempted_L10",
        "threePointersAttempted_L3", "threePointersAttempted_L5", "threePointersAttempted_L10",
        "freeThrowsAttempted_L3", "freeThrowsAttempted_L5", "freeThrowsAttempted_L10",
        # PHASE 1.1: shot volume per-minute rates
        "rate_fga", "rate_3pa", "rate_fta",
        # PHASE 1.2: efficiency features (True Shooting % + shooting %s)
        "ts_pct_L5", "ts_pct_L10", "ts_pct_season",
        "three_pct_L5", "ft_pct_L5",
        # PHASE 2: matchup & context features
        "matchup_pace", "pace_factor",
        "def_matchup_difficulty", "offensive_environment",
        # PHASE 3: advanced rate stats
        "usage_rate_L5", "rebound_rate_L5", "assist_rate_L5",
        # NEW: home/away splits
        "points_home_avg", "points_away_avg",
        "rebounds_home_avg", "rebounds_away_avg",
        "assists_home_avg", "assists_away_avg",
        "threepoint_goals_home_avg", "threepoint_goals_away_avg",
    ]
    # Stacking feature priority: minutes_oof first if available, else minutes if present
    if "minutes_oof" in df.columns:
        features.insert(0, "minutes_oof")
    elif "minutes" in df.columns:
        features.append("minutes")

    # DEBUG: Log which Phase features are missing
    if verbose:
        phase1_requested = ["fieldGoalsAttempted_L5", "threePointersAttempted_L5", "freeThrowsAttempted_L5",
                           "rate_fga", "rate_3pa", "rate_fta", "ts_pct_L5", "ts_pct_L10", "three_pct_L5"]
        phase2_requested = ["matchup_pace", "pace_factor", "def_matchup_difficulty", "offensive_environment"]
        phase3_requested = ["usage_rate_L5", "rebound_rate_L5", "assist_rate_L5"]
        phase4_requested = ["opp_def_vs_points", "rest_days", "is_b2b", "mins_trend", "expected_margin"]
        phase5_requested = ["is_guard", "is_center", "starter_prob", "likely_injury_return", "games_since_injury"]

        phase1_missing = [f for f in phase1_requested if f not in df.columns]
        phase2_missing = [f for f in phase2_requested if f not in df.columns]
        phase3_missing = [f for f in phase3_requested if f not in df.columns]
        phase4_missing = [f for f in phase4_requested if f not in df.columns]
        phase5_missing = [f for f in phase5_requested if f not in df.columns]

        if phase1_missing or phase2_missing or phase3_missing or phase4_missing or phase5_missing:
            log(f"  [DEBUG] Phase features MISSING from dataframe:", True)
            if phase1_missing:
                log(f"    Phase 1 missing ({len(phase1_missing)}): {phase1_missing}", True)
            if phase2_missing:
                log(f"    Phase 2 missing ({len(phase2_missing)}): {phase2_missing}", True)
            if phase3_missing:
                log(f"    Phase 3 missing ({len(phase3_missing)}): {phase3_missing}", True)
            if phase4_missing:
                log(f"    Phase 4 missing ({len(phase4_missing)}): {phase4_missing}", True)
            if phase5_missing:
                log(f"    Phase 5 missing ({len(phase5_missing)}): {phase5_missing}", True)
        else:
            log(f"  [DEBUG] ✅ ALL Phase 1-5 features present in dataframe!", True)

    features = [f for f in features if f in df.columns]

    X = df[features].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32")
    y = pd.to_numeric(df["label"], errors="coerce").astype(float).values
    w = pd.to_numeric(df.get("sample_weight", pd.Series(np.ones(len(df)))), errors="coerce").fillna(1.0).values

    n = len(X)
    split = max(1, min(int(n * 0.8), n - 1))
    X_tr, X_val = X.iloc[:split, :], X.iloc[split:, :]
    y_tr, y_val = pd.Series(y[:split]), pd.Series(y[split:])
    w_tr, w_val = w[:split], w[split:]

    # Use neural hybrid if requested and available
    if use_neural and TABNET_AVAILABLE and TORCH_AVAILABLE:
        print(f"\n{'='*60}")
        print(f"Training Neural Hybrid Model for {name}")
        print(f"{'='*60}")
        
        model = NeuralHybridPredictor(name, use_gpu=use_gpu)
        model.fit(X_tr, y_tr, X_val, y_val, epochs=neural_epochs, batch_size=1024)
        
        y_pred = model.predict(X_val)
        rmse = float(math.sqrt(mean_squared_error(y_val, y_pred))) if len(y_pred) else float("nan")
        mae = float(mean_absolute_error(y_val, y_pred)) if len(y_pred) else float("nan")
        
        # Get sigma model from neural hybrid
        reg_sig = model.sigma_model
        
        print(_sec(f"{name.capitalize()} neural hybrid metrics (validation)"))
        print(f"- RMSE={_fmt(rmse)}, MAE={_fmt(mae)}")
        return model, reg_sig, {"rows": int(n), "rmse": rmse, "mae": mae}

    # Fallback to LightGBM
    if _HAS_LGB:
        # PHASE 1: Heavy regularization to prevent overfitting with volume+efficiency features
        reg = lgb.LGBMRegressor(
            objective="regression",
            learning_rate=0.1,        # increased from 0.05
            num_leaves=31,
            max_depth=3,              # NEW - shallow trees (was -1)
            min_child_samples=100,    # NEW - require more data per leaf
            colsample_bytree=0.7,     # reduced from 0.9
            subsample=0.7,            # reduced from 0.8
            subsample_freq=5,
            reg_alpha=0.5,            # NEW - L1 regularization
            reg_lambda=0.5,           # NEW - L2 regularization
            n_estimators=50,          # reduced from 800
            random_state=seed,
            n_jobs=N_JOBS,
            force_col_wise=True,
            verbosity=-1
        )
    else:
        reg = HistGradientBoostingRegressor(
            learning_rate=0.06, max_depth=None, max_iter=400, random_state=seed,
            validation_fraction=0.2, early_stopping=True
        )
    reg.fit(X_tr, y_tr, sample_weight=w_tr)
    y_pred = reg.predict(X_val) if len(X_val) else np.array([])
    rmse = float(math.sqrt(mean_squared_error(y_val, y_pred))) if len(y_pred) else float("nan")
    mae = float(mean_absolute_error(y_val, y_pred)) if len(y_pred) else float("nan")

    # Train heteroskedastic sigma model on training residuals
    try:
        y_tr_pred = reg.predict(X_tr)
        sigma_target = np.abs(y_tr - y_tr_pred)
        if _HAS_LGB:
            reg_sig = lgb.LGBMRegressor(objective="regression_l1", learning_rate=0.05, num_leaves=31, max_depth=-1,
                                        colsample_bytree=0.9, subsample=0.8, subsample_freq=5,
                                        n_estimators=600, random_state=seed + 999, n_jobs=-1,
                                        force_col_wise=True, verbosity=-1)
        else:
            reg_sig = HistGradientBoostingRegressor(learning_rate=0.06, max_depth=None, max_iter=300, random_state=seed + 999,
                                                    validation_fraction=0.2, early_stopping=True)
        reg_sig.fit(X_tr, sigma_target, sample_weight=w_tr)
    except Exception:
        reg_sig = None

    print(_sec(f"{name.capitalize()} model metrics (validation)"))
    print(f"- RMSE={_fmt(rmse)}, MAE={_fmt(mae)}")
    return reg, reg_sig, {"rows": int(n), "rmse": rmse, "mae": mae}

# ---------------- Betting Odds and Priors Loaders ----------------

def load_team_abbrev_map(source_root: Path, verbose: bool, source_name: str = "reference data") -> Dict[Tuple[int, str], str]:
    """
    Load Team Abbrev.csv and return dict: (season, team_name) -> abbreviation
    Handles renames across seasons (NOH->NOP, CHH/CHA, SEA->OKC, etc.)
    
    This is a general utility that can load from any source (priors, odds, etc.)
    """
    # Accept a variety of filename patterns (case-insensitive)
    candidates = list(source_root.glob("*abbrev*.csv")) + list(source_root.glob("*team*abbrev*.csv"))
    abbrev_path = None
    if candidates:
        # prefer exact match "Team Abbrev.csv" if present
        for c in candidates:
            if c.name.lower() == "team abbrev.csv" or c.name.lower() == "team_abbrev.csv":
                abbrev_path = c
                break
        if abbrev_path is None:
            abbrev_path = candidates[0]
    if abbrev_path is None or not abbrev_path.exists():
        log(f"Info: Team Abbrev CSV not found in {source_name} at {source_root}. Returning empty map (team matching may be limited).", verbose)
        return {}

    df = pd.read_csv(abbrev_path, low_memory=False)
    # Filter NBA non-playoff
    if "lg" in df.columns:
        df = df[df["lg"] == "NBA"]
    if "playoffs" in df.columns:
        df = df[df["playoffs"] == False]

    abbrev_map: Dict[Tuple[int, str], str] = {}
    for _, row in df.iterrows():
        # Accept alternate column names for season/team/abbreviation
        season = 0
        if "season" in row and pd.notna(row["season"]):
            try:
                season = int(row["season"])
            except Exception:
                season = int(pd.to_numeric(row["season"], errors="coerce") or 0)
        team = ""
        for key in ("team", "team_name", "teamname", "teamfull"):
            if key in row and pd.notna(row[key]):
                team = str(row[key]).strip()
                break
        abbr = ""
        for key in ("abbreviation", "abbr", "team_abbr"):
            if key in row and pd.notna(row[key]):
                abbr = str(row[key]).strip()
                break
        if season > 0 and team and abbr:
            abbrev_map[(season, _norm(team))] = abbr

    log(f"Loaded {len(abbrev_map)} team abbreviation mappings from {source_name}", verbose)
    return abbrev_map

def resolve_team_abbrev(team_name: str, season: int, abbrev_map: Dict[Tuple[int, str], str]) -> Optional[str]:
    """Resolve team name to abbreviation for a given season"""
    key = (season, _norm(team_name))
    if key in abbrev_map:
        return abbrev_map[key]
    # Fallback: try finding any season with same team name
    for (s, tn), abbr in abbrev_map.items():
        if tn == _norm(team_name):
            return abbr
    return None

def load_betting_odds(odds_path: Path, verbose: bool) -> pd.DataFrame:
    """
    Load betting odds dataset and normalize to canonical schema.
    Returns DataFrame with columns: gid, game_date_utc, season_end_year, home_abbrev, away_abbrev,
    market_home_ml, market_away_ml, market_spread, spread_move, market_total, total_move,
    market_implied_home, market_implied_away
    
    Note: Team abbreviations come from the odds file itself or TeamStatistics mapping,
    NOT from Basketball Reference priors (those are separate statistical context).
    """
    if not odds_path.exists():
        log(f"Betting odds file not found: {odds_path}", verbose)
        return pd.DataFrame()

    log(_sec("Loading betting odds dataset"), verbose)
    df = pd.read_csv(odds_path, low_memory=False)
    log(f"Loaded {len(df):,} odds rows from {odds_path.name}", verbose)

    # Detect and normalize columns (flexible mapping)
    def find_col(patterns: List[str]) -> Optional[str]:
        for pat in patterns:
            matches = [c for c in df.columns if pat.lower() in c.lower()]
            if matches:
                return matches[0]
        return None

    gid_col = find_col(["gameid", "game_id", "gid"])
    date_col = find_col(["date", "game_date", "gamedate"])
    home_col = find_col(["hometeam", "home_team", "home"])
    away_col = find_col(["awayteam", "away_team", "away"])
    book_col = find_col(["book", "bookmaker", "sportsbook"])

    # Check if odds file already has abbreviations
    home_abbr_col = find_col(["home_abbrev", "home_abbr", "home_team_abbr"])
    away_abbr_col = find_col(["away_abbrev", "away_abbr", "away_team_abbr"])

    # Moneyline
    home_ml_open_col = find_col(["home_ml_open", "homemlopen", "home_moneyline_open"])
    home_ml_close_col = find_col(["home_ml_close", "homemlclose", "home_moneyline_close", "home_ml"])
    away_ml_open_col = find_col(["away_ml_open", "awaymlopen", "away_moneyline_open"])
    away_ml_close_col = find_col(["away_ml_close", "awaymlclose", "away_moneyline_close", "away_ml"])

    # Spread
    spread_open_col = find_col(["spread_open", "spreadopen", "line_open", "handicap_open"])
    spread_close_col = find_col(["spread_close", "spreadclose", "line_close", "handicap_close", "spread", "line"])

    # Total
    total_open_col = find_col(["total_open", "totalopen", "ou_open", "over_under_open"])
    total_close_col = find_col(["total_close", "totalclose", "ou_close", "over_under_close", "total", "ou"])

    # Build canonical df
    canonical = pd.DataFrame()

    if gid_col:
        canonical["gid"] = _id_to_str(df[gid_col])

    if date_col:
        canonical["game_date_utc"] = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_convert(None)
        canonical["season_end_year"] = _season_from_date(canonical["game_date_utc"]).astype("float32")

    if home_col:
        canonical["home_team"] = df[home_col].astype(str).str.strip()
    if away_col:
        canonical["away_team"] = df[away_col].astype(str).str.strip()

    # Use abbreviations from odds file if present
    if home_abbr_col:
        canonical["home_abbrev"] = df[home_abbr_col].astype(str).str.strip().str.upper()
    if away_abbr_col:
        canonical["away_abbrev"] = df[away_abbr_col].astype(str).str.strip().str.upper()

    if book_col:
        canonical["book"] = df[book_col].astype(str).str.strip()

    # Odds columns (numeric)
    for src, dst in [
        (home_ml_open_col, "home_ml_open"), (home_ml_close_col, "home_ml_close"),
        (away_ml_open_col, "away_ml_open"), (away_ml_close_col, "away_ml_close"),
        (spread_open_col, "spread_open"), (spread_close_col, "spread_close"),
        (total_open_col, "total_open"), (total_close_col, "total_close"),
    ]:
        if src:
            canonical[dst] = pd.to_numeric(df[src], errors="coerce")

    # If odds file lacks team identifiers but gid looks like YYYYMMDD_HOME_AWAY, parse abbrevs
    if ("home_abbrev" not in canonical.columns or canonical["home_abbrev"].isna().all()) and "gid" in canonical.columns:
        try:
            parts = canonical["gid"].astype(str).str.split("_", n=2, expand=True)
            if parts.shape[1] == 3:
                date_part = parts[0]
                home_part = parts[1].str.strip().str.upper()
                away_part = parts[2].str.strip().str.upper()
                # Simple date sanity: first part is 8 digits
                mask = date_part.str.match(r"^\d{8}$")
                canonical.loc[mask, "home_abbrev"] = home_part[mask]
                canonical.loc[mask, "away_abbrev"] = away_part[mask]
        except Exception:
            pass

    # Compute consensus closing values per game (median across books)
    if "gid" not in canonical.columns and "game_date_utc" in canonical.columns:
        # Generate synthetic gid from date + teams
        home_part = canonical.get("home_abbrev", canonical.get("home_team", pd.Series(["UNK"] * len(canonical))))
        away_part = canonical.get("away_abbrev", canonical.get("away_team", pd.Series(["UNK"] * len(canonical))))
        canonical["gid"] = (
            canonical["game_date_utc"].dt.strftime("%Y%m%d") + "_" +
            home_part.fillna("UNK").astype(str) + "_" +
            away_part.fillna("UNK").astype(str)
        )

    # Aggregate to one row per game (consensus medians)
    agg_dict = {}
    for col in ["season_end_year", "home_ml_close", "away_ml_close", "spread_close", "total_close",
                "home_ml_open", "away_ml_open", "spread_open", "total_open"]:
        if col in canonical.columns:
            agg_dict[col] = "median"

    # Keep first non-null for identifiers
    for col in ["game_date_utc", "home_team", "away_team", "home_abbrev", "away_abbrev"]:
        if col in canonical.columns:
            agg_dict[col] = "first"

    if not agg_dict:
        log("Warning: No odds columns found to aggregate", verbose)
        return pd.DataFrame()

    game_odds = canonical.groupby("gid", as_index=False).agg(agg_dict)

    # Compute market priors and movement
    def american_to_prob(ml: float) -> float:
        """Convert American odds to implied probability"""
        if pd.isna(ml):
            return np.nan
        if ml < 0:
            return -ml / (-ml + 100)
        else:
            return 100 / (ml + 100)

    if "home_ml_close" in game_odds.columns:
        game_odds["market_home_ml"] = game_odds["home_ml_close"]
        game_odds["market_implied_home"] = game_odds["home_ml_close"].apply(american_to_prob)
    if "away_ml_close" in game_odds.columns:
        game_odds["market_away_ml"] = game_odds["away_ml_close"]
        game_odds["market_implied_away"] = game_odds["away_ml_close"].apply(american_to_prob)

    # No-vig normalization
    if "market_implied_home" in game_odds.columns and "market_implied_away" in game_odds.columns:
        total_vig = game_odds["market_implied_home"] + game_odds["market_implied_away"]
        game_odds["market_implied_home"] = game_odds["market_implied_home"] / total_vig
        game_odds["market_implied_away"] = game_odds["market_implied_away"] / total_vig

    if "spread_close" in game_odds.columns:
        game_odds["market_spread"] = game_odds["spread_close"]
        if "spread_open" in game_odds.columns:
            game_odds["spread_move"] = game_odds["spread_close"] - game_odds["spread_open"]

    if "total_close" in game_odds.columns:
        game_odds["market_total"] = game_odds["total_close"]
        if "total_open" in game_odds.columns:
            game_odds["total_move"] = game_odds["total_close"] - game_odds["total_open"]

    # Keep only needed columns
    keep_cols = ["gid", "game_date_utc", "season_end_year", "home_abbrev", "away_abbrev",
                 "market_home_ml", "market_away_ml", "market_spread", "spread_move",
                 "market_total", "total_move", "market_implied_home", "market_implied_away"]
    keep_cols = [c for c in keep_cols if c in game_odds.columns]
    game_odds = game_odds[keep_cols]

    log(f"Processed {len(game_odds):,} unique games with market odds", verbose)
    return game_odds

def load_basketball_reference_priors(priors_root: Path, verbose: bool, seasons_to_keep: Optional[Set[int]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Basketball Reference priors bundle from 7 CSVs (2 team + 4 player + 1 mapping).
    Returns: (priors_players, priors_teams) with season_for_game = season + 1 (shifted for leakage safety)
    
    Team Priors (2 CSVs):
    - Team Summaries.csv: O/D ratings, pace, SRS, four factors, W/L records
    - Team Abbrev.csv: Team name → abbreviation mapping (loaded separately by load_team_abbrev_map)
    
    Player Priors (4 CSVs, ~60-65 features total):
    - Per 100 Poss.csv: Core rate stats, shooting %, O/D ratings
    - Advanced.csv: PER, TS%, USG%, Win Shares, BPM, VORP
    - Player Shooting.csv: Shot zones, corner 3%, dunks, assisted rates
    - Player Play By Play.csv: Position %, on-court +/-, fouls
    
    These priors provide GENERAL STATISTICAL CONTEXT (O/D ratings, pace, SRS, advanced metrics)
    that augment the rolling game-level features. They are NOT betting odds.
    """
    log(_sec("Loading Basketball Reference priors"), verbose)

    # Helper: parse Basketball Reference season values to end-year integers
    def _parse_season_end_year(season_series: pd.Series) -> pd.Series:
        def _one(x) -> Optional[int]:
            if pd.isna(x):
                return None
            try:
                # If numeric-like
                val = pd.to_numeric(x, errors="ignore")
                if isinstance(val, (int, np.integer)):
                    return int(val)
                if isinstance(val, float) and not np.isnan(val):
                    return int(val)
            except Exception:
                pass
            s = str(x).strip()
            # Examples: '2011-12', '2011-2012', '2012'
            if "-" in s:
                parts = s.split("-")
                last = parts[-1]
                last = ''.join(ch for ch in last if ch.isdigit())
                if len(last) == 2:
                    yy = int(last)
                    return 2000 + yy if yy < 50 else 1900 + yy
                if len(last) == 4:
                    return int(last)
            # Fallback: digits only
            digits = ''.join(ch for ch in s if ch.isdigit())
            return int(digits) if digits else None
        parsed = season_series.apply(_one)
        return parsed.astype('float64')

    # Team priors
    # Accept multiple filename variants for team/player priors
    team_summaries_candidates = list(priors_root.glob("*team*summary*.csv")) + list(priors_root.glob("*team*summaries*.csv"))
    team_summaries_path = team_summaries_candidates[0] if team_summaries_candidates else (priors_root / "Team Summaries.csv")
    team_per100_path = priors_root / "Team Stats Per 100 Poss.csv"

    priors_teams = pd.DataFrame()

    if team_summaries_path and team_summaries_path.exists():
        ts = pd.read_csv(team_summaries_path, low_memory=False)
        # Filter NBA
        if "lg" in ts.columns:
            ts = ts[ts["lg"] == "NBA"]
        # Prefer non-playoff rows, but for incomplete seasons keep playoff rows as fallback
        if "playoffs" in ts.columns:
            # For each team-season, prefer playoffs=False if available, otherwise keep playoffs=True
            ts_non_playoff = ts[ts["playoffs"] == False].copy()
            ts_playoff = ts[ts["playoffs"] == True].copy()
            # Mark teams that have non-playoff data
            if "season" in ts.columns:
                non_playoff_keys = set(zip(ts_non_playoff["season"], ts_non_playoff.get("team", [""] * len(ts_non_playoff))))
                ts_playoff_fallback = ts_playoff[
                    ~ts_playoff.apply(lambda r: (r.get("season"), r.get("team")) in non_playoff_keys, axis=1)
                ]
                ts = pd.concat([ts_non_playoff, ts_playoff_fallback], ignore_index=True)
                log(f"Team priors: {len(ts_non_playoff)} non-playoff + {len(ts_playoff_fallback)} playoff fallback = {len(ts)} total", verbose)
            else:
                ts = ts_non_playoff
        # Season filter (Option B): keep only seasons we will train on (±1 padding applied by caller)
        if seasons_to_keep and "season" in ts.columns:
            try:
                end_year = _parse_season_end_year(ts["season"]).astype("Int64")
                before = len(ts)
                ts = ts[end_year.isin(list(seasons_to_keep))].copy()
                if verbose:
                    log(f"Team priors season filter: {before:,} -> {len(ts):,} rows", True)
            except Exception:
                pass

        # Keep key columns - detect abbreviation vs team-name columns explicitly
        abbr_candidates = [c for c in ("abbreviation", "abbr", "team_abbr") if c in ts.columns]
        team_name_candidates = [c for c in ("team", "team_name") if c in ts.columns]
        abbr_col = abbr_candidates[0] if abbr_candidates else None
        team_name_col = team_name_candidates[0] if team_name_candidates else None

        team_cols = ["season"]
        if abbr_col:
            team_cols.append(abbr_col)
        elif team_name_col:
            team_cols.append(team_name_col)
        # Expand to include Four Factors and other key stats
        for c in ["w", "l", "mov", "sos", "srs", "o_rtg", "d_rtg", "n_rtg", "pace",
                  "ts_percent", "e_fg_percent", "tov_percent", "orb_percent", "ft_fga",
                  "opp_e_fg_percent", "opp_tov_percent", "drb_percent", "opp_ft_fga",
                  "f_tr", "x3p_ar"]:
            if c in ts.columns:
                team_cols.append(c)
        priors_teams = ts[team_cols].copy() if team_cols else pd.DataFrame()

        # Ensure we have an 'abbreviation' column: either rename existing or map from team name using Team Abbrev.csv
        if abbr_col and abbr_col != "abbreviation":
            priors_teams = priors_teams.rename(columns={abbr_col: "abbreviation"})
        
        # Normalize abbreviation formatting
        if "abbreviation" in priors_teams.columns:
            priors_teams["abbreviation"] = (
                priors_teams["abbreviation"].astype(str).str.strip().str.upper()
            )
        
        # If we still don't have abbreviations but do have team names, build them from mapping
        if "abbreviation" not in priors_teams.columns and team_name_col:
            # Load mapping from BR bundle
            abbrev_map = load_team_abbrev_map(priors_root, verbose, source_name="Basketball Reference Team Abbrev")
            if not abbrev_map:
                log("Warning: Could not find Team Abbrev mapping; team priors merge may fail (no abbreviations).", verbose)
            else:
                def _map_abbrev(row):
                    try:
                        season_val = int(pd.to_numeric(row.get("season", None), errors="coerce")) if pd.notna(row.get("season", None)) else None
                    except Exception:
                        season_val = None
                    team_name = row.get(team_name_col, None)
                    if season_val is None or team_name is None or pd.isna(team_name):
                        return None
                    ab = resolve_team_abbrev(str(team_name), int(season_val), abbrev_map)
                    return ab
                priors_teams["abbreviation"] = priors_teams.apply(_map_abbrev, axis=1)
                # Clean formatting
                priors_teams["abbreviation"] = priors_teams["abbreviation"].astype(str).str.strip().str.upper()

        # Normalize percents (0-100 -> 0-1)
        for col in priors_teams.columns:
            if "percent" in col.lower() or col in ["f_tr", "x3p_ar", "ft_fga", "opp_ft_fga"]:
                vals = pd.to_numeric(priors_teams[col], errors="coerce")
                # If max > 1.5, assume 0-100 scale
                if vals.max() > 1.5:
                    priors_teams[col] = vals / 100.0

        # Check for duplicate columns before shifting season
        if priors_teams.columns.duplicated().any():
            dup_cols = priors_teams.columns[priors_teams.columns.duplicated()].tolist()
            log(f"Warning: Duplicate columns found in team priors: {dup_cols}. Removing duplicates.", verbose)
            priors_teams = priors_teams.loc[:, ~priors_teams.columns.duplicated()]

        # Shift: season S priors are used in season S+1
        if not priors_teams.empty and "season" in priors_teams.columns:
            try:
                season_series = priors_teams["season"]
                if isinstance(season_series, pd.DataFrame):
                    season_series = season_series.iloc[:, 0]
                end_year = _parse_season_end_year(season_series)
                # BR season=2025 means 2024-2025 season, end_year IS the season_for_game (no +1 needed)
                priors_teams["season_for_game"] = end_year
                priors_teams = priors_teams.drop(columns=["season"])
            except Exception as e:
                log(f"Error creating season_for_game for teams: {e}", verbose)
                log(f"season column type: {type(priors_teams['season'])}", verbose)
                raise
        else:
            log("Warning: No 'season' column found in Team Summaries", verbose)

        # Final sanity: ensure abbreviation column present
        if "abbreviation" not in priors_teams.columns:
            log("Warning: Team priors missing 'abbreviation' column after processing. Available columns: " + str(list(priors_teams.columns)), verbose)
        else:
            # Drop rows where abbreviation is missing
            missing_abbr = priors_teams["abbreviation"].isna().sum()
            if missing_abbr:
                log(f"Info: Dropping {missing_abbr} team-season rows with missing abbreviation in priors", verbose)
                priors_teams = priors_teams[priors_teams["abbreviation"].notna()].reset_index(drop=True)

        log(f"Loaded {len(priors_teams):,} team-season statistical priors from Team Summaries", verbose)

    # Player priors - comprehensive integration of multiple CSVs
    def load_and_filter_player_csv(path: Path, csv_name: str) -> Optional[pd.DataFrame]:
        """Helper to load player CSV with NBA filtering and TOT preference"""
        if not path.exists():
            log(f"  Skipping {csv_name} (not found)", verbose)
            return None

        df = pd.read_csv(path, low_memory=False)

        # Filter NBA
        if "lg" in df.columns:
            df = df[df["lg"] == "NBA"]

        # Prefer TOT rows for multi-team seasons
        if "team" in df.columns and "player_id" in df.columns:
            tot_rows = df[df["team"] == "TOT"]
            non_tot = df[df["team"] != "TOT"]
            has_tot = set(tot_rows["player_id"])
            non_tot = non_tot[~non_tot["player_id"].isin(has_tot)]
            df = pd.concat([tot_rows, non_tot], ignore_index=True)

        # Season filter (Option B): keep only seasons we will train on (±1 padding applied by caller)
        before = len(df)
        if seasons_to_keep and "season" in df.columns and before:
            try:
                end_year = _parse_season_end_year(df["season"]).astype("Int64")
                df = df[end_year.isin(list(seasons_to_keep))].copy()
                if verbose:
                    log(f"  {csv_name}: filtered by season {before:,} -> {len(df):,}", True)
            except Exception:
                pass

        log(f"  Loaded {len(df):,} rows from {csv_name}", verbose)
        return df

    priors_players = pd.DataFrame()

    # 1. Per 100 Poss - core rate stats
    per100_path = priors_root / "Per 100 Poss.csv"
    per100 = load_and_filter_player_csv(per100_path, "Per 100 Poss.csv")
    if per100 is not None:
        cols = ["season", "player_id", "player", "age", "pos", "g", "mp",
                "pts_per_100_poss", "trb_per_100_poss", "ast_per_100_poss",
                "stl_per_100_poss", "blk_per_100_poss", "tov_per_100_poss",
                "fg_per_100_poss", "fga_per_100_poss", "fg_percent",
                "x3p_per_100_poss", "x3pa_per_100_poss", "x3p_percent",
                "ft_per_100_poss", "fta_per_100_poss", "ft_percent",
                "orb_per_100_poss", "drb_per_100_poss",
                "o_rtg", "d_rtg"]
        cols = [c for c in cols if c in per100.columns]
        priors_players = per100[cols].copy()

    # 2. Advanced - PER, WS, BPM, TS%, USG%
    advanced_path = priors_root / "Advanced.csv"
    advanced = load_and_filter_player_csv(advanced_path, "Advanced.csv")
    if advanced is not None and not priors_players.empty:
        adv_cols = ["season", "player_id", "per", "ts_percent", "usg_percent",
                    "ws", "ws_per_48", "bpm", "obpm", "dbpm", "vorp"]
        adv_cols = [c for c in adv_cols if c in advanced.columns]
        if len(adv_cols) > 2:  # Need at least season + player_id + 1 stat
            priors_players = priors_players.merge(
                advanced[adv_cols], on=["season", "player_id"], how="left"
            )

    # 3. Player Shooting - shooting zones, corner 3%, dunks
    shooting_path = priors_root / "Player Shooting.csv"
    shooting = load_and_filter_player_csv(shooting_path, "Player Shooting.csv")
    if shooting is not None and not priors_players.empty:
        shoot_cols = ["season", "player_id", "avg_dist_fga",
                      "percent_fga_from_x2p_range", "percent_fga_from_x0_3_range",
                      "percent_fga_from_x3_10_range", "percent_fga_from_x10_16_range",
                      "percent_fga_from_x16_3p_range", "percent_fga_from_x3p_range",
                      "fg_percent_from_x2p_range", "fg_percent_from_x0_3_range",
                      "fg_percent_from_x3_10_range", "fg_percent_from_x10_16_range",
                      "fg_percent_from_x16_3p_range", "fg_percent_from_x3p_range",
                      "percent_assisted_x2p_fg", "percent_assisted_x3p_fg",
                      "percent_dunks_of_fga", "num_of_dunks",
                      "percent_corner_3s_of_3pa", "corner_3_point_percent"]
        shoot_cols = [c for c in shoot_cols if c in shooting.columns]
        if len(shoot_cols) > 2:
            priors_players = priors_players.merge(
                shooting[shoot_cols], on=["season", "player_id"], how="left"
            )

    # 4. Player Play By Play - position %, on-court +/-, fouls
    pbp_path = priors_root / "Player Play By Play.csv"
    pbp = load_and_filter_player_csv(pbp_path, "Player Play By Play.csv")
    if pbp is not None and not priors_players.empty:
        pbp_cols = ["season", "player_id",
                    "pg_percent", "sg_percent", "sf_percent", "pf_percent", "c_percent",
                    "on_court_plus_minus_per_100_poss", "net_plus_minus_per_100_poss",
                    "bad_pass_turnover", "lost_ball_turnover",
                    "shooting_foul_committed", "offensive_foul_committed",
                    "shooting_foul_drawn", "offensive_foul_drawn",
                    "points_generated_by_assists", "and1"]
        pbp_cols = [c for c in pbp_cols if c in pbp.columns]
        if len(pbp_cols) > 2:
            priors_players = priors_players.merge(
                pbp[pbp_cols], on=["season", "player_id"], how="left"
            )

    if priors_players.empty:
        log("Warning: No player priors loaded (no matching CSVs or required columns). Expected files: 'Per 100 Poss.csv', 'Advanced.csv', 'Player Shooting.csv', 'Player Play By Play.csv' with at least columns ['season','player_id'] plus stats. Returning empty priors.", verbose)
        return priors_players, priors_teams

    # Normalize all percent columns (0-100 -> 0-1)
    for col in priors_players.columns:
        if "percent" in col.lower() or "_pct" in col.lower():
            vals = pd.to_numeric(priors_players[col], errors="coerce")
            if vals.notna().any() and vals.max() > 1.5:
                priors_players[col] = vals / 100.0

    # Check for duplicate columns before shifting season
    if priors_players.columns.duplicated().any():
        dup_cols = priors_players.columns[priors_players.columns.duplicated()].tolist()
        log(f"Warning: Duplicate columns found: {dup_cols}. Removing duplicates.", verbose)
        priors_players = priors_players.loc[:, ~priors_players.columns.duplicated()]

    # Shift to next season
    if "season" in priors_players.columns:
        try:
            season_series = priors_players["season"]
            if isinstance(season_series, pd.DataFrame):
                season_series = season_series.iloc[:, 0]
            end_year = _parse_season_end_year(season_series)
            # BR season=2025 means 2024-2025 season, end_year IS the season_for_game (no +1 needed)
            priors_players["season_for_game"] = end_year
            priors_players = priors_players.drop(columns=["season"])
        except Exception as e:
            log(f"Error creating season_for_game: {e}", verbose)
            log(f"season column type: {type(priors_players['season'])}", verbose)
            log(f"season column shape: {priors_players['season'].shape if hasattr(priors_players['season'], 'shape') else 'N/A'}", verbose)
            raise
    else:
        log("Warning: No 'season' column found in player priors", verbose)

    # Summary of all loaded priors
    log(_sec("Basketball Reference Priors Summary"), verbose)
    log(f"📁 Total CSVs loaded: 7 (2 team + 4 player + 1 mapping)", verbose)
    log(f"", verbose)
    log(f"Team Priors:", verbose)
    log(f"  ✓ Team Summaries: {len(priors_teams):,} team-seasons", verbose)
    log(f"  ✓ Team Abbrev: (loaded separately for name mapping)", verbose)
    log(f"", verbose)
    log(f"Player Priors: {len(priors_players):,} player-seasons with {len(priors_players.columns)} features", verbose)
    log(f"  ✓ Per 100 Poss.csv", verbose)
    log(f"  ✓ Advanced.csv", verbose)
    log(f"  ✓ Player Shooting.csv", verbose)
    log(f"  ✓ Player Play By Play.csv", verbose)

    return priors_players, priors_teams

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(description="Train NBA game and player models (always fetch from Kaggle).")
    ap.add_argument("--dataset", type=str, default="eoinamoore/historical-nba-data-and-player-box-scores", help="Kaggle dataset ref")
    ap.add_argument("--models-dir", type=str, default="models", help="Output dir for models")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging")
    ap.add_argument("--skip-rest", action="store_true", help="Skip rest/b2b features (useful when dates missing)")
    ap.add_argument("--fresh", action="store_true", help="Copy CSVs into a new per-run folder to avoid any stale artifacts")
    ap.add_argument("--lgb-log-period", type=int, default=0, help="Print LightGBM eval metrics every N iterations (0 = silent)")
    ap.add_argument("--n-jobs", type=int, default=-1, help="Threads for LightGBM/HistGBM (-1=all cores). Reduce to lower RAM usage.")
    ap.add_argument("--disable-neural", action="store_true", help="Disable neural hybrid and use only LightGBM (not recommended)")
    ap.add_argument("--neural-epochs", type=int, default=100, help="Number of epochs for TabNet training (default: 100)")
    ap.add_argument("--neural-device", type=str, default="auto", choices=["auto", "cpu", "gpu"], help="Device for neural training: auto (detect GPU), cpu, or gpu")

    # Era controls
    ap.add_argument("--game-season-cutoff", type=str, default="2002",
                    help="Minimum season_end_year to include for game models. Presets: classic=1997, balanced=2002. Also accepts an int like 1997/2001/2004.")
    ap.add_argument("--player-season-cutoff", type=str, default="2002",
                    help="Minimum season_end_year to include for player models. Presets: min=1998, balanced=2002, modern=2005. Also accepts an int.")
    ap.add_argument("--decay", type=float, default=0.97, help="Time-decay factor for sample weights (0.90-0.99 typical).")
    ap.add_argument("--min-weight", type=float, default=0.30, help="Minimum sample weight after decay.")
    ap.add_argument("--lockout-weight", type=float, default=0.90, help="Extra multiplier applied to lockout seasons (1999, 2012).")

    # Additional datasets (now enabled by default with caching)
    ap.add_argument("--odds-dataset", type=str, default="cviaxmiwnptr/nba-betting-data-october-2007-to-june-2024",
                    help="Path or Kaggle dataset ref for betting odds (default: cviaxmiwnptr/nba-betting-data-october-2007-to-june-2024)")
    ap.add_argument("--skip-odds", action="store_true", help="Do not load or merge betting odds; keep training independent of odds")
    ap.add_argument("--priors-dataset", type=str, default=None,
                    help="Path or Kaggle dataset ref for Basketball Reference priors bundle (local path or Kaggle dataset). Optional - training works without priors.")
    ap.add_argument("--enable-window-ensemble", action="store_true", default=True,
                    help="Train 5-year window ensembles (default: enabled, use --no-window-ensemble to disable)")
    ap.add_argument("--no-window-ensemble", action="store_false", dest="enable_window_ensemble",
                    help="Disable window ensemble training")

    args = ap.parse_args()

    verbose = args.verbose
    seed = args.seed
    # Apply global threads limit for learners
    global N_JOBS
    N_JOBS = int(args.n_jobs)

    print(_sec("🏀 NBA Training Pipeline Configuration"))
    print("\n📊 DATASETS (auto-cached):")
    print(f"  1. Main:   {args.dataset}")
    print(f"  2. Odds:   {'(skipped)' if args.skip_odds else args.odds_dataset}")
    print(f"  3. Priors: {args.priors_dataset}")

    print("\n🎯 FEATURES INCLUDED:")
    print("  • Betting market odds → Implied probabilities, spreads, totals, line movement (from odds dataset)")
    print("\n  • Basketball Reference Statistical Priors (7 CSVs):")
    print("\n    📁 TEAM PRIORS (2 CSVs):")
    print("       1. Team Summaries.csv")
    print("          - O/D ratings, pace, SRS")
    print("          - Four factors (eFG%, TOV%, ORB%, FT/FGA)")
    print("          - Win/loss records, margin of victory")
    print("       2. Team Abbrev.csv")
    print("          - Team name → abbreviation mapping across seasons")
    print("          - Handles team relocations/rebrands (SEA→OKC, NOH→NOP, etc.)")
    print("\n    👤 PLAYER PRIORS (4 CSVs, ~60-65 features):")
    print("       3. Per 100 Poss.csv (~20 features)")
    print("          - Core rate stats: pts, reb, ast, stl, blk, tov per 100 possessions")
    print("          - Shooting %: FG%, 3P%, FT%")
    print("          - O/D ratings, ORB, DRB")
    print("       4. Advanced.csv (~10 features) ⭐")
    print("          - PER (Player Efficiency Rating)")
    print("          - TS% (True Shooting % - better than FG%)")
    print("          - USG% (Usage Rate - volume indicator)")
    print("          - Win Shares: WS, WS/48")
    print("          - BPM: Box Plus/Minus (OBPM, DBPM)")
    print("          - VORP (Value Over Replacement Player)")
    print("       5. Player Shooting.csv (~20 features) ⭐⭐⭐")
    print("          - Shot distribution by zone: 0-3ft, 3-10ft, 10-16ft, 16ft-3P, 3P")
    print("          - FG% by zone (efficiency from each distance)")
    print("          - Corner 3% and corner 3 rate - CRITICAL for 3PM props")
    print("          - Dunks per game, average shot distance")
    print("          - Assisted FG rates (how often assisted on 2P vs 3P)")
    print("       6. Player Play By Play.csv (~15 features) ⭐")
    print("          - Position %: PG, SG, SF, PF, C distribution")
    print("          - On-court +/- per 100 possessions")
    print("          - Net +/- per 100 possessions")
    print("          - Fouls: shooting/offensive fouls committed and drawn")
    print("          - Assists points generated, And-1s")
    print("\n  📈 Predictive Power:")
    print("     ✅ Shooting zones → 3PM prediction")
    print("     ✅ Position % → Role identification")
    print("     ✅ Advanced stats → Overall talent baseline")
    print("     ✅ Usage % → Volume expectations")
    print("     ✅ On-court impact → Context-aware performance")
    print("\n  Note: Priors are STATISTICAL CONTEXT (not betting odds) - they provide historical performance baselines")

    print("\n⚙️  TRAINING SETTINGS:")
    print(f"  • Output directory: {args.models_dir}")
    print(f"  • Random seed: {seed}")
    print(f"  • Season cutoffs: Games ≥{args.game_season_cutoff}, Players ≥{args.player_season_cutoff}")
    print(f"  • Time-decay weights: {args.decay} decay, {args.min_weight} min, {args.lockout_weight}x lockout penalty")
    print(f"  • Rest/B2B features: {'Disabled' if args.skip_rest else 'Enabled'}")
    print(f"  • LightGBM logging: {'Silent' if args.lgb_log_period == 0 else f'Every {args.lgb_log_period} iterations'}")
    
    # Neural network configuration (now default)
    use_neural = not args.disable_neural
    
    # Determine device
    use_gpu = False
    if args.neural_device == "gpu":
        use_gpu = True
    elif args.neural_device == "auto":
        # Auto-detect GPU
        if TORCH_AVAILABLE:
            import torch
            use_gpu = torch.cuda.is_available()
    
    if use_neural:
        if TABNET_AVAILABLE and TORCH_AVAILABLE:
            print(f"  • 🧠 Neural Hybrid: ENABLED (default)")
            print(f"    - TabNet + LightGBM architecture")
            print(f"    - Epochs: {args.neural_epochs}")
            if TORCH_AVAILABLE:
                import torch
                device_info = "GPU (CUDA)" if use_gpu and torch.cuda.is_available() else "CPU"
                print(f"    - Device: {device_info}")
            else:
                print(f"    - Device: CPU")
        else:
            print(f"  • 🧠 Neural Hybrid: ENABLED but libraries missing")
            print(f"    - PyTorch: {'✓' if TORCH_AVAILABLE else '✗ (pip install torch)'}")
            print(f"    - TabNet: {'✓' if TABNET_AVAILABLE else '✗ (pip install pytorch-tabnet)'}")
            print(f"    - ⚠️  Falling back to LightGBM only")
            use_neural = False  # Force disable if libraries missing
    else:
        print(f"  • Model: LightGBM only (neural disabled with --disable-neural)")

    if kagglehub is None:
        raise RuntimeError("kagglehub is required. Install with: pip install kagglehub")

    # Download Kaggle dataset (always)
    print(_sec("Fetching latest from Kaggle"))
    ds_root = Path(kagglehub.dataset_download(args.dataset))
    print(f"- Downloaded to cache: {ds_root}")

    # Optionally copy to a per-run folder (ensures clean, fresh artifacts)
    ds_use_root = ds_root
    if args.fresh:
        run_dir = _fresh_run_dir(Path(".kaggle_runs"))
        teams_src, players_src = _find_dataset_files(ds_root)
        teams_dst = _copy_if_exists(teams_src, run_dir)
        players_dst = _copy_if_exists(players_src, run_dir)
        ds_use_root = run_dir
        print(f"- Copied CSVs to: {run_dir}")
        if not teams_dst:
            raise FileNotFoundError("TeamStatistics CSV not found in Kaggle dataset.")
        teams_path, players_path = teams_dst, players_dst
    else:
        teams_path, players_path = _find_dataset_files(ds_root)
        if not teams_path:
            raise FileNotFoundError("TeamStatistics CSV not found in Kaggle dataset.")

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Build games and team context
    print(_sec("Building game dataset"))
    games_df, context_map, team_id_to_abbrev = build_games_from_teamstats(teams_path, verbose=verbose, skip_rest=args.skip_rest)

    # Augment with current season completed games (2025-26) from nba_api
    print(_sec("Augmenting with current season data"))
    current_season_df = fetch_current_season_games(season="2025-26", verbose=verbose)

    if current_season_df is not None and not current_season_df.empty:
        # Convert current season data to match historical format
        # The function already returns data in the right format, but we need to process it through build_games_from_teamstats logic

        # Save current season data to temp CSV
        temp_csv = Path(".current_season_temp.csv")
        current_season_df.to_csv(temp_csv, index=False)

        # Process through same pipeline
        current_games_df, current_context, current_abbrevs = build_games_from_teamstats(temp_csv, verbose=verbose, skip_rest=args.skip_rest)

        # Merge abbreviations
        if current_abbrevs:
            team_id_to_abbrev.update(current_abbrevs)

        # Append current season games to historical data
        before_len = len(games_df)
        games_df = pd.concat([games_df, current_games_df], ignore_index=True)

        # Merge context maps
        for tid, ctx_df in current_context.items():
            if tid in context_map:
                context_map[tid] = pd.concat([context_map[tid], ctx_df], ignore_index=True)
            else:
                context_map[tid] = ctx_df

        print(f"- Added {len(current_games_df):,} games from 2025-26 season (total: {before_len:,} -> {len(games_df):,})")

        # Clean up temp file
        temp_csv.unlink(missing_ok=True)
        # Free temp frames
        del current_season_df, current_games_df, current_context, current_abbrevs
        gc.collect()
    else:
        print("- No current season games available yet (season may not have started)")

    # SKIP historical odds fetching for now - only needed for 2022+ seasons
    # Historical odds (pre-2022) are not available from The Odds API anyway
    # This saves significant time and RAM, especially for cached historical windows (2002-2021)
    print(_sec("Skipping historical odds fetch - only available for 2022+ seasons"))
    print("- Historical windows (2002-2021) don't need odds data")
    print("- Odds will be loaded later only for current season window if needed")

    # Add team abbreviations for ALL games (needed for team priors merge)
    if team_id_to_abbrev:
        games_df["home_abbrev"] = games_df["home_tid"].map(team_id_to_abbrev)
        games_df["away_abbrev"] = games_df["away_tid"].map(team_id_to_abbrev)
        matched = games_df["home_abbrev"].notna().sum()
        log(f"- Mapped {matched:,} / {len(games_df):,} games to team abbreviations ({matched/len(games_df)*100:.1f}%)", verbose)

    # Ensure home_team and away_team columns exist for exhaustion features
    games_df['home_team'] = games_df['home_abbrev']
    games_df['away_team'] = games_df['away_abbrev']

    # AUTOMATED 5-YEAR WINDOW TRAINING (RAM-efficient with smart caching)
    # Strategy: Only train missing windows + current season window
    # This prevents retraining from 2002 onwards every time
    # NOTE: This has been moved to AFTER base LGB model training (see line ~3600)
    # Keeping this comment here for reference
    if False:  # DISABLED - moved to after clf_final is trained
        # args.enable_window_ensemble:
        if "season_end_year" in games_df.columns:
            import pickle
            cache_dir = "model_cache"
            os.makedirs(cache_dir, exist_ok=True)

            min_year = int(games_df["season_end_year"].min())
            max_year = int(games_df["season_end_year"].max())
            window_size = 5

            # Get all unique seasons and split into exact 5-year windows
            # Convert to integers to avoid numpy float issues
            all_seasons = sorted([int(s) for s in games_df["season_end_year"].dropna().unique()])

            print(f"\n{'='*70}")
            print(f"5-YEAR WINDOW TRAINING (RAM-Efficient Mode)")
            print(f"Data range: {min_year}-{max_year}")
            print(f"Total unique seasons: {len(all_seasons)}")
            print(f"{'='*70}")

            # Determine which window contains the current season
            current_season_year = max_year
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
                            # Check if cache has all expected seasons
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
                    # Always retrain current season window (new data may have arrived)
                    print(f"[TRAIN] Window {start_year}-{end_year}: Current season - will train")
                    windows_to_process.append({
                        'seasons': window_seasons,
                        'start_year': start_year,
                        'end_year': end_year,
                        'cache_path': cache_path,
                        'cache_meta_path': cache_meta_path,
                        'is_current': True
                    })
                elif not cache_valid:
                    # Historical window missing or invalid - need to train
                    status = "missing" if not cache_exists else "invalid"
                    print(f"[TRAIN] Window {start_year}-{end_year}: Cache {status} - will train")
                    windows_to_process.append({
                        'seasons': window_seasons,
                        'start_year': start_year,
                        'end_year': end_year,
                        'cache_path': cache_path,
                        'cache_meta_path': cache_meta_path,
                        'is_current': False
                    })

            if not windows_to_process:
                print("\n[OK] All windows cached and up-to-date. No training needed!")
            else:
                print(f"\n{'='*70}")
                print(f"Will process {len(windows_to_process)} window(s) sequentially to minimize RAM")
                print(f"{'='*70}\n")

                # Process windows sequentially (not all at once) to save RAM
                for idx, window_info in enumerate(windows_to_process, 1):
                    window_seasons = window_info['seasons']
                    start_year = window_info['start_year']
                    end_year = window_info['end_year']
                    cache_path = window_info['cache_path']
                    cache_meta_path = window_info['cache_meta_path']
                    is_current = window_info['is_current']

                    print(f"\n{'='*70}")
                    print(f"Training window {idx}/{len(windows_to_process)}: {start_year}-{end_year}")
                    print(f"Seasons: {list(window_seasons)}")
                    print(f"{'='*70}")

                    # Extract only this window's data to minimize RAM
                    window_mask = games_df["season_end_year"].isin(window_seasons)
                    games_window = games_df[window_mask].reset_index(drop=True)
                    print(f"Window contains {len(games_window):,} games")

                    # Train the model for this window
                    use_player_props = end_year >= 2022
                    if use_player_props:
                        print("Using historic player props for this window.")
                    else:
                        print("Skipping historic player props for this window.")

                    game_weights = _compute_sample_weights(
                        games_window["season_end_year"].to_numpy(dtype="float64"),
                        decay=args.decay, min_weight=args.min_weight, lockout_weight=args.lockout_weight
                    )

                    result = train_all_ensemble_components(
                        games_df=games_window,
                        game_features=GAME_FEATURES,
                        game_defaults=GAME_DEFAULTS,
                        lgb_model=clf_final,
                        optimal_refit_freq=20,
                        verbose=verbose
                    )

                    # Cache the result (even for current window, for faster subsequent runs)
                    print(f"Saving trained model to cache: {cache_path}")
                    with open(cache_path, "wb") as f:
                        pickle.dump(result, f)

                    # Save metadata for cache validation
                    meta = {
                        'seasons': list(map(int, window_seasons)),
                        'start_year': start_year,
                        'end_year': end_year,
                        'trained_date': datetime.now().isoformat(),
                        'num_games': len(games_window),
                        'is_current_season': is_current
                    }
                    with open(cache_meta_path, 'w') as f:
                        json.dump(meta, f, indent=2)

                    print(f"[OK] Window {start_year}-{end_year} complete and cached")

                    # Free memory after each window
                    del games_window, result, game_weights
                    gc.collect()
                    print(f"Memory freed for next window")

                print(f"\n{'='*70}")
                print(f"[OK] All required windows trained and cached")
                print(f"{'='*70}\n")
        else:
            print("No season_end_year column found; cannot automate 5-year window training.")
    # Load and merge betting odds (if provided)
    # NOTE: This section has been DISABLED - odds loading moved to after window training
    # Betting odds (2008-2025) should NOT be loaded into historical windows (2002-2006, etc.)
    odds_df = pd.DataFrame()
    if False:  # DISABLED - moved to after window ensemble training
        # args.odds_dataset and not args.skip_odds:
        odds_path_str = args.odds_dataset
        # Check if it's a Kaggle dataset ref or local path
        if "/" in odds_path_str and not os.path.exists(odds_path_str):
            # Assume Kaggle dataset
            if not kagglehub:
                log("Warning: kagglehub not available, skipping odds dataset", verbose)
                odds_df = pd.DataFrame()
            else:
                try:
                    log(f"Downloading odds dataset from Kaggle: {odds_path_str}", verbose)
                    odds_root = Path(kagglehub.dataset_download(odds_path_str))
                    # Find CSV file in downloaded dataset
                    odds_csvs = list(odds_root.glob("*.csv"))
                    log(f"Found {len(odds_csvs)} CSV files in odds dataset: {[f.name for f in odds_csvs]}", verbose)
                    if odds_csvs:
                        odds_path = odds_csvs[0]  # Use first CSV found
                        odds_df = load_betting_odds(odds_path, verbose)
                    else:
                        log(f"Warning: No CSV files found in {odds_root}", verbose)
                        odds_df = pd.DataFrame()
                except Exception as e:
                    log(f"Warning: Failed to load odds dataset: {e}", verbose)
                    odds_df = pd.DataFrame()
        else:
            odds_path = Path(odds_path_str)
            if odds_path.exists():
                odds_df = load_betting_odds(odds_path, verbose)
            else:
                log(f"Warning: Odds dataset path does not exist: {odds_path}", verbose)
                odds_df = pd.DataFrame()

        # Merge odds into games_df
        if not odds_df.empty:
            log(f"Attempting to merge {len(odds_df):,} odds rows into {len(games_df):,} games", verbose)
            
            # Try multiple merge strategies
            merged = False
            
            # Strategy 1: Match by gid if present in both (normalize to strings first)
            if "gid" in games_df.columns and "gid" in odds_df.columns:
                games_df["gid_norm"] = _id_to_str(games_df["gid"]).str.strip()
                odds_df["gid_norm"] = _id_to_str(odds_df["gid"]).str.strip()
                before_merge = len(games_df)
                common_gids = set(games_df["gid_norm"]) & set(odds_df["gid_norm"])
                if verbose:
                    try:
                        sample_g = list(games_df["gid_norm"].dropna().unique())[:5]
                        sample_o = list(odds_df["gid_norm"].dropna().unique())[:5]
                        log(f"  Games gid samples: {sample_g}", True)
                        log(f"  Odds  gid samples: {sample_o}", True)
                    except Exception:
                        pass
                log(f"  Strategy 1 (gid match): {len(common_gids):,} common game IDs", verbose)
                
                if len(common_gids) > 0:
                    games_df = games_df.merge(
                        odds_df.drop(columns=["game_date_utc", "season_end_year", "gid"], errors="ignore").rename(columns={"gid_norm":"gid_norm_odds"}),
                        left_on="gid_norm", right_on="gid_norm_odds", how="left", suffixes=("", "_odds")
                    )
                    games_df = games_df.drop(columns=["gid_norm", "gid_norm_odds"], errors="ignore")
                    merged = True
                    actual_merged = (games_df["market_implied_home"].notna() & 
                                   (games_df["market_implied_home"] != GAME_DEFAULTS.get("market_implied_home", 0.5))).sum()
                    log(f"  ✓ Merged by gid: {actual_merged:,} games have real odds", verbose)
            
            # Strategy 2: Match by date + team abbreviations
            if not merged and "date" in games_df.columns and "home_abbrev" in games_df.columns and "home_abbrev" in odds_df.columns:
                # Create merge key
                games_df["_merge_key"] = (
                    games_df["date"].dt.strftime("%Y%m%d").fillna("") + "_" +
                    games_df["home_abbrev"].fillna("") + "_" +
                    games_df["away_abbrev"].fillna("")
                )
                odds_df["_merge_key"] = (
                    odds_df["game_date_utc"].dt.strftime("%Y%m%d").fillna("") + "_" +
                    odds_df["home_abbrev"].fillna("") + "_" +
                    odds_df["away_abbrev"].fillna("")
                )
                
                common_keys = set(games_df["_merge_key"]) & set(odds_df["_merge_key"])
                log(f"  Strategy 2 (date+abbrev match): {len(common_keys):,} common keys", verbose)
                
                if len(common_keys) > 0:
                    games_df = games_df.merge(
                        odds_df.drop(columns=["gid", "game_date_utc", "season_end_year"], errors="ignore"),
                        on="_merge_key", how="left", suffixes=("", "_odds")
                    )
                    merged = True
                    actual_merged = (games_df["market_implied_home"].notna() & 
                                   (games_df["market_implied_home"] != GAME_DEFAULTS.get("market_implied_home", 0.5))).sum()
                    log(f"  ✓ Merged by date+abbrev: {actual_merged:,} games have real odds", verbose)
                
                # Clean up merge key
                games_df = games_df.drop(columns=["_merge_key"], errors="ignore")
            
            if not merged:
                log("  ⚠️  Could not merge odds - no common keys found", verbose)
                log(f"     Games has: {list(games_df.columns[:10])}", verbose)
                log(f"     Odds has: {list(odds_df.columns)}", verbose)

        # Ensure betting odds columns always exist (fill with defaults if not merged)
        for col in ["market_implied_home", "market_implied_away", "market_spread", "spread_move", "market_total", "total_move"]:
            if col not in games_df.columns:
                games_df[col] = GAME_DEFAULTS.get(col, 0.0)
            else:
                games_df[col] = games_df[col].fillna(GAME_DEFAULTS.get(col, 0.0))

    # Load and merge Basketball Reference priors (if provided)
    priors_players = pd.DataFrame()
    priors_teams = pd.DataFrame()

    if args.priors_dataset:
        # Compute target seasons to keep for priors (games seasons ±1 for safety)
        seasons_to_keep: Optional[Set[int]] = None
        if "season_end_year" in games_df.columns:
            try:
                base_seasons = set(int(x) for x in pd.to_numeric(games_df["season_end_year"], errors="coerce").dropna().astype(int).unique())
                padded = set()
                for s in base_seasons:
                    padded.update([s-1, s, s+1])
                seasons_to_keep = padded
                if verbose:
                    log(f"Priors season filter prepared: keeping {len(seasons_to_keep)} seasons (sample: {sorted(list(seasons_to_keep))[:5]}…)", True)
            except Exception:
                seasons_to_keep = None
        priors_path_str = args.priors_dataset
        # Check if Kaggle dataset or local path
        if "/" in priors_path_str and not os.path.exists(priors_path_str):
            if not kagglehub:
                log("Warning: kagglehub not available, skipping priors dataset", verbose)
                priors_players, priors_teams = pd.DataFrame(), pd.DataFrame()
            else:
                try:
                    log(f"Downloading priors dataset from Kaggle: {priors_path_str}", verbose)
                    priors_root = Path(kagglehub.dataset_download(priors_path_str))
                    log(f"Priors downloaded to: {priors_root}", verbose)
                    
                    # List all CSV files found
                    csv_files = list(priors_root.glob("**/*.csv"))
                    log(f"Found {len(csv_files)} CSV files in priors dataset:", verbose)
                    for f in csv_files:
                        log(f"  - {f.relative_to(priors_root)}", verbose)
                    
                    priors_players, priors_teams = load_basketball_reference_priors(priors_root, verbose, seasons_to_keep=seasons_to_keep)
                except Exception as e:
                    log(f"Warning: Failed to load priors dataset: {e}", verbose)
                    import traceback
                    log(f"Traceback: {traceback.format_exc()}", verbose)
                    priors_players, priors_teams = pd.DataFrame(), pd.DataFrame()
        else:
            priors_root = Path(priors_path_str)
            if priors_root.exists():
                log(f"Loading priors from local path: {priors_root}", verbose)
                
                # List all CSV files found
                csv_files = list(priors_root.glob("*.csv"))
                log(f"Found {len(csv_files)} CSV files in priors directory:", verbose)
                for f in csv_files:
                    log(f"  - {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)", verbose)
                
                priors_players, priors_teams = load_basketball_reference_priors(priors_root, verbose, seasons_to_keep=seasons_to_keep)
            else:
                log(f"Warning: Priors dataset path does not exist: {priors_root}", verbose)
                priors_players, priors_teams = pd.DataFrame(), pd.DataFrame()
        
    # Diagnostic: Check what we loaded
        if not priors_teams.empty:
            log(f"✓ Loaded team priors: {len(priors_teams):,} rows, columns: {list(priors_teams.columns)}", verbose)
            if "abbreviation" in priors_teams.columns:
                # Filter out NaN values before sorting
                team_abbrevs = [a for a in priors_teams['abbreviation'].unique() if pd.notna(a)]
                log(f"  Team abbreviations ({len(team_abbrevs)} unique): {sorted(team_abbrevs)[:10]}", verbose)
            if "season_for_game" in priors_teams.columns:
                log(f"  Seasons: {priors_teams['season_for_game'].min():.0f} to {priors_teams['season_for_game'].max():.0f}", verbose)
        else:
            log("⚠️  No team priors loaded (priors_teams is empty)", verbose)
        
        if not priors_players.empty:
            log(f"✓ Loaded player priors: {len(priors_players):,} rows, {len(priors_players.columns)} columns", verbose)
        else:
            log("⚠️  No player priors loaded (priors_players is empty)", verbose)

        # Fallback: if games_df still lacks abbreviations but we have team names and a priors root,
        # build abbreviations using Basketball Reference Team Abbrev mapping (post-load cross-implementation)
        if ("home_abbrev" not in games_df.columns or games_df["home_abbrev"].isna().all()) and ("home_name" in games_df.columns):
            try:
                br_map = load_team_abbrev_map(priors_root, verbose, source_name="BR Team Abbrev")
            except Exception:
                br_map = {}
            if br_map:
                def _map_row(row):
                    try:
                        s = int(pd.to_numeric(row.get("season_end_year"), errors="coerce"))
                    except Exception:
                        return pd.Series({"home_abbrev": None, "away_abbrev": None})
                    hn = row.get("home_name", None)
                    an = row.get("away_name", None)
                    ha = resolve_team_abbrev(str(hn), s, br_map) if hn is not None else None
                    aa = resolve_team_abbrev(str(an), s, br_map) if an is not None else None
                    return pd.Series({"home_abbrev": ha, "away_abbrev": aa})
                mapped = games_df.apply(_map_row, axis=1)
                for col in ["home_abbrev", "away_abbrev"]:
                    games_df[col] = mapped[col].astype(str).str.strip().str.upper()
                matched_h = games_df["home_abbrev"].notna().sum()
                matched_a = games_df["away_abbrev"].notna().sum()
                log(f"- Mapped abbreviations via team names using BR table: home {matched_h:,}, away {matched_a:,} (rows: {len(games_df):,})", verbose)

        # Merge team priors into games_df
        if not priors_teams.empty and "abbreviation" in priors_teams.columns and "season_end_year" in games_df.columns:
            # home_abbrev/away_abbrev are now available for ALL games (from team ID mapping)
            if "home_abbrev" in games_df.columns and "away_abbrev" in games_df.columns:
                log(f"Attempting to merge team priors: {len(priors_teams):,} team-seasons into {len(games_df):,} games", verbose)
                log(f"  Games abbrevs: {games_df['home_abbrev'].notna().sum():,} home, {games_df['away_abbrev'].notna().sum():,} away", verbose)
                log(f"  Games seasons: {games_df['season_end_year'].min():.0f} to {games_df['season_end_year'].max():.0f}", verbose)
                log(f"  Priors seasons: {priors_teams['season_for_game'].min():.0f} to {priors_teams['season_for_game'].max():.0f}", verbose)
                log(f"  Priors teams: {priors_teams['abbreviation'].nunique()} unique", verbose)
                
                # Check for common teams/seasons
                game_abbrevs = set(games_df["home_abbrev"].dropna()) | set(games_df["away_abbrev"].dropna())
                prior_abbrevs = set(priors_teams["abbreviation"].dropna())
                common_abbrevs = game_abbrevs & prior_abbrevs
                log(f"  Common team abbrevs: {len(common_abbrevs)} (e.g., {list(common_abbrevs)[:5]})", verbose)
                
                game_seasons = set(games_df["season_end_year"].dropna())
                prior_seasons = set(priors_teams["season_for_game"].dropna())
                common_seasons = game_seasons & prior_seasons
                log(f"  Common seasons: {len(common_seasons)} (e.g., {sorted(common_seasons)[:5]})", verbose)
                
                # Merge home team priors
                # Try exact season match first, then fall back to previous season (for future games)
                home_priors = priors_teams.rename(columns={
                    "abbreviation": "home_abbrev",
                    "season_for_game": "season_end_year",
                    "o_rtg": "home_o_rtg_prior",
                    "d_rtg": "home_d_rtg_prior",
                    "pace": "home_pace_prior",
                    "srs": "home_srs_prior",
                    "e_fg_percent": "home_efg_prior",
                    "tov_percent": "home_tov_pct_prior",
                    "orb_percent": "home_orb_pct_prior",
                    "ft_fga": "home_ftr_prior",
                    "opp_e_fg_percent": "home_opp_efg_prior",
                    "opp_tov_percent": "home_opp_tov_pct_prior",
                    "drb_percent": "home_drb_pct_prior",
                    "opp_ft_fga": "home_opp_ftr_prior",
                    "ts_percent": "home_ts_pct_prior",
                    "x3p_ar": "home_3par_prior",
                    "mov": "home_mov_prior"
                })
                home_prior_cols = ["home_abbrev", "season_end_year"]
                for col in ["home_o_rtg_prior", "home_d_rtg_prior", "home_pace_prior", "home_srs_prior",
                           "home_efg_prior", "home_tov_pct_prior", "home_orb_pct_prior", "home_ftr_prior",
                           "home_opp_efg_prior", "home_opp_tov_pct_prior", "home_drb_pct_prior", "home_opp_ftr_prior",
                           "home_ts_pct_prior", "home_3par_prior", "home_mov_prior"]:
                    if col in home_priors.columns:
                        home_prior_cols.append(col)
                
                before_merge = len(games_df)
                games_df = games_df.merge(
                    home_priors[home_prior_cols],
                    on=["home_abbrev", "season_end_year"], how="left"
                )
                
                # Ensure columns exist even if merge found no matches
                for col in ["home_o_rtg_prior", "home_d_rtg_prior", "home_pace_prior", "home_srs_prior",
                           "home_efg_prior", "home_tov_pct_prior", "home_orb_pct_prior", "home_ftr_prior",
                           "home_opp_efg_prior", "home_opp_tov_pct_prior", "home_drb_pct_prior", "home_opp_ftr_prior",
                           "home_ts_pct_prior", "home_3par_prior", "home_mov_prior"]:
                    if col not in games_df.columns:
                        games_df[col] = np.nan
                
                # Fallback: For unmatched rows, try previous season
                stat_cols = [c for c in home_prior_cols if c not in ["home_abbrev", "season_end_year"] and c in games_df.columns]
                if stat_cols:
                    unmatched_mask = games_df[stat_cols[0]].isna()
                    if unmatched_mask.sum() > 0:
                        log(f"  Trying fallback: {unmatched_mask.sum()} unmatched home teams, attempting season-1", verbose)
                        games_unmatched = games_df[unmatched_mask].copy()
                        games_unmatched["_prev_season"] = games_unmatched["season_end_year"] - 1
                        if verbose:
                            sample_fallback = games_unmatched[["home_abbrev", "season_end_year", "_prev_season"]].head(3)
                            log(f"    Sample fallback keys: {sample_fallback.to_dict(orient='records')}", True)
                            avail_seasons = set(home_priors["season_end_year"].dropna().unique())
                            log(f"    Available priors seasons (sample): {sorted(avail_seasons)[-5:]}", True)
                            # Check if LAL + 2025 exists in home_priors
                            lal_2025 = home_priors[(home_priors["home_abbrev"] == "LAL") & (home_priors["season_end_year"] == 2025.0)]
                            log(f"    LAL + season 2025 in home_priors: {len(lal_2025)} rows", True)
                            if len(lal_2025) > 0:
                                log(f"      {lal_2025[['home_abbrev', 'season_end_year', 'home_o_rtg_prior']].to_dict(orient='records')}", True)
                            # Check the renamed version
                            hp_renamed = home_priors[home_prior_cols].rename(columns={"season_end_year": "_prev_season"})
                            lal_2025_renamed = hp_renamed[(hp_renamed["home_abbrev"] == "LAL") & (hp_renamed["_prev_season"] == 2025.0)]
                            log(f"    After rename, LAL + _prev_season 2025: {len(lal_2025_renamed)} rows", True)
                            if len(lal_2025_renamed) > 0 and 'home_o_rtg_prior' in lal_2025_renamed.columns:
                                log(f"      Sample: {lal_2025_renamed[['home_abbrev', '_prev_season', 'home_o_rtg_prior']].iloc[0].to_dict()}", True)
                        fallback_merge = games_unmatched.merge(
                            home_priors[home_prior_cols].rename(columns={"season_end_year": "_prev_season"}),
                            on=["home_abbrev", "_prev_season"], how="left", suffixes=("", "_fb")
                        )
                        if verbose:
                            log(f"    Fallback merge shape: {fallback_merge.shape}", True)
                            # Check if stat columns got renamed with _fb suffix
                            for sc in stat_cols[:3]:
                                if sc in fallback_merge.columns and sc + "_fb" in fallback_merge.columns:
                                    val_no_suffix = fallback_merge[sc].iloc[0] if len(fallback_merge) > 0 else None
                                    val_with_suffix = fallback_merge[sc + "_fb"].iloc[0] if len(fallback_merge) > 0 else None
                                    log(f"      {sc}: both versions exist! no_suffix={val_no_suffix}, with_suffix={val_with_suffix}", True)
                                elif sc in fallback_merge.columns:
                                    val = fallback_merge[sc].iloc[0] if len(fallback_merge) > 0 else None
                                    log(f"      {sc}: exists (no _fb version), value={val}", True)
                                elif sc + "_fb" in fallback_merge.columns:
                                    val = fallback_merge[sc + "_fb"].iloc[0] if len(fallback_merge) > 0 else None
                                    log(f"      {sc}_fb: exists (no plain version), value={val}", True)
                        # Copy fallback values - need to handle suffix!
                        for col in stat_cols:
                            col_with_fb = col + "_fb"
                            if col_with_fb in fallback_merge.columns:
                                # Use the _fb suffixed column which has the actual merged data
                                idx = games_df[unmatched_mask].index
                                games_df.loc[idx, col] = fallback_merge[col_with_fb].values
                            elif col in fallback_merge.columns:
                                idx = games_df[unmatched_mask].index
                                games_df.loc[idx, col] = fallback_merge[col].values
                        # Check if fallback worked
                        post_fallback_matched = games_df.loc[unmatched_mask, stat_cols[0]].notna().sum()
                        if verbose and post_fallback_matched > 0:
                            log(f"    Fallback matched {post_fallback_matched} teams", True)
                
                home_matched = games_df["home_o_rtg_prior"].notna().sum()
                log(f"  Home priors matched: {home_matched:,} / {len(games_df):,} ({home_matched/len(games_df)*100:.1f}%)", verbose)

                # Merge away team priors
                away_priors = priors_teams.rename(columns={
                    "abbreviation": "away_abbrev",
                    "season_for_game": "season_end_year",
                    "o_rtg": "away_o_rtg_prior",
                    "d_rtg": "away_d_rtg_prior",
                    "pace": "away_pace_prior",
                    "srs": "away_srs_prior",
                    "e_fg_percent": "away_efg_prior",
                    "tov_percent": "away_tov_pct_prior",
                    "orb_percent": "away_orb_pct_prior",
                    "ft_fga": "away_ftr_prior",
                    "opp_e_fg_percent": "away_opp_efg_prior",
                    "opp_tov_percent": "away_opp_tov_pct_prior",
                    "drb_percent": "away_drb_pct_prior",
                    "opp_ft_fga": "away_opp_ftr_prior",
                    "ts_percent": "away_ts_pct_prior",
                    "x3p_ar": "away_3par_prior",
                    "mov": "away_mov_prior"
                })
                away_prior_cols = ["away_abbrev", "season_end_year"]
                for col in ["away_o_rtg_prior", "away_d_rtg_prior", "away_pace_prior", "away_srs_prior",
                           "away_efg_prior", "away_tov_pct_prior", "away_orb_pct_prior", "away_ftr_prior",
                           "away_opp_efg_prior", "away_opp_tov_pct_prior", "away_drb_pct_prior", "away_opp_ftr_prior",
                           "away_ts_pct_prior", "away_3par_prior", "away_mov_prior"]:
                    if col in away_priors.columns:
                        away_prior_cols.append(col)
                
                games_df = games_df.merge(
                    away_priors[away_prior_cols],
                    on=["away_abbrev", "season_end_year"], how="left"
                )
                
                # Ensure columns exist even if merge found no matches
                for col in ["away_o_rtg_prior", "away_d_rtg_prior", "away_pace_prior", "away_srs_prior",
                           "away_efg_prior", "away_tov_pct_prior", "away_orb_pct_prior", "away_ftr_prior",
                           "away_opp_efg_prior", "away_opp_tov_pct_prior", "away_drb_pct_prior", "away_opp_ftr_prior",
                           "away_ts_pct_prior", "away_3par_prior", "away_mov_prior"]:
                    if col not in games_df.columns:
                        games_df[col] = np.nan
                
                # Fallback: For unmatched rows, try previous season
                stat_cols = [c for c in away_prior_cols if c not in ["away_abbrev", "season_end_year"] and c in games_df.columns]
                if stat_cols:
                    unmatched_mask = games_df[stat_cols[0]].isna()
                    if unmatched_mask.sum() > 0:
                        log(f"  Trying fallback: {unmatched_mask.sum()} unmatched away teams, attempting season-1", verbose)
                        games_unmatched = games_df[unmatched_mask].copy()
                        games_unmatched["_prev_season"] = games_unmatched["season_end_year"] - 1
                        fallback_merge = games_unmatched.merge(
                            away_priors[away_prior_cols].rename(columns={"season_end_year": "_prev_season"}),
                            on=["away_abbrev", "_prev_season"], how="left", suffixes=("", "_fb")
                        )
                        # Copy fallback values - need to handle suffix!
                        for col in stat_cols:
                            col_with_fb = col + "_fb"
                            if col_with_fb in fallback_merge.columns:
                                idx = games_df[unmatched_mask].index
                                games_df.loc[idx, col] = fallback_merge[col_with_fb].values
                            elif col in fallback_merge.columns:
                                idx = games_df[unmatched_mask].index
                                games_df.loc[idx, col] = fallback_merge[col].values
                
                away_matched = games_df["away_o_rtg_prior"].notna().sum()
                log(f"  Away priors matched: {away_matched:,} / {len(games_df):,} ({away_matched/len(games_df)*100:.1f}%)", verbose)
                
                log(f"✓ Merged team priors for {len(priors_teams):,} team-seasons", verbose)
            else:
                log("Warning: Missing home_abbrev or away_abbrev in games_df - cannot merge team priors", verbose)
                log(f"  Available columns: {list(games_df.columns)}", verbose)
        else:
            if priors_teams.empty:
                log("Warning: priors_teams is empty - no team priors to merge", verbose)
            elif "abbreviation" not in priors_teams.columns:
                log(f"Warning: 'abbreviation' column not in priors_teams. Available: {list(priors_teams.columns)}", verbose)
            elif "season_end_year" not in games_df.columns:
                log("Warning: 'season_end_year' not in games_df - cannot merge team priors", verbose)

        # Ensure team priors columns always exist (fill with defaults if not merged)
        for col in ["home_o_rtg_prior", "home_d_rtg_prior", "home_pace_prior", "home_srs_prior",
                   "away_o_rtg_prior", "away_d_rtg_prior", "away_pace_prior", "away_srs_prior",
                   "home_efg_prior", "home_tov_pct_prior", "home_orb_pct_prior", "home_ftr_prior",
                   "away_efg_prior", "away_tov_pct_prior", "away_orb_pct_prior", "away_ftr_prior",
                   "home_opp_efg_prior", "home_opp_tov_pct_prior", "home_drb_pct_prior", "home_opp_ftr_prior",
                   "away_opp_efg_prior", "away_opp_tov_pct_prior", "away_drb_pct_prior", "away_opp_ftr_prior",
                   "home_ts_pct_prior", "home_3par_prior", "home_mov_prior",
                   "away_ts_pct_prior", "away_3par_prior", "away_mov_prior"]:
            if col not in games_df.columns:
                games_df[col] = GAME_DEFAULTS.get(col, 0.0)
            else:
                games_df[col] = games_df[col].fillna(GAME_DEFAULTS.get(col, 0.0))

        # Optional: If odds didn't merge earlier and we now have abbreviations, try date+abbrev merge once
        if ("market_implied_home" in games_df.columns and (games_df["market_implied_home"] == GAME_DEFAULTS.get("market_implied_home", 0.5)).all()) and not odds_df.empty:
            if "date" in games_df.columns and "home_abbrev" in games_df.columns and "home_abbrev" in odds_df.columns:
                games_df["_merge_key"] = (
                    games_df["date"].dt.strftime("%Y%m%d").fillna("") + "_" +
                    games_df["home_abbrev"].fillna("") + "_" +
                    games_df["away_abbrev"].fillna("")
                )
                odds_df["_merge_key"] = (
                    odds_df["game_date_utc"].dt.strftime("%Y%m%d").fillna("") + "_" +
                    odds_df["home_abbrev"].fillna("") + "_" +
                    odds_df["away_abbrev"].fillna("")
                )
                common_keys = set(games_df["_merge_key"]) & set(odds_df["_merge_key"])
                log(f"Retrying odds merge by date+abbrev after abbrev mapping: {len(common_keys):,} common keys", verbose)
                if len(common_keys) > 0:
                    games_df = games_df.merge(
                        odds_df.drop(columns=["gid", "game_date_utc", "season_end_year"], errors="ignore"),
                        on="_merge_key", how="left", suffixes=("", "_odds")
                    ).drop(columns=["_merge_key"], errors="ignore")

    # Diagnostic: Check priors data availability
    if verbose:
        print(_sec("Data Availability Diagnostic"))
        print(f"Total games: {len(games_df):,}")

        # Check betting odds
        odds_cols = ["market_implied_home", "market_spread", "market_total", "home_abbrev", "away_abbrev"]
        print("\n📊 BETTING ODDS:")
        for col in odds_cols:
            if col in games_df.columns:
                if col in ["home_abbrev", "away_abbrev"]:
                    non_null = games_df[col].notna().sum()
                    print(f"  {col}: {non_null:,} games ({non_null/len(games_df)*100:.1f}%)")
                else:
                    default_val = GAME_DEFAULTS.get(col, 0.0)
                    non_default = (games_df[col] != default_val).sum()
                    print(f"  {col}: {non_default:,} non-default ({non_default/len(games_df)*100:.1f}%)")

        # Check team priors
        priors_cols = ["home_o_rtg_prior", "home_d_rtg_prior", "home_pace_prior", "home_srs_prior",
                      "home_efg_prior", "home_tov_pct_prior", "home_orb_pct_prior", "home_ftr_prior"]
        print("\n🏀 BASKETBALL REFERENCE STATISTICAL PRIORS (Team - showing 8 key features):")
        for col in priors_cols:
            if col in games_df.columns:
                default_val = GAME_DEFAULTS.get(col, 0.0)
                non_default = (games_df[col] != default_val).sum()
                print(f"  {col}: {non_default:,} non-default ({non_default/len(games_df)*100:.1f}%)")

        # Summary
        if "home_o_rtg_prior" in games_df.columns:
            with_priors = (games_df["home_o_rtg_prior"] != GAME_DEFAULTS.get("home_o_rtg_prior", 0.0)).sum()
            # Additional diagnostic: show actual values
            if verbose and len(games_df) > 0:
                sample_priors = games_df[["home_abbrev", "away_abbrev", "season_end_year", 
                                          "home_o_rtg_prior", "home_d_rtg_prior", "home_pace_prior"]].head(5)
                log(f"\n  Sample merged values:", True)
                log(f"{sample_priors.to_string()}", True)
                log(f"\n  Value ranges:", True)
                log(f"    home_o_rtg_prior: {games_df['home_o_rtg_prior'].min():.2f} to {games_df['home_o_rtg_prior'].max():.2f}", True)
                log(f"    Default value: {GAME_DEFAULTS.get('home_o_rtg_prior', 0.0)}", True)
            if with_priors == 0:
                print("\n⚠️  WARNING: NO games have real Basketball Reference statistical priors - all using defaults!")
                print("   This means statistical priors are NOT being used in training.")
                print("   Possible causes:")
                print("   • Season mismatch between games and Basketball Reference data")
                print("   • Missing team abbreviation column in Team Summaries CSV")
                print("   • Priors dataset path is incorrect or files not found")
            else:
                print(f"\n✓ {with_priors:,} games ({with_priors/len(games_df)*100:.1f}%) have Basketball Reference statistical priors")

    # Train game models + OOF
    print(_sec("Training game models"))

    # Compute sample weights for games (era-decay), even if window ensemble is disabled
    if "season_end_year" in games_df.columns:
        game_weights = _compute_sample_weights(
            games_df["season_end_year"].to_numpy(dtype="float64"),
            decay=args.decay, min_weight=args.min_weight, lockout_weight=args.lockout_weight
        )
    else:
        game_weights = np.ones(len(games_df), dtype="float64")

    clf_final, calibrator, reg_final, spread_sigma, oof_games, game_metrics = _fit_game_models(
        games_df, seed=seed, verbose=verbose, folds=5, lgb_log_period=args.lgb_log_period, sample_weights=game_weights
    )

# Save game models
    import pickle
    with open(models_dir / "moneyline_model.pkl", "wb") as f:
        pickle.dump(clf_final, f)
    if calibrator is not None:
        with open(models_dir / "moneyline_calibrator.pkl", "wb") as f:
            pickle.dump(calibrator, f)
    with open(models_dir / "spread_model.pkl", "wb") as f:
        pickle.dump(reg_final, f)
    with open(models_dir / "spread_sigma.json", "w", encoding="utf-8") as f:
        json.dump({"spread_sigma": spread_sigma}, f)

    # ========================================================================
    # AUTOMATED 5-YEAR WINDOW TRAINING (RAM-efficient with smart caching)
    # ========================================================================
    # Now that clf_final is trained, we can use it for window ensemble training
    if args.enable_window_ensemble:
        if "season_end_year" in games_df.columns:
            import pickle
            cache_dir = "model_cache"
            os.makedirs(cache_dir, exist_ok=True)

            # Filter games to only include seasons >= game_season_cutoff (2002)
            game_cutoff_year = int(args.game_season_cutoff)
            original_game_count = len(games_df)
            games_df = games_df[games_df["season_end_year"] >= game_cutoff_year].copy()
            if verbose:
                print(f"\n{'='*70}")
                print(f"SEASON FILTERING FOR ENSEMBLE TRAINING")
                print(f"Game season cutoff: {game_cutoff_year}")
                print(f"Games before filter: {original_game_count:,}")
                print(f"Games after filter: {len(games_df):,} (seasons {game_cutoff_year}-{int(games_df['season_end_year'].max())})")
                print(f"{'='*70}")

            min_year = int(games_df["season_end_year"].min())
            max_year = int(games_df["season_end_year"].max())
            window_size = 5

            # Get all unique seasons and split into exact 5-year windows
            # Convert to integers to avoid numpy float issues
            all_seasons = sorted([int(s) for s in games_df["season_end_year"].dropna().unique()])

            print(f"\n{'='*70}")
            print(f"5-YEAR WINDOW TRAINING (RAM-Efficient Mode)")
            print(f"Data range: {min_year}-{max_year}")
            print(f"Total unique seasons: {len(all_seasons)}")
            if verbose:
                print(f"DEBUG - All seasons: {all_seasons}")
                print(f"DEBUG - Season value counts (first 10):")
                vc = games_df["season_end_year"].value_counts().sort_index()
                print(vc.head(10))
            print(f"{'='*70}")

            # Determine which window contains the current season
            current_season_year = max_year
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
                            # Check if cache has all expected seasons
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
                    # Always retrain current season window (new data may have arrived)
                    print(f"[TRAIN] Window {start_year}-{end_year}: Current season - will train")
                    windows_to_process.append({
                        'seasons': window_seasons,
                        'start_year': start_year,
                        'end_year': end_year,
                        'cache_path': cache_path,
                        'cache_meta_path': cache_meta_path,
                        'is_current': True
                    })
                elif not cache_valid:
                    # Historical window missing or invalid - need to train
                    status = "missing" if not cache_exists else "invalid"
                    print(f"[TRAIN] Window {start_year}-{end_year}: Cache {status} - will train")
                    windows_to_process.append({
                        'seasons': window_seasons,
                        'start_year': start_year,
                        'end_year': end_year,
                        'cache_path': cache_path,
                        'cache_meta_path': cache_meta_path,
                        'is_current': False
                    })

            if not windows_to_process:
                print("\n[OK] All windows cached and up-to-date. No training needed!")
            else:
                print(f"\n{'='*70}")
                print(f"Will process {len(windows_to_process)} window(s) sequentially to minimize RAM")
                print(f"{'='*70}\n")

                # Process windows sequentially (not all at once) to save RAM
                for idx, window_info in enumerate(windows_to_process, 1):
                    window_seasons = window_info['seasons']
                    start_year = window_info['start_year']
                    end_year = window_info['end_year']
                    cache_path = window_info['cache_path']
                    cache_meta_path = window_info['cache_meta_path']
                    is_current = window_info['is_current']

                    print(f"\n{'='*70}")
                    print(f"Training window {idx}/{len(windows_to_process)}: {start_year}-{end_year}")
                    print(f"Seasons: {list(window_seasons)}")
                    print(f"{'='*70}")

                    # Extract only this window's data to minimize RAM
                    window_mask = games_df["season_end_year"].isin(window_seasons)
                    games_window = games_df[window_mask].reset_index(drop=True)
                    print(f"Window contains {len(games_window):,} games")

                    # Train the model for this window
                    use_player_props = end_year >= 2022
                    if use_player_props:
                        print("Using historic player props for this window.")
                    else:
                        print("Skipping historic player props for this window.")

                    game_weights = _compute_sample_weights(
                        games_window["season_end_year"].to_numpy(dtype="float64"),
                        decay=args.decay, min_weight=args.min_weight, lockout_weight=args.lockout_weight
                    )

                    result = train_all_ensemble_components(
                        games_df=games_window,
                        game_features=GAME_FEATURES,
                        game_defaults=GAME_DEFAULTS,
                        lgb_model=clf_final,
                        optimal_refit_freq=20,
                        verbose=verbose
                    )

                    # Cache the result (even for current window, for faster subsequent runs)
                    print(f"Saving trained model to cache: {cache_path}")
                    with open(cache_path, "wb") as f:
                        pickle.dump(result, f)

                    # Save metadata for cache validation
                    meta = {
                        'seasons': list(map(int, window_seasons)),
                        'start_year': start_year,
                        'end_year': end_year,
                        'trained_date': datetime.now().isoformat(),
                        'num_games': len(games_window),
                        'is_current_season': is_current
                    }
                    with open(cache_meta_path, 'w') as f:
                        json.dump(meta, f, indent=2)

                    print(f"[OK] Window {start_year}-{end_year} complete and cached")

                    # Free memory after each window
                    del games_window, result, game_weights
                    gc.collect()
                    print(f"Memory freed for next window")

                print(f"\n{'='*70}")
                print(f"[OK] All required windows trained and cached")
                print(f"{'='*70}\n")
        else:
            print("No season_end_year column found; cannot automate 5-year window training.")

    # ========================================================================
    # ENHANCED ENSEMBLE: Ridge + Elo + Four Factors + Meta-Learner (All Improvements)
    # ========================================================================
    print(_sec("Training Enhanced Ensemble (All Improvements)"))
    try:
        # Train all components in one pipeline
        ridge_model, elo_model, ff_model, ensembler, games_enhanced, ensemble_metrics = \
            train_all_ensemble_components(
                games_df=games_df,
                game_features=GAME_FEATURES,
                game_defaults=GAME_DEFAULTS,
                lgb_model=clf_final,  # Your existing LGB model
                optimal_refit_freq=20,  # Tested optimal: 20 games (~1.2 weeks)
                verbose=verbose
            )
        
        # Save all ensemble models
        pickle.dump(ridge_model, open(models_dir / "ridge_model_enhanced.pkl", "wb"))
        pickle.dump(elo_model, open(models_dir / "elo_model_enhanced.pkl", "wb"))
        pickle.dump(ff_model, open(models_dir / "four_factors_model_enhanced.pkl", "wb"))
        pickle.dump(ensembler, open(models_dir / "ensemble_meta_learner_enhanced.pkl", "wb"))
        
        # Update training metadata
        game_metrics.update({
            'ridge': ensemble_metrics.get('ridge', {}),
            'elo': ensemble_metrics.get('elo', {}),
            'four_factors': ensemble_metrics.get('four_factors', {}),
            'refit_frequency_tests': ensemble_metrics.get('refit_frequency_tests', {}),
            'optimal_refit_frequency': ensemble_metrics.get('optimal_refit_frequency', 20),
            'ensemble': ensemble_metrics.get('ensemble', {}),
        })
        
        print(f"✓ All ensemble models saved")
        print(f"  Ridge: ridge_model_enhanced.pkl")
        print(f"  Elo: elo_model_enhanced.pkl")
        print(f"  Four Factors: four_factors_model_enhanced.pkl")
        print(f"  Meta-Learner: ensemble_meta_learner_enhanced.pkl")
        
    except Exception as e:
        print(f"⚠ Ensemble training failed: {e}")
        import traceback
        traceback.print_exc()
        ridge_model, elo_model, ff_model, ensembler = None, None, None, None

    # ========================================================================
    # PLAYER MODELS (Per-Window Training for Memory Optimization)
    # ========================================================================
    player_metrics: Dict[str, Dict[str, float]] = {}
    if players_path and players_path.exists():
        print(_sec("Training player models per window"))

        # Define cache directory
        cache_dir = "model_cache"
        os.makedirs(cache_dir, exist_ok=True)

        # OPTIMIZATION: Check which windows need training BEFORE loading any data
        # This saves massive memory when most windows are cached (5x reduction per window)
        all_seasons = sorted([int(s) for s in games_df["season_end_year"].dropna().unique()])
        max_year = int(games_df["season_end_year"].max())
        window_size = 5
        player_windows_to_process = []
        
        print("\n" + "="*70)
        print("CHECKING PLAYER WINDOW CACHES")
        print("="*70)
        
        for i in range(0, len(all_seasons), window_size):
            window_seasons = all_seasons[i:i+window_size]
            start_year = int(window_seasons[0])
            end_year = int(window_seasons[-1])
            cache_path_check = f"{cache_dir}/player_models_{start_year}_{end_year}.pkl"
            is_current_window = max_year in window_seasons
            
            # Decide whether to train this window
            if is_current_window:
                # Always retrain current season
                print(f"[TRAIN] Window {start_year}-{end_year}: Current season - will train")
                player_windows_to_process.append({
                    'seasons': window_seasons,
                    'start_year': start_year,
                    'end_year': end_year,
                    'is_current': True
                })
            elif not os.path.exists(cache_path_check):
                # Historical window not cached - need to train
                print(f"[TRAIN] Window {start_year}-{end_year}: Not cached - will train")
                player_windows_to_process.append({
                    'seasons': window_seasons,
                    'start_year': start_year,
                    'end_year': end_year,
                    'is_current': False
                })
            else:
                print(f"[SKIP] Window {start_year}-{end_year}: Using existing cache")
        
        windows_to_process = player_windows_to_process
        
        if not windows_to_process:
            print("\n✅ All player windows cached! Skipping player data loading entirely.")
            print("="*70)
        else:
            print(f"\n📊 Will train {len(windows_to_process)} window(s)")
            print("="*70)

        # Process each window
        for idx, window_info in enumerate(windows_to_process, 1):
            window_seasons = set(window_info['seasons'])
            start_year = window_info['start_year']
            end_year = window_info['end_year']
            cache_path = f"{cache_dir}/player_models_{start_year}_{end_year}.pkl"
            cache_meta_path = f"{cache_dir}/player_models_{start_year}_{end_year}_meta.json"
            is_current = window_info['is_current']

            print(f"\n{'='*70}")
            print(f"Training player models: Window {idx}/{len(windows_to_process)}")
            print(f"Seasons: {start_year}-{end_year} ({'CURRENT' if is_current else 'historical'})")
            print(f"{'='*70}")

            # Check cache (skip historical windows if cached)
            if os.path.exists(cache_path) and not is_current:
                print(f"[SKIP] Using cached models from {cache_path}")
                continue

            # ================================================================
            # WINDOW-SPECIFIC DATA LOADING (MAJOR OPTIMIZATION!)
            # Only load player data needed for THIS window
            # Saves 5x memory vs loading all 25 years at once
            # ================================================================
            print(f"Loading player data for window {start_year}-{end_year}...")
            
            # Fetch current season data if this is the current window
            current_player_df = None
            if is_current:
                current_player_df = fetch_current_season_player_stats(season="2025-26", verbose=verbose)
                if current_player_df is not None and not current_player_df.empty:
                    print(f"  • Fetched {len(current_player_df):,} current season player-games")
            
            # Load historical player data (will be filtered to window inside build_players_from_playerstats)
            # Note: We still need to load full CSV here, but build_players_from_playerstats
            # will filter early (line 1656-1673) using window_seasons parameter
            if current_player_df is not None and not current_player_df.empty:
                temp_player_csv = Path(f".window_{start_year}_{end_year}_players.csv")
                hist_players_df = pd.read_csv(players_path, low_memory=False)
                
                # Filter historical to window immediately (before concat to save memory)
                date_col = [c for c in hist_players_df.columns if 'date' in c.lower()][0]
                hist_players_df[date_col] = pd.to_datetime(hist_players_df[date_col], errors="coerce")
                hist_players_df['_temp_season'] = _season_from_date(hist_players_df[date_col])
                padded_seasons = set(window_seasons) | {start_year-1, end_year+1}
                hist_players_df = hist_players_df[hist_players_df['_temp_season'].isin(padded_seasons)].copy()
                hist_players_df = hist_players_df.drop(columns=['_temp_season'])
                
                print(f"  • Loaded {len(hist_players_df):,} historical player-games for window")
                
                combined_players_df = pd.concat([hist_players_df, current_player_df], ignore_index=True)
                combined_players_df.to_csv(temp_player_csv, index=False)
                player_data_path = temp_player_csv
                
                # Clean up
                del hist_players_df, current_player_df
                gc.collect()
            else:
                # No current season data, create window-specific CSV from historical only
                temp_player_csv = Path(f".window_{start_year}_{end_year}_players.csv")
                hist_players_df = pd.read_csv(players_path, low_memory=False)
                
                # Filter to window
                date_col = [c for c in hist_players_df.columns if 'date' in c.lower()][0]
                hist_players_df[date_col] = pd.to_datetime(hist_players_df[date_col], errors="coerce")
                hist_players_df['_temp_season'] = _season_from_date(hist_players_df[date_col])
                padded_seasons = set(window_seasons) | {start_year-1, end_year+1}
                hist_players_df = hist_players_df[hist_players_df['_temp_season'].isin(padded_seasons)].copy()
                hist_players_df = hist_players_df.drop(columns=['_temp_season'])
                
                print(f"  • Loaded {len(hist_players_df):,} player-games for window")
                
                hist_players_df.to_csv(temp_player_csv, index=False)
                player_data_path = temp_player_csv
                
                del hist_players_df
                gc.collect()
            
            print(f"  ✓ Window-specific player data prepared")

            # Filter game context to window
            context_window = context_map[context_map["season_end_year"].isin(window_seasons)].copy()

            # Filter OOF games to window (oof_games only has gid, need to filter by gid from context)
            window_gids = set(context_window['gid'].unique())
            oof_window = oof_games[oof_games['gid'].isin(window_gids)].copy()

            # Filter priors to window (±1 for context)
            padded_seasons = window_seasons | {start_year-1, end_year+1}
            priors_window = priors_players[
                priors_players["season_for_game"].isin(padded_seasons)
            ].copy() if priors_players is not None and not priors_players.empty else None

            print(f"Window data: {len(context_window):,} games, {len(priors_window) if priors_window is not None else 0:,} player-season priors")

            # Build frames for this window (window_seasons triggers internal filtering)
            frames = build_players_from_playerstats(
                player_data_path,
                context_window,
                oof_window,
                verbose=verbose,
                priors_players=priors_window,
                window_seasons=window_seasons
            )

            # Load historical player props for this window
            print(_sec(f"Loading player props for {start_year}-{end_year}"))
            player_props_cache = Path("data/historical_player_props_cache.csv")

            # Filter raw player data to window for prop fetching
            raw_players_df = pd.read_csv(player_data_path, low_memory=False)
            date_col = [c for c in raw_players_df.columns if 'date' in c.lower()][0] if any('date' in c.lower() for c in raw_players_df.columns) else None

            if date_col:
                raw_players_df[date_col] = pd.to_datetime(raw_players_df[date_col], errors="coerce", format='mixed', utc=True).dt.tz_convert(None)
                raw_players_df['season_end_year'] = _season_from_date(raw_players_df[date_col])
                raw_players_df_window = raw_players_df[raw_players_df["season_end_year"].isin(window_seasons)]
            else:
                raw_players_df_window = raw_players_df

            historical_player_props = load_or_fetch_historical_player_props(
                players_df=raw_players_df_window,
                api_key=THEODDS_API_KEY,
                cache_path=player_props_cache,
                verbose=verbose,
                max_requests=100
            )

            # Merge player props into frames
            if not historical_player_props.empty:
                for stat_name, stat_df in frames.items():
                    if stat_df is None or stat_df.empty:
                        continue

                    prop_type_map = {
                        'points': 'points',
                        'rebounds': 'rebounds',
                        'assists': 'assists',
                        'threes': 'threes',
                        'minutes': None
                    }

                    prop_type = prop_type_map.get(stat_name)
                    if prop_type is None:
                        continue

                    stat_props = historical_player_props[historical_player_props['prop_type'] == prop_type].copy()
                    if stat_props.empty:
                        continue

                    # Prepare merge columns
                    if 'date' in stat_df.columns:
                        stat_df['date_str'] = pd.to_datetime(stat_df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
                    else:
                        continue

                    stat_props['date_str'] = stat_props['date'].astype(str)

                    # Normalize names
                    stat_df['player_name_norm'] = stat_df.get('playerName', stat_df.get('player_name', '')).str.lower().str.strip()
                    stat_props['player_name_norm'] = stat_props['player_name'].str.lower().str.strip()

                    # Merge
                    before_merge = len(stat_df)
                    stat_df = stat_df.merge(
                        stat_props[['date_str', 'player_name_norm', 'market_line', 'market_over_odds', 'market_under_odds']],
                        on=['date_str', 'player_name_norm'],
                        how='left'
                    )

                    props_matched = stat_df['market_line'].notna().sum()
                    print(f"- {stat_name}: merged {props_matched:,} / {before_merge:,} prop odds ({props_matched/before_merge*100:.1f}%)")

                    # Clean up
                    stat_df = stat_df.drop(columns=['date_str', 'player_name_norm'], errors='ignore')
                    frames[stat_name] = stat_df

            # Era filter + weights per frame
            player_cut = _parse_season_cutoff(args.player_season_cutoff, kind="player")
            for k, df in list(frames.items()):
                if df is None or df.empty:
                    continue
                if "season_end_year" in df.columns:
                    before = len(df)
                    df = df[(df["season_end_year"].fillna(player_cut)) >= player_cut].reset_index(drop=True)
                    frames[k] = df
                    if verbose:
                        print(f"- {k}: filtered by season >= {player_cut}: {before:,} -> {len(df):,}")
                    # weights
                    df["sample_weight"] = _compute_sample_weights(
                        df["season_end_year"].to_numpy(dtype="float64"),
                        decay=args.decay, min_weight=args.min_weight, lockout_weight=args.lockout_weight
                    )
                    frames[k] = df
                else:
                    df["sample_weight"] = 1.0
                    frames[k] = df

            # Train models for this window
            print(_sec(f"Training models for {start_year}-{end_year}"))

            minutes_model, m_metrics = _fit_minutes_model(frames.get("minutes", pd.DataFrame()), seed=seed + 10, verbose=verbose)
            points_model, points_sigma_model, p_metrics = _fit_stat_model(frames.get("points", pd.DataFrame()), seed=seed + 20, verbose=verbose, name="points", use_neural=use_neural, neural_epochs=args.neural_epochs, use_gpu=use_gpu)
            rebounds_model, rebounds_sigma_model, r_metrics = _fit_stat_model(frames.get("rebounds", pd.DataFrame()), seed=seed + 30, verbose=verbose, name="rebounds", use_neural=use_neural, neural_epochs=args.neural_epochs, use_gpu=use_gpu)
            assists_model, assists_sigma_model, a_metrics = _fit_stat_model(frames.get("assists", pd.DataFrame()), seed=seed + 40, verbose=verbose, name="assists", use_neural=use_neural, neural_epochs=args.neural_epochs, use_gpu=use_gpu)
            threes_model, threes_sigma_model, t_metrics = _fit_stat_model(frames.get("threes", pd.DataFrame()), seed=seed + 50, verbose=verbose, name="threes", use_neural=use_neural, neural_epochs=args.neural_epochs, use_gpu=use_gpu)

            # Save per-window models
            # Check if we're using neural hybrid models
            is_neural = isinstance(points_model, NeuralHybridPredictor)
            
            if is_neural:
                # Save neural hybrid models using their own save method
                cache_base = cache_path.with_suffix('')  # Remove .pkl
                for prop, model in [('points', points_model), ('rebounds', rebounds_model), 
                                    ('assists', assists_model), ('threes', threes_model)]:
                    if isinstance(model, NeuralHybridPredictor):
                        model.save(cache_base.parent / f"{cache_base.stem}_{prop}.pkl")
                
                # Save minutes model separately (not neural)
                window_models = {
                    'minutes': minutes_model,
                    'points': None,  # Saved separately
                    'rebounds': None,  # Saved separately
                    'assists': None,  # Saved separately
                    'threes': None,  # Saved separately
                    'points_sigma': None,  # Included in neural model
                    'rebounds_sigma': None,  # Included in neural model
                    'assists_sigma': None,  # Included in neural model
                    'threes_sigma': None,  # Included in neural model
                    'window_seasons': list(window_seasons),
                    'is_neural': True,
                    'metrics': {
                        'minutes': m_metrics,
                        'points': p_metrics,
                        'rebounds': r_metrics,
                        'assists': a_metrics,
                        'threes': t_metrics
                    }
                }
            else:
                window_models = {
                    'minutes': minutes_model,
                    'points': points_model,
                    'rebounds': rebounds_model,
                    'assists': assists_model,
                    'threes': threes_model,
                    'points_sigma': points_sigma_model,
                    'rebounds_sigma': rebounds_sigma_model,
                    'assists_sigma': assists_sigma_model,
                    'threes_sigma': threes_sigma_model,
                    'window_seasons': list(window_seasons),
                    'is_neural': False,
                    'metrics': {
                        'minutes': m_metrics,
                        'points': p_metrics,
                        'rebounds': r_metrics,
                        'assists': a_metrics,
                        'threes': t_metrics
                    }
                }

            with open(cache_path, 'wb') as f:
                pickle.dump(window_models, f)

            # Save metadata
            meta = {
                'seasons': list(map(int, window_seasons)),
                'start_year': start_year,
                'end_year': end_year,
                'trained_date': datetime.now().isoformat(),
                'num_player_games': sum(len(df) for df in frames.values() if df is not None and not df.empty),
                'is_current_season': is_current,
                'metrics': {k: {mk: float(mv) for mk, mv in v.items()} if v else {} for k, v in window_models['metrics'].items()}
            }

            with open(cache_meta_path, 'w') as f:
                json.dump(meta, f, indent=2)

            print(f"[OK] Player models for {start_year}-{end_year} saved to {cache_path}")

            # Free memory before next window
            del context_window, oof_window, priors_window, frames, raw_players_df, raw_players_df_window
            del minutes_model, points_model, rebounds_model, assists_model, threes_model
            del points_sigma_model, rebounds_sigma_model, assists_sigma_model, threes_sigma_model
            
            # Clean up window-specific temp file
            if player_data_path.exists():
                player_data_path.unlink()
                print(f"  • Cleaned up temp file: {player_data_path.name}")
            
            gc.collect()

            print(f"✓ Window {start_year}-{end_year} complete, memory freed")

        # Save global models using most recent window (backward compatibility)
        print("\n" + "="*70)
        print("Saving global models (using most recent window)")
        print("="*70)

        # Find latest window (whether trained now or cached)
        all_window_caches = sorted([f for f in os.listdir(cache_dir) if f.startswith("player_models_") and f.endswith(".pkl")])
        
        if all_window_caches:
            # Extract years from filename player_models_2022_2026.pkl
            latest_cache_file = all_window_caches[-1]  # Last in sorted order
            latest_cache = f"{cache_dir}/{latest_cache_file}"
            
            if os.path.exists(latest_cache):
                with open(latest_cache, 'rb') as f:
                    latest_models = pickle.load(f)

                is_neural = latest_models.get('is_neural', False)
                
                if is_neural:
                    # Load neural hybrid models from their separate files
                    cache_base = Path(latest_cache).with_suffix('')
                    
                    # Save minutes model (non-neural)
                    if 'minutes' in latest_models and latest_models['minutes'] is not None:
                        model_path = models_dir / "minutes_model.pkl"
                        with open(model_path, 'wb') as f:
                            pickle.dump(latest_models['minutes'], f)
                        print(f"  ✓ minutes_model.pkl")
                    
                    # Copy neural hybrid models
                    for stat_name in ['points', 'rebounds', 'assists', 'threes']:
                        neural_path = cache_base.parent / f"{cache_base.stem}_{stat_name}.pkl"
                        if neural_path.exists():
                            # Load and re-save to models directory
                            model = NeuralHybridPredictor.load(neural_path)
                            model_path = models_dir / f"{stat_name}_model.pkl"
                            model.save(model_path)
                            print(f"  ✓ {stat_name}_model.pkl (neural hybrid)")
                            
                            # TabNet model is saved alongside
                            tabnet_src = neural_path.parent / f"{neural_path.stem}_tabnet.zip"
                            tabnet_dst = models_dir / f"{stat_name}_model_tabnet.zip"
                            if tabnet_src.exists():
                                import shutil
                                shutil.copy(tabnet_src, tabnet_dst)
                                print(f"  ✓ {stat_name}_model_tabnet.zip")
                else:
                    # Standard LightGBM models
                    for stat_name in ['minutes', 'points', 'rebounds', 'assists', 'threes']:
                        if stat_name in latest_models and latest_models[stat_name] is not None:
                            model_path = models_dir / f"{stat_name}_model.pkl"
                            with open(model_path, 'wb') as f:
                                pickle.dump(latest_models[stat_name], f)
                            print(f"  ✓ {stat_name}_model.pkl")

                            if f"{stat_name}_sigma" in latest_models and latest_models[f"{stat_name}_sigma"] is not None:
                                sigma_model_path = models_dir / f"{stat_name}_sigma_model.pkl"
                                with open(sigma_model_path, 'wb') as f:
                                    pickle.dump(latest_models[f"{stat_name}_sigma"], f)
                                print(f"  ✓ {stat_name}_sigma_model.pkl")

                # Aggregate metrics from latest window
                player_metrics = latest_models.get('metrics', {})
        else:
            print("⚠️  No player window caches found - player models not saved")
            player_metrics = {}

        # Clean up temp files
        if 'temp_player_csv' in locals():
            temp_player_csv.unlink(missing_ok=True)
        if 'temp_combined_csv' in locals():
            temp_combined_csv.unlink(missing_ok=True)
    else:
        print(_sec("Player models"))
        print("- Skipped (PlayerStatistics.csv not found in Kaggle dataset).")

    # Save metadata
    meta = {
        "game_features": GAME_FEATURES,
        "game_defaults": GAME_DEFAULTS,
        "game_metrics": game_metrics,
        "player_metrics": player_metrics,
        "era": {
            "game_season_cutoff": _parse_season_cutoff(args.game_season_cutoff, "game"),
            "player_season_cutoff": _parse_season_cutoff(args.player_season_cutoff, "player"),
            "decay": args.decay,
            "min_weight": args.min_weight,
            "lockout_weight": args.lockout_weight,
        },
        "versions": {
            "lightgbm": getattr(lgb, "__version__", None) if _HAS_LGB else None,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        }
    }
    with open(models_dir / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Final report
    print(_sec("Saved artifacts"))
    print(f"- { (models_dir / 'moneyline_model.pkl').as_posix() }")
    if os.path.exists(models_dir / "moneyline_calibrator.pkl"):
        print(f"- { (models_dir / 'moneyline_calibrator.pkl').as_posix() }")
        print(f"- { (models_dir / 'spread_model.pkl').as_posix() }")
    else:
        print(f"- { (models_dir / 'spread_model.pkl').as_posix() }")
    print(f"- { (models_dir / 'spread_sigma.json').as_posix() }")
    
    print(f"\n💡 View detailed metrics anytime:")
    print(f"   python show_metrics.py")
    if player_metrics:
        print(f"- { (models_dir / 'minutes_model.pkl').as_posix() }")
        print(f"- { (models_dir / 'points_model.pkl').as_posix() }")
        print(f"- { (models_dir / 'rebounds_model.pkl').as_posix() }")
        print(f"- { (models_dir / 'assists_model.pkl').as_posix() }")
        print(f"- { (models_dir / 'threes_model.pkl').as_posix() }")

    print(_sec("Summary"))
    print(f"- Games: {len(games_df):,}")
    
    # Display game metrics prominently
    print(f"\n🏀 GAME PREDICTIONS:")
    print(f"   Moneyline: logloss={_fmt(game_metrics['ml_logloss'])}, Brier={_fmt(game_metrics['ml_brier'])}")
    if 'ml_accuracy' in game_metrics:
        ml_acc = game_metrics['ml_accuracy']
        print(f"   Moneyline Accuracy: {ml_acc*100:.1f}% {'🟢' if ml_acc >= 0.55 else '🟡' if ml_acc >= 0.52 else '🔴'}")
    print(f"   Spread:    RMSE={_fmt(game_metrics['sp_rmse'])}, MAE={_fmt(game_metrics['sp_mae'])}, sigma={_fmt(game_metrics['spread_sigma'])}")
    if 'sp_accuracy' in game_metrics:
        sp_acc = game_metrics['sp_accuracy']
        print(f"   Spread Accuracy: {sp_acc*100:.1f}% {'🟢' if sp_acc >= 0.53 else '🟡' if sp_acc >= 0.50 else '🔴'}")
    
    print(f"\n👤 PLAYER PROPS:")
    if player_metrics:
        for k, mm in player_metrics.items():
            print(f"- {k.capitalize()}: rows={mm.get('rows')}, RMSE={_fmt(mm.get('rmse'))}, MAE={_fmt(mm.get('mae'))}")
    
    # ========================================================================
    # DYNAMIC WINDOW SELECTOR: Train context-aware ensemble selector
    # ========================================================================
    # Only train if window ensembles exist (from --enable-window-ensemble)
    cache_dir = Path("model_cache")
    ensemble_files = list(cache_dir.glob("player_ensemble_*.pkl")) if cache_dir.exists() else []
    
    if len(ensemble_files) >= 2:  # Need at least 2 windows to select between
        print(_sec("Training Dynamic Window Selector"))
        print(f"Found {len(ensemble_files)} window ensembles - training selector...")
        
        try:
            # Run the dynamic selector training script
            result = subprocess.run(
                [sys.executable, "train_dynamic_selector_enhanced.py"],
                capture_output=True,
                text=True,
                timeout=900  # 15 minute timeout
            )
            
            if result.returncode == 0:
                print("✓ Dynamic selector trained successfully")
                print(f"  Selector saved to: {cache_dir / 'dynamic_selector_enhanced.pkl'}")
            else:
                print(f"⚠ Dynamic selector training failed:")
                print(result.stderr[:500] if result.stderr else "No error output")
        
        except subprocess.TimeoutExpired:
            print("⚠ Dynamic selector training timed out (>15 min)")
        except Exception as e:
            print(f"⚠ Dynamic selector training error: {e}")
    else:
        print(_sec("Skipping Dynamic Window Selector"))
        print(f"Reason: Need 2+ window ensembles (found {len(ensemble_files)})")
        print("To enable: Run with --enable-window-ensemble flag first")

if __name__ == "__main__":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    main()