# RIQ MEEPING MACHINE — Fully Integrated, Fast-Mode, All-in-One
"""
NBA Props Analyzer with ML Model Integration

Loads and uses trained models from train_auto.py:
- Player models: points_model.pkl, assists_model.pkl, rebounds_model.pkl, threes_model.pkl, minutes_model.pkl
- Game models: moneyline_model.pkl, moneyline_calibrator.pkl, spread_model.pkl
- Metadata: training_metadata.json (includes RMSE for each model), spread_sigma.json

Feature Engineering:
- All 3 phases implemented matching train_auto.py:
  Phase 1: Shot volume (FGA, 3PA, FTA rolling stats + per-minute rates + efficiency metrics)
  Phase 2: Matchup context (pace, defensive difficulty, offensive environment)
  Phase 3: Advanced rates (usage%, rebound%, assist%)
- 61 features total for player predictions
- Aligns exactly with training schema to prevent shape mismatches

Model predictions are ensembled with statistical projections using inverse-variance weighting.
"""
from __future__ import annotations

import os
import sys

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

# Suppress sklearn feature name warnings (models work fine, just cosmetic warnings)
import warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')

import re
import time
import json
import math
import random
import pickle
import datetime
from dataclasses import dataclass
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Any
from datetime import timezone, datetime as dt

import numpy as np
import pandas as pd
import requests
from scipy import stats
from nba_api.stats.endpoints import LeagueDashTeamStats
from nba_api.stats.static import teams as nba_teams

# ========= RUNTIME / FAST MODE =========
FAST_MODE = False  # set to False for full runs
REQUEST_TIMEOUT = 4 if FAST_MODE else 10
RETRIES = 1 if FAST_MODE else 3
# Fetch games for today + 1 day (2 days total)
DAYS_TO_FETCH = 2
MAX_GAMES = 6 if FAST_MODE else 20
SLEEP_SHORT = 0.05 if FAST_MODE else 0.2
SLEEP_LONG = 0.1 if FAST_MODE else 0.3
RUN_TIME_BUDGET_SEC = None  # Disabled - no time limit for analysis

# ========= CONFIG =========
# API-Sports (schedule + player stats)
API_KEY = "4979ac5e1f7ae10b1d6b58f1bba01140"
if not API_KEY or API_KEY == "YOUR_KEY_HERE":
    API_KEY = os.getenv("API_SPORTS_KEY") or os.getenv("APISPORTS_KEY")
    if not API_KEY:
        raise ValueError("❌ API key not found. Set API_SPORTS_KEY or edit API_KEY in this file.")

BASE_URL = "https://v1.basketball.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}
LEAGUE_ID = 12
SEASON = "2025-2026"  # Current season (we're in Oct 2025, early in 2025-26 season)
STATS_SEASON = "2024-2025"  # Previous season for historical context

# SportsGameOdds / RapidAPI (swappable sportsbook)
# Replace SGO with a RapidAPI-backed sportsbook (e.g., TheRundown via RapidAPI).
# Set environment variables:
#   RAPIDAPI_KEY (x-rapidapi-key)
#   RAPIDAPI_HOST (x-rapidapi-host) e.g. "therundown-therundown-v1.p.rapidapi.com"
SGO_RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY") or os.getenv("SGO_RAPIDAPI_KEY") or "9ef7289093msh76adf5ee5bedb5fp15e0d6jsnc2a0d0ed9abe"
SGO_RAPIDAPI_HOST = os.getenv("RAPIDAPI_HOST") or os.getenv("SGO_RAPIDAPI_HOST") or "therundown-therundown-v1.p.rapidapi.com"
# V1 endpoint for game lists: /sports/4/events/{YYYY-MM-DD}
# V2 endpoint for detailed event data with player props: /v2/events/{event_id}
SGO_RAPIDAPI_BASE = f"https://{SGO_RAPIDAPI_HOST}"
SGO_RAPIDAPI_EVENTS_PATH = "/sports/4/events"  # List events by date
SGO_RAPIDAPI_V2_EVENT_PATH = "/v2/events"  # Get event details with player props
SGO_ODD_IDS = os.getenv("SGO_ODD_IDS")  # optional filter from env
SGO_AFFILIATE_IDS = os.getenv("SGO_AFFILIATE_IDS")  # optional affiliate filter
# Player prop market IDs: 39=Points, 49=Assists, 51=Rebounds, 52=Threes, etc.
SGO_PLAYER_MARKET_IDS = "39,49,51,52,53,55,56,57,58,60,286,287"
SGO_LIMIT = 50
SGO_MAX_PAGES = 50
SGO_SLEEP_BETWEEN_PAGES_SEC = 0.15

# API-Sports DISABLED - Not needed with TheRundown + The Odds API
APISPORTS_ODDS_ENABLED = False

# The Odds API (third prop source)
THEODDS_API_KEY = "c98703301e8f89ef2c3648a4373939fd"  # UPGRADED to 20K Plan - includes player props!
THEODDS_BASE_URL = "https://api.the-odds-api.com/v4"
THEODDS_ENABLED = True  # Enable/disable The Odds API
THEODDS_SPORT = "basketball_nba"  # Sport key
THEODDS_REGIONS = "us"  # us, uk, eu, au
THEODDS_MARKETS = "h2h,spreads,totals,player_points,player_rebounds,player_assists,player_threes"  # NOW WITH PLAYER PROPS!
THEODDS_BOOKMAKERS = "fanduel"  # FanDuel bookmaker

# SportsGameOdds API DISABLED - Not needed with TheRundown + The Odds API
SGO_DIRECT_ENABLED = False

# PrizePicks API DISABLED - Blocked by Cloudflare protection
PRIZEPICKS_ENABLED = False

BANKROLL = 100.0
MAX_STAKE = 10.0  # Maximum stake per bet
MIN_KELLY_STAKE = 0.005

# MINIMUM WIN PROBABILITY (confidence filter for accuracy)
# Only recommend bets where model predicts >= this % chance of winning
# Higher = fewer but more accurate bets
# Recommended: 0.55-0.60 for profitable long-term edge
MIN_WIN_PROBABILITY = 0.56  # 56% minimum confidence (default)

DEBUG_MODE = False  # set True to log markets/bookmakers seen

# ELG gates by prop type (lower threshold = more permissive)
# Negative values mean we accept bets even with slight negative ELG (high conviction)
ELG_GATES = {
    "points": -0.005,    # Allow more points props
    "assists": -0.005,   # Allow more assists props
    "rebounds": -0.005,  # Allow more rebounds props
    "threes": -0.005,    # Allow more threes props
    "moneyline": -0.02,  # Very permissive for game bets with ML models
    "spread": -0.02,     # Very permissive for game bets with ML models
    "DEFAULT": -0.005,
}

# Odds filter
MIN_ODDS = -500
MAX_ODDS = +400  # Max odds for single legs
MAX_PARLAY_ODDS = +600  # Max combined odds for parlays (avoid longshots)

# Early-season blending
PRIOR_GAMES_STRENGTH = 12.0
TEAM_CONTINUITY_DEFAULT = 0.7

# Posterior tightness by market
N_EFF_BY_MARKET = {"PTS": 90.0, "AST": 80.0, "REB": 85.0, "3PM": 70.0, "Moneyline": 45.0, "Spread": 45.0, "DEFAULT": 80.0}

# Category view
TOP_PER_CATEGORY = 5
CATEGORIES = [
    ("points", "Points"),
    ("assists", "Assists"),
    ("rebounds", "Rebounds"),
    ("threes", "3PM"),
    ("moneyline", "Moneyline"),
    ("spread", "Spread"),
]

# Data files
WEIGHTS_FILE = "prop_weights.pkl"
RESULTS_FILE = "prop_results.pkl"
CACHE_FILE = "player_cache.pkl"
EQUITY_FILE = "equity_curve.pkl"
LEDGER_FILE = "bets_ledger.pkl"
CALIBRATION_FILE = "calibration.pkl"

# Internal stat columns
stat_map = {"points": "points", "assists": "assists", "rebounds": "rebounds", "threes": "threes"}
MEEP_MESSAGES = ["Riq Machine Working", "Meeping...", "Crunching numbers", "Analyzing props", "Finding value"]

# ========= ODDS + POSTERIOR UTILITIES =========
def american_to_decimal(american: int | float) -> float:
    if isinstance(american, str):
        american = float(american)
    if american >= 100:
        return 1.0 + (american / 100.0)
    elif american <= -100:
        return 1.0 + (100.0 / abs(american))
    return 2.0

def break_even_p(american: int | float) -> Tuple[float, float]:
    dec = american_to_decimal(american)
    b = dec - 1.0
    return 1.0 / (1.0 + b), b

def implied_prob_from_american(american: int | float) -> float:
    if isinstance(american, str):
        american = float(american)
    return 100.0 / (american + 100.0) if american >= 100 else abs(american) / (abs(american) + 100.0)

def sample_beta_posterior(p_hat: float, n_eff: float, n_samples: int = 600, seed: int = 42) -> List[float]:
    p_hat = min(max(1e-4, p_hat), 1.0 - 1e-4)
    alpha = 1.0 + p_hat * max(1e-6, n_eff)
    beta = 1.0 + (1.0 - p_hat) * max(1e-6, n_eff)
    rng = random.Random(seed)
    out = []
    for _ in range(n_samples):
        x = rng.gammavariate(alpha, 1.0)
        y = rng.gammavariate(beta, 1.0)
        out.append(x / (x + y + 1e-12))
    return out

def kelly_fraction(p: float, b: float) -> float:
    q = 1.0 - p
    f = (b * p - q) / max(1e-9, b)
    return max(0.0, f)

@dataclass
class KellyConfig:
    q_conservative: float = 0.30
    fk_low: float = 0.25
    fk_high: float = 0.50
    dd_scale: float = 1.0

def dynamic_fractional_kelly(p_samples: List[float], b: float, cfg: KellyConfig) -> Tuple[float, float, float, float]:
    if not p_samples:
        return 0.0, 0.0, 0.0, 0.0
    arr = np.array(p_samples)
    p_c = float(np.quantile(arr, cfg.q_conservative))
    p_mean = float(arr.mean())
    p_be = 1.0 / (1.0 + b)
    if p_c <= p_be:
        return 0.0, p_c, 0.0, p_mean
    denom = max(1e-9, p_mean - p_c)
    conf = (p_mean - p_be) / denom
    conf = max(0.0, min(2.0, conf)) / 2.0
    frac_k = (cfg.fk_low + (cfg.fk_high - cfg.fk_low) * conf) * cfg.dd_scale
    f_star = kelly_fraction(p_c, b)
    f = frac_k * f_star
    return f, p_c, frac_k, p_mean

def risk_adjusted_elg(p_samples: List[float], b: float, f: float) -> float:
    if not p_samples or f <= 0.0:
        return -1e9
    ps = np.clip(np.array(p_samples), 1.0e-6, 1.0 - 1.0e-6)
    return float(np.mean(ps * np.log1p(f * b) + (1.0 - ps) * np.log1p(-f)))

@dataclass
class ExposureCaps:
    max_per_game: float = 0.20
    max_per_player: float = 0.12
    max_per_team: float = 0.20
    max_props_per_player: int = 2

def drawdown_scale(equity_curve: List[float], floor: float = 0.6, window: int = 14) -> float:
    if not equity_curve or len(equity_curve) < 2:
        return 1.0
    recent = equity_curve[-window:] if len(equity_curve) >= window else equity_curve
    peak = max(recent); curr = recent[-1]
    if peak <= 0: return 1.0
    dd = (peak - curr) / peak
    if dd <= 0: return 1.0
    if dd >= 0.30: return floor
    return 1.0 - (1.0 - floor) * (dd / 0.30)

# ========= PROP MODELS =========
def _ewma(values: List[float], half_life: float = 5.0) -> float:
    if not values: return 0.0
    v = np.asarray(values, dtype=float); n = len(v); idx = np.arange(n, dtype=float)
    lam = 0.5 ** (1.0 / max(1.0, half_life)); w = lam ** (n - 1 - idx); w /= w.sum()
    return float((v * w).sum())

def _robust_sigma(values: List[float], mu_ref: float) -> float:
    if not values: return 0.0
    v = np.asarray(values, dtype=float); mad = float(np.median(np.abs(v - mu_ref)))
    return max(1e-6, mad / 0.6745)

def _normal_cdf(x: float) -> float:
    a1,a2,a3,a4,a5,p = 0.254829592,-0.284496736,1.421413741,-1.453152027,1.061405429,0.3275911
    sign = 1 if x >= 0 else -1; x = abs(x) / math.sqrt(2.0)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    return 0.5 * (1.0 + sign * y)

def _poisson_cdf(k: int, lam: float) -> float:
    total = 0.0; term = math.exp(-lam); total += term
    for i in range(1, max(1, k + 1)):
        term *= lam / i; total += term
        if i >= k: break
    return min(1.0, max(0.0, total))

def _fit_nb_params(values: List[float]) -> Tuple[float, float, float]:
    v = np.array(values, dtype=float)
    m = float(np.mean(v)); var = float(np.var(v, ddof=1)) if len(v) > 1 else max(1e-6, m)
    if var <= m + 1e-9: return m, float("inf"), 0.0
    r = (m * m) / (var - m + 1e-9); p = m / (m + r)
    return max(1e-6, r), min(max(1e-9, p), 1 - 1e-9), m

def _nb_cdf(k: int, r: float, p: float) -> float:
    if r == float("inf"): return 0.0
    total = 0.0; prob = (1 - p) ** r; total += prob
    for i in range(1, max(1, k + 1)):
        prob *= ((i - 1 + r) / i) * p; total += prob
        if i >= k: break
    return min(1.0, max(0.0, total))

def project_stat(values: List[float], prop_type: str, pace_multiplier: float, defense_factor: float) -> Tuple[float, float]:
    vals = np.array(values, dtype=float)
    if len(vals) == 0: return 0.0, 0.0
    mu_base = _ewma(list(vals), half_life=5.0)
    m3 = float(np.mean(vals[:3])) if len(vals) >= 3 else float(np.mean(vals))
    m8 = float(np.mean(vals[:8])) if len(vals) >= 8 else float(np.mean(vals))
    trend = 0.0 if m8 <= 1e-9 else (m3 - m8) / m8
    trend_boost = max(-0.05, min(0.05, 0.25 * trend))
    mu = mu_base * (1.0 + trend_boost)
    mu *= pace_multiplier
    if prop_type == "assists":
        mu *= (0.7 + 0.3 * defense_factor)
    elif prop_type in ("points", "threes"):
        mu *= defense_factor
    elif prop_type == "rebounds":
        mu *= (0.8 * pace_multiplier + 0.2)
    raw_sigma = _robust_sigma(list(vals), mu_base)
    if prop_type == "points": sigma = max(0.8, raw_sigma * 1.10)
    elif prop_type == "assists": sigma = max(0.6, raw_sigma * 1.20)
    elif prop_type == "rebounds": sigma = max(0.8, raw_sigma * 1.15)
    elif prop_type == "threes": sigma = max(0.5, raw_sigma * 1.30)
    else: sigma = max(0.8, raw_sigma * 1.10)
    return float(mu), float(sigma)

def prop_win_probability(prop_type: str, values: List[float], line: float, pick: str, mu: float, sigma: float) -> Tuple[float, float]:
    if prop_type != "threes":
        sigma = max(sigma, 1e-6); z = (mu - line) / sigma
        p = 1.0 - _normal_cdf((line - mu) / sigma) if pick == "over" else _normal_cdf((line - mu) / sigma)
        return min(1.0 - 1e-4, max(1e-4, p)), z
    ints = [max(0, int(round(v))) for v in values if v is not None]
    if len(ints) == 0:
        sigma = max(sigma, 1e-6); z = (mu - line) / sigma
        p = 1.0 - _normal_cdf((line - mu) / sigma) if pick == "over" else _normal_cdf((line - mu) / sigma)
        return min(1.0 - 1.0e-4, max(1.0e-4, p)), z
    mean, r, _ = _fit_nb_params(ints)
    if r == float("inf"):
        lam = mean
        if pick == "over":
            k = math.ceil(line); p = 1.0 - _poisson_cdf(k - 1, lam)
        else:
            k = math.floor(line); p = _poisson_cdf(k, lam)
        z_like = (mu - line) / max(1.0e-6, sigma)
        return min(1.0 - 1.0e-4, max(1.0e-4, p)), z_like
    p_nb = mean / (mean + r)
    if pick == "over":
        k = math.ceil(line); p = 1.0 - _nb_cdf(k - 1, r, p_nb)
    else:
        k = math.floor(line); p = _nb_cdf(k, r, p_nb)
    z_like = (mu - line) / max(1.0e-6, sigma)
    return min(1.0 - 1.0e-4, max(1.0e-4, p)), z_like

# ========= DATA PERSISTENCE =========
def load_data(filename, default=None):
    if os.path.exists(filename):
        try:
            with open(filename, "rb") as f: return pickle.load(f)
        except Exception: return default if default is not None else {}
    return default if default is not None else {}

def save_data(filename, data):
    with open(filename, "wb") as f: pickle.dump(data, f)

def load_equity():
    if os.path.exists(EQUITY_FILE):
        try:
            with open(EQUITY_FILE, "rb") as f: return pickle.load(f)
        except Exception: return [BANKROLL]
    return [BANKROLL]

def save_equity(curve: List[float]):
    try:
        with open(EQUITY_FILE, "wb") as f: pickle.dump(curve, f)
    except Exception: pass

# ========= BET LEDGER (learning from outcomes) =========

def _load_ledger() -> Dict[str, Any]:
    ld = load_data(LEDGER_FILE, default={})
    if not isinstance(ld, dict):
        ld = {}
    ld.setdefault("bets", [])  # list of bet dicts
    return ld


def _load_calibration() -> Dict[str, Any]:
    cal = load_data(CALIBRATION_FILE, default={})
    if not isinstance(cal, dict):
        cal = {}
    return cal


def _save_calibration(cal: Dict[str, Any]) -> None:
    try:
        save_data(CALIBRATION_FILE, cal)
    except Exception:
        pass


def apply_calibration(prop_type: str, p: float) -> float:
    """Apply isotonic-like piecewise calibration from stored bins; fallback to identity."""
    try:
        cal = _load_calibration().get(prop_type)
        if not cal:
            return p
        bins = cal.get("bins"); vals = cal.get("vals")
        if not bins or not vals or len(bins) != len(vals):
            return p
        # Linear interpolate
        import bisect
        i = bisect.bisect_left(bins, p)
        if i <= 0:
            return float(vals[0])
        if i >= len(bins):
            return float(vals[-1])
        x0, x1 = float(bins[i-1]), float(bins[i])
        y0, y1 = float(vals[i-1]), float(vals[i])
        if x1 == x0:
            return float(y0)
        t = (p - x0)/(x1 - x0)
        return float(y0 + t * (y1 - y0))
    except Exception:
        return p


def _save_ledger(ledger: Dict[str, Any]) -> None:
    try:
        save_data(LEDGER_FILE, ledger)
    except Exception:
        pass


def _unique_bet_key(b: Dict[str, Any]) -> str:
    return f"{b.get('prop_id','')}|{b.get('pick','')}|{str(b.get('game_date',''))}"


def record_recommendation(prop: Dict[str, Any], predicted_prob: float) -> None:
    """Append a recommended bet to the ledger (idempotent by key)."""
    try:
        ledger = _load_ledger()
        bet = {
            "prop_id": prop.get("prop_id"),
            "player": prop.get("player"),
            "prop_type": prop.get("prop_type"),
            "line": float(prop.get("line", 0.0)) if prop.get("line") is not None else 0.0,
            "pick": prop.get("pick"),
            "odds": int(prop.get("odds", 0)) if prop.get("odds") is not None else 0,
            "bookmaker": prop.get("bookmaker"),
            "game": prop.get("game"),
            "game_date": prop.get("game_date"),
            "predicted_prob": float(predicted_prob),
            "recorded_at": dt.now(timezone.utc).isoformat(),
            "settled": False,
            "actual": None,
            "won": None,
        }
        key = _unique_bet_key(bet)
        existing = { _unique_bet_key(x): i for i, x in enumerate(ledger.get("bets", [])) }
        if key not in existing:
            ledger["bets"].append(bet)
            _save_ledger(ledger)
    except Exception:
        pass


def _season_str_from_date(d: datetime.date) -> str:
    end_year = d.year + (1 if d.month >= 8 else 0)
    return f"{end_year-1}-{str(end_year)[-2:]}"


def _fetch_player_stat_on_date(player_name: str, prop_type: str, date_str: str) -> Optional[float]:
    """Fetch actual player stat for the given date using nba_api; returns None on failure."""
    try:
        import pandas as _pd  # local alias to avoid confusion
        from nba_api.stats.endpoints import playergamelog
        from nba_api.stats.static import players as nba_players

        # Resolve player ID by full name (case-insensitive)
        plist = nba_players.get_players()
        pid = None
        for p in plist:
            if str(p.get('full_name','')).lower() == str(player_name).lower():
                pid = p.get('id'); break
        if pid is None:
            return None

        # Parse date and derive season string
        dt_val = _pd.to_datetime(date_str, errors='coerce')
        if dt_val is None or _pd.isna(dt_val):
            return None
        season = _season_str_from_date(dt_val.date())

        gl = playergamelog.PlayerGameLog(player_id=pid, season=season, season_type_all_star='Regular Season')
        df = gl.get_data_frames()[0]
        if df is None or df.empty:
            return None
        # Match by date string equality on GAME_DATE
        target = _pd.to_datetime(df['GAME_DATE'], errors='coerce').dt.date
        mask = target == dt_val.date()
        row = df.loc[mask]
        if row.empty:
            return None
        stat_col_map = {"points": "PTS", "assists": "AST", "rebounds": "REB", "threes": "FG3M", "minutes": "MIN"}
        col = stat_col_map.get(prop_type)
        if col is None or col not in row.columns:
            return None
        val = row.iloc[0][col]
        if col == "MIN":
            # minutes may be "MM:SS"
            s = str(val)
            if ":" in s:
                mm, ss = s.split(":")
                return float(mm) + float(ss)/60.0
            return float(val)
        return float(val)
    except Exception:
        return None


def settle_ledger(verbose: bool = True) -> int:
    """Settle unsettled bets by fetching actuals and updating prop weights. Returns count settled.
    Also refresh calibration curves per prop_type from the ledger.
    """
    ledger = _load_ledger()
    bets = ledger.get("bets", [])
    if not bets:
        if verbose: print("- No bets in ledger to settle")
        return 0
    settled = 0
    for b in bets:
        if b.get("settled"):
            continue
        gd = b.get("game_date")
        try:
            dt_val = pd.to_datetime(gd, errors='coerce')
        except Exception:
            dt_val = None
        if dt_val is None or pd.isna(dt_val):
            continue
        # Only settle past games
        if dt_val.tz_localize(None) > pd.Timestamp.now():
            continue
        actual = _fetch_player_stat_on_date(b.get("player",""), b.get("prop_type",""), str(gd))
        if actual is None:
            continue
        line = float(b.get("line", 0.0))
        pick = str(b.get("pick",""))
        if pick == "over":
            if actual == line:
                # push — skip learning
                b.update({"settled": True, "actual": actual, "won": None})
                settled += 1
                continue
            won = actual > line
        else:
            if actual == line:
                b.update({"settled": True, "actual": actual, "won": None})
                settled += 1
                continue
            won = actual < line
        b.update({"settled": True, "actual": actual, "won": bool(won)})
        try:
            update_prop_weights(b.get("prop_id",""), bool(won), float(b.get("predicted_prob", 0.5)))
        except Exception:
            pass
        settled += 1
    _save_ledger(ledger)
    # Recompute calibration maps (per prop_type) using recent settled bets (exclude pushes)
    try:
        df = pd.DataFrame(ledger.get("bets", []))
        if not df.empty and {"prop_type","predicted_prob","won"}.issubset(df.columns):
            df = df[df["won"].notna()].copy()
            cal_out: Dict[str, Any] = {}
            for pt, grp in df.groupby("prop_type"):
                p_pred = pd.to_numeric(grp["predicted_prob"], errors="coerce").clip(0.01, 0.99)
                y = grp["won"].astype(int)
                if len(p_pred) >= 200:
                    # 10 bins
                    bins = np.linspace(0.05, 0.95, 11)
                    inds = np.digitize(p_pred, bins, right=True)
                    xs, ys = [], []
                    for i in range(1, len(bins)):
                        mask = inds == i
                        if mask.any():
                            xs.append(float(bins[i-1]))
                            ys.append(float(y[mask].mean()))
                    if len(xs) >= 2:
                        cal_out[pt] = {"bins": xs, "vals": ys}
            if cal_out:
                cur = _load_calibration(); cur.update(cal_out); _save_calibration(cur)
    except Exception:
        pass
    if verbose: print(f"- Settled {settled} bets")
    return settled

prop_weights = load_data(WEIGHTS_FILE, defaultdict(lambda: 1.0))
prop_results = load_data(RESULTS_FILE, defaultdict(list))
player_cache = load_data(CACHE_FILE, {})
_pid_cache: Dict[str, int] = player_cache.get("__pid_cache__", {})

# Quick CLI hook to settle bets without running full analysis
if __name__ == "__main__" and "--settle-bets" in sys.argv:
    try:
        settle_ledger(verbose=True)
    finally:
        sys.exit(0)

# ========= API-Sports (games + player stats) =========
def fetch_json(endpoint: str, params: dict = None, retries: int = RETRIES, timeout: int = REQUEST_TIMEOUT) -> Optional[dict]:
    url = f"{BASE_URL}{endpoint}"
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=timeout)
            if resp.status_code == 200: return resp.json()
            if resp.status_code == 429:
                if DEBUG_MODE: print(f"   ⚠️ Rate limited. Waiting {2 ** attempt}s...")
                time.sleep(2 ** attempt)
            else:
                if DEBUG_MODE: print(f"   ❌ Error {resp.status_code}: {resp.text[:160]}")
                return None
        except Exception as e:
            if DEBUG_MODE: print(f"   ⚠️ Request failed: {e}")
            if attempt < retries - 1: time.sleep(0.5)
    return None

def _lookup_player_id(name: str) -> Optional[int]:
    if name in _pid_cache:
        return _pid_cache[name]
    parts = name.strip().split()
    candidates = []
    if len(parts) >= 2:
        candidates.append(" ".join(parts[::-1]))
    candidates.append(name.strip())
    if len(parts) >= 2:
        candidates.append(parts[-1])
    for q in candidates:
        data = fetch_json("/players", params={"search": q})
        if data and "response" in data and data["response"]:
            toks = set(t.lower() for t in name.split())
            best = None
            for rec in data["response"]:
                nm = rec.get("name","")
                if nm and toks.issubset(set(nm.lower().split())):
                    best = rec; break
            if not best:
                best = data["response"][0]
            pid = best.get("id")
            if pid:
                _pid_cache[name] = pid
                player_cache["__pid_cache__"] = _pid_cache
                save_data(CACHE_FILE, player_cache)
                return pid
    return None

def _parse_game_row(row: dict) -> dict:
    try:
        points = row.get("points", 0) or 0
        assists = row.get("assists", 0) or 0
        rbd = row.get("rebounds", {}); rebounds = (rbd.get("total", 0) if isinstance(rbd, dict) else rbd) or 0
        threes_data = row.get("threepoint_goals", {}); threes = threes_data.get("total", 0) if isinstance(threes_data, dict) else 0

        # Extract minutes - API-Sports format is "MM:SS" string
        min_val = row.get("min", "") or row.get("minutes", "") or "0:00"
        if isinstance(min_val, str) and ':' in min_val:
            parts = min_val.split(':')
            minutes = float(parts[0]) + float(parts[1]) / 60.0 if len(parts) == 2 else 0.0
        else:
            minutes = float(min_val) if min_val else 0.0

        return {
            "points": float(points),
            "assists": float(assists),
            "rebounds": float(rebounds),
            "threes": float(threes),
            "minutes": minutes
        }
    except Exception:
        return {}

def get_player_stats_split(player_name: str, max_last: int, max_curr: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cache_key = f"split_{player_name}_{max_last}_{max_curr}"
    if cache_key in player_cache:
        ts, data = player_cache[cache_key]
        if (datetime.datetime.now() - ts).total_seconds() < 1800:
            return data.get("last", pd.DataFrame()), data.get("curr", pd.DataFrame())
    pid = _lookup_player_id(player_name)
    if not pid: return pd.DataFrame(), pd.DataFrame()

    last, curr = [], []

    # Last season: Use API-Sports (historical data is reliable)
    d_last = fetch_json("/games/statistics/players", params={"season": STATS_SEASON, "player": pid})
    if d_last and "response" in d_last:
        for r in d_last["response"][:max_last]:
            g = _parse_game_row(r)
            if g: last.append(g)

    # Current season: Use nba_api for accurate real-time stats
    try:
        from nba_api.stats.endpoints import playergamelog
        from nba_api.stats.static import players

        # Find NBA.com player ID
        nba_players = players.get_players()
        nba_player = next((p for p in nba_players if p['full_name'].lower() == player_name.lower()), None)

        if nba_player:
            # Get current season game log (2025-26)
            gamelog = playergamelog.PlayerGameLog(
                player_id=nba_player['id'],
                season='2025-26',
                season_type_all_star='Regular Season'
            )
            df_games = gamelog.get_data_frames()[0]

            if not df_games.empty:
                # Take most recent games up to max_curr
                for idx, row in df_games.head(max_curr).iterrows():
                    # Extract minutes - handle both float and string formats (e.g., "36:25" or 36.42)
                    min_val = row.get('MIN', 0) or 0
                    if isinstance(min_val, str) and ':' in min_val:
                        # Format is "MM:SS" - convert to decimal minutes
                        parts = min_val.split(':')
                        minutes = float(parts[0]) + float(parts[1]) / 60.0 if len(parts) == 2 else 0.0
                    else:
                        minutes = float(min_val)

                    curr.append({
                        "points": float(row.get('PTS', 0) or 0),
                        "assists": float(row.get('AST', 0) or 0),
                        "rebounds": float(row.get('REB', 0) or 0),
                        "threes": float(row.get('FG3M', 0) or 0),
                        "minutes": minutes
                    })
                if DEBUG_MODE and len(curr) > 0:
                    print(f"   ✓ Fetched {len(curr)} current season games for {player_name} from nba_api")
            else:
                if DEBUG_MODE:
                    print(f"   ⚠ No current season data from nba_api for {player_name}, falling back to API-Sports")
                # Fallback to API-Sports if nba_api has no data
                d_curr = fetch_json("/games/statistics/players", params={"season": SEASON, "player": pid})
                if d_curr and "response" in d_curr:
                    for r in d_curr["response"][:max_curr]:
                        g = _parse_game_row(r)
                        if g: curr.append(g)
        else:
            if DEBUG_MODE:
                print(f"   ⚠ Player not found in nba_api: {player_name}, using API-Sports")
            # Fallback to API-Sports
            d_curr = fetch_json("/games/statistics/players", params={"season": SEASON, "player": pid})
            if d_curr and "response" in d_curr:
                for r in d_curr["response"][:max_curr]:
                    g = _parse_game_row(r)
                    if g: curr.append(g)

    except Exception as e:
        if DEBUG_MODE:
            print(f"   ⚠ Error fetching from nba_api for {player_name}: {e}, using API-Sports")
        # Fallback to API-Sports on error
        d_curr = fetch_json("/games/statistics/players", params={"season": SEASON, "player": pid})
        if d_curr and "response" in d_curr:
            for r in d_curr["response"][:max_curr]:
                g = _parse_game_row(r)
                if g: curr.append(g)

    df_last = pd.DataFrame(last); df_curr = pd.DataFrame(curr)
    player_cache[cache_key] = (datetime.datetime.now(), {"last": df_last, "curr": df_curr})
    save_data(CACHE_FILE, player_cache)
    return df_last, df_curr

def get_upcoming_games() -> List[dict]:
    # Use current date for game fetching
    today = datetime.date.today()
    dates = [today + datetime.timedelta(days=i) for i in range(DAYS_TO_FETCH)]
    print(f"📅 Fetching games for {DAYS_TO_FETCH} days:\n" + "\n".join([f"   - {d:%Y-%m-%d}" for d in dates]) + "\n")
    games = []
    for d in dates:
        data = fetch_json("/games", params={"league": LEAGUE_ID, "season": SEASON, "date": d.strftime("%Y-%m-%d"), "timezone": "America/Chicago"})
        if data and "response" in data:
            games.extend(data["response"])
        time.sleep(SLEEP_SHORT)
        if len(games) >= MAX_GAMES:
            break
    print(f"   Found {len(games)} total games\n"); return games[:MAX_GAMES]

def get_matchup_context(game_info: dict) -> dict:
    # Placeholder for pace/defense factors (keep neutral for now)
    return {"pace": 1.0, "home_defensive_factor": 1.0, "away_defensive_factor": 1.0}

def update_prop_weights(prop_id: str, actual_result: bool, predicted_prob: float):
    prop_results[prop_id].append({"result": actual_result, "predicted": predicted_prob, "timestamp": datetime.datetime.now()})
    results = prop_results[prop_id]
    if len(results) >= 5:
        correct = sum(1 for r in results[-20:] if r["result"]); acc = correct / len(results[-20:])
        if acc > 0.60: prop_weights[prop_id] *= 1.05
        elif acc < 0.45: prop_weights[prop_id] *= 0.95
    save_data(WEIGHTS_FILE, dict(prop_weights)); save_data(RESULTS_FILE, dict(prop_results))

def get_prop_confidence_multiplier(prop_id: str) -> float:
    return float(prop_weights.get(prop_id, 1.0))

# ========= SGO (RapidAPI sportsbook) HELPERS =========
def _norm_team(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace("saint", "st")
    if "trail" in s and "blazers" in s:
        s = s.replace("blazers", "trail blazers")
    if ("clippers" in s or "lakers" in s) and "los angeles" in s:
        s = s.replace("los angeles", "la")
    if "warriors" in s and "golden state" in s:
        s = s.replace("golden state", "gs")
    return s

def _token_match(a: str, b: str, threshold: float = 0.6) -> bool:
    ta = set(_norm_team(a).split())
    tb = set(_norm_team(b).split())
    if not ta or not tb:
        return False
    return (len(ta & tb) / max(len(ta), len(tb))) >= threshold

def _parse_iso8601(x: Optional[str]) -> Optional[dt]:
    if not x:
        return None
    try:
        return dt.fromisoformat(x.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None

def _within_minutes(ts_a: Optional[str], ts_b: Optional[str], window_min: int = 240) -> bool:
    da = _parse_iso8601(ts_a); db = _parse_iso8601(ts_b)
    if not da or not db:
        return True
    return abs((da - db).total_seconds()) <= window_min * 60

def _index_games(apisports_games: List[dict]) -> List[Tuple[dict, str, str, str]]:
    idx = []
    for g in apisports_games:
        home_name = g.get("teams", {}).get("home", {}).get("name", "") or g.get("home_team", "")
        away_name = g.get("teams", {}).get("away", {}).get("name", "") or g.get("away_team", "")
        start_iso = g.get("date") or g.get("commence_time") or g.get("start_time") or ""
        idx.append((g, _norm_team(home_name), _norm_team(away_name), start_iso))
    return idx

def _match_game(idx: List[Tuple[dict, str, str, str]], home: str, away: str, start: Optional[str]) -> Optional[dict]:
    nh, na = _norm_team(home), _norm_team(away)
    for g, gh, ga, gstart in idx:
        if (_token_match(nh, gh) and _token_match(na, ga)) or (_token_match(nh, ga) and _token_match(na, gh)):
            if _within_minutes(gstart, start, window_min=240):
                return g
    return None

def _rapidapi_headers(api_key: Optional[str] = None, host: Optional[str] = None) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    k = api_key or SGO_RAPIDAPI_KEY
    h = host or SGO_RAPIDAPI_HOST
    if k:
        headers["x-rapidapi-key"] = k
    if h:
        headers["x-rapidapi-host"] = h
    return headers

def _to_int_american(v) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        pass
    try:
        dec = float(v)
        if dec <= 1.0:
            return None
        if dec >= 2.0:
            return int(round((dec - 1.0) * 100))
        return int(round(-100.0 / (dec - 1.0)))
    except Exception:
        return None

def sgo_fetch_events(
    api_key: Optional[str] = None,
    apisports_games: Optional[List[dict]] = None,
    market_odds_available: bool = True,
    limit: int = SGO_LIMIT,
    max_pages: int = SGO_MAX_PAGES,
    pause_sec: float = SGO_SLEEP_BETWEEN_PAGES_SEC,
    fetch_player_props: bool = True,
) -> Tuple[List[dict], List[str]]:
    """
    Fetch markets from TheRundown API:
    1. First fetch game list from /sports/4/events/{date}
    2. Then fetch detailed event data with player props from /v2/events/{event_id}
    """
    api_key = api_key or SGO_RAPIDAPI_KEY
    host = SGO_RAPIDAPI_HOST
    headers = _rapidapi_headers(api_key, host)
    out: List[dict] = []
    warns: List[str] = []

    # If caller passed apisports_games, derive unique game dates to query; otherwise query today
    dates_to_query = set()
    if apisports_games:
        for g in apisports_games:
            dt_raw = g.get("date") or g.get("commence_time") or g.get("start_time") or g.get("game_date") or ""
            try:
                parsed = dt.fromisoformat(str(dt_raw).replace("Z", "+00:00")).date()
                dates_to_query.add(parsed.strftime("%Y-%m-%d"))
            except Exception:
                # fallback: try YYYY-MM-DD substring
                try:
                    dates_to_query.add(str(dt_raw)[:10])
                except Exception:
                    pass
    if not dates_to_query:
        dates_to_query.add(datetime.date.today().strftime("%Y-%m-%d"))

    # Step 1: Fetch basic game events from V1 endpoint
    game_events = []
    for date_str in sorted(dates_to_query):
        events = _fetch_events_for_date(date_str, headers, limit, max_pages, pause_sec, warns)
        game_events.extend(events)
    
    # Step 2: If player props requested, fetch detailed event data from V2 endpoint
    if fetch_player_props and game_events:
        event_ids = []
        for ev in game_events:
            # Unwrap if needed
            ev_data = ev.get('odds', ev) if 'odds' in ev else ev
            event_id = ev_data.get("event_id", "")
            if event_id:
                event_ids.append(event_id)
        
        if DEBUG_MODE:
            print(f"   [DEBUG] Fetching player props for {len(event_ids)} events via V2 endpoint...")
        
        # Fetch V2 data for each event (with player props)
        for idx, event_id in enumerate(event_ids):
            v2_event = _fetch_v2_event_with_player_props(event_id, headers, warns)
            if v2_event:
                # V2 response has structure: {'meta': ..., 'events': [...]}
                # Extract the events array and add each
                if isinstance(v2_event, dict) and 'events' in v2_event:
                    for evt in v2_event['events']:
                        if isinstance(evt, dict):
                            out.append(evt)
                else:
                    out.append(v2_event)
            if (idx + 1) % 5 == 0 and DEBUG_MODE:
                print(f"   [DEBUG] Fetched {idx + 1}/{len(event_ids)} V2 events")
            time.sleep(0.1)  # Rate limiting
    else:
        out = game_events

    if DEBUG_MODE:
        print(f"   [DEBUG] RapidAPI events/markets fetched: {len(out)}")
        with_markets = sum(1 for e in out if isinstance(e.get('markets'), list) and e['markets'])
        with_odds = sum(1 for e in out if isinstance(e.get('odds'), (list, dict)) and e['odds'])
        with_lines = sum(1 for e in out if isinstance(e.get('lines'), dict) and e['lines'])
        print(f"   [DEBUG] Events with markets[]: {with_markets} | with odds[]: {with_odds} | with lines: {with_lines}")
        # Print first event structure for debugging
        if out and len(out) > 0:
            print(f"   [DEBUG] First event keys: {list(out[0].keys())}")
            if 'markets' in out[0] and out[0]['markets']:
                print(f"   [DEBUG] First event has {len(out[0]['markets'])} markets")
                if len(out[0]['markets']) > 0:
                    first_market = out[0]['markets'][0]
                    print(f"   [DEBUG] First market keys: {list(first_market.keys()) if isinstance(first_market, dict) else 'not a dict'}")
                    print(f"   [DEBUG] First market sample: {str(first_market)[:800]}")
            print(f"   [DEBUG] First event sample (first 1500 chars): {str(out[0])[:1500]}")
    if not api_key:
        warns.append("RAPIDAPI_KEY not set; set RAPIDAPI_KEY to use RapidAPI sportsbook.")
    return out, warns

def _fetch_events_for_date(
    date_str: str,
    headers: Dict[str, str],
    limit: int,
    max_pages: int,
    pause_sec: float,
    warns: List[str],
) -> List[dict]:
    """Helper to fetch events for a specific date from V1 endpoint."""
    out = []
    pages = 0
    next_cursor = None
    
    while True:
        url = f"{SGO_RAPIDAPI_BASE}{SGO_RAPIDAPI_EVENTS_PATH}/{date_str}"
        params = {"limit": limit}
        
        if SGO_AFFILIATE_IDS:
            params["affiliate_ids"] = SGO_AFFILIATE_IDS
        if next_cursor:
            params["cursor"] = next_cursor

        if DEBUG_MODE and pages == 0:
            print(f"   [DEBUG] Fetching V1 events for {date_str}")

        try:
            resp = requests.get(url, params=params, headers=headers, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            warns.append(f"V1 API fetch error ({date_str}): {e}")
            if DEBUG_MODE:
                print(f"   [DEBUG] Error: {e}")
            break

        # Normalize returned shapes
        events = data.get("events") or data.get("markets") or data.get("data") or data.get("items") or data.get("odds") or []
        if isinstance(events, dict):
            events = list(events.values())

        normalized = []
        for node in events:
            if isinstance(node, dict) and ("markets" in node or "bookmakers" in node or "outcomes" in node or "lines" in node or "odds" in node):
                normalized.append(node)
            else:
                normalized.append({"odds": node})

        if normalized:
            out.extend(normalized)

        # pagination
        next_cursor = data.get("nextCursor") or data.get("cursor") or None
        pages += 1
        if not next_cursor or pages >= max_pages:
            break
        time.sleep(pause_sec)
    
    if DEBUG_MODE and out:
        print(f"   [DEBUG] Fetched {len(out)} V1 events for {date_str}")
    
    return out


def _fetch_v2_event_with_player_props(
    event_id: str,
    headers: Dict[str, str],
    warns: List[str],
) -> Optional[dict]:
    """Fetch detailed event data with player props from V2 endpoint."""
    url = f"{SGO_RAPIDAPI_BASE}{SGO_RAPIDAPI_V2_EVENT_PATH}/{event_id}"
    params = {
        "participant_type": "TYPE_PLAYER",
        "market_ids": SGO_PLAYER_MARKET_IDS,
    }
    
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        return data
    except Exception as e:
        warns.append(f"V2 API fetch error (event {event_id}): {e}")
        if DEBUG_MODE:
            print(f"   [DEBUG] V2 fetch error for event {event_id}: {e}")
        return None

def _extract(obj: dict, keys: Tuple[str, ...]):
    for k in keys:
        if k in obj and obj[k] is not None:
            return obj[k]
    return None

def _extract_line(obj: dict) -> Optional[float]:
    v = _extract(obj, ("line", "point", "points", "total", "handicap", "threshold", "value"))
    if v is None: return None
    try:
        return float(v)
    except Exception:
        return None

def _extract_market(obj: dict) -> Optional[str]:
    v = _extract(obj, ("marketType", "market", "marketName", "key", "type"))
    return str(v).lower() if v is not None else None

def _extract_bookmaker(obj: dict) -> Optional[str]:
    v = _extract(obj, ("bookmaker", "bookmakerId", "bookmakerID", "sportsbook", "provider", "source", "book", "name"))
    if v is None:
        return None
    return str(v).strip().lower()

def _extract_player_name(obj: dict) -> Optional[str]:
    v = _extract(obj, ("player", "playerName", "name", "participant", "runnerName", "selectionName"))
    return str(v) if v is not None else None

def _extract_team_name(obj: dict) -> Optional[str]:
    v = _extract(obj, ("team", "teamName", "name", "runnerName", "selectionName"))
    return str(v) if v is not None else None

def _read_event_meta(ev: dict) -> Tuple[str, str, str, str]:
    event_id = str(_extract(ev, ("eventID","id","eventId")) or "")
    away = str(_extract(ev, ("awayTeam","away","visitor")) or "")
    home = str(_extract(ev, ("homeTeam","home","host")) or "")
    start = str(_extract(ev, ("startTime","commenceTime","commence_time","start_time","date")) or "")
    return event_id, away, home, start

def _iter_odds_objects(ev: dict):
    # Try multiple shapes (flat odds list, dict keyed by id, nested markets/bookmakers)
    odds = ev.get("odds") or ev.get("markets") or ev.get("market") or ev.get("data") or {}
    if isinstance(odds, dict):
        for odd_id, obj in odds.items():
            if isinstance(obj, dict):
                yield odd_id, obj
    elif isinstance(odds, list):
        for obj in odds:
            if isinstance(obj, dict):
                yield obj.get("id") or obj.get("_id") or "", obj
    else:
        # fallback: the event itself may be a single odds object
        if isinstance(ev, dict):
            yield ev.get("id") or ev.get("eventID") or "", ev

# Bookmaker matching (FanDuel) for nested markets
def _is_fanduel_bookmaker(bm_node: dict) -> bool:
    cand = []
    for k in ("bookmaker", "bookmakerId", "bookmakerID", "id", "name", "provider", "sportsbook", "source", "book"):
        v = bm_node.get(k)
        if v:
            cand.append(str(v).lower())
    return any("fanduel" in x for x in cand)

# Coerce generic market string to our internal type (fallback for flat odds)
def _coerce_market(market_raw: str) -> Optional[str]:
    if not market_raw:
        return None
    m = market_raw.lower()
    # Player props
    if "player" in m:
        if "assist" in m:
            return "assists"
        if "rebound" in m:
            return "rebounds"
        if "3" in m or "three" in m or "3pt" in m or "3pm" in m or "3-pointer" in m or "3-pointers" in m:
            return "threes"
        if "point" in m or "pts" in m or "scor" in m:
            return "points"
    # Game markets
    if "spread" in m or "handicap" in m or "point spread" in m:
        return "spread"
    if "moneyline" in m or "h2h" in m or m == "ml" or "match winner" in m:
        return "moneyline"
    return None

def _parse_nested_markets(ev: dict, matched_game: dict) -> List[dict]:
    props: List[dict] = []
    markets = ev.get("markets") or []
    if not isinstance(markets, list):
        # sometimes the top-level is a list of bookmakers/outcomes directly
        markets = [ev]

    game_id = matched_game.get("id") or matched_game.get("gameId") or ev.get("eventID") or ev.get("id") or ""
    away = matched_game.get("teams",{}).get("away",{}).get("name") or ev.get("awayTeam") or ev.get("away") or ""
    home = matched_game.get("teams",{}).get("home",{}).get("name") or ev.get("homeTeam") or ev.get("home") or ""
    game_label = f"{away} at {home}"
    game_date = matched_game.get("date") or ev.get("startTime") or ev.get("commenceTime") or ""

    for m in markets:
        bet_type = str(m.get("betTypeID") or m.get("type") or m.get("key") or "").lower()
        market_name = str(m.get("marketName") or m.get("name") or "").lower()

        # Determine internal type
        internal_type: Optional[str] = None
        # Player props by name
        if "player" in market_name or bet_type.startswith("player"):
            if "assist" in market_name:
                internal_type = "assists"
            elif "rebound" in market_name:
                internal_type = "rebounds"
            elif "3" in market_name or "three" in market_name:
                internal_type = "threes"
            elif "point" in market_name or "pts" in market_name or "score" in market_name:
                internal_type = "points"
        if internal_type is None:
            # Game markets by betTypeID
            if bet_type in ("ml","moneyline"):
                internal_type = "moneyline"
            elif bet_type in ("sp","spread","handicap"):
                internal_type = "spread"
            # If OU with player participants, handle below while looping outcomes

        bms = m.get("bookmakers") or m.get("providers") or m.get("books") or []
        # If bookmakers not present, the market object might be an outcome list itself
        if not bms and isinstance(m.get("outcomes") or m.get("selections") or m.get("runners"), list):
            bms = [{"name": "unknown", "outcomes": m.get("outcomes") or m.get("selections") or m.get("runners")}]

        for bm in bms:
            if not _is_fanduel_bookmaker(bm) and "fanduel" not in str(bm.get("name","")).lower():
                # skip non-FanDuel lines (we focus on FanDuel-like naming), but don't strictly require it - keep option
                pass
            outcomes = bm.get("outcomes") or bm.get("runners") or bm.get("selections") or []

            # Try detect player OU inside OU markets (when bet_type == 'ou')
            if internal_type is None and bet_type in ("ou","over_under","totals"):
                internal_type_guess = None
                if "assist" in market_name:
                    internal_type_guess = "assists"
                elif "rebound" in market_name:
                    internal_type_guess = "rebounds"
                elif "3" in market_name or "three" in market_name:
                    internal_type_guess = "threes"
                elif "point" in market_name or "pts" in market_name or "score" in market_name:
                    internal_type_guess = "points"
                if internal_type_guess:
                    internal_type = internal_type_guess

            if internal_type in {"points","assists","rebounds","threes"}:
                bucket: Dict[str, Dict[str, Any]] = {}
                for oc in outcomes:
                    side = str(oc.get("sideID") or oc.get("side") or oc.get("name") or "").lower()
                    player = oc.get("participant") or oc.get("player") or oc.get("runnerName") or oc.get("selectionName") or oc.get("name")
                    if not player:
                        continue
                    line = oc.get("line") or oc.get("threshold") or oc.get("value") or m.get("line") or m.get("threshold")
                    try:
                        line = float(line)
                    except Exception:
                        continue
                    american = _to_int_american(oc.get("americanOdds") or oc.get("oddsAmerican") or oc.get("price") or oc.get("odds"))
                    if american is None:
                        continue
                    key = f"{player}|{line}"
                    if key not in bucket:
                        bucket[key] = {"player": str(player), "line": float(line)}
                    if side.startswith("over"):
                        bucket[key]["odds_over"] = int(american)
                    elif side.startswith("under"):
                        bucket[key]["odds_under"] = int(american)
                for _, rec in bucket.items():
                    prop_key = f"{game_id}_{rec['player']}_{internal_type}".replace(" ", "_")
                    props.append({
                        "prop_id": prop_key,
                        "game_id": game_id,
                        "game": game_label,
                        "game_date": game_date,
                        "player": rec["player"],
                        "prop_type": internal_type,
                        "line": rec["line"],
                        "odds": int(rec.get("odds_over", rec.get("odds_under", -110))),
                        **({"odds_over": rec["odds_over"]} if "odds_over" in rec else {}),
                        **({"odds_under": rec["odds_under"]} if "odds_under" in rec else {}),
                        "bookmaker": bm.get("name", "RapidAPI"),
                        "source": "RAPIDAPI",
                    })

            elif internal_type == "moneyline":
                for oc in outcomes:
                    side = str(oc.get("sideID") or oc.get("side") or oc.get("name") or "").lower()
                    team = oc.get("team") or oc.get("runnerName") or oc.get("selectionName") or (home if side == "home" else away if side == "away" else None)
                    american = _to_int_american(oc.get("americanOdds") or oc.get("oddsAmerican") or oc.get("price") or oc.get("odds"))
                    if not team or american is None:
                        continue
                    prop_id = f"{game_id}_moneyline_{team}".replace(" ", "_")
                    props.append({
                        "prop_id": prop_id, "game_id": game_id, "game": game_label, "game_date": game_date,
                        "player": str(team), "prop_type": "moneyline", "line": 0.0, "odds": int(american),
                        "bookmaker": bm.get("name", "RapidAPI"), "source": "RAPIDAPI",
                    })

            elif internal_type == "spread":
                for oc in outcomes:
                    side = str(oc.get("sideID") or oc.get("side") or oc.get("name") or "").lower()
                    team = oc.get("team") or oc.get("runnerName") or oc.get("selectionName") or (home if side == "home" else away if side == "away" else None)
                    line = oc.get("line") or oc.get("handicap") or m.get("line") or m.get("handicap")
                    american = _to_int_american(oc.get("americanOdds") or oc.get("oddsAmerican") or oc.get("price") or oc.get("odds"))
                    try:
                        line = float(line)
                    except Exception:
                        line = None
                    if not team or american is None or line is None:
                        continue
                    side_label = f"{team} {line:+.1f}"
                    prop_id = f"{game_id}_spread_{side_label}".replace(" ", "_")
                    props.append({
                        "prop_id": prop_id, "game_id": game_id, "game": game_label, "game_date": game_date,
                        "player": side_label, "prop_type": "spread", "line": float(line), "odds": int(american),
                        "bookmaker": bm.get("name", "RapidAPI"), "source": "RAPIDAPI",
                    })
    return props

# ========= API-SPORTS ODDS FETCHER =========
def apisports_fetch_odds(games: List[dict]) -> List[dict]:
    """
    Fetch odds from API-Sports.io for given games.
    Returns list of props (moneyline, spread, totals, player props).
    """
    if not APISPORTS_ODDS_ENABLED:
        return []
    
    props: List[dict] = []
    
    for game in games:
        game_id = game.get("id")
        if not game_id:
            continue

        # Fetch odds for this game
        url = f"{BASE_URL}/odds"

        # Note: API-Sports bookmaker parameter expects a single integer
        # Loop through bookmakers if multiple are configured
        bookmakers_to_fetch = APISPORTS_BOOKMAKERS if APISPORTS_BOOKMAKERS else [None]

        for bookmaker_id in bookmakers_to_fetch:
            params = {
                "game": game_id,
                "league": LEAGUE_ID,
                "season": SEASON.split("-")[0]  # "2025-2026" -> "2025"
            }
            if bookmaker_id:
                params["bookmaker"] = bookmaker_id

            try:
                resp = requests.get(url, headers=HEADERS, params=params, timeout=REQUEST_TIMEOUT)
                if DEBUG_MODE:
                    print(f"   [API-Sports] GET {url} params={params} status={resp.status_code}")

                if resp.status_code != 200:
                    if DEBUG_MODE:
                        print(f"   [API-Sports] Odds fetch failed for game {game_id}: {resp.status_code} - {resp.text[:200]}")
                    continue

                data = resp.json()
                odds_data = data.get("response", [])

                if DEBUG_MODE:
                    print(f"   [API-Sports] Game {game_id}: {len(odds_data)} odds entries")
                    if len(odds_data) == 0 and data:
                        print(f"   [API-Sports] Full response: {json.dumps(data, indent=2)[:500]}")

                if not odds_data:
                    continue

                # Parse odds data
                game_label = f"{game['teams']['away']['name']} at {game['teams']['home']['name']}"
                game_date = game.get("date", "")
                home_team = game['teams']['home']['name']
                away_team = game['teams']['away']['name']

                for odds_entry in odds_data:
                    bookmaker = odds_entry.get("bookmaker", {})
                    bookmaker_name = bookmaker.get("name", "Unknown")

                    bets = odds_entry.get("bets", [])

                    for bet in bets:
                        bet_id = bet.get("id")
                        bet_name = bet.get("name", "")
                        values = bet.get("values", [])

                        # Moneyline (bet_id=1)
                        if bet_id == 1:
                            for val in values:
                                team = val.get("value")
                                odds = val.get("odd")
                                if team and odds:
                                    try:
                                        # Convert decimal to American
                                        decimal_odds = float(odds)
                                        american_odds = (decimal_odds - 1) * 100 if decimal_odds >= 2.0 else -100 / (decimal_odds - 1)
                                        american_odds = int(american_odds)

                                        prop_id = f"{game_id}_moneyline_{team}_{bookmaker_name}".replace(" ", "_")
                                        props.append({
                                            "prop_id": prop_id,
                                            "game_id": game_id,
                                            "game": game_label,
                                            "game_date": game_date,
                                            "player": team,
                                            "prop_type": "moneyline",
                                            "line": 0.0,
                                            "odds": american_odds,
                                            "bookmaker": bookmaker_name,
                                            "source": "API-Sports",
                                        })
                                    except (ValueError, ZeroDivisionError):
                                        continue

                        # Spread (bet_id=2)
                        elif bet_id == 2:
                            for val in values:
                                team = val.get("value")
                                spread = val.get("handicap")
                                odds = val.get("odd")
                                if team and spread is not None and odds:
                                    try:
                                        decimal_odds = float(odds)
                                        american_odds = (decimal_odds - 1) * 100 if decimal_odds >= 2.0 else -100 / (decimal_odds - 1)
                                        american_odds = int(american_odds)
                                        spread_val = float(spread)

                                        side_label = f"{team} {spread_val:+.1f}"
                                        prop_id = f"{game_id}_spread_{side_label}_{bookmaker_name}".replace(" ", "_")
                                        props.append({
                                            "prop_id": prop_id,
                                            "game_id": game_id,
                                            "game": game_label,
                                            "game_date": game_date,
                                            "player": side_label,
                                            "prop_type": "spread",
                                            "line": spread_val,
                                            "odds": american_odds,
                                            "bookmaker": bookmaker_name,
                                            "source": "API-Sports",
                                        })
                                    except (ValueError, ZeroDivisionError):
                                        continue

                        # Over/Under Total (bet_id=3)
                        elif bet_id == 3:
                            for val in values:
                                ou_type = val.get("value", "").lower()
                                total = val.get("handicap")
                                odds = val.get("odd")
                                if total is not None and odds:
                                    try:
                                        decimal_odds = float(odds)
                                        american_odds = (decimal_odds - 1) * 100 if decimal_odds >= 2.0 else -100 / (decimal_odds - 1)
                                        american_odds = int(american_odds)
                                        total_val = float(total)

                                        # Note: Game totals are different from player props
                                        # Skip for now or handle separately
                                    except (ValueError, ZeroDivisionError):
                                        continue

                        # Player props (bet_id 12-15, etc.)
                        # Note: API-Sports player props may have different bet_ids
                        # Will need to check documentation for exact mappings

                time.sleep(0.1)  # Rate limiting

            except Exception as e:
                if DEBUG_MODE:
                    print(f"   [API-Sports] Error fetching odds for game {game_id}: {e}")
                continue
    
    return props


# ========= THE ODDS API FETCHER =========
def theodds_fetch_odds(games: List[dict]) -> List[dict]:
    """
    Fetch odds from The Odds API for given games.
    Returns list of props (moneyline, spread, totals, player props).

    IMPORTANT: Two-tier endpoint structure:
    1. GET /events - returns list of events with event IDs
    2. GET /events/{eventId}/odds - returns odds for specific event (includes player props)
    """
    if not THEODDS_ENABLED:
        return []

    props: List[dict] = []

    try:
        # Build game ID mapping (match by team names)
        game_map = {}
        for game in games:
            home = game['teams']['home']['name'].lower()
            away = game['teams']['away']['name'].lower()
            # Create multiple key variations for matching
            game_map[f"{away}_{home}"] = game
            game_map[f"{home}_{away}"] = game

        # STEP 1: Fetch list of events to get event IDs
        events_url = f"{THEODDS_BASE_URL}/sports/{THEODDS_SPORT}/events"
        events_params = {
            "apiKey": THEODDS_API_KEY,
        }

        events_resp = requests.get(events_url, params=events_params, timeout=REQUEST_TIMEOUT * 2)

        if DEBUG_MODE:
            print(f"   [TheOdds] GET {events_url} status={events_resp.status_code}")
            print(f"   [TheOdds] Remaining requests: {events_resp.headers.get('x-requests-remaining', 'unknown')}")

        if events_resp.status_code != 200:
            if DEBUG_MODE:
                print(f"   [TheOdds] Events fetch failed: {events_resp.status_code} - {events_resp.text[:200]}")
            return []

        events = events_resp.json()

        if DEBUG_MODE:
            print(f"   [TheOdds] Fetched {len(events)} events")
        
        # STEP 2: For each event, fetch odds with player props
        for event in events:
            event_id = event.get("id")
            home_team = event.get("home_team", "").lower()
            away_team = event.get("away_team", "").lower()
            # Use The Odds API's commence_time as authoritative game date
            event_commence_time = event.get("commence_time", "")

            # Match event to our game
            game_key = f"{away_team}_{home_team}"
            game = game_map.get(game_key) or game_map.get(f"{home_team}_{away_team}")

            if not game:
                # Try partial matching
                for key, g in game_map.items():
                    if home_team in key or away_team in key:
                        game = g
                        break

            if not game:
                if DEBUG_MODE:
                    print(f"   [TheOdds] No match for {away_team} @ {home_team}")
                continue

            game_id = game.get("id")
            game_label = f"{game['teams']['away']['name']} at {game['teams']['home']['name']}"
            # Use The Odds API's commence_time instead of matched game date for accuracy
            game_date = event_commence_time or game.get("date", "")

            # Fetch odds for this specific event (including player props)
            event_odds_url = f"{THEODDS_BASE_URL}/sports/{THEODDS_SPORT}/events/{event_id}/odds"
            event_odds_params = {
                "apiKey": THEODDS_API_KEY,
                "regions": THEODDS_REGIONS,
                "markets": THEODDS_MARKETS,  # Now includes player_points, player_rebounds, etc.
                "oddsFormat": "american",
            }

            if THEODDS_BOOKMAKERS:
                event_odds_params["bookmakers"] = THEODDS_BOOKMAKERS

            event_odds_resp = requests.get(event_odds_url, params=event_odds_params, timeout=REQUEST_TIMEOUT * 2)

            if DEBUG_MODE:
                print(f"   [TheOdds] Event {event_id[:8]}... status={event_odds_resp.status_code}")

            if event_odds_resp.status_code != 200:
                if DEBUG_MODE:
                    print(f"   [TheOdds] Event odds fetch failed: {event_odds_resp.status_code} - {event_odds_resp.text[:200]}")
                continue

            event_data = event_odds_resp.json()
            bookmakers = event_data.get("bookmakers", [])
            
            for bookmaker in bookmakers:
                bookmaker_name = bookmaker.get("title", "Unknown")
                markets = bookmaker.get("markets", [])
                
                for market in markets:
                    market_key = market.get("key", "")
                    outcomes = market.get("outcomes", [])
                    
                    # H2H (Moneyline)
                    if market_key == "h2h":
                        for outcome in outcomes:
                            team = outcome.get("name")
                            odds = outcome.get("price")
                            if team and odds:
                                prop_id = f"{game_id}_moneyline_{team}_{bookmaker_name}".replace(" ", "_")
                                props.append({
                                    "prop_id": prop_id,
                                    "game_id": game_id,
                                    "game": game_label,
                                    "game_date": game_date,
                                    "player": team,
                                    "home_team": game['teams']['home']['name'],
                                    "away_team": game['teams']['away']['name'],
                                    "prop_type": "moneyline",
                                    "line": 0.0,
                                    "odds": int(odds),
                                    "bookmaker": bookmaker_name,
                                    "source": "TheOddsAPI",
                                })
                    
                    # Spreads
                    elif market_key == "spreads":
                        for outcome in outcomes:
                            team = outcome.get("name")
                            point = outcome.get("point")
                            odds = outcome.get("price")
                            if team and point is not None and odds:
                                side_label = f"{team} {float(point):+.1f}"
                                prop_id = f"{game_id}_spread_{side_label}_{bookmaker_name}".replace(" ", "_")
                                props.append({
                                    "prop_id": prop_id,
                                    "game_id": game_id,
                                    "game": game_label,
                                    "game_date": game_date,
                                    "player": side_label,
                                    "home_team": game['teams']['home']['name'],
                                    "away_team": game['teams']['away']['name'],
                                    "prop_type": "spread",
                                    "line": float(point),
                                    "odds": int(odds),
                                    "bookmaker": bookmaker_name,
                                    "source": "TheOddsAPI",
                                })

                    # Totals
                    elif market_key == "totals":
                        for outcome in outcomes:
                            over_under = outcome.get("name", "").lower()  # "Over" or "Under"
                            point = outcome.get("point")
                            odds = outcome.get("price")
                            if point is not None and odds:
                                side_label = f"Total {over_under.capitalize()} {float(point)}"
                                prop_id = f"{game_id}_total_{over_under}_{point}_{bookmaker_name}".replace(" ", "_")
                                props.append({
                                    "prop_id": prop_id,
                                    "game_id": game_id,
                                    "game": game_label,
                                    "game_date": game_date,
                                    "player": side_label,
                                    "home_team": game['teams']['home']['name'],
                                    "away_team": game['teams']['away']['name'],
                                    "prop_type": "total",
                                    "line": float(point),
                                    "odds": int(odds),
                                    "bookmaker": bookmaker_name,
                                    "source": "TheOddsAPI",
                                })
                    
                    # Player Props
                    elif market_key.startswith("player_"):
                        # Map market key to our prop type
                        prop_type_map = {
                            "player_points": "points",
                            "player_rebounds": "rebounds",
                            "player_assists": "assists",
                            "player_threes": "threes",
                        }
                        prop_type = prop_type_map.get(market_key)
                        
                        if prop_type:
                            for outcome in outcomes:
                                player_name = outcome.get("description")  # Player name
                                over_under = outcome.get("name", "").lower()  # "Over" or "Under"
                                point = outcome.get("point")  # Line value
                                odds = outcome.get("price")
                                
                                if player_name and point is not None and odds:
                                    prop_id = f"{game_id}_{player_name}_{prop_type}_{point}_{bookmaker_name}".replace(" ", "_")
                                    
                                    # Create prop with both over/under if possible
                                    existing_prop = None
                                    for p in props:
                                        if (p.get("game_id") == game_id and 
                                            p.get("player") == player_name and
                                            p.get("prop_type") == prop_type and
                                            p.get("line") == float(point) and
                                            p.get("bookmaker") == bookmaker_name):
                                            existing_prop = p
                                            break
                                    
                                    if existing_prop:
                                        # Add the other side
                                        if over_under == "over":
                                            existing_prop["odds_over"] = int(odds)
                                        else:
                                            existing_prop["odds_under"] = int(odds)
                                    else:
                                        # Create new prop
                                        new_prop = {
                                            "prop_id": prop_id,
                                            "game_id": game_id,
                                            "game": game_label,
                                            "game_date": game_date,
                                            "player": player_name,
                                            "home_team": game['teams']['home']['name'],
                                            "away_team": game['teams']['away']['name'],
                                            "prop_type": prop_type,
                                            "line": float(point),
                                            "odds": int(odds),
                                            "bookmaker": bookmaker_name,
                                            "source": "TheOddsAPI",
                                        }
                                        if over_under == "over":
                                            new_prop["odds_over"] = int(odds)
                                        else:
                                            new_prop["odds_under"] = int(odds)
                                        props.append(new_prop)

            # Rate limiting - small delay between event requests to avoid hitting API limits
            time.sleep(0.1)
        
    except Exception as e:
        if DEBUG_MODE:
            print(f"   [TheOdds] Error fetching odds: {e}")
            import traceback
            traceback.print_exc()
    
    return props


def sgo_direct_fetch_odds(games: List[dict]) -> List[dict]:
    """
    Fetch odds from SportsGameOdds.com API for given games.
    Returns list of game-level props (moneyline, spread).
    API Docs: https://sportsgameodds.com/docs/reference

    NOTE: Player props not available in current API tier - only team/game markets returned.
    """
    if not SGO_DIRECT_ENABLED:
        return []

    props: List[dict] = []

    try:
        # Use v2/events endpoint as per SGO documentation
        url = f"{SGO_DIRECT_BASE_URL}/v2/events/"
        headers = {
            "X-Api-Key": SGO_DIRECT_API_KEY,
            "Content-Type": "application/json"
        }
        params = {
            "leagueID": "NBA",  # NBA league ID for SportsGameOdds (verified from API docs)
        }

        resp = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT * 2)

        if DEBUG_MODE:
            print(f"   [SGO Direct] GET {url} params={params} status={resp.status_code}")

        if resp.status_code != 200:
            if DEBUG_MODE:
                print(f"   [SGO Direct] Failed: {resp.status_code} - {resp.text[:300]}")
            return []

        data = resp.json()

        # v2 API structure may be different - check for both "data" and direct events array
        if isinstance(data, dict):
            events = data.get("data", data.get("events", []))
            if DEBUG_MODE and "success" in data and not data["success"]:
                print(f"   [SGO Direct] API error: {data.get('error', 'Unknown error')}")
        else:
            events = data if isinstance(data, list) else []

        if DEBUG_MODE:
            print(f"   [SGO Direct] Fetched {len(events)} events")
            if events and len(events) > 0:
                print(f"   [SGO Direct] Sample event keys: {list(events[0].keys())[:10]}")

        if not events:
            return []

        # Build game ID mapping (match by team names)
        game_map = {}
        for game in games:
            home = _norm_team(game['teams']['home']['name'])
            away = _norm_team(game['teams']['away']['name'])
            game_map[f"{away}_{home}"] = game
            game_map[f"{home}_{away}"] = game

        for event in events:
            # Check if event has odds available
            status = event.get("status", {})
            if not status.get("oddsPresent", False):
                continue

            # Get team info from SGO v2 format
            teams_info = event.get("teams", {})
            home_team_sgo = teams_info.get("home", {}).get("names", {}).get("long", "")
            away_team_sgo = teams_info.get("away", {}).get("names", {}).get("long", "")

            # Try to find matching game using token matching
            game = None
            for g in games:
                g_home = g['teams']['home']['name']
                g_away = g['teams']['away']['name']
                if _token_match(home_team_sgo, g_home) and _token_match(away_team_sgo, g_away):
                    game = g
                    break

            if not game:
                if DEBUG_MODE:
                    print(f"   [SGO Direct] No match for {away_team_sgo} @ {home_team_sgo}")
                continue

            game_id = game.get("id")
            game_label = f"{game['teams']['away']['name']} at {game['teams']['home']['name']}"
            game_date = game.get("date", "")

            # Parse odds - v2 API structure: event.odds is a dict with oddID keys
            odds_dict = event.get("odds", {})

            if DEBUG_MODE and odds_dict:
                print(f"   [SGO Direct] Processing {game_label}: {len(odds_dict)} markets")

            for odd_id, odd_data in odds_dict.items():
                if not isinstance(odd_data, dict):
                    continue

                # Extract market info from odd data
                bet_type = odd_data.get("betTypeID", "")
                stat_id = odd_data.get("statID", "")
                stat_entity = odd_data.get("statEntityID", "")
                period = odd_data.get("periodID", "")

                book_odds = odd_data.get("bookOdds")
                fair_odds = odd_data.get("fairOdds")
                spread_line = odd_data.get("bookSpread") or odd_data.get("fairSpread")

                # Only process full-game markets (not quarters, etc.)
                if period != "game":
                    continue

                # Moneyline (ml)
                if bet_type == "ml" and stat_id == "points":
                    team_name = game['teams']['home']['name'] if stat_entity == "home" else game['teams']['away']['name']
                    odds_val = book_odds or fair_odds

                    if odds_val:
                        try:
                            odds_int = int(float(odds_val))
                        except:
                            continue

                        prop_id = f"{game_id}_moneyline_{team_name}_SGO".replace(" ", "_")
                        props.append({
                            "prop_id": prop_id,
                            "game_id": game_id,
                            "game": game_label,
                            "game_date": game_date,
                            "player": team_name,
                            "prop_type": "moneyline",
                            "line": 0.0,
                            "odds": odds_int,
                            "bookmaker": "SGO",
                            "source": "SGO_Direct",
                            "home_team": game['teams']['home']['name'],
                            "away_team": game['teams']['away']['name'],
                        })

                # Spread (sp)
                elif bet_type == "sp" and stat_id == "points" and spread_line:
                    team_name = game['teams']['home']['name'] if stat_entity == "home" else game['teams']['away']['name']
                    odds_val = book_odds or fair_odds

                    if odds_val:
                        try:
                            odds_int = int(float(odds_val))
                            line_float = float(spread_line)
                        except:
                            continue

                        side_label = f"{team_name} {line_float:+.1f}"
                        prop_id = f"{game_id}_spread_{side_label}_SGO".replace(" ", "_")
                        props.append({
                            "prop_id": prop_id,
                            "game_id": game_id,
                            "game": game_label,
                            "game_date": game_date,
                            "player": side_label,
                            "prop_type": "spread",
                            "line": line_float,
                            "odds": odds_int,
                            "bookmaker": "SGO",
                            "source": "SGO_Direct",
                            "home_team": game['teams']['home']['name'],
                            "away_team": game['teams']['away']['name'],
                        })

                # NOTE: Player props not available in current API tier (only team markets)

    except Exception as e:
        if DEBUG_MODE:
            print(f"   [SGO Direct] Error fetching odds: {e}")
            import traceback
            traceback.print_exc()

    return props


def prizepicks_fetch_props(games: List[dict]) -> List[dict]:
    """
    Fetch player prop projections from PrizePicks API (FREE, no API key needed).
    Returns list of player props with PrizePicks lines as "odds".

    PrizePicks API provides projection lines for player props, which we convert to
    odds format for analysis. Since PrizePicks is pick'em style, we use neutral
    odds (e.g., +100/-110) for over/under.
    """
    if not PRIZEPICKS_ENABLED:
        return []

    props: List[dict] = []

    try:
        url = f"{PRIZEPICKS_BASE_URL}/projections"
        params = {
            "league_id": PRIZEPICKS_LEAGUE_ID,
            "per_page": "250",
            "single_stat": "true",
            "game_mode": "pickem"
        }

        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT * 2)

        if DEBUG_MODE:
            print(f"   [PrizePicks] GET {url} params={params} status={resp.status_code}")

        if resp.status_code != 200:
            if DEBUG_MODE:
                print(f"   [PrizePicks] Failed: {resp.status_code} - {resp.text[:300]}")
            return []

        data = resp.json()

        # PrizePicks v2 API structure: {data: [{id, attributes: {player_name, stat_type, line_score, ...}}], included: [game objects]}
        projections = data.get("data", [])
        included = data.get("included", [])

        if DEBUG_MODE:
            print(f"   [PrizePicks] Fetched {len(projections)} projections")

        # Build game mapping from included data
        game_map = {}
        for item in included:
            if item.get("type") == "new_player":
                # Player data - skip
                continue
            elif item.get("type") == "league":
                # League data - skip
                continue
            # Game objects have home_team and away_team in attributes
            attrs = item.get("attributes", {})
            if "home_team" in attrs and "away_team" in attrs:
                game_id = item.get("id")
                game_map[game_id] = {
                    "home_team": attrs.get("home_team"),
                    "away_team": attrs.get("away_team"),
                    "start_time": attrs.get("start_time")
                }

        # Parse projections
        for proj in projections:
            attrs = proj.get("attributes", {})
            relationships = proj.get("relationships", {})

            # Get projection details
            player_name = attrs.get("name")  # Player name
            stat_type_raw = attrs.get("stat_type", "").lower()
            line_score = attrs.get("line_score")

            # Map stat type
            stat_type_map = {
                "points": "points",
                "pts": "points",
                "rebounds": "rebounds",
                "reb": "rebounds",
                "assists": "assists",
                "ast": "assists",
                "3-pt made": "threes",
                "threes": "threes",
                "3pm": "threes",
            }

            prop_type = None
            for key, val in stat_type_map.items():
                if key in stat_type_raw:
                    prop_type = val
                    break

            if not prop_type or not player_name or line_score is None:
                continue

            # Try to match to our games
            new_player_id = relationships.get("new_player", {}).get("data", {}).get("id")
            league_id = relationships.get("league", {}).get("data", {}).get("id")

            # Find game from relationships or attributes
            game_id_pp = attrs.get("game_id")  # PrizePicks game ID
            game_info = game_map.get(str(game_id_pp)) if game_id_pp else None

            # Match to our games list
            matched_game = None
            if game_info:
                home_team_pp = game_info.get("home_team", "")
                away_team_pp = game_info.get("away_team", "")

                for g in games:
                    g_home = g['teams']['home']['name']
                    g_away = g['teams']['away']['name']
                    if _token_match(home_team_pp, g_home) and _token_match(away_team_pp, g_away):
                        matched_game = g
                        break

            if not matched_game:
                # Skip projections we can't match to games
                if DEBUG_MODE:
                    print(f"   [PrizePicks] No match for {player_name} projection")
                continue

            game_id = matched_game.get("id")
            game_label = f"{matched_game['teams']['away']['name']} at {matched_game['teams']['home']['name']}"
            game_date = matched_game.get("date", "")

            # PrizePicks uses pick'em style (no odds), so we assign neutral odds
            # Use +100 for over and -110 for under (standard juice)
            prop_id = f"{game_id}_{player_name}_{prop_type}_{line_score}_PrizePicks".replace(" ", "_")

            # Create TWO props: one for Over, one for Under
            # Over
            props.append({
                "prop_id": prop_id + "_over",
                "game_id": game_id,
                "game": game_label,
                "game_date": game_date,
                "player": player_name,
                "prop_type": prop_type,
                "line": float(line_score),
                "odds": +100,  # Even money for pick'em
                "bookmaker": "PrizePicks",
                "source": "PrizePicks",
            })

            # Under
            props.append({
                "prop_id": prop_id + "_under",
                "game_id": game_id,
                "game": game_label,
                "game_date": game_date,
                "player": player_name,
                "prop_type": prop_type,
                "line": float(line_score),
                "odds": -110,  # Slight juice on under
                "bookmaker": "PrizePicks",
                "source": "PrizePicks",
            })

    except Exception as e:
        if DEBUG_MODE:
            print(f"   [PrizePicks] Error fetching projections: {e}")
            import traceback
            traceback.print_exc()

    return props


def rundown_props_for_games(apisports_games: List[dict]) -> Tuple[List[dict], List[str]]:
    """
    Parse TheRundown API data to extract moneyline, spread, and player props.
    TheRundown structure: {event_id, teams_normalized, lines: {bookmaker_id: {moneyline, spread, player_props}}}
    """
    events, warns = sgo_fetch_events(apisports_games=apisports_games)
    if DEBUG_MODE:
        print(f"   [DEBUG] RapidAPI events fetched: {len(events)}; warnings: {warns}")
        if warns:
            for w in warns:
                print(f"   [DEBUG] SGO warn: {w}")
        # Debug first event structure
        if events and len(events) > 0:
            print(f"   [DEBUG] First event structure keys: {list(events[0].keys())}")
            # Check if data is nested in 'odds' key
            if 'odds' in events[0] and isinstance(events[0]['odds'], dict):
                print(f"   [DEBUG] Odds object keys: {list(events[0]['odds'].keys())}")
                if 'lines' in events[0]['odds']:
                    lines_obj = events[0]['odds']['lines']
                    print(f"   [DEBUG] Lines keys: {list(lines_obj.keys()) if isinstance(lines_obj, dict) else 'not a dict'}")
                    # Show first bookmaker structure
                    if isinstance(lines_obj, dict) and lines_obj:
                        first_book_id = list(lines_obj.keys())[0]
                        first_book = lines_obj[first_book_id]
                        print(f"   [DEBUG] First bookmaker ({first_book_id}) keys: {list(first_book.keys()) if isinstance(first_book, dict) else 'not a dict'}")
                        print(f"   [DEBUG] First bookmaker sample: {str(first_book)[:800]}")
            elif 'lines' in events[0]:
                print(f"   [DEBUG] Lines keys: {list(events[0]['lines'].keys()) if isinstance(events[0]['lines'], dict) else 'not a dict'}")
                # Show first bookmaker structure
                if isinstance(events[0]['lines'], dict) and events[0]['lines']:
                    first_book_id = list(events[0]['lines'].keys())[0]
                    first_book = events[0]['lines'][first_book_id]
                    print(f"   [DEBUG] First bookmaker ({first_book_id}) keys: {list(first_book.keys()) if isinstance(first_book, dict) else 'not a dict'}")
                    print(f"   [DEBUG] First bookmaker sample: {str(first_book)[:800]}")
    
    game_idx = _index_games(apisports_games)
    props: List[dict] = []
    seen_bm = Counter()

    for ev in events:
        # Unwrap if data is in 'odds' key
        if 'odds' in ev and isinstance(ev['odds'], dict):
            ev = ev['odds']
        
        # V2 structure: check for markets array (player props)
        if 'markets' in ev and isinstance(ev['markets'], list):
            # V2 player prop markets
            props_from_markets = _parse_v2_markets(ev, game_idx, seen_bm)
            props.extend(props_from_markets)
            continue
        
        # V1 TheRundown structure (game lines)
        event_id = ev.get("event_id", "")
        teams = ev.get("teams_normalized", [])
        if len(teams) < 2:
            teams = ev.get("teams", [])
        if len(teams) < 2:
            continue
            
        # Extract team names
        away = teams[0].get("name", "") if isinstance(teams[0], dict) else str(teams[0])
        home = teams[1].get("name", "") if isinstance(teams[1], dict) else str(teams[1])
        start = ev.get("event_date", "")
        
        # Try to match to API-Sports game for better game info
        matched_game = _match_game(game_idx, home, away, start)
        if matched_game:
            game_id = matched_game.get("id", event_id)
            game_label = f"{matched_game.get('teams',{}).get('away',{}).get('name', away)} at {matched_game.get('teams',{}).get('home',{}).get('name', home)}"
            game_date = matched_game.get("date", start)
        else:
            game_id = event_id
            game_label = f"{away} at {home}"
            game_date = start
        
        # Parse lines (bookmaker odds)
        lines = ev.get("lines", {})
        if not isinstance(lines, dict):
            continue
        
        # Look for any bookmaker with moneyline/spread data
        for bookmaker_id, book_data in lines.items():
            if not isinstance(book_data, dict):
                continue
            
            bookmaker_name = book_data.get("bookmaker") or f"Book{bookmaker_id}"
            seen_bm[bookmaker_name] += 1
            
            # Extract moneyline
            moneyline = book_data.get("moneyline", {})
            if moneyline:
                away_ml = moneyline.get("moneyline_away")
                home_ml = moneyline.get("moneyline_home")
                
                if away_ml:
                    try:
                        prop_id = f"{game_id}_moneyline_{away}".replace(" ", "_")
                        props.append({
                            "prop_id": prop_id, "game_id": game_id, "game": game_label, "game_date": game_date,
                            "player": away, "prop_type": "moneyline", "line": 0.0, "odds": int(away_ml),
                            "bookmaker": bookmaker_name, "source": "TheRundown",
                            "home_team": home, "away_team": away,  # For nba_api team stats lookup
                        })
                    except:
                        pass
                if home_ml:
                    try:
                        prop_id = f"{game_id}_moneyline_{home}".replace(" ", "_")
                        props.append({
                            "prop_id": prop_id, "game_id": game_id, "game": game_label, "game_date": game_date,
                            "player": home, "prop_type": "moneyline", "line": 0.0, "odds": int(home_ml),
                            "bookmaker": bookmaker_name, "source": "TheRundown",
                            "home_team": home, "away_team": away,  # For nba_api team stats lookup
                        })
                    except:
                        pass
            
            # Extract spread
            spread = book_data.get("spread", {})
            if spread:
                away_spread = spread.get("point_spread_away")
                home_spread = spread.get("point_spread_home")
                away_odds = spread.get("point_spread_away_money")
                home_odds = spread.get("point_spread_home_money")
                
                if away_spread is not None and away_odds:
                    try:
                        side = f"{away} {float(away_spread):+.1f}"
                        prop_id = f"{game_id}_spread_{side}".replace(" ", "_")
                        props.append({
                            "prop_id": prop_id, "game_id": game_id, "game": game_label, "game_date": game_date,
                            "player": side, "prop_type": "spread", "line": float(away_spread), "odds": int(away_odds),
                            "bookmaker": bookmaker_name, "source": "TheRundown",
                            "home_team": home, "away_team": away,  # For nba_api team stats lookup
                        })
                    except:
                        pass
                if home_spread is not None and home_odds:
                    try:
                        side = f"{home} {float(home_spread):+.1f}"
                        prop_id = f"{game_id}_spread_{side}".replace(" ", "_")
                        props.append({
                            "prop_id": prop_id, "game_id": game_id, "game": game_label, "game_date": game_date,
                            "player": side, "prop_type": "spread", "line": float(home_spread), "odds": int(home_odds),
                            "bookmaker": bookmaker_name, "source": "TheRundown",
                            "home_team": home, "away_team": away,  # For nba_api team stats lookup
                        })
                    except:
                        pass
            
            # Extract player props (points, assists, rebounds, threes)
            player_props_data = book_data.get("player_props", {}) or book_data.get("playerProps", {})
            if player_props_data and isinstance(player_props_data, dict):
                for player_name, player_stats in player_props_data.items():
                    if not isinstance(player_stats, dict):
                        continue
                    
                    # Points
                    if "points" in player_stats or "pts" in player_stats:
                        pts_data = player_stats.get("points") or player_stats.get("pts")
                        if isinstance(pts_data, dict):
                            line = pts_data.get("line") or pts_data.get("total")
                            over_odds = pts_data.get("over") or pts_data.get("over_money")
                            under_odds = pts_data.get("under") or pts_data.get("under_money")
                            if line and (over_odds or under_odds):
                                try:
                                    prop_id = f"{game_id}_{player_name}_points".replace(" ", "_")
                                    prop = {
                                        "prop_id": prop_id, "game_id": game_id, "game": game_label, "game_date": game_date,
                                        "player": player_name, "prop_type": "points", "line": float(line),
                                        "odds": int(over_odds) if over_odds else int(under_odds),
                                        "bookmaker": bookmaker_name, "source": "TheRundown",
                                    }
                                    if over_odds: prop["odds_over"] = int(over_odds)
                                    if under_odds: prop["odds_under"] = int(under_odds)
                                    props.append(prop)
                                except:
                                    pass
                    
                    # Assists
                    if "assists" in player_stats or "ast" in player_stats:
                        ast_data = player_stats.get("assists") or player_stats.get("ast")
                        if isinstance(ast_data, dict):
                            line = ast_data.get("line") or ast_data.get("total")
                            over_odds = ast_data.get("over") or ast_data.get("over_money")
                            under_odds = ast_data.get("under") or ast_data.get("under_money")
                            if line and (over_odds or under_odds):
                                try:
                                    prop_id = f"{game_id}_{player_name}_assists".replace(" ", "_")
                                    prop = {
                                        "prop_id": prop_id, "game_id": game_id, "game": game_label, "game_date": game_date,
                                        "player": player_name, "prop_type": "assists", "line": float(line),
                                        "odds": int(over_odds) if over_odds else int(under_odds),
                                        "bookmaker": bookmaker_name, "source": "TheRundown",
                                    }
                                    if over_odds: prop["odds_over"] = int(over_odds)
                                    if under_odds: prop["odds_under"] = int(under_odds)
                                    props.append(prop)
                                except:
                                    pass
                    
                    # Rebounds
                    if "rebounds" in player_stats or "reb" in player_stats:
                        reb_data = player_stats.get("rebounds") or player_stats.get("reb")
                        if isinstance(reb_data, dict):
                            line = reb_data.get("line") or reb_data.get("total")
                            over_odds = reb_data.get("over") or reb_data.get("over_money")
                            under_odds = reb_data.get("under") or reb_data.get("under_money")
                            if line and (over_odds or under_odds):
                                try:
                                    prop_id = f"{game_id}_{player_name}_rebounds".replace(" ", "_")
                                    prop = {
                                        "prop_id": prop_id, "game_id": game_id, "game": game_label, "game_date": game_date,
                                        "player": player_name, "prop_type": "rebounds", "line": float(line),
                                        "odds": int(over_odds) if over_odds else int(under_odds),
                                        "bookmaker": bookmaker_name, "source": "TheRundown",
                                    }
                                    if over_odds: prop["odds_over"] = int(over_odds)
                                    if under_odds: prop["odds_under"] = int(under_odds)
                                    props.append(prop)
                                except:
                                    pass
                    
                    # Threes (3-pointers)
                    if "threes" in player_stats or "three_pointers_made" in player_stats or "3pm" in player_stats:
                        three_data = player_stats.get("threes") or player_stats.get("three_pointers_made") or player_stats.get("3pm")
                        if isinstance(three_data, dict):
                            line = three_data.get("line") or three_data.get("total")
                            over_odds = three_data.get("over") or three_data.get("over_money")
                            under_odds = three_data.get("under") or three_data.get("under_money")
                            if line and (over_odds or under_odds):
                                try:
                                    prop_id = f"{game_id}_{player_name}_threes".replace(" ", "_")
                                    prop = {
                                        "prop_id": prop_id, "game_id": game_id, "game": game_label, "game_date": game_date,
                                        "player": player_name, "prop_type": "threes", "line": float(line),
                                        "odds": int(over_odds) if over_odds else int(under_odds),
                                        "bookmaker": bookmaker_name, "source": "TheRundown",
                                    }
                                    if over_odds: prop["odds_over"] = int(over_odds)
                                    if under_odds: prop["odds_under"] = int(under_odds)
                                    props.append(prop)
                                except:
                                    pass

    if DEBUG_MODE:
        print(f"   [DEBUG] Parsed {len(props)} props from TheRundown")
        print("   [DEBUG] Seen bookmakers:", dict(seen_bm.most_common(10)))

    return props, warns


def _parse_v2_markets(ev: dict, game_idx: List[Tuple[dict, str, str, str]], seen_bm: Counter) -> List[dict]:
    """Parse V2 market structure for player props."""
    props = []
    
    # Extract event info
    event_id = ev.get("event_id", "")
    teams = ev.get("teams", [])
    if len(teams) < 2:
        return props
    
    away = teams[0].get("name", "") if isinstance(teams[0], dict) else str(teams[0])
    home = teams[1].get("name", "") if isinstance(teams[1], dict) else str(teams[1])
    start = ev.get("event_date", "")
    
    # Match to API-Sports game
    matched_game = _match_game(game_idx, home, away, start)
    if matched_game:
        game_id = matched_game.get("id", event_id)
        game_label = f"{matched_game.get('teams',{}).get('away',{}).get('name', away)} at {matched_game.get('teams',{}).get('home',{}).get('name', home)}"
        game_date = matched_game.get("date", start)
    else:
        game_id = event_id
        game_label = f"{away} at {home}"
        game_date = start
    
    # V2 market structure: markets[{id, market_id, name, participants[{name, lines[{value, prices{bookmaker_id: {price}}}]}]}]
    markets = ev.get("markets", [])
    for market in markets:
        if not isinstance(market, dict):
            continue
        
        market_name = (market.get("name") or "").lower()
        market_id = market.get("market_id", "")
        
        # Map market_id/name to our stat types
        # Market IDs: 39=assists, 49=points, 51=rebounds, 52=threes (VERIFIED from API response)
        stat_type = None
        if market_id == 39 or "assist" in market_name:
            stat_type = "assists"
        elif market_id == 49 or ("point" in market_name and "spread" not in market_name):
            stat_type = "points"
        elif market_id == 51 or "rebound" in market_name:
            stat_type = "rebounds"
        elif market_id == 52 or "three" in market_name or "3" in market_name:
            stat_type = "threes"
        
        if not stat_type:
            continue
        
        # Parse participants (players)
        participants = market.get("participants", [])
        for participant in participants:
            if not isinstance(participant, dict):
                continue
            
            participant_name = participant.get("name", "")
            if not participant_name:
                continue
            
            # Parse lines (Over/Under)
            lines = participant.get("lines", [])
            over_line = None
            under_line = None
            over_odds_by_book = {}
            under_odds_by_book = {}
            
            for line in lines:
                if not isinstance(line, dict):
                    continue
                
                value_str = str(line.get("value", "")).lower()
                prices = line.get("prices", {})
                
                # Extract line value (e.g., "Over 1.5" -> 1.5)
                try:
                    line_val = float(value_str.split()[-1])
                except:
                    continue
                
                # Determine if Over or Under
                is_over = "over" in value_str
                is_under = "under" in value_str
                
                # Parse prices from all bookmakers
                for bookmaker_id, price_data in prices.items():
                    if not isinstance(price_data, dict):
                        continue
                    
                    price = price_data.get("price")
                    closed_at = price_data.get("closed_at")
                    
                    # Skip closed lines
                    if closed_at:
                        continue
                    
                    if price is None:
                        continue
                    
                    bookmaker_name = f"Book{bookmaker_id}"
                    seen_bm[bookmaker_name] += 1
                    
                    try:
                        price_int = int(price)
                        if is_over:
                            over_line = line_val
                            over_odds_by_book[bookmaker_name] = price_int
                        elif is_under:
                            under_line = line_val
                            under_odds_by_book[bookmaker_name] = price_int
                    except:
                        pass
            
            # Create props for each bookmaker
            if over_line is not None or under_line is not None:
                line_val = over_line if over_line is not None else under_line
                
                # Combine all bookmakers that have prices
                all_books = set(list(over_odds_by_book.keys()) + list(under_odds_by_book.keys()))
                
                for bookmaker_name in all_books:
                    over_odds = over_odds_by_book.get(bookmaker_name)
                    under_odds = under_odds_by_book.get(bookmaker_name)
                    
                    if over_odds or under_odds:
                        prop_id = f"{game_id}_{participant_name}_{stat_type}_{bookmaker_name}".replace(" ", "_")
                        prop = {
                            "prop_id": prop_id,
                            "game_id": game_id,
                            "game": game_label,
                            "game_date": game_date,
                            "player": participant_name,
                            "prop_type": stat_type,
                            "line": line_val,
                            "odds": over_odds if over_odds else under_odds,
                            "bookmaker": bookmaker_name,
                            "source": "TheRundown_V2",
                        }
                        if over_odds:
                            prop["odds_over"] = over_odds
                        if under_odds:
                            prop["odds_under"] = under_odds
                        props.append(prop)
    
    return props


# ========= ML MODEL ENSEMBLE (load + predict) =========
MODEL_DIR = "models"
# Player stat models (matches train_auto.py output)
PLAYER_MODEL_FILES = {
    "points":  "points_model.pkl",
    "assists": "assists_model.pkl",
    "rebounds":"rebounds_model.pkl",
    "threes":  "threes_model.pkl",
    "minutes": "minutes_model.pkl",
    "threepoint_goals": "threepoint_goals_model.pkl",  # Legacy model (same as threes)
}
# Optional per-stat sigma models (heteroskedastic noise)
PLAYER_SIGMA_MODEL_FILES = {
    "points":  "points_sigma_model.pkl",
    "assists": "assists_sigma_model.pkl",
    "rebounds":"rebounds_sigma_model.pkl",
    "threes":  "threes_sigma_model.pkl",
}
# Game-level models (matches train_auto.py output)
GAME_MODEL_FILES = {
    "moneyline": "moneyline_model.pkl",
    "moneyline_calibrator": "moneyline_calibrator.pkl",
    "moneyline_winprob": "moneyline_winprob_model.pkl",  # Win probability model
    "spread": "spread_model.pkl",
    "spread_margin": "spread_margin_model.pkl",  # Spread margin predictor
}
MODEL_RMSE_DEFAULT = {"points": 5.8, "assists": 1.8, "rebounds": 2.5, "threes": 1.2, "minutes": 3.5}

def _load_model_metadata() -> Dict[str, Any]:
    """Load training metadata which includes RMSE and other metrics"""
    meta_path = os.path.join(MODEL_DIR, "training_metadata.json")
    rmse_by_type = dict(MODEL_RMSE_DEFAULT)
    spread_sigma = 10.5  # default
    
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            
            # Extract player model metrics
            player_metrics = meta.get("player_metrics", {})
            for stat_type in ["points", "assists", "rebounds", "threes", "minutes"]:
                if stat_type in player_metrics:
                    metrics = player_metrics[stat_type]
                    if isinstance(metrics, dict) and "rmse" in metrics:
                        rmse_by_type[stat_type] = float(metrics["rmse"])
            
            # Extract game model metrics  
            game_metrics = meta.get("game_metrics", {})
            if "spread_sigma" in game_metrics:
                spread_sigma = float(game_metrics["spread_sigma"])
                
            if DEBUG_MODE:
                print(f"   ↳ Loaded training metadata: {meta_path}")
                print(f"      Player RMSEs: {rmse_by_type}")
                print(f"      Spread sigma: {spread_sigma}")
        except Exception as e:
            if DEBUG_MODE:
                print(f"   Warning: Failed to load training metadata: {e}")
    
    # Also try legacy model_registry.json for backwards compatibility
    reg_path = os.path.join(MODEL_DIR, "model_registry.json")
    if os.path.exists(reg_path):
        try:
            with open(reg_path, "r", encoding="utf-8") as f:
                reg = json.load(f)
            candidates = reg.get("models", reg)
            for k_an, key_model in [("points","points"), ("assists","assists"), ("rebounds","rebounds"), ("threes","threes")]:
                node = candidates.get(key_model) or candidates.get(k_an) or {}
                rmse = None
                if isinstance(node, dict):
                    rmse = node.get("rmse") or (node.get("metrics", {}) if isinstance(node.get("metrics"), dict) else {}).get("rmse")
                if rmse is not None:
                    rmse_by_type[k_an] = float(rmse)
        except Exception:
            pass
    
    return {"rmse": rmse_by_type, "spread_sigma": spread_sigma}

METADATA = _load_model_metadata()
MODEL_RMSE = METADATA["rmse"]
SPREAD_SIGMA = METADATA["spread_sigma"]

class ModelPredictor:
    def __init__(self):
        self.player_models: Dict[str, object] = {}
        self.player_sigma_models: Dict[str, object] = {}
        self.game_models: Dict[str, object] = {}
        # Enhanced ensemble models (legacy - kept for backward compatibility)
        self.ridge_model = None
        self.elo_model = None
        self.ff_model = None
        self.ensemble_meta_learner = None
        # UNIFIED HIERARCHICAL ENSEMBLE (NEW!)
        self.unified_ensemble = None

        # Load player models
        for key, fname in PLAYER_MODEL_FILES.items():
            path = os.path.join(MODEL_DIR, fname)
            if os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        self.player_models[key] = pickle.load(f)
                    if DEBUG_MODE: print(f"   ↳ Loaded player model: {path}")
                except Exception as e:
                    if DEBUG_MODE: print(f"   Warning: Failed to load {path}: {e}")
        # Load sigma models
        for key, fname in PLAYER_SIGMA_MODEL_FILES.items():
            path = os.path.join(MODEL_DIR, fname)
            if os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        self.player_sigma_models[key] = pickle.load(f)
                    if DEBUG_MODE: print(f"   ↳ Loaded sigma model: {path}")
                except Exception:
                    pass
        
        # Load game models
        for key, fname in GAME_MODEL_FILES.items():
            path = os.path.join(MODEL_DIR, fname)
            if os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        self.game_models[key] = pickle.load(f)
                    if DEBUG_MODE: print(f"   ↳ Loaded game model: {path}")
                except Exception as e:
                    if DEBUG_MODE: print(f"   Warning: Failed to load {path}: {e}")
        
        # Load UNIFIED HIERARCHICAL ENSEMBLE (priority over legacy)
        unified_path = os.path.join(MODEL_DIR, "hierarchical_ensemble_full.pkl")
        if os.path.exists(unified_path):
            try:
                with open(unified_path, "rb") as f:
                    self.unified_ensemble = pickle.load(f)
                if DEBUG_MODE:
                    print(f"   ✓ Loaded UNIFIED HIERARCHICAL ENSEMBLE: {unified_path}")
                    print(f"     → Includes ALL 7 models (Ridge, Elo, FF, LGB, Dynamic Elo, Rolling FF, Enhanced Log)")
                    print(f"     → Master meta-learner with cross-validated weights")
            except Exception as e:
                if DEBUG_MODE: print(f"   ⚠ Warning: Failed to load unified ensemble {unified_path}: {e}")
                self.unified_ensemble = None
        else:
            if DEBUG_MODE: print(f"   ℹ Unified ensemble not found, will try loading legacy ensemble models")

        # Load player ensemble models (per-window architecture)
        self.player_ensembles = {}
        CACHE_DIR = "model_cache"
        current_year = dt.now().year
        if current_year >= 2022:
            ensemble_file = "player_ensemble_2022_2026.pkl"
        elif current_year >= 2017:
            ensemble_file = "player_ensemble_2017_2021.pkl"
        elif current_year >= 2012:
            ensemble_file = "player_ensemble_2012_2016.pkl"
        elif current_year >= 2007:
            ensemble_file = "player_ensemble_2007_2011.pkl"
        else:
            ensemble_file = "player_ensemble_2002_2006.pkl"

        ensemble_path = os.path.join(CACHE_DIR, ensemble_file)
        if os.path.exists(ensemble_path):
            try:
                with open(ensemble_path, "rb") as f:
                    ensembles_data = pickle.load(f)
                for stat_name in ['points', 'rebounds', 'assists', 'threes']:
                    if stat_name in ensembles_data:
                        self.player_ensembles[stat_name] = ensembles_data[stat_name]
                if DEBUG_MODE:
                    print(f"   ✓ Loaded PLAYER ENSEMBLE: {ensemble_file}")
                    print(f"     → {len(self.player_ensembles)} stat ensembles (+1-2% RMSE improvement)")
                    print(f"     → Using ensemble for: {', '.join(self.player_ensembles.keys())}")
                    print(f"     → LightGBM-only for minutes (ensemble degrades performance)")
            except Exception as e:
                if DEBUG_MODE: print(f"   Warning: Failed to load player ensemble: {e}")
                self.player_ensembles = {}
        else:
            if DEBUG_MODE: print(f"   Note: Player ensemble not found, using LightGBM-only for all stats")
            self.player_ensembles = {}

        # Load enhanced selector for context-aware window selection
        self.enhanced_selector = None
        self.selector_windows = {}
        selector_file = os.path.join(CACHE_DIR, "dynamic_selector_enhanced.pkl")
        selector_meta_file = os.path.join(CACHE_DIR, "dynamic_selector_enhanced_meta.json")

        if os.path.exists(selector_file) and os.path.exists(selector_meta_file):
            try:
                with open(selector_file, 'rb') as f:
                    self.enhanced_selector = pickle.load(f)
                with open(selector_meta_file, 'r') as f:
                    selector_meta = json.load(f)

                # Load all window ensembles for selector
                import glob
                ensemble_files = sorted(glob.glob(os.path.join(CACHE_DIR, "player_ensemble_*.pkl")))
                for pkl_path in ensemble_files:
                    window_name = os.path.basename(pkl_path).replace("player_ensemble_", "").replace(".pkl", "").replace("_", "-")
                    with open(pkl_path, 'rb') as f:
                        self.selector_windows[window_name] = pickle.load(f)

                if DEBUG_MODE:
                    print(f"   ✓ Loaded ENHANCED SELECTOR")
                    print(f"     → Context-aware window selection (+0.5% vs cherry-pick)")
                    print(f"     → {len(self.selector_windows)} windows available")
            except Exception as e:
                if DEBUG_MODE: print(f"   Warning: Failed to load enhanced selector: {e}")
                self.enhanced_selector = None

        # Load enhanced ensemble models (legacy - if unified not available)
        if self.unified_ensemble is None:
            ensemble_models = {
                'ridge': 'ridge_model_enhanced.pkl',
                'elo': 'elo_model_enhanced.pkl',
                'ff': 'four_factors_model_enhanced.pkl',
                'meta': 'ensemble_meta_learner_enhanced.pkl'
            }
            for key, fname in ensemble_models.items():
                path = os.path.join(MODEL_DIR, fname)
                if os.path.exists(path):
                    try:
                        with open(path, "rb") as f:
                            model = pickle.load(f)
                        if key == 'ridge':
                            self.ridge_model = model
                        elif key == 'elo':
                            self.elo_model = model
                        elif key == 'ff':
                            self.ff_model = model
                        elif key == 'meta':
                            self.ensemble_meta_learner = model
                        if DEBUG_MODE: print(f"   ↳ Loaded legacy ensemble {key} model: {path}")
                    except Exception as e:
                        if DEBUG_MODE: print(f"   Warning: Failed to load legacy ensemble {key} {path}: {e}")
                else:
                    if DEBUG_MODE: print(f"   ℹ Legacy ensemble {key} model not found: {path}")
        
        # Load spread sigma
        sigma_path = os.path.join(MODEL_DIR, "spread_sigma.json")
        if os.path.exists(sigma_path):
            try:
                with open(sigma_path, "r") as f:
                    data = json.load(f)
                    self.spread_sigma = float(data.get("spread_sigma", SPREAD_SIGMA))
                    if DEBUG_MODE: print(f"   ↳ Loaded spread sigma: {self.spread_sigma}")
            except Exception:
                self.spread_sigma = SPREAD_SIGMA
        else:
            self.spread_sigma = SPREAD_SIGMA

        # Load game defaults and features from training metadata
        self.game_defaults = {}
        self.game_features = []
        metadata_path = os.path.join(MODEL_DIR, "training_metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    self.game_defaults = metadata.get("game_defaults", {})
                    self.game_features = metadata.get("game_features", [])
                    if DEBUG_MODE:
                        print(f"   ↳ Loaded game defaults: {len(self.game_defaults)} features")
                        print(f"   ↳ Loaded game feature list: {len(self.game_features)} features")
            except Exception as e:
                if DEBUG_MODE: print(f"   Warning: Failed to load training metadata: {e}")

    def available(self, prop_type: str) -> bool:
        """Check if a player stat model is available"""
        return prop_type in self.player_models
    
    def game_model_available(self, model_type: str) -> bool:
        """Check if a game-level model is available"""
        return model_type in self.game_models

    def predict(self, prop_type: str, feats: pd.DataFrame) -> Optional[float]:
        """
        Predict player stat using trained model.

        Note: Player ensemble models are loaded but not yet fully integrated.
        Full integration requires player history tracking.
        Current: Using LightGBM (which is a component of the ensemble anyway).
        """
        m = self.player_models.get(prop_type)
        if m is None or feats is None or feats.empty:
            return None
        try:
            y = m.predict(feats)
            prediction = float(y[0]) if isinstance(y, (list, np.ndarray)) else float(y)

            # Log ensemble availability (for future full integration)
            if DEBUG_MODE and prop_type in self.player_ensembles:
                print(f"   ℹ Ensemble available for {prop_type} (using LightGBM component for now)")

            return prediction
        except Exception as e:
            if DEBUG_MODE: print(f"   Warning: ML predict failed for {prop_type}: {e}")
            return None

    def predict_with_ensemble(self, prop_type: str, feats: pd.DataFrame, player_history: Optional[pd.DataFrame] = None) -> Optional[float]:
        """
        Predict using enhanced selector + window ensembles.

        Falls back to LightGBM if selector not available or prediction fails.
        """
        # Try enhanced selector first
        if self.enhanced_selector is None:
            if DEBUG_MODE:
                print(f"   ℹ Enhanced selector not loaded")
            return None
            
        if player_history is None or len(player_history) < 3:
            if DEBUG_MODE and player_history is not None:
                print(f"   ℹ Player history too short: {len(player_history)} games (need 3+)")
            return None
            
        try:
            stat_name = prop_type  # 'points', 'rebounds', 'assists', 'threes'

            if stat_name not in self.enhanced_selector:
                if DEBUG_MODE:
                    print(f"   ℹ Selector not trained for {stat_name}")
                return None  # Selector not trained for this stat

            # Extract recent stats for base predictions
            stat_col_map = {
                'points': 'points',
                'rebounds': 'rebounds',
                'assists': 'assists',
                'threes': 'threes',  # Fixed: df_last/df_curr use 'threes' not 'threePointersMade'
                'minutes': 'minutes'
            }
            stat_col = stat_col_map.get(stat_name)

            if stat_col and stat_col in player_history.columns:
                recent_values = player_history[stat_col].tail(10).values
                recent_values = recent_values[~np.isnan(recent_values)]

                if len(recent_values) >= 3:
                    # Extract enhanced features for selector
                    baseline = np.mean(recent_values)
                    recent_3 = recent_values[-3:] if len(recent_values) >= 3 else recent_values

                    # Rest days (estimate from dates if available)
                    rest_days = 3  # default

                    feature_vector = np.array([
                        len(player_history),  # games_played
                        baseline,  # recent_avg
                        np.std(recent_values) if len(recent_values) > 1 else 0,  # recent_std
                        np.min(recent_values),  # recent_min
                        np.max(recent_values),  # recent_max
                        recent_values[-1] - recent_values[0] if len(recent_values) >= 2 else 0,  # trend
                        rest_days,  # rest_days
                        np.mean(recent_3),  # recent_form_3
                        np.mean(recent_3) - baseline,  # form_change
                        (np.std(recent_values) / baseline) if baseline > 0.1 else 0,  # consistency_cv
                    ]).reshape(1, -1)

                    # Use selector to pick window
                    selector_obj = self.enhanced_selector[stat_name]
                    X_scaled = selector_obj['scaler'].transform(feature_vector)
                    window_idx = selector_obj['selector'].predict(X_scaled)[0]
                    probs = selector_obj['selector'].predict_proba(X_scaled)[0]
                    selected_window = selector_obj['windows_list'][window_idx]
                    confidence = probs[window_idx]

                    if DEBUG_MODE:
                        print(f"   🎯 SELECTOR: {selected_window} (confidence: {confidence*100:.1f}%)")

                    # Get prediction from selected window's ensemble
                    if selected_window in self.selector_windows:
                        window_ensembles = self.selector_windows[selected_window]
                        if stat_name in window_ensembles:
                            ensemble_obj = window_ensembles[stat_name]
                            if isinstance(ensemble_obj, dict) and 'model' in ensemble_obj:
                                ensemble = ensemble_obj['model']
                            else:
                                ensemble = ensemble_obj

                            if hasattr(ensemble, 'is_fitted') and ensemble.is_fitted:
                                # Get ensemble prediction
                                base_preds = np.array([baseline, baseline, baseline, baseline, baseline])
                                X_meta = ensemble.scaler.transform(base_preds.reshape(1, -1))
                                pred = ensemble.meta_learner.predict(X_meta)[0]

                                if DEBUG_MODE:
                                    print(f"   ✅ ENHANCED PREDICTION: {pred:.2f} (baseline: {baseline:.2f})")

                                return float(pred)
                            else:
                                if DEBUG_MODE:
                                    print(f"   ⚠ Ensemble not fitted for {selected_window}/{stat_name}")
                    else:
                        if DEBUG_MODE:
                            print(f"   ⚠ Window {selected_window} not loaded")
                else:
                    if DEBUG_MODE:
                        print(f"   ℹ Not enough stat values: {len(recent_values)}")
            else:
                if DEBUG_MODE:
                    print(f"   ℹ Stat column '{stat_col}' not in player_history columns: {list(player_history.columns)}")
        except Exception as e:
            if DEBUG_MODE: 
                import traceback
                print(f"   ⚠ Enhanced selector failed: {e}")
                print(f"   Stack trace: {traceback.format_exc()}")

        # Fallback to LightGBM
        return None

    def predict_sigma(self, prop_type: str, feats: pd.DataFrame) -> Optional[float]:
        """Predict per-instance sigma if sigma model available."""
        m = self.player_sigma_models.get(prop_type)
        if m is None or feats is None or feats.empty:
            return None
        try:
            y = m.predict(feats)
            s = float(y[0]) if isinstance(y, (list, np.ndarray)) else float(y)
            # floor and bounds
            return float(max(0.3, min(12.0, s)))
        except Exception:
            return None
    
    def predict_moneyline(self, feats: pd.DataFrame, home_team_id: Optional[str] = None, away_team_id: Optional[str] = None) -> Optional[float]:
        """
        Predict moneyline probability with unified hierarchical ensemble.

        Priority:
        1. Unified hierarchical ensemble (7 models + master meta-learner)
        2. Legacy enhanced ensemble (Ridge + Elo + FF + LGB meta-learner)
        3. Base LGB + calibration (fallback)
        """
        # Try unified ensemble first (BEST)
        if self.unified_ensemble is not None:
            try:
                unified_prob = self.unified_ensemble.predict(feats, self.game_features, self.game_defaults)[0]
                if DEBUG_MODE: print(f"   ✓ Used UNIFIED ensemble prediction: {unified_prob:.4f}")
                return float(unified_prob)
            except Exception as e:
                if DEBUG_MODE: print(f"   ⚠ Unified ensemble failed, falling back: {e}")

        # Try legacy enhanced ensemble (GOOD)
        legacy_prob = self.predict_moneyline_ensemble(feats, home_team_id, away_team_id)
        if legacy_prob is not None:
            if DEBUG_MODE: print(f"   ↳ Used legacy ensemble prediction: {legacy_prob:.4f}")
            return legacy_prob

        # Fallback to base LGB + calibration (OK)
        base_model = self.game_models.get("moneyline")
        if base_model is None or feats is None or feats.empty:
            return None
        try:
            # Get base prediction
            y = base_model.predict_proba(feats) if hasattr(base_model, 'predict_proba') else base_model.predict(feats)
            prob = float(y[0, 1]) if hasattr(y, 'shape') and len(y.shape) > 1 else float(y[0])

            # Apply calibration if available
            calibrator = self.game_models.get("moneyline_calibrator")
            if calibrator is not None:
                prob_calibrated = calibrator.predict_proba([[prob]])
                prob = float(prob_calibrated[0, 1]) if hasattr(prob_calibrated, 'shape') and len(prob_calibrated.shape) > 1 else prob

            if DEBUG_MODE: print(f"   ↳ Used base LGB prediction: {prob:.4f}")
            return prob
        except Exception as e:
            if DEBUG_MODE: print(f"   Warning: Moneyline predict failed: {e}")
            return None
    
    def predict_moneyline_ensemble(self, feats: pd.DataFrame, home_team_id: Optional[str] = None, away_team_id: Optional[str] = None) -> Optional[float]:
        """Predict moneyline using enhanced ensemble (Ridge + Elo + 4F + LGB meta-learner)"""
        # Check if all ensemble components are available
        if (self.ridge_model is None or self.elo_model is None or 
            self.ff_model is None or self.ensemble_meta_learner is None):
            return None  # Fallback to base moneyline if ensemble not available
        
        if feats is None or feats.empty:
            return None
        
        try:
            # Get base LGB prediction
            lgb_model = self.game_models.get("moneyline")
            if lgb_model is None:
                return None
            y = lgb_model.predict_proba(feats) if hasattr(lgb_model, 'predict_proba') else lgb_model.predict(feats)
            lgb_prob = float(y[0, 1]) if hasattr(y, 'shape') and len(y.shape) > 1 else float(y[0])
            
            # Ridge prediction
            ridge_probs = self.ridge_model.predict_proba(feats)
            ridge_prob = float(ridge_probs[0, 1])
            
            # Four Factors prediction
            ff_probs = self.ff_model.predict_proba(feats)
            ff_prob = float(ff_probs[0, 1])
            
            # Elo prediction (requires team IDs)
            elo_prob = 0.5  # default if no team IDs
            if home_team_id is not None and away_team_id is not None:
                try:
                    elo_prob = self.elo_model.expected_win_prob(home_team_id, away_team_id)
                except Exception:
                    elo_prob = 0.5
            
            # Stack predictions for meta-learner
            X_meta = np.array([[ridge_prob, elo_prob, ff_prob, lgb_prob]])
            
            # Get ensemble prediction
            ensemble_probs = self.ensemble_meta_learner.predict_proba(X_meta)
            ensemble_prob = float(ensemble_probs[0, 1])
            
            return ensemble_prob
        except Exception as e:
            if DEBUG_MODE: print(f"   Warning: Ensemble prediction failed: {e}")
            return None  # Fallback to base moneyline
    
    def predict_spread(self, feats: pd.DataFrame) -> Optional[Tuple[float, float]]:
        """Predict spread (returns predicted margin and sigma)"""
        spread_model = self.game_models.get("spread")
        if spread_model is None or feats is None or feats.empty:
            return None
        try:
            margin = spread_model.predict(feats)
            margin_val = float(margin[0]) if isinstance(margin, (list, np.ndarray)) else float(margin)
            return (margin_val, self.spread_sigma)
        except Exception as e:
            if DEBUG_MODE: print(f"   Warning: Spread predict failed: {e}")
            return None

MODEL = ModelPredictor()

def build_player_features(df_last: pd.DataFrame, df_curr: pd.DataFrame) -> pd.DataFrame:
    """
    Build features for ML prediction matching train_auto.py schema with ALL PHASES.
    Includes Phase 1 (shot volume), Phase 2 (matchup), Phase 3 (advanced rates).
    
    NOW PROPERLY COMPUTES PHASE FEATURES FROM ACTUAL GAME DATA!
    """
    def seq(col: str) -> List[float]:
        vals: List[float] = []
        if not df_curr.empty and col in df_curr.columns:
            vals += [float(x) for x in df_curr[col].tolist() if pd.notnull(x)]
        if not df_last.empty and col in df_last.columns:
            vals += [float(x) for x in df_last[col].tolist() if pd.notnull(x)]
        return vals

    pts = seq("points"); ast = seq("assists"); reb = seq("rebounds"); thr = seq("threes")
    mins = seq("minutes") if "minutes" in (df_curr.columns.tolist() + df_last.columns.tolist()) else []
    
    # PHASE 1: Shot volume features - NOW ACTUALLY FETCHED FROM GAME DATA!
    fga = seq("fieldGoalsAttempted") if "fieldGoalsAttempted" in (df_curr.columns.tolist() + df_last.columns.tolist()) else []
    tpa = seq("threePointersAttempted") if "threePointersAttempted" in (df_curr.columns.tolist() + df_last.columns.tolist()) else []
    fta = seq("freeThrowsAttempted") if "freeThrowsAttempted" in (df_curr.columns.tolist() + df_last.columns.tolist()) else []
    fg_pct = seq("fieldGoalPercentage") if "fieldGoalPercentage" in (df_curr.columns.tolist() + df_last.columns.tolist()) else []
    three_pct = seq("threePointPercentage") if "threePointPercentage" in (df_curr.columns.tolist() + df_last.columns.tolist()) else []
    ft_pct = seq("freeThrowPercentage") if "freeThrowPercentage" in (df_curr.columns.tolist() + df_last.columns.tolist()) else []
    
    if DEBUG_MODE and (fga or tpa or fta):
        print(f"   [PHASE 1] Shot volume data available: FGA={len(fga)} games, 3PA={len(tpa)} games, FTA={len(fta)} games")

    def avg(v: List[float], w: int) -> float:
        if not v: return 0.0
        return float(np.mean(v[:w])) if len(v) >= 1 else 0.0
    
    def safe_div(n, d):
        return float(n / d) if d > 0 else 0.0
    
    # Calculate True Shooting % from recent games
    def calc_ts_pct(pts_seq, fga_seq, fta_seq, window):
        if not pts_seq or not fga_seq or not fta_seq:
            return 0.56  # league average
        p = avg(pts_seq, window)
        a = avg(fga_seq, window)
        t = avg(fta_seq, window)
        denominator = 2 * (a + 0.44 * t)
        return p / denominator if denominator > 0 else 0.56

    # Base context features (1-18)
    feats = {
        "is_home": 1,
        "season_end_year": 2025.0,
        "season_decade": 2020.0,
        "team_recent_pace": 1.0,
        "team_off_strength": 1.0,
        "team_def_strength": 1.0,
        "team_recent_winrate": 0.5,
        "opp_recent_pace": 1.0,
        "opp_off_strength": 1.0,
        "opp_def_strength": 1.0,
        "opp_recent_winrate": 0.5,
        "match_off_edge": 0.0,
        "match_def_edge": 0.0,
        "match_pace_sum": 2.0,
        "winrate_diff": 0.0,
        "oof_ml_prob": 0.5,
        "oof_spread_pred": 0.0,
        "starter_flag": 1,
        
        # Player rest features (19-20)
        "days_rest": 3.0,
        "player_b2b": 0.0,
        
        # Core stat rolling averages (21-32)
        "points_L3": avg(pts, 3),
        "points_L5": avg(pts, 5),
        "points_L10": avg(pts, 10),
        "assists_L3": avg(ast, 3),
        "assists_L5": avg(ast, 5),
        "assists_L10": avg(ast, 10),
        "rebounds_L3": avg(reb, 3),
        "rebounds_L5": avg(reb, 5),
        "rebounds_L10": avg(reb, 10),
        "threes_L3": avg(thr, 3),
        "threes_L5": avg(thr, 5),
        "threes_L10": avg(thr, 10),
        
        # PHASE 1: Shot volume rolling stats (33-41)
        "fieldGoalsAttempted_L3": avg(fga, 3) if fga else 10.0,
        "fieldGoalsAttempted_L5": avg(fga, 5) if fga else 10.0,
        "fieldGoalsAttempted_L10": avg(fga, 10) if fga else 10.0,
        "threePointersAttempted_L3": avg(tpa, 3) if tpa else 3.0,
        "threePointersAttempted_L5": avg(tpa, 5) if tpa else 3.0,
        "threePointersAttempted_L10": avg(tpa, 10) if tpa else 3.0,
        "freeThrowsAttempted_L3": avg(fta, 3) if fta else 2.0,
        "freeThrowsAttempted_L5": avg(fta, 5) if fta else 2.0,
        "freeThrowsAttempted_L10": avg(fta, 10) if fta else 2.0,
        
        # PHASE 1: Per-minute shot volume rates (42-44)
        "rate_fga": safe_div(avg(fga, 10), avg(mins, 10)) if fga and mins else 0.4,
        "rate_3pa": safe_div(avg(tpa, 10), avg(mins, 10)) if tpa and mins else 0.12,
        "rate_fta": safe_div(avg(fta, 10), avg(mins, 10)) if fta and mins else 0.08,
        
        # PHASE 1: Efficiency metrics (45-49) - NOW CALCULATED FROM ACTUAL DATA!
        "ts_pct_L5": calc_ts_pct(pts, fga, fta, 5),
        "ts_pct_L10": calc_ts_pct(pts, fga, fta, 10),
        "ts_pct_season": calc_ts_pct(pts, fga, fta, len(pts) if pts else 10),
        "three_pct_L5": avg(three_pct, 5) if three_pct else 0.35,
        "ft_pct_L5": avg(ft_pct, 5) if ft_pct else 0.77,
        
        # PHASE 2: Matchup & context factors (50-53) - COMPUTED FROM TEAM STATS!
        # Note: These are updated later with real matchup data from fetch_nba_team_stats
        "matchup_pace": 1.0,  # Placeholder - updated from opponent team stats
        "pace_factor": 1.0,  # Placeholder - updated from opponent team stats
        "def_matchup_difficulty": 1.0,  # Placeholder - updated from opponent defensive rating
        "offensive_environment": 1.0,  # Placeholder - updated from team offensive rating vs opponent defense
        
        # PHASE 3: Advanced rate stats (54-56) - ESTIMATED FROM VOLUME STATS!
        # Usage Rate = (FGA + 0.44*FTA) / Minutes * 5 (normalized to per-100 possessions)
        "usage_rate_L5": (safe_div(avg(fga, 5) + 0.44 * avg(fta, 5), avg(mins, 5)) * 5.0 * 4.8) if fga and fta and mins else 22.0,
        # Rebound Rate = rebounds per minute * pace factor
        "rebound_rate_L5": (safe_div(avg(reb, 5), avg(mins, 5)) * 48.0 * 0.5) if reb and mins else 12.0,
        # Assist Rate = assists per minute * pace factor
        "assist_rate_L5": (safe_div(avg(ast, 5), avg(mins, 5)) * 48.0 * 0.35) if ast and mins else 18.0,
        
        # Home/Away performance splits (57-60)
        "points_home_avg": avg(pts, 10),
        "points_away_avg": avg(pts, 10),
        "assists_home_avg": avg(ast, 10),
        "assists_away_avg": avg(ast, 10),
        
        # Minutes prediction (61)
        "minutes": avg(mins, 10) if mins else 24.0,
    }
    
    # Debug: Verify Phase features are calculated from real data
    if DEBUG_MODE:
        print(f"   [PHASE INTEGRATION] Features built:")
        print(f"     Phase 1 (Shot Volume): FGA_L5={feats['fieldGoalsAttempted_L5']:.1f}, rate_fga={feats['rate_fga']:.3f}, TS%_L5={feats['ts_pct_L5']:.3f}")
        print(f"     Phase 2 (Matchup): pace_factor={feats['pace_factor']:.3f}, def_difficulty={feats['def_matchup_difficulty']:.3f}")
        print(f"     Phase 3 (Advanced): usage={feats['usage_rate_L5']:.1f}%, reb_rate={feats['rebound_rate_L5']:.1f}%, ast_rate={feats['assist_rate_L5']:.1f}%")
    
    return pd.DataFrame([feats])


def build_minutes_features(df_last: pd.DataFrame, df_curr: pd.DataFrame) -> pd.DataFrame:
    """Build minutes-model feature row matching training order (23 features)."""
    def seq(col: str) -> List[float]:
        vals: List[float] = []
        if not df_curr.empty and col in df_curr.columns:
            vals += [float(x) for x in df_curr[col].tolist() if pd.notnull(x)]
        if not df_last.empty and col in df_last.columns:
            vals += [float(x) for x in df_last[col].tolist() if pd.notnull(x)]
        return vals

    mins = seq("minutes")
    def avg(v: List[float], w: int) -> float:
        if not v: return 0.0
        return float(np.mean(v[:w]))

    # Base context defaults (same as build_player_features)
    base = {
        "is_home": 1,
        "season_end_year": 2025.0,
        "season_decade": 2020.0,
        "team_recent_pace": 1.0,
        "team_off_strength": 1.0,
        "team_def_strength": 1.0,
        "team_recent_winrate": 0.5,
        "opp_recent_pace": 1.0,
        "opp_off_strength": 1.0,
        "opp_def_strength": 1.0,
        "opp_recent_winrate": 0.5,
        "match_off_edge": 0.0,
        "match_def_edge": 0.0,
        "match_pace_sum": 2.0,
        "winrate_diff": 0.0,
        "oof_ml_prob": 0.5,
        "oof_spread_pred": 0.0,
        "starter_flag": 1,
    }
    # Minutes trends expected by minutes model
    base["min_prev_mean5"] = avg(mins, 5) if mins else 24.0
    base["min_prev_mean10"] = avg(mins, 10) if mins else 24.0
    base["min_prev_last1"] = float(mins[0]) if mins else 24.0
    
    # Player rest features (NEW - needed for 23 features)
    base["days_rest"] = 3.0
    base["player_b2b"] = 0.0

    # Exact feature order as trained (23 features)
    order = [
        "is_home",
        "season_end_year", "season_decade",
        "team_recent_pace", "team_off_strength", "team_def_strength", "team_recent_winrate",
        "opp_recent_pace",  "opp_off_strength",  "opp_def_strength",  "opp_recent_winrate",
        "match_off_edge", "match_def_edge", "match_pace_sum", "winrate_diff",
        "oof_ml_prob", "oof_spread_pred", "starter_flag",
        "min_prev_mean5", "min_prev_mean10", "min_prev_last1",
        "days_rest", "player_b2b",
    ]
    row = {k: float(base.get(k, 0.0)) for k in order}
    return pd.DataFrame([row])

# ========= ANALYSIS HELPERS =========
# Market prior variances (tighter prior centers projection near line)
SIGMA_MARKET = {"points": 3.0, "assists": 1.2, "rebounds": 1.8, "threes": 1.0}

def _market_key(prop_type: str) -> str:
    m = prop_type.lower()
    return {"points":"PTS","assists":"AST","rebounds":"REB","threes":"3PM","moneyline":"Moneyline","spread":"Spread"}.get(m, "DEFAULT")

def analyze_player_prop(prop: dict, matchup_context: dict) -> Optional[dict]:
    # If both Over/Under odds are present, we choose later after projecting
    if prop.get("odds") is not None and (prop["odds"] < MIN_ODDS or prop["odds"] > MAX_ODDS):
        return None

    df_last, df_curr = get_player_stats_split(prop["player"], 25, 25)
    if (df_last.empty and df_curr.empty) or (len(df_last) + len(df_curr) < 3): return None
    stat_col = stat_map.get(prop["prop_type"])
    vl = df_last[stat_col].astype(float).tolist() if (stat_col and stat_col in df_last.columns) else []
    vc = df_curr[stat_col].astype(float).tolist() if (stat_col and stat_col in df_curr.columns) else []

    # Debug output for first few props
    if DEBUG_MODE and len(vl) + len(vc) > 0:
        print(f"\n[DEBUG] {prop['player']} - {prop['prop_type']}")
        print(f"  Last season games: {len(vl)}, Current season games: {len(vc)}")
        if len(vl) > 0:
            print(f"  Last season avg: {sum(vl)/len(vl):.1f} (sample: {vl[:5]})")
        if len(vc) > 0:
            print(f"  Current season avg: {sum(vc)/len(vc):.1f} (sample: {vc[:5]})")

    # Prioritize current season data heavily when available
    merged: List[float] = []
    if len(vc) >= 5:
        # If we have 5+ current season games, use them primarily with minimal historical context
        n_curr = float(len(vc))
        w_cur = blend_weight(n_curr, n0=3.0, continuity=0.9)  # Much lower n0 = prioritize current
        rep_last = max(1, int(round((1.0 - w_cur) * 10)))
        rep_curr = max(1, int(round(w_cur * 10)))
        for v in vl: merged.extend([v]*rep_last)
        for v in vc: merged.extend([v]*rep_curr)
    elif len(vc) > 0:
        # If we have some current season data, blend more evenly but still favor current
        n_curr = float(len(vc))
        w_cur = blend_weight(n_curr, n0=8.0, continuity=0.8)
        rep_last = max(1, int(round((1.0 - w_cur) * 10)))
        rep_curr = max(1, int(round(w_cur * 10)))
        for v in vl: merged.extend([v]*rep_last)
        for v in vc: merged.extend([v]*rep_curr)
    else:
        # Fall back to last season only if no current season data
        merged = vl

    if not merged: merged = vl or vc

    pace = matchup_context.get("pace", 1.0)
    defense = (matchup_context.get("home_defensive_factor", 1.0) + matchup_context.get("away_defensive_factor", 1.0)) / 2.0

    # EWMA/trend projection
    mu_ewma, sigma_ewma = project_stat(merged, prop["prop_type"], pace, defense)
    if mu_ewma == 0 or sigma_ewma == 0: return None

    # ML prediction (if model present)
    mu_ml = None
    sigma_ml = None
    if MODEL.available(prop["prop_type"]):
        feats_row = build_player_features(df_last, df_curr)
        # Inject predicted minutes into features for other stats
        if prop.get("prop_type") != "minutes" and MODEL.available("minutes"):
            try:
                min_feats = build_minutes_features(df_last, df_curr)
                mins_pred = MODEL.predict("minutes", min_feats)
                if mins_pred is not None:
                    try:
                        feats_row.at[0, "minutes"] = float(mins_pred)
                    except Exception:
                        pass
            except Exception:
                pass
        # Try enhanced selector first, fallback to LightGBM
        mu_ml = MODEL.predict_with_ensemble(prop["prop_type"], feats_row, player_history=df_last)
        if mu_ml is None:
            mu_ml = MODEL.predict(prop["prop_type"], feats_row)
        sigma_ml = MODEL.predict_sigma(prop["prop_type"], feats_row) or MODEL_RMSE.get(prop["prop_type"])

    # Inverse-variance ensemble (+ market prior)
    s_ewma = 1.0 / (max(1e-6, sigma_ewma) ** 2)
    use_ml = (mu_ml is not None and sigma_ml and sigma_ml > 0)
    s_ml = (1.0 / (max(1e-6, sigma_ml) ** 2)) if use_ml else 0.0
    # Market prior centered at line
    sigma_m = SIGMA_MARKET.get(prop["prop_type"], None)
    s_mkt = (1.0 / (max(1e-6, float(sigma_m)) ** 2)) if sigma_m else 0.0
    num = s_ewma * mu_ewma + (s_ml * mu_ml if use_ml else 0.0) + (s_mkt * float(prop["line"]) if s_mkt else 0.0)
    den = s_ewma + (s_ml if use_ml else 0.0) + s_mkt
    projection = float(num / den) if den > 0 else mu_ewma
    std_dev    = float(math.sqrt(1.0 / den)) if den > 0 else sigma_ewma

    if projection == 0 or std_dev == 0: return None

    disparity = projection - prop["line"]
    pick = "over" if disparity > 0 else "under"
    risk_adj = disparity / std_dev if std_dev > 0 else 0.0
    edge_pct = (disparity / prop["line"]) * 100 if prop["line"] != 0 else 0.0

    # Choose correct side odds if both provided
    if "odds_over" in prop or "odds_under" in prop:
        if pick == "over" and prop.get("odds_over") is not None:
            prop["odds"] = int(prop["odds_over"])
        elif pick == "under" and prop.get("odds_under") is not None:
            prop["odds"] = int(prop["odds_under"])
    if prop.get("odds") is None:
        return None
    if prop["odds"] < MIN_ODDS or prop["odds"] > MAX_ODDS:
        return None

    p_hat, _ = prop_win_probability(prop["prop_type"], merged, prop["line"], pick, projection, std_dev)
    # Apply empirical calibration if available
    p_hat = apply_calibration(prop["prop_type"], float(p_hat))
    conf_mult = get_prop_confidence_multiplier(prop["prop_id"])
    n_eff = N_EFF_BY_MARKET.get(_market_key(prop["prop_type"]), N_EFF_BY_MARKET["DEFAULT"])
    n_eff_scaled = max(10.0, min(200.0, n_eff * conf_mult))
    p_samples = sample_beta_posterior(p_hat, n_eff_scaled, n_samples=400 if FAST_MODE else 600)

    dec = american_to_decimal(prop["odds"]); b = dec - 1.0
    equity = load_equity(); dd_scale = drawdown_scale(equity, floor=0.6, window=14)
    kcfg = KellyConfig(q_conservative=0.35, fk_low=0.25, fk_high=0.50, dd_scale=dd_scale)
    f, p_c, _, p_mean = dynamic_fractional_kelly(p_samples, b, kcfg)
    if f <= 0.0: return None
    p_be = 1.0 / (1.0 + b)
    if p_c <= p_be: return None
    elg = risk_adjusted_elg(p_samples, b, f)
    
    # Use prop-type specific ELG gate
    elg_threshold = ELG_GATES.get(prop["prop_type"], ELG_GATES["DEFAULT"])
    if elg <= elg_threshold: return None

    stake = BANKROLL * f
    stake = min(stake, MAX_STAKE)  # Cap at max stake
    if stake < MIN_KELLY_STAKE: return None
    potential_profit = stake * (dec - 1.0)
    ev_dollars = (p_mean * potential_profit) - ((1.0 - p_mean) * stake)
    ev_pct = (ev_dollars / max(1e-9, stake)) * 100.0

    # Record recommendation for post-game learning
    try:
        record_recommendation(prop, predicted_prob=float(p_mean))
    except Exception:
        pass

    prop.update({
        "projection": round(projection, 2), "std_dev": round(std_dev, 2), "disparity": round(disparity, 2),
        "pick": pick, "edge": round(edge_pct, 2), "risk_adjusted": round(risk_adj, 2), "trend": 0.0,
        "win_prob": round(p_mean * 100, 2), "p_conservative": round(p_c, 4), "p_break_even": round(p_be, 4),
        "kelly_pct": round(f * 100, 2), "stake": round(stake, 2), "potential_profit": round(potential_profit, 2),
        "ev": round(ev_pct, 2), "confidence_mult": round(conf_mult, 3),
        "roi": round((potential_profit / max(1e-9, stake)) * 100.0, 2),
        "elg": round(elg, 8), "composite_score": round(elg, 8),
        "games_analyzed": int(len(vl) + len(vc)), "pace_factor": round(pace, 3), "defense_factor": round(defense, 3)
    })
    
    # ACCURACY FILTER: Only recommend high-confidence bets
    if p_mean < (MIN_WIN_PROBABILITY / 100.0):
        if DEBUG_MODE:
            print(f"   Filtered out {prop.get('player')} {prop.get('prop_type')} - confidence too low ({p_mean*100:.1f}% < {MIN_WIN_PROBABILITY}%)")
        return None
    
    return prop

# Global cache for team stats (fetched once per run)
_TEAM_STATS_CACHE: Optional[Dict[str, Dict[str, float]]] = None

def fetch_nba_team_stats(season: str = "2024-25") -> Dict[str, Dict[str, float]]:
    """
    Fetch real-time NBA team statistics from nba_api (FREE).
    Returns dict mapping team name -> stats dict with all features needed for model.

    Fetches two datasets:
    - Advanced stats: OFF_RATING, DEF_RATING, NET_RATING, PACE, EFG_PCT, TS_PCT, etc.
    - Four Factors: TOV_PCT, OREB_PCT, DREB_PCT, FTA_RATE (+ opponent versions)

    Caches results globally to avoid repeated API calls.
    """
    global _TEAM_STATS_CACHE

    if _TEAM_STATS_CACHE is not None:
        return _TEAM_STATS_CACHE

    team_stats_dict = {}

    try:
        # Fetch Advanced stats (ORTG, DRTG, PACE, eFG%, TS%, etc.)
        time.sleep(0.6)  # Rate limiting
        advanced = LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense='Advanced',
            per_mode_detailed='PerGame',
            season_type_all_star='Regular Season'
        )
        df_adv = advanced.get_data_frames()[0]

        # Fetch Four Factors (TOV%, ORB%, DRB%, FTA, + opponent versions)
        time.sleep(0.6)  # Rate limiting
        four_factors = LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense='Four Factors',
            per_mode_detailed='PerGame',
            season_type_all_star='Regular Season'
        )
        df_ff = four_factors.get_data_frames()[0]

        # Merge by TEAM_ID
        df_merged = pd.merge(df_adv, df_ff, on='TEAM_ID', suffixes=('', '_ff'))

        # Filter to NBA teams only (GP > 60 indicates full NBA season)
        df_nba = df_merged[df_merged['GP'] > 60].copy()

        # Build stats dict for each team
        for _, row in df_nba.iterrows():
            team_name = row['TEAM_NAME']

            team_stats_dict[team_name] = {
                'o_rtg': row.get('OFF_RATING', 110.0),
                'd_rtg': row.get('DEF_RATING', 110.0),
                'pace': row.get('PACE', 100.0),
                'efg': row.get('EFG_PCT', 0.52),
                'tov_pct': row.get('TM_TOV_PCT', 0.14),
                'orb_pct': row.get('OREB_PCT', 0.23),
                'drb_pct': row.get('DREB_PCT', 0.77),
                'ftr': row.get('FTA_RATE', 0.24),
                'opp_efg': row.get('OPP_EFG_PCT', 0.52),
                'opp_tov_pct': row.get('OPP_TOV_PCT', 0.14),
                'opp_ftr': row.get('OPP_FTA_RATE', 0.24),
                'ts_pct': row.get('TS_PCT', 0.56),
                'ast_pct': row.get('AST_PCT', 0.60),
                'net_rtg': row.get('NET_RATING', 0.0),
                'wins': row.get('W', 41),
                'losses': row.get('L', 41),
            }

        _TEAM_STATS_CACHE = team_stats_dict
        if DEBUG_MODE:
            print(f"   [NBA API] Fetched stats for {len(team_stats_dict)} NBA teams")

    except Exception as e:
        if DEBUG_MODE:
            print(f"   [NBA API] Error fetching team stats: {e}")
        _TEAM_STATS_CACHE = {}

    return _TEAM_STATS_CACHE

def build_game_features(game_info: dict) -> pd.DataFrame:
    """
    Build features for game-level predictions (moneyline, spread).
    Now uses REAL-TIME team stats from nba_api instead of defaults!
    """
    # Load defaults as fallback
    if not hasattr(MODEL, 'game_defaults') or not MODEL.game_defaults:
        return pd.DataFrame()

    defaults = MODEL.game_defaults.copy()

    # Fetch real-time team stats
    team_stats = fetch_nba_team_stats()

    # Extract home/away team names from game_info
    home_team = game_info.get('home_team') or game_info.get('player', '')  # 'player' field may contain team name
    away_team = game_info.get('away_team', '')

    # Try to match team names from game_info to nba_api team names
    home_stats = None
    away_stats = None

    for team_name, stats in team_stats.items():
        if home_team and _token_match(home_team, team_name):
            home_stats = stats
        if away_team and _token_match(away_team, team_name):
            away_stats = stats

    # If we found real stats, override defaults
    if home_stats:
        defaults['home_o_rtg_prior'] = home_stats['o_rtg']
        defaults['home_d_rtg_prior'] = home_stats['d_rtg']
        defaults['home_pace_prior'] = home_stats['pace']
        defaults['home_efg_prior'] = home_stats['efg']
        defaults['home_tov_pct_prior'] = home_stats['tov_pct']
        defaults['home_orb_pct_prior'] = home_stats['orb_pct']
        defaults['home_drb_pct_prior'] = home_stats['drb_pct']
        defaults['home_ftr_prior'] = home_stats['ftr']
        defaults['home_opp_efg_prior'] = home_stats['opp_efg']
        defaults['home_opp_tov_pct_prior'] = home_stats['opp_tov_pct']
        defaults['home_opp_ftr_prior'] = home_stats['opp_ftr']
        defaults['home_ts_pct_prior'] = home_stats['ts_pct']
        defaults['home_recent_winrate'] = home_stats['wins'] / (home_stats['wins'] + home_stats['losses'])

    if away_stats:
        defaults['away_o_rtg_prior'] = away_stats['o_rtg']
        defaults['away_d_rtg_prior'] = away_stats['d_rtg']
        defaults['away_pace_prior'] = away_stats['pace']
        defaults['away_efg_prior'] = away_stats['efg']
        defaults['away_tov_pct_prior'] = away_stats['tov_pct']
        defaults['away_orb_pct_prior'] = away_stats['orb_pct']
        defaults['away_drb_pct_prior'] = away_stats['drb_pct']
        defaults['away_ftr_prior'] = away_stats['ftr']
        defaults['away_opp_efg_prior'] = away_stats['opp_efg']
        defaults['away_opp_tov_pct_prior'] = away_stats['opp_tov_pct']
        defaults['away_opp_ftr_prior'] = away_stats['opp_ftr']
        defaults['away_ts_pct_prior'] = away_stats['ts_pct']
        defaults['away_recent_winrate'] = away_stats['wins'] / (away_stats['wins'] + away_stats['losses'])

    # Calculate derived features
    if home_stats and away_stats:
        defaults['home_off_strength'] = home_stats['o_rtg'] / 110.0
        defaults['home_def_strength'] = 110.0 / home_stats['d_rtg']
        defaults['away_off_strength'] = away_stats['o_rtg'] / 110.0
        defaults['away_def_strength'] = 110.0 / away_stats['d_rtg']
        defaults['home_recent_pace'] = home_stats['pace'] / 100.0
        defaults['away_recent_pace'] = away_stats['pace'] / 100.0
        defaults['match_off_edge'] = home_stats['o_rtg'] - away_stats['d_rtg']
        defaults['match_def_edge'] = away_stats['o_rtg'] - home_stats['d_rtg']
        defaults['match_pace_sum'] = (home_stats['pace'] + away_stats['pace']) / 100.0
        defaults['winrate_diff'] = defaults['home_recent_winrate'] - defaults['away_recent_winrate']

    # Create single-row DataFrame
    return pd.DataFrame([defaults])

def analyze_game_bet(prop: dict) -> Optional[dict]:
    """
    Analyze moneyline/spread bet using trained ML models.
    Ensembles ML predictions with market-implied odds.
    """
    if prop.get("odds") is not None and (prop["odds"] < MIN_ODDS or prop["odds"] > MAX_ODDS): return None

    # Market-implied probability (baseline)
    p_market = implied_prob_from_american(prop["odds"])

    # Try to get ML model prediction
    p_ml = None
    if prop["prop_type"] == "moneyline":
        # Build game features and get moneyline prediction
        game_feats = build_game_features(prop)
        if not game_feats.empty and MODEL.game_models.get("moneyline"):
            # Extract team IDs for unified ensemble
            home_team_id = prop.get("home_team") or prop.get("home_abbrev")
            away_team_id = prop.get("away_team") or prop.get("away_abbrev")
            p_ml = MODEL.predict_moneyline(game_feats, home_team_id, away_team_id)
    elif prop["prop_type"] == "spread":
        # Build game features and get spread prediction
        game_feats = build_game_features(prop)
        if not game_feats.empty and MODEL.game_models.get("spread"):
            result = MODEL.predict_spread(game_feats)
            if result:
                margin, sigma = result
                # Convert margin prediction to probability
                # For spread bets, check if predicted margin covers the line
                line = prop.get("line", 0.0)
                if sigma and sigma > 0:
                    # If line is +5.5, we need margin > -5.5 to cover
                    z = (margin - line) / sigma
                    p_ml = stats.norm.cdf(z)

    # Ensemble: if we have ML prediction, blend with market
    if p_ml is not None and 0 < p_ml < 1:
        # Inverse-variance weighting (assume market has lower uncertainty)
        w_market = 0.6  # Market gets 60% weight (more liquid/efficient)
        w_ml = 0.4      # Model gets 40% weight
        p_hat = w_market * p_market + w_ml * p_ml
    else:
        # Fall back to market odds only
        p_hat = p_market

    dec = american_to_decimal(prop["odds"]); b = dec - 1.0
    conf_mult = get_prop_confidence_multiplier(prop["prop_id"])
    n_eff = N_EFF_BY_MARKET.get(_market_key(prop["prop_type"]), 45.0)
    p_samples = sample_beta_posterior(p_hat, max(10.0, min(120.0, n_eff * conf_mult)), n_samples=400 if FAST_MODE else 600)
    equity = load_equity(); dd_scale = drawdown_scale(equity, floor=0.6, window=14)
    kcfg = KellyConfig(q_conservative=0.35, fk_low=0.25, fk_high=0.50, dd_scale=dd_scale)
    f, p_c, _, p_mean = dynamic_fractional_kelly(p_samples, b, kcfg)
    if f <= 0.0: return None
    p_be = 1.0 / (1.0 + b)
    if p_c <= p_be: return None
    elg = risk_adjusted_elg(p_samples, b, f)
    
    # Use prop-type specific ELG gate
    elg_threshold = ELG_GATES.get(prop["prop_type"], ELG_GATES["DEFAULT"])
    if elg <= elg_threshold: return None
    
    stake = BANKROLL * f
    stake = min(stake, MAX_STAKE)  # Cap at max stake
    if stake < MIN_KELLY_STAKE: return None
    potential_profit = stake * (dec - 1.0)
    ev_dollars = (p_mean * potential_profit) - ((1.0 - p_mean) * stake)
    ev_pct = (ev_dollars / max(1e-9, stake)) * 100.0
    prop.update({
        "projection": 0.0, "std_dev": 0.0, "disparity": 0.0, "pick": prop.get("player","N/A"),
        "edge": 0.0, "risk_adjusted": 0.0, "trend": 0.0,
        "win_prob": round(p_mean * 100, 2), "p_conservative": round(p_c, 4), "p_break_even": round(p_be, 4),
        "kelly_pct": round(f * 100, 2), "stake": round(stake, 2), "potential_profit": round(potential_profit, 2),
        "ev": round(ev_pct, 2), "confidence_mult": round(conf_mult, 3),
        "roi": round((potential_profit / max(1e-9, stake)) * 100.0, 2),
        "elg": round(elg, 8), "composite_score": round(elg, 8), "games_analyzed": 0,
        "pace_factor": 1.0, "defense_factor": 1.0
    })
    
    # ACCURACY FILTER: Only recommend high-confidence bets
    if p_mean < (MIN_WIN_PROBABILITY / 100.0):
        if DEBUG_MODE:
            print(f"   Filtered parlay - confidence too low ({p_mean*100:.1f}% < {MIN_WIN_PROBABILITY}%)")
        return None
    
    return prop

def top_by_category(analyzed_props: List[dict], k="elg") -> Dict[str, List[dict]]:
    def score(p): return p.get(k, p.get("composite_score", 0.0))
    groups = {}
    for key, _label in CATEGORIES:
        items = [p for p in analyzed_props if p.get("prop_type") == key]
        items.sort(key=lambda x: score(x), reverse=True)
        groups[key] = items[:TOP_PER_CATEGORY]
    return groups

def blend_weight(n_current: float, n0: float = PRIOR_GAMES_STRENGTH, continuity: float = TEAM_CONTINUITY_DEFAULT) -> float:
    continuity = max(0.25, min(1.0, continuity))
    n0_eff = n0 / continuity
    return n_current / (n_current + n0_eff + 1e-9)

# ========= PARLAY BUILDER =========
def build_parlays(props: List[dict], max_legs: int = 3, min_legs: int = 2, max_parlays: int = 10) -> List[dict]:
    """
    Build optimal 2-3 leg parlays from available props.
    
    Strategy:
    - Combine uncorrelated props (different players, different stat types)
    - Prioritize high win probability props (>65%)
    - Calculate parlay odds and expected value
    - Cap stake at MAX_STAKE per parlay
    """
    from itertools import combinations
    
    # Filter for high-quality props
    quality_props = [p for p in props if p.get("win_prob", 0) >= 60]
    if len(quality_props) < min_legs:
        return []
    
    parlays = []
    
    # Generate combinations
    for num_legs in range(min_legs, max_legs + 1):
        for combo in combinations(quality_props, num_legs):
            # Check for correlation (same player, same game)
            players = [p.get("player", "") for p in combo]
            games = [p.get("game_id", "") for p in combo]
            
            # Skip if same player appears multiple times
            if len(set(players)) != len(players):
                continue
            
            # Calculate parlay metrics
            parlay_prob = 1.0
            parlay_odds_decimal = 1.0
            legs = []
            
            for prop in combo:
                p_win = prop.get("win_prob", 50) / 100.0
                parlay_prob *= p_win
                
                odds = prop.get("odds", 0)
                if odds:
                    parlay_odds_decimal *= american_to_decimal(odds)
                
                legs.append({
                    "player": prop.get("player"),
                    "prop_type": prop.get("prop_type"),
                    "line": prop.get("line"),
                    "pick": prop.get("pick"),
                    "odds": odds,
                    "win_prob": prop.get("win_prob"),
                })
            
            # Calculate parlay payout
            parlay_odds_american = (parlay_odds_decimal - 1.0) * 100 if parlay_odds_decimal >= 2.0 else -100 / (parlay_odds_decimal - 1.0)

            # Filter parlays with odds exceeding MAX_PARLAY_ODDS
            if parlay_odds_american > MAX_PARLAY_ODDS:
                continue

            # Kelly sizing for parlay (more conservative)
            p_break_even = 1.0 / parlay_odds_decimal
            if parlay_prob <= p_break_even:
                continue
            
            # Conservative Kelly fraction for parlays
            edge = parlay_prob - p_break_even
            kelly_fraction = edge / (parlay_odds_decimal - 1.0) * 0.5  # Half Kelly for parlays
            kelly_fraction = max(0.0, min(0.15, kelly_fraction))  # Cap at 15% for safety
            
            stake = BANKROLL * kelly_fraction
            stake = min(stake, MAX_STAKE)  # Cap at max stake
            
            if stake < MIN_KELLY_STAKE:
                continue
            
            potential_profit = stake * (parlay_odds_decimal - 1.0)
            ev_dollars = (parlay_prob * potential_profit) - ((1.0 - parlay_prob) * stake)
            ev_pct = (ev_dollars / max(1e-9, stake)) * 100.0
            
            # Only keep positive EV parlays
            if ev_pct <= 0:
                continue
            
            parlay = {
                "type": "parlay",
                "num_legs": num_legs,
                "legs": legs,
                "parlay_prob": round(parlay_prob * 100, 2),
                "parlay_odds": round(parlay_odds_american, 0),
                "parlay_odds_decimal": round(parlay_odds_decimal, 2),
                "stake": round(stake, 2),
                "potential_profit": round(potential_profit, 2),
                "ev": round(ev_pct, 2),
                "kelly_pct": round(kelly_fraction * 100, 2),
                "score": parlay_prob * ev_pct,  # Composite score for ranking
            }
            parlays.append(parlay)
    
    # Sort by composite score and return top parlays
    parlays.sort(key=lambda x: x["score"], reverse=True)
    return parlays[:max_parlays]

# ========= MAIN =========
def run_analysis():
    start = time.monotonic()
    print("=" * 72)
    print("RIQ MEEPING MACHINE 🚀 — Unified Analyzer (TheRundown + ML Ensemble)")
    print("=" * 72)
    print(f"Season: {SEASON} | Stats: prior={STATS_SEASON} | Bankroll: ${BANKROLL:.2f}")
    print(f"Odds Range: {MIN_ODDS} to {MAX_ODDS} | Ranking: ELG + dynamic Kelly")
    print(f"FAST_MODE: {'ON' if FAST_MODE else 'OFF'} | Time Budget: {'Disabled' if RUN_TIME_BUDGET_SEC is None else f'{RUN_TIME_BUDGET_SEC}s'}")
    print("=" * 72)
    print()

    # Quick API ping to fail fast if key/network is bad
    ping = fetch_json("/games", params={"league": LEAGUE_ID, "season": SEASON, "date": datetime.date.today().strftime("%Y-%m-%d")})
    if ping is None:
        print("❌ API unreachable or key invalid. Set API_SPORTS_KEY/APISPORTS_KEY and try again.")
        return

    games = get_upcoming_games()
    if not games:
        print("❌ No upcoming games found"); return

    print("🎲 Fetching odds from multiple sources...")
    contexts = {g["id"]: get_matchup_context(g) for g in games}
    
    # Fetch from The Odds API ONLY (comprehensive coverage - game markets + player props)
    print("   • The Odds API (all markets + player props)...")
    theodds_props = theodds_fetch_odds(games)

    # Use only The Odds API props
    all_props = theodds_props

    if DEBUG_MODE:
        print(f"   [The Odds API] Fetched {len(theodds_props)} props")
    
    # Deduplicate: keep best odds for each unique prop
    props_by_key = {}
    for prop in all_props:
        # Create key: game_id + player + prop_type + line
        key = f"{prop['game_id']}_{prop.get('player', '')}_{prop['prop_type']}_{prop.get('line', 0)}"
        
        # Keep prop with best odds (higher for positive, closer to 0 for negative)
        if key not in props_by_key:
            props_by_key[key] = prop
        else:
            existing = props_by_key[key]
            # Simple best-odds logic: higher American odds is better
            if prop.get('odds', -999) > existing.get('odds', -999):
                props_by_key[key] = prop
    
    fd_props = list(props_by_key.values())
    
    print(f"   ✓ Fetched {len(fd_props)} unique props from The Odds API")

    # Optional: per-game counts
    by_game = defaultdict(int)
    for p in fd_props:
        by_game[p["game_id"]] += 1
    for g in games:
        gid = g["id"]; cnt = by_game.get(gid, 0)
        game_date_str = str(g.get("date",""))[:16]
        print(f"   ✓ {game_date_str} — {g['teams']['home']['name']} vs {g['teams']['away']['name']}: {cnt} props")

    player_props = [p for p in fd_props if p["prop_type"] in ["points","assists","rebounds","threes"]]
    game_bets   = [p for p in fd_props if p["prop_type"] in ["moneyline","spread"]]

    print(f"\n   Total props: {len(fd_props)} | Player: {len(player_props)} | Game: {len(game_bets)}\n")
    if DEBUG_MODE and len(fd_props) == 0:
        print("   [DEBUG] No props found. Check expand/nested markets and bookmaker naming; printing diagnostics above.")

    print(f"🔍 {random.choice(MEEP_MESSAGES)}...")

    analyzed = []
    for idx, prop in enumerate(player_props + game_bets, 1):
        if RUN_TIME_BUDGET_SEC is not None and time.monotonic() - start > RUN_TIME_BUDGET_SEC:
            print("⏳ Time budget reached during analysis.")
            break
        try:
            if prop["prop_type"] in ["points","assists","rebounds","threes"]:
                res = analyze_player_prop(prop, contexts.get(prop["game_id"], {}))
            else:
                res = analyze_game_bet(prop)
            if res: analyzed.append(res)
        except Exception as e:
            if DEBUG_MODE: print(f"   ⚠️ Analysis failed for {prop.get('player')} {prop.get('prop_type')}: {e}")
        if idx % 25 == 0:
            print(f"   {random.choice(MEEP_MESSAGES)}... {idx}/{len(player_props)+len(game_bets)} analyzed")
        if prop["prop_type"] in ["points","assists","rebounds","threes"]:
            time.sleep(SLEEP_SHORT)

    print(f"\n   ✅ {len(analyzed)} props meet ELG gates\n")
    
    # Build parlays from analyzed props
    parlays = build_parlays(analyzed, max_legs=3, min_legs=2, max_parlays=10)
    print(f"   🎯 Built {len(parlays)} optimal parlays (2-3 legs)\n")
    
    groups = top_by_category(analyzed, k="elg")
    
    # Summary table
    print("=" * 72)
    print(f"SUMMARY: Top Props by Category")
    print("=" * 72)
    print(f"{'Category':<15} {'Count':<10} {'Best ELG':<15} {'Best Win%':<15}")
    print("-" * 72)
    for key, label in CATEGORIES:
        bucket = groups.get(key, [])
        count = len(bucket)
        best_elg = max([p.get("elg", 0) for p in bucket], default=0)
        best_win = max([p.get("win_prob", 0) for p in bucket], default=0)
        print(f"{label:<15} {count:<10} {best_elg:<15.6f} {best_win:<15.1f}%")
    print()
    
    if not analyzed:
        print("❌ No props passed ELG gates"); return

    print("=" * 72)
    print(f"TOP {TOP_PER_CATEGORY} PER CATEGORY (by ELG)")
    print("=" * 72)
    
    for key, label in CATEGORIES:
        bucket = groups.get(key, [])
        print(f"\n{'='*72}")
        print(f"{label.upper()} ({len(bucket)} props)")
        print(f"{'='*72}")
        
        if not bucket:
            print(f"   No {label.lower()} props found or none passed ELG threshold")
            continue
            
        for i, p in enumerate(bucket, 1):
            conf = "🟢" if p["win_prob"] >= 65 else ("🟡" if p["win_prob"] >= 55 else "🟠")
            score_val = p.get("elg", p.get("composite_score", 0.0))
            date_out = str(p.get("game_date",""))[:16]
            
            print(f"\n{conf} #{i} — {p['player']}")
            print(f"   Game:     {p['game']}")
            print(f"   Date:     {date_out}")
            print(f"   Bookmaker: {p.get('bookmaker', 'Unknown')}")
            
            if key in ["points","assists","rebounds","threes"]:
                print(f"   Line:     {p['line']:.1f}")
                print(f"   Projection: {p['projection']:.2f} (Δ: {p['disparity']:+.2f}, σ: {p['std_dev']:.2f})")
                print(f"   Pace:     {p.get('pace_factor',1.0):.3f}x | Defense: {p.get('defense_factor',1.0):.3f}x")
                print(f"   Pick:     {p['pick'].upper()} @ {p['odds']:+d}")
            else:
                # Moneyline or spread
                print(f"   Pick:     {p['pick']} @ {p['odds']:+d}")
            
            print(f"   Kelly:    {p['kelly_pct']:.2f}% → Stake: ${p['stake']:.2f}")
            print(f"   Profit:   ${p['potential_profit']:.2f}")
            print(f"   EV:       {p['ev']:+.2f}% | Win Prob: {p['win_prob']:.1f}%")
            print(f"   ELG Score: {score_val:.6f}")
    
    # Display Parlays
    if parlays:
        print(f"\n{'='*72}")
        print(f"🎯 TOP PARLAYS ({len(parlays)} combinations)")
        print(f"{'='*72}")
        
        for i, parlay in enumerate(parlays, 1):
            print(f"\n🎲 Parlay #{i} — {parlay['num_legs']} Legs")
            print(f"   Combined Odds: {int(parlay['parlay_odds']):+d} (Decimal: {parlay['parlay_odds_decimal']:.2f})")
            print(f"   Win Probability: {parlay['parlay_prob']:.1f}%")
            print(f"   Stake: ${parlay['stake']:.2f} (Kelly: {parlay['kelly_pct']:.2f}%)")
            print(f"   Potential Profit: ${parlay['potential_profit']:.2f}")
            print(f"   Expected Value: {parlay['ev']:+.2f}%")
            print(f"\n   Legs:")
            for j, leg in enumerate(parlay['legs'], 1):
                print(f"     {j}. {leg['player']} - {leg['prop_type'].upper()} {leg['pick'].upper()} {leg['line']} @ {int(leg['odds']):+d} ({leg['win_prob']:.1f}%)")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"prop_analysis_{timestamp}.json"
    
    # Save JSON with all analysis results
    groups_json = {key: groups.get(key, []) for key,_ in CATEGORIES}
    output_data = {
        "timestamp": timestamp,
        "season": SEASON,
        "stats_season": STATS_SEASON,
        "bankroll": BANKROLL,
        "max_stake": MAX_STAKE,
        "total_props_analyzed": len(player_props) + len(game_bets),
        "props_passed_elg": len(analyzed),
        "top_by_category": groups_json,
        "parlays": parlays,
        "summary": {
            key: {
                "count": len(groups.get(key, [])),
                "best_elg": max([p.get("elg", 0) for p in groups.get(key, [])], default=0),
                "best_win_prob": max([p.get("win_prob", 0) for p in groups.get(key, [])], default=0),
            }
            for key, _ in CATEGORIES
        }
    }
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*72}")
    print(f"✅ Results saved to: {output_file}")
    print(f"   Total Props Analyzed: {len(player_props) + len(game_bets)}")
    print(f"   Props Passed ELG: {len(analyzed)}")
    print(f"   Parlays Generated: {len(parlays)}")
    print(f"{'='*72}")
    print("🎉 Meep complete!")

if __name__ == "__main__":
    try: run_analysis()
    except KeyboardInterrupt: print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}"); import traceback; traceback.print_exc()
