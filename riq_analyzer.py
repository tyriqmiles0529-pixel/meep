# RIQ MEEPING MACHINE — Fully Integrated, Fast-Mode, All-in-One
from __future__ import annotations

import os
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

# ========= RUNTIME / FAST MODE =========
FAST_MODE = False  # set to False for full runs
REQUEST_TIMEOUT = 4 if FAST_MODE else 10
RETRIES = 1 if FAST_MODE else 3
# Loosen: fetch games 3 days out even in FAST_MODE
DAYS_TO_FETCH = 3
MAX_GAMES = 6 if FAST_MODE else 20
SLEEP_SHORT = 0.05 if FAST_MODE else 0.2
SLEEP_LONG = 0.1 if FAST_MODE else 0.3
RUN_TIME_BUDGET_SEC = 50 if FAST_MODE else 300

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
SEASON = "2025-2026"
STATS_SEASON = "2024-2025"

# SportsGameOdds (FanDuel odds)
SGO_BASE = "https://api.sportsgameodds.com/v2"
SGO_EVENTS_ENDPOINT = f"{SGO_BASE}/events"
# Some accounts use different league identifiers
SGO_LEAGUE_IDS = ["NBA", "basketball_nba", "nba"]
# Prefer env var; fallback demo key
SGO_API_KEY = os.getenv("SGO_API_KEY") or os.getenv("SPORTSGAMEODDS_API_KEY") or "3ee00eb314b80853c6c77920c5bf74f7"
SGO_LIMIT = 50
SGO_MAX_PAGES = 200
SGO_SLEEP_BETWEEN_PAGES_SEC = 0.15

BANKROLL = 100.0
MIN_KELLY_STAKE = 0.005
DEBUG_MODE = True  # set True to log markets/bookmakers seen

# Odds filter
MIN_ODDS = -500
MAX_ODDS = +500

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

prop_weights = load_data(WEIGHTS_FILE, defaultdict(lambda: 1.0))
prop_results = load_data(RESULTS_FILE, defaultdict(list))
player_cache = load_data(CACHE_FILE, {})
_pid_cache: Dict[str, int] = player_cache.get("__pid_cache__", {})

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
        return {"points": float(points), "assists": float(assists), "rebounds": float(rebounds), "threes": float(threes)}
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
    d_last = fetch_json("/games/statistics/players", params={"season": STATS_SEASON, "player": pid})
    if d_last and "response" in d_last:
        for r in d_last["response"][:max_last]:
            g = _parse_game_row(r)
            if g: last.append(g)

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

# ========= SGO (FanDuel odds) HELPERS =========
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

def _sgo_headers_params(api_key: Optional[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    headers: Dict[str, str] = {}
    params: Dict[str, str] = {}
    if api_key:
        headers["x-api-key"] = api_key
        # API key goes in header only, not in query params
    return headers, params

def sgo_fetch_events(
    api_key: Optional[str] = None,
    market_odds_available: bool = True,
    limit: int = SGO_LIMIT,
    max_pages: int = SGO_MAX_PAGES,
    pause_sec: float = SGO_SLEEP_BETWEEN_PAGES_SEC,
) -> Tuple[List[dict], List[str]]:
    api_key = api_key or SGO_API_KEY
    headers, base_params = _sgo_headers_params(api_key)
    out: List[dict] = []
    warns: List[str] = []
    for league_id_try in SGO_LEAGUE_IDS:
        next_cursor: Optional[str] = None
        pages = 0
        while True:
            params = {
                **base_params,
                "leagueID": league_id_try,
                "marketOddsAvailable": str(market_odds_available).lower(),
                "limit": str(limit),
                "expand": "markets,bookmakers,outcomes",
            }
            if next_cursor:
                params["cursor"] = next_cursor
            try:
                resp = requests.get(SGO_EVENTS_ENDPOINT, params=params, headers=headers, timeout=20)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                warns.append(f"SGO fetch error ({league_id_try}): {e}")
                break
            events = data.get("events", [])
            out.extend(events)
            next_cursor = data.get("nextCursor")
            pages += 1
            if not next_cursor or pages >= max_pages:
                break
            time.sleep(pause_sec)
        if out:
            break
    if DEBUG_MODE:
        print(f"   [DEBUG] SGO events fetched: {len(out)} (league tried: {','.join(SGO_LEAGUE_IDS)})")
        with_markets = sum(1 for e in out if isinstance(e.get('markets'), list) and e['markets'])
        print(f"   [DEBUG] Events with markets[]: {with_markets}")
        for e in out[:3]:
            mkts = e.get("markets") or []
            if not mkts: continue
            kinds = [str(m.get('betTypeID') or m.get('type') or m.get('key')).lower() for m in mkts[:5]]
            bms0 = (mkts[0].get("bookmakers") or [])
            bm_names = [str(bm.get('name') or bm.get('bookmaker') or bm.get('id')).lower() for bm in bms0[:5]]
            print(f"   [DEBUG] Event {e.get('id') or e.get('eventID')}: markets={len(mkts)} types={kinds} first-bm={bm_names}")
    return out, warns

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
    v = _extract(obj, ("bookmaker", "bookmakerId", "bookmakerID", "sportsbook", "provider", "source", "book"))
    if v is None:
        return None
    return str(v).strip().lower()

def _extract_player_name(obj: dict) -> Optional[str]:
    v = _extract(obj, ("player", "playerName", "name", "participant"))
    return str(v) if v is not None else None

def _extract_team_name(obj: dict) -> Optional[str]:
    v = _extract(obj, ("team", "teamName", "name", "runnerName", "selectionName"))
    return str(v) if v is not None else None

def _read_event_meta(ev: dict) -> Tuple[str, str, str, str]:
    event_id = str(_extract(ev, ("eventID","id","eventId")) or "")
    away = str(_extract(ev, ("awayTeam","away","visitor")) or "")
    home = str(_extract(ev, ("homeTeam","home","host")) or "")
    start = str(_extract(ev, ("startTime","commenceTime","commence_time","start_time")) or "")
    return event_id, away, home, start

def _iter_odds_objects(ev: dict):
    odds = ev.get("odds") or {}
    if isinstance(odds, dict):
        for odd_id, obj in odds.items():
            if isinstance(obj, dict):
                yield odd_id, obj
    elif isinstance(odds, list):
        for obj in odds:
            if isinstance(obj, dict):
                yield obj.get("id") or obj.get("_id") or "", obj

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
        if "three" in m or "3pt" in m or "3pm" in m or "3-pointer" in m or "3-pointers" in m:
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
        return props

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

        bms = m.get("bookmakers") or []
        for bm in bms:
            if not _is_fanduel_bookmaker(bm):
                continue
            outcomes = bm.get("outcomes") or []

            # Try detect player OU inside OU markets (when bet_type == 'ou')
            if internal_type is None and bet_type in ("ou","over_under","totals"):
                # If outcomes carry participant fields, treat as player OU
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
                        "bookmaker": "FanDuel",
                        "source": "SGO",
                    })

            elif internal_type == "moneyline":
                for oc in outcomes:
                    side = str(oc.get("sideID") or oc.get("side") or "").lower()
                    team = oc.get("team") or oc.get("runnerName") or oc.get("selectionName") or (home if side == "home" else away if side == "away" else None)
                    american = _to_int_american(oc.get("americanOdds") or oc.get("oddsAmerican") or oc.get("price") or oc.get("odds"))
                    if not team or american is None:
                        continue
                    prop_id = f"{game_id}_moneyline_{team}".replace(" ", "_")
                    props.append({
                        "prop_id": prop_id, "game_id": game_id, "game": game_label, "game_date": game_date,
                        "player": str(team), "prop_type": "moneyline", "line": 0.0, "odds": int(american),
                        "bookmaker": "FanDuel", "source": "SGO",
                    })

            elif internal_type == "spread":
                for oc in outcomes:
                    side = str(oc.get("sideID") or oc.get("side") or "").lower()
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
                        "bookmaker": "FanDuel", "source": "SGO",
                    })
    return props

def sgo_fanduel_props_for_games(apisports_games: List[dict]) -> Tuple[List[dict], List[str]]:
    events, warns = sgo_fetch_events()
    game_idx = _index_games(apisports_games)
    props: List[dict] = []

    seen_bm = Counter()
    seen_mk = Counter()

    for ev in events:
        event_id, away, home, start = _read_event_meta(ev)
        matched_game = _match_game(game_idx, home, away, start)
        if not matched_game:
            continue

        # 1) Try nested markets path (preferred)
        nested_props = _parse_nested_markets(ev, matched_game)
        props.extend(nested_props)

        # 2) Fallback: flat odds objects (older/simple feeds)
        game_id = matched_game.get("id") or matched_game.get("gameId") or event_id
        game_label = f"{matched_game.get('teams',{}).get('away',{}).get('name', away)} at {matched_game.get('teams',{}).get('home',{}).get('name', home)}"
        game_date = matched_game.get("date") or start

        for odd_id, obj in _iter_odds_objects(ev):
            bm = _extract_bookmaker(obj)
            if bm:
                seen_bm[bm] += 1
            # Require FanDuel-like bookmaker value
            if not bm or "fanduel" not in bm:
                continue

            market_raw = _extract_market(obj)
            if market_raw:
                seen_mk[market_raw] += 1
            prop_type = _coerce_market(market_raw)
            if not prop_type:
                continue

            selections = obj.get("selections") or obj.get("outcomes") or []
            american = _to_int_american(_extract(obj, ("americanOdds","closeOdds","odds","price")))
            line = _extract_line(obj)

            # Player markets
            if prop_type in {"points","assists","rebounds","threes"}:
                player = _extract_player_name(obj)
                if not player and isinstance(selections, list):
                    for sel in selections:
                        maybe_player = _extract(sel, ("player","playerName","participant","name"))
                        if maybe_player:
                            player = str(maybe_player); break
                if not player:
                    continue
                # Prefer selections for over/under odds
                over_odds = None; under_odds = None
                if isinstance(selections, list) and selections:
                    for sel in selections:
                        sel_name = str(_extract(sel, ("name","selectionName","label")) or "").lower()
                        sel_odds = _to_int_american(_extract(sel, ("americanOdds","odds","price","closeOdds")))
                        if line is None:
                            line = _extract_line(sel) or line
                        if sel_name in {"over","o"}:
                            over_odds = sel_odds
                        elif sel_name in {"under","u"}:
                            under_odds = sel_odds
                if line is None:
                    continue
                prop_key = f"{game_id}_{player}_{prop_type}".replace(" ", "_")
                rec = {
                    "prop_id": prop_key,
                    "game_id": game_id,
                    "game": game_label,
                    "game_date": game_date,
                    "player": player,
                    "prop_type": prop_type,
                    "line": float(line),
                    "bookmaker": "FanDuel",
                    "source": "SGO",
                }
                if over_odds is not None or under_odds is not None:
                    if over_odds is not None: rec["odds_over"] = int(over_odds)
                    if under_odds is not None: rec["odds_under"] = int(under_odds)
                    rec["odds"] = int(over_odds if over_odds is not None else (under_odds if under_odds is not None else -110))
                elif american is not None:
                    rec["odds"] = int(american)
                else:
                    continue
                props.append(rec)

            elif prop_type == "moneyline":
                if isinstance(selections, list) and selections:
                    for sel in selections:
                        team = _extract_team_name(sel) or _extract(sel, ("name","selectionName"))
                        odds_val = _to_int_american(_extract(sel, ("americanOdds","odds","price","closeOdds")))
                        if not team or odds_val is None:
                            continue
                        prop_id = f"{game_id}_moneyline_{team}".replace(" ", "_")
                        props.append({
                            "prop_id": prop_id, "game_id": game_id, "game": game_label, "game_date": game_date,
                            "player": str(team), "prop_type": "moneyline", "line": 0.0, "odds": int(odds_val),
                            "bookmaker": "FanDuel", "source": "SGO",
                        })
                else:
                    team = _extract_team_name(obj) or _extract(obj, ("name","selectionName"))
                    if team and american is not None:
                        prop_id = f"{game_id}_moneyline_{team}".replace(" ", "_")
                        props.append({
                            "prop_id": prop_id, "game_id": game_id, "game": game_label, "game_date": game_date,
                            "player": str(team), "prop_type": "moneyline", "line": 0.0, "odds": int(american),
                            "bookmaker": "FanDuel", "source": "SGO",
                        })

            elif prop_type == "spread":
                if isinstance(selections, list) and selections:
                    for sel in selections:
                        team = _extract_team_name(sel) or _extract(sel, ("name","selectionName"))
                        odds_val = _to_int_american(_extract(sel, ("americanOdds","odds","price","closeOdds")))
                        line_val = _extract_line(sel) if _extract_line(sel) is not None else (line if line is not None else None)
                        if not team or odds_val is None or line_val is None:
                            continue
                        side = f"{team} {float(line_val):+}"
                        prop_id = f"{game_id}_spread_{side}".replace(" ", "_")
                        props.append({
                            "prop_id": prop_id, "game_id": game_id, "game": game_label, "game_date": game_date,
                            "player": side, "prop_type": "spread", "line": float(line_val), "odds": int(odds_val),
                            "bookmaker": "FanDuel", "source": "SGO",
                        })
                else:
                    team = _extract_team_name(obj) or _extract(obj, ("name","selectionName"))
                    if team and american is not None and line is not None:
                        side = f"{team} {float(line):+}"
                        prop_id = f"{game_id}_spread_{side}".replace(" ", "_")
                        props.append({
                            "prop_id": prop_id, "game_id": game_id, "game": game_label, "game_date": game_date,
                            "player": side, "prop_type": "spread", "line": float(line), "odds": int(american),
                            "bookmaker": "FanDuel", "source": "SGO",
                        })

    if DEBUG_MODE:
        print("   [DEBUG] Seen bookmakers (flat path):", dict(seen_bm.most_common(10)))
        print("   [DEBUG] Seen markets (flat path):", dict(seen_mk.most_common(20)))

    if not (os.getenv("SGO_API_KEY") or os.getenv("SPORTSGAMEODDS_API_KEY")) and SGO_API_KEY.endswith("bf74f7"):
        warns.append("SGO_API_KEY not set; using inline fallback key. Set SGO_API_KEY in your environment for safety.")
    return props, warns

# ========= ML MODEL ENSEMBLE (load + predict) =========
MODEL_DIR = "models"
MODEL_FILES = {
    "points":  "points_model.pkl",
    "assists": "assists_model.pkl",
    "rebounds":"rebounds_model.pkl",
    "threes":  "threepoint_goals_model.pkl",
}
MODEL_RMSE_DEFAULT = {"points": 5.8, "assists": 1.8, "rebounds": 2.5, "threes": 1.2}

def _load_model_registry() -> Dict[str, float]:
    reg_path = os.path.join(MODEL_DIR, "model_registry.json")
    rmse_by_type = dict(MODEL_RMSE_DEFAULT)
    if not os.path.exists(reg_path):
        return rmse_by_type
    try:
        with open(reg_path, "r", encoding="utf-8") as f:
            reg = json.load(f)
        candidates = reg.get("models", reg)
        for k_an, key_model in [("points","points"), ("assists","assists"), ("rebounds","rebounds"), ("threes","threepoint_goals")]:
            node = candidates.get(key_model) or candidates.get(k_an) or {}
            rmse = None
            if isinstance(node, dict):
                rmse = node.get("rmse") or (node.get("metrics", {}) if isinstance(node.get("metrics"), dict) else {}).get("rmse")
            if rmse is not None:
                rmse_by_type[k_an] = float(rmse)
    except Exception:
        return rmse_by_type
    return rmse_by_type

MODEL_RMSE = _load_model_registry()

class ModelPredictor:
    def __init__(self):
        self.models: Dict[str, object] = {}
        for key, fname in MODEL_FILES.items():
            path = os.path.join(MODEL_DIR, fname)
            if os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        self.models[key] = pickle.load(f)
                    if DEBUG_MODE: print(f"   ↳ Loaded ML model: {path}")
                except Exception as e:
                    if DEBUG_MODE: print(f"   ⚠️ Failed to load {path}: {e}")

    def available(self, prop_type: str) -> bool:
        return prop_type in self.models

    def predict(self, prop_type: str, feats: pd.DataFrame) -> Optional[float]:
        m = self.models.get(prop_type)
        if m is None or feats is None or feats.empty:
            return None
        try:
            y = m.predict(feats)
            return float(y[0]) if isinstance(y, (list, np.ndarray)) else float(y)
        except Exception as e:
            if DEBUG_MODE: print(f"   ⚠️ ML predict failed for {prop_type}: {e}")
            return None

MODEL = ModelPredictor()

def build_player_features(df_last: pd.DataFrame, df_curr: pd.DataFrame) -> pd.DataFrame:
    def seq(col: str) -> List[float]:
        vals: List[float] = []
        if not df_curr.empty and col in df_curr.columns:
            vals += [float(x) for x in df_curr[col].tolist() if pd.notnull(x)]
        if not df_last.empty and col in df_last.columns:
            vals += [float(x) for x in df_last[col].tolist() if pd.notnull(x)]
        return vals

    pts = seq("points"); ast = seq("assists"); reb = seq("rebounds"); thr = seq("threes")

    def avg(v: List[float], w: int) -> float:
        if not v: return 0.0
        return float(np.mean(v[:w])) if len(v) >= 1 else 0.0

    def trend(v: List[float]) -> float:
        a3, a10 = avg(v, 3), avg(v, 10)
        return 0.0 if a10 == 0 else float((a3 - a10) / max(1e-6, a10))

    feats = {
        "points_avg_3g": avg(pts, 3),
        "points_avg_5g": avg(pts, 5),
        "points_avg_10g": avg(pts, 10),
        "points_trend": trend(pts),
        "assists_avg_3g": avg(ast, 3),
        "assists_avg_5g": avg(ast, 5),
        "assists_avg_10g": avg(ast, 10),
        "assists_trend": trend(ast),
        "rebounds_avg_3g": avg(reb, 3),
        "rebounds_avg_5g": avg(reb, 5),
        "rebounds_avg_10g": avg(reb, 10),
        "rebounds_trend": trend(reb),
        "threepoint_goals_avg_3g": avg(thr, 3),
        "threepoint_goals_avg_5g": avg(thr, 5),
        "threepoint_goals_avg_10g": avg(thr, 10),
        "threepoint_goals_trend": trend(thr),
    }
    return pd.DataFrame([feats])

# ========= ANALYSIS HELPERS =========
def _market_key(prop_type: str) -> str:
    m = prop_type.lower()
    return {"points":"PTS","assists":"AST","rebounds":"REB","threes":"3PM","moneyline":"Moneyline","spread":"Spread"}.get(m, "DEFAULT")

def analyze_player_prop(prop: dict, matchup_context: dict) -> Optional[dict]:
    # If both Over/Under odds are present, we choose later after projecting
    if prop.get("odds") is not None and (prop["odds"] < MIN_ODDS or prop["odds"] > MAX_ODDS):
        return None

    df_last, df_curr = get_player_stats_split(prop["player"], 25, 5)
    if (df_last.empty and df_curr.empty) or (len(df_last) + len(df_curr) < 3): return None
    stat_col = stat_map.get(prop["prop_type"])
    vl = df_last[stat_col].astype(float).tolist() if (stat_col and stat_col in df_last.columns) else []
    vc = df_curr[stat_col].astype(float).tolist() if (stat_col and stat_col in df_curr.columns) else []
    n_curr = float(len(vc)); w_cur = blend_weight(n_curr, n0=PRIOR_GAMES_STRENGTH, continuity=TEAM_CONTINUITY_DEFAULT)
    merged: List[float] = []
    rep_last = max(1, int(round((1.0 - w_cur) * 10))); rep_curr = max(1, int(round(w_cur * 10)))
    for v in vl: merged.extend([v]*rep_last)
    for v in vc: merged.extend([v]*rep_curr)
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
        mu_ml = MODEL.predict(prop["prop_type"], feats_row)
        sigma_ml = MODEL_RMSE.get(prop["prop_type"])

    # Inverse-variance ensemble
    if mu_ml is not None and sigma_ml and sigma_ml > 0:
        s_ewma = 1.0 / (max(1e-6, sigma_ewma) ** 2)
        s_ml   = 1.0 / (max(1e-6, sigma_ml) ** 2)
        projection = float((s_ewma * mu_ewma + s_ml * mu_ml) / (s_ewma + s_ml))
        std_dev    = float(math.sqrt(1.0 / (s_ewma + s_ml)))
    else:
        projection = mu_ewma
        std_dev    = sigma_ewma

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
    if elg <= -0.0005: return None

    stake = BANKROLL * f
    if stake < MIN_KELLY_STAKE: return None
    potential_profit = stake * (dec - 1.0)
    ev_dollars = (p_mean * potential_profit) - ((1.0 - p_mean) * stake)
    ev_pct = (ev_dollars / max(1e-9, stake)) * 100.0

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
    return prop

def analyze_game_bet(prop: dict) -> Optional[dict]:
    if prop.get("odds") is not None and (prop["odds"] < MIN_ODDS or prop["odds"] > MAX_ODDS): return None
    p_hat = implied_prob_from_american(prop["odds"])
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
    if elg <= -0.0005: return None
    stake = BANKROLL * f
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

# ========= MAIN =========
def run_analysis():
    start = time.monotonic()
    print("=" * 72)
    print("RIQ MEEPING MACHINE 🚀 — Unified Analyzer (FanDuel via SGO + ML Ensemble)")
    print("=" * 72)
    print(f"Season: {SEASON} | Stats: prior={STATS_SEASON} | Bankroll: ${BANKROLL:.2f}")
    print(f"Odds Range: {MIN_ODDS} to {MAX_ODDS} | Ranking: ELG + dynamic Kelly")
    print(f"FAST_MODE: {'ON' if FAST_MODE else 'OFF'} | Time Budget: {RUN_TIME_BUDGET_SEC}s")
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

    print("🎲 Fetching FanDuel odds and props from SportsGameOdds...")
    contexts = {g["id"]: get_matchup_context(g) for g in games}
    fd_props, fd_warns = sgo_fanduel_props_for_games(games)
    for w in fd_warns:
        if DEBUG_MODE: print(f"[SGO] {w}")

    # Optional: per-game counts
    by_game = defaultdict(int)
    for p in fd_props:
        by_game[p["game_id"]] += 1
    for g in games:
        gid = g["id"]; cnt = by_game.get(gid, 0)
        game_date_str = str(g.get("date",""))[:16]
        print(f"   ✓ {game_date_str} — {g['teams']['home']['name']} vs {g['teams']['away']['name']}: {cnt} FanDuel markets")

    player_props = [p for p in fd_props if p["prop_type"] in ["points","assists","rebounds","threes"]]
    game_bets   = [p for p in fd_props if p["prop_type"] in ["moneyline","spread"]]

    print(f"\n   Total props: {len(fd_props)} | Player: {len(player_props)} | Game: {len(game_bets)}\n")
    if DEBUG_MODE and len(fd_props) == 0:
        print("   [DEBUG] No props found. Check expand/nested markets and bookmaker naming; printing diagnostics above.")

    print(f"🔍 {random.choice(MEEP_MESSAGES)}...")

    analyzed = []
    for idx, prop in enumerate(player_props + game_bets, 1):
        if time.monotonic() - start > RUN_TIME_BUDGET_SEC:
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
    if not analyzed:
        print("❌ No props passed ELG gates"); return

    groups = top_by_category(analyzed, k="elg")
    print("=" * 72); print(f"TOP {TOP_PER_CATEGORY} PER CATEGORY (by ELG)"); print("=" * 72)
    for key, label in CATEGORIES:
        bucket = groups.get(key, [])
        if not bucket: continue
        print(f"\n{label}\n" + "-" * len(label))
        for i, p in enumerate(bucket, 1):
            conf = "🟢" if p["win_prob"] >= 65 else ("🟡" if p["win_prob"] >= 55 else "🟠")
            score_val = p.get("elg", p.get("composite_score", 0.0))
            date_out = str(p.get("game_date",""))[:16]
            print(f"{conf} #{i:2d} | {p['player']:<25s} | {label:<8s} | ELG: {score_val:.6f}")
            print(f"     Game: {p['game']} | Date: {date_out}")
            if key in ["points","assists","rebounds","threes"]:
                print(f"     Line: {p['line']:<6.1f} | Proj: {p['projection']:<6.2f} | Δ: {p['disparity']:+.2f} | σ: {p['std_dev']:.2f}")
                print(f"     🏀 Pace: {p.get('pace_factor',1.0):.3f}x | 🛡️ Defense: {p.get('defense_factor',1.0):.3f}x")
            print(f"     Pick: {p['pick'].upper() if key!='moneyline' else p['pick']:<6s} | Odds: {p['odds']:+d}")
            print(f"     Kelly: {p['kelly_pct']:.2f}% | Stake: ${p['stake']:.2f} | Profit: ${p['potential_profit']:.2f}")
            print(f"     EV: {p['ev']:+.2f}% | Win Prob: {p['win_prob']:.1f}%")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S"); output_file = f"prop_analysis_{timestamp}.json"
    groups_json = {key: groups.get(key, []) for key,_ in CATEGORIES}
    with open(output_file, "w") as f:
        json.dump({"timestamp": timestamp, "season": SEASON, "stats_season": STATS_SEASON, "bankroll": BANKROLL, "top_by_category": groups_json}, f, indent=2)
    print(f"\n✅ Results saved to: {output_file}\n🎉 Meep complete!")

if __name__ == "__main__":
    try: run_analysis()
    except KeyboardInterrupt: print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}"); import traceback; traceback.print_exc()