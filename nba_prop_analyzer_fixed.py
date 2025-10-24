# RIQ MEEPING MACHINE — Fully Integrated, Fast-Mode, All-in-One
from __future__ import annotations
import requests, pandas as pd, numpy as np, datetime, json, time, os, pickle, random, math
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# ========= RUNTIME / FAST MODE =========
FAST_MODE = True
REQUEST_TIMEOUT = 4 if FAST_MODE else 10
RETRIES = 1 if FAST_MODE else 3
DAYS_TO_FETCH = 3  # Always scan 3 days ahead
MAX_GAMES = 12 if FAST_MODE else 20  # Increased from 6 to see more games
MAX_PLAYER_PROPS_ANALYZE = 50 if FAST_MODE else 200  # Increased from 24
SLEEP_SHORT = 0.05 if FAST_MODE else 0.2
SLEEP_LONG = 0.1 if FAST_MODE else 0.3
RUN_TIME_BUDGET_SEC = 50 if FAST_MODE else 300

# ========= CONFIG =========
# Hardcoded API key (for venv compatibility - can't import from other files)
API_KEY = "4979ac5e1f7ae10b1d6b58f1bba01140"  # Your API-Sports.io key
# Fallback to env var if key is placeholder
if not API_KEY or API_KEY == "YOUR_KEY_HERE":
    API_KEY = os.getenv("API_SPORTS_KEY") or os.getenv("APISPORTS_KEY")
    if not API_KEY:
        raise ValueError("❌ API key not found. Edit line 20 in this file and add your key.")
BASE_URL = "https://v1.basketball.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}
LEAGUE_ID = 12
SEASON = "2025-2026"
STATS_SEASON = "2024-2025"
BOOKMAKER_ID = 4
BANKROLL = 100.0
MIN_KELLY_STAKE = 0.01
DEBUG_MODE = False

# Odds filter
MIN_ODDS = -500
MAX_ODDS = +500

# Early-season blending
PRIOR_GAMES_STRENGTH = 12.0
TEAM_CONTINUITY_DEFAULT = 0.7

# Posterior tightness by market
N_EFF_BY_MARKET = {"PTS": 90.0, "AST": 80.0, "REB": 85.0, "3PM": 70.0, "Moneyline": 45.0, "Spread": 45.0, "DEFAULT": 80.0}

# Category view (Top 5 per)
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

stat_map = {"points": "points", "assists": "assists", "rebounds": "rebounds", "threes": "threepoint_goals"}
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
    q_conservative: float = 0.25  # Slightly less conservative (was 0.30)
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
    ps = np.clip(np.array(p_samples), 1e-6, 1.0 - 1e-6)
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

# ========= PROP MODELS (distributional tails) =========
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
        return min(1.0 - 1e-4, max(1e-4, p)), z
    mean, r, _ = _fit_nb_params(ints)
    if r == float("inf"):
        lam = mean
        if pick == "over":
            k = math.ceil(line); p = 1.0 - _poisson_cdf(k - 1, lam)
        else:
            k = math.floor(line); p = _poisson_cdf(k, lam)
        z_like = (mu - line) / max(1e-6, sigma)
        return min(1.0 - 1e-4, max(1e-4, p)), z_like
    p_nb = mean / (mean + r)
    if pick == "over":
        k = math.ceil(line); p = 1.0 - _nb_cdf(k - 1, r, p_nb)
    else:
        k = math.floor(line); p = _nb_cdf(k, r, p_nb)
    z_like = (mu - line) / max(1e-6, sigma)
    return min(1.0 - 1e-4, max(1e-4, p)), z_like

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
# In-memory PID cache (also persisted in player_cache)
_pid_cache: Dict[str, int] = player_cache.get("__pid_cache__", {})

# ========= API FUNCTIONS =========
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
    # Fast path: local cache
    if name in _pid_cache:
        return _pid_cache[name]
    # Strategy 1: reversed name
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
            # Pick the best match that contains all tokens
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

def get_player_stats_split(player_name: str, max_last: int = 25, max_curr: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

def get_game_odds(game_id: int) -> Optional[dict]:
    data = fetch_json("/odds", params={"game": game_id, "bookmaker": BOOKMAKER_ID})
    if not data or "response" not in data or not data["response"]: return None
    return data["response"][0]

def get_matchup_context(game_info: dict) -> dict:
    home_team_id = game_info.get("teams", {}).get("home", {}).get("id")
    away_team_id = game_info.get("teams", {}).get("away", {}).get("id")
    if not home_team_id or not away_team_id: return {"pace": 1.0, "home_defensive_factor": 1.0, "away_defensive_factor": 1.0}
    # Lightweight team context (avoid heavy loops)
    params = {"league": LEAGUE_ID, "season": STATS_SEASON}
    pace = 1.0; h_def = 1.0; a_def = 1.0
    # Optional: you can enrich with /statistics per team if needed
    return {"pace": pace, "home_defensive_factor": h_def, "away_defensive_factor": a_def}

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

def extract_props_from_odds(odds_data: dict, game_info: dict) -> List[dict]:
    if not odds_data or "bookmakers" not in odds_data: return []
    bookmakers = odds_data.get("bookmakers", [])
    if not bookmakers: return []
    props = []; bookmaker = bookmakers[0]

    ALLOWED = {"moneyline","money line","match winner","spread","point spread","handicap","points","point","player points","assists","assist","player assists","rebounds","rebound","player rebounds","total rebounds","threes","three","3-point","3-pointers","player threes"}
    PER_GAME_LIMIT = 20 if FAST_MODE else 200

    for bet in bookmaker.get("bets", []):
        if len(props) >= PER_GAME_LIMIT:
            break
        bet_name = bet.get("name", "").lower()
        if not any(a in bet_name for a in ALLOWED): continue

        is_game = any(x in bet_name for x in ["moneyline","money line","match winner","spread","point spread","handicap"])
        if is_game:
            bet_type = "moneyline" if any(x in bet_name for x in ["moneyline","money line","match winner"]) else ("spread" if any(x in bet_name for x in ["spread","point spread","handicap"]) else None)
            if not bet_type: continue
            for value in bet.get("values", [])[:6]:
                prop_text = value.get("value", ""); odds_value = value.get("odd", -110)
                try:
                    if isinstance(odds_value, str): odds_value = float(odds_value)
                    if isinstance(odds_value, float) and 1.0 <= odds_value <= 100.0:
                        odds = int((odds_value - 1) * 100) if odds_value >= 2.0 else int(-100 / (odds_value - 1))
                    else:
                        odds = int(odds_value)
                except Exception:
                    continue
                prop_id = f"{game_info['id']}_{bet_type}_{prop_text}".replace(" ", "_")
                props.append({"prop_id": prop_id, "game_id": game_info["id"], "game": f"{game_info['teams']['home']['name']} vs {game_info['teams']['away']['name']}", "game_date": game_info["date"], "player": prop_text, "prop_type": bet_type, "line": 0.0, "odds": odds, "bookmaker": "DraftKings"})
                if len(props) >= PER_GAME_LIMIT: break
        else:
            prop_type = None
            if "point" in bet_name and "spread" not in bet_name: prop_type = "points"
            elif "assist" in bet_name: prop_type = "assists"
            elif "rebound" in bet_name: prop_type = "rebounds"
            elif "three" in bet_name or "3-point" in bet_name or "3-pointer" in bet_name: prop_type = "threes"
            if not prop_type: continue
            for value in bet.get("values", [])[:10]:
                text = value.get("value", ""); parts = text.split()
                if len(parts) < 2: continue
                try:
                    line = float(parts[-1]); player_name = " ".join(parts[:-1])
                except ValueError: continue
                odds_value = value.get("odd", -110)
                try:
                    if isinstance(odds_value, str): odds_value = float(odds_value)
                    if isinstance(odds_value, float) and 1.0 <= odds_value <= 100.0:
                        odds = int((odds_value - 1) * 100) if odds_value >= 2.0 else int(-100 / (odds_value - 1))
                    else:
                        odds = int(odds_value)
                except Exception:
                    continue
                prop_id = f"{game_info['id']}_{player_name}_{prop_type}".replace(" ", "_")
                props.append({"prop_id": prop_id, "game_id": game_info["id"], "game": f"{game_info['teams']['home']['name']} vs {game_info['teams']['away']['name']}", "game_date": game_info["date"], "player": player_name, "prop_type": prop_type, "line": line, "odds": odds, "bookmaker": "DraftKings"})
                if len(props) >= PER_GAME_LIMIT: break
    return props

def _market_key(prop_type: str) -> str:
    m = prop_type.lower()
    return {"points":"PTS","assists":"AST","rebounds":"REB","threes":"3PM","moneyline":"Moneyline","spread":"Spread"}.get(m, "DEFAULT")

def analyze_player_prop(prop: dict, matchup_context: dict) -> Optional[dict]:
    if prop.get("odds") is not None and (prop["odds"] < MIN_ODDS or prop["odds"] > MAX_ODDS): return None
    df_last, df_curr = get_player_stats_split(prop["player"], 25, 5)
    if (df_last.empty and df_curr.empty) or (len(df_last) + len(df_curr) < 3): return None
    stat_col = stat_map.get(prop["prop_type"]);
    vl = df_last[stat_col].astype(float).tolist() if (stat_col and stat_col in df_last.columns) else []
    vc = df_curr[stat_col].astype(float).tolist() if (stat_col and stat_col in df_curr.columns) else []
    n_curr = float(len(vc)); w_cur = blend_weight(n_curr, n0=PRIOR_GAMES_STRENGTH, continuity=TEAM_CONTINUITY_DEFAULT)
    # Replication-based blend (fast and simple)
    merged = []
    rep_last = max(1, int(round((1.0 - w_cur) * 10))); rep_curr = max(1, int(round(w_cur * 10)))
    for v in vl: merged.extend([v]*rep_last)
    for v in vc: merged.extend([v]*rep_curr)
    if not merged: merged = vl or vc

    pace = matchup_context.get("pace", 1.0)
    defense = (matchup_context.get("home_defensive_factor", 1.0) + matchup_context.get("away_defensive_factor", 1.0)) / 2.0
    projection, std_dev = project_stat(merged, prop["prop_type"], pace, defense)
    if projection == 0 or std_dev == 0: return None

    disparity = projection - prop["line"]; pick = "over" if disparity > 0 else "under"
    risk_adj = disparity / std_dev if std_dev > 0 else 0.0
    edge_pct = (disparity / prop["line"]) * 100 if prop["line"] != 0 else 0.0

    p_hat, _ = prop_win_probability(prop["prop_type"], merged, prop["line"], pick, projection, std_dev)
    conf_mult = get_prop_confidence_multiplier(prop["prop_id"])
    n_eff = N_EFF_BY_MARKET.get(_market_key(prop["prop_type"]), N_EFF_BY_MARKET["DEFAULT"])
    n_eff_scaled = max(10.0, min(200.0, n_eff * conf_mult))
    p_samples = sample_beta_posterior(p_hat, n_eff_scaled, n_samples=400 if FAST_MODE else 600)

    dec = american_to_decimal(prop["odds"]); b = dec - 1.0
    equity = load_equity(); dd_scale = drawdown_scale(equity, floor=0.6, window=14)
    kcfg = KellyConfig(q_conservative=0.30, fk_low=0.25, fk_high=0.50, dd_scale=dd_scale)
    f, p_c, _, p_mean = dynamic_fractional_kelly(p_samples, b, kcfg)
    if f <= 0.0: return None
    p_be = 1.0 / (1.0 + b)
    if p_c <= p_be: return None
    elg = risk_adjusted_elg(p_samples, b, f)
    if elg <= 0.0: return None

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
    kcfg = KellyConfig(q_conservative=0.30, fk_low=0.25, fk_high=0.50, dd_scale=dd_scale)
    f, p_c, _, p_mean = dynamic_fractional_kelly(p_samples, b, kcfg)
    if f <= 0.0: return None
    p_be = 1.0 / (1.0 + b)
    if p_c <= p_be: return None
    elg = risk_adjusted_elg(p_samples, b, f)
    if elg <= 0.0: return None
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

def run_analysis():
    start = time.monotonic()
    print("=" * 72)
    print("RIQ MEEPING MACHINE 🚀 — Unified Analyzer")
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

    print("🎲 Fetching odds and props...")
    print("   ✅ Points, Assists, Rebounds, 3PM, Moneyline, Spread")
    print("   ❌ Excluding: Totals, Blocks, Steals, TOs, Minutes")
    all_props = []; contexts = {}
    for g in games[:MAX_GAMES]:
        if time.monotonic() - start > RUN_TIME_BUDGET_SEC:
            print("⏳ Time budget reached while reading odds.")
            break
        ctx = get_matchup_context(g); contexts[g["id"]] = ctx
        od = get_game_odds(g["id"])
        if od:
            props = extract_props_from_odds(od, g)
            all_props.extend(props)
            info = f"Pace: {ctx['pace']:.2f}x" if ctx['pace'] != 1.0 else ""
            print(f"   ✓ {g['teams']['home']['name']} vs {g['teams']['away']['name']}: {len(props)} props {info}")
        time.sleep(SLEEP_LONG)

    player_props = [p for p in all_props if p["prop_type"] in ["points","assists","rebounds","threes"]]
    game_bets = [p for p in all_props if p["prop_type"] in ["moneyline","spread"]]

    # Limit for speed
    if len(player_props) > MAX_PLAYER_PROPS_ANALYZE:
        print(f"⚡ Limiting player props to first {MAX_PLAYER_PROPS_ANALYZE} for speed.")
        player_props = player_props[:MAX_PLAYER_PROPS_ANALYZE]

    print(f"\n   Total props: {len(all_props)} | Player: {len(player_props)} | Game: {len(game_bets)}\n")
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
            print(f"{conf} #{i:2d} | {p['player']:<25s} | {label:<8s} | ELG: {score_val:.6f}")
            print(f"     Game: {p['game']}")
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
