import requests
import pandas as pd
import numpy as np
import datetime
import json
import time
import os
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import threading

# ============================================
# API ENDPOINT STRUCTURE
# ============================================
# Team Stats:       /statistics?league=12&season=2025-2026&team={team_id}
# Players:          /players?search={name}
# Player Stats:     /players/statistics?season=2025-2026&player={player_id}
# Odds:             /odds?game={game_id}&bookmaker={bookmaker_id}
# Games:            /games?league={league_id}&season={season}
# NBA Team IDs: 132-161 (alphabetical)
# ============================================

# ============================================
# CONFIGURATION
# ============================================
API_KEY = os.getenv("API_SPORTS_KEY", "")
BASE_URL = "https://v1.basketball.api-sports.io"
HEADERS = {
    "x-apisports-key": API_KEY
}

# Settings
LEAGUE_ID = 12
SEASON = "2025-2026"  # Current season for games/odds
STATS_SEASON = "2024-2025"  # Last season for player/team stats
BOOKMAKER_ID = 4
TOP_PROPS = 15
LOOKBACK_GAMES = 10
KELLY_FRACTION = 0.25
BANKROLL = 100.0
MIN_CONFIDENCE = 0.40  # Lowered to 40%
MIN_KELLY_STAKE = 0.01  # Allow bets as small as 1 cent
MIN_GAMES_REQUIRED = 1  # Only need 1 game of data
DEBUG_MODE = True  # Enable debug logging
MAX_WORKERS = 8  # Parallel API calls

# Data persistence
WEIGHTS_FILE = "prop_weights.pkl"
RESULTS_FILE = "prop_results.pkl"
CACHE_FILE = "player_cache.pkl"

# Stat mapping
stat_map = {
    "points": "points",
    "assists": "assists",
    "rebounds": "totReb",
    "threes": "fgm"
}

# Thread-safe lock for cache updates
cache_lock = threading.Lock()

# ============================================
# DATA PERSISTENCE - OPTIMIZED
# ============================================
def load_data(filename, default=None):
    """Load pickled data with error handling"""
    if os.path.exists(filename):
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except (pickle.PickleError, EOFError):
            return default if default is not None else {}
    return default if default is not None else {}


def save_data(filename, data):
    """Save data to pickle file atomically"""
    temp_file = f"{filename}.tmp"
    with open(temp_file, "wb") as f:
        pickle.dump(data, f)
    os.replace(temp_file, filename)


# Load global data structures
prop_weights = load_data(WEIGHTS_FILE, defaultdict(lambda: 1.0))
prop_results = load_data(RESULTS_FILE, defaultdict(list))
player_cache = load_data(CACHE_FILE, {})


# ============================================
# OPTIMIZED CACHING WITH TTL
# ============================================
class CacheEntry:
    """Cache entry with timestamp for TTL support"""
    __slots__ = ('data', 'timestamp')

    def __init__(self, data):
        self.data = data
        self.timestamp = datetime.datetime.now()

    def is_valid(self, ttl_seconds):
        """Check if cache entry is still valid"""
        return (datetime.datetime.now() - self.timestamp).total_seconds() < ttl_seconds


# Thread-safe cache wrapper
class ThreadSafeCache:
    """Thread-safe cache with TTL support"""
    def __init__(self):
        self._cache = {}
        self._lock = threading.Lock()

    def get(self, key, ttl_seconds=3600):
        """Get cached value if valid"""
        with self._lock:
            entry = self._cache.get(key)
            if entry and entry.is_valid(ttl_seconds):
                return entry.data
            return None

    def set(self, key, value):
        """Set cache value"""
        with self._lock:
            self._cache[key] = CacheEntry(value)

    def clear_expired(self, ttl_seconds=3600):
        """Clear expired entries"""
        with self._lock:
            now = datetime.datetime.now()
            self._cache = {
                k: v for k, v in self._cache.items()
                if (now - v.timestamp).total_seconds() < ttl_seconds
            }


# Global cache instances
api_cache = ThreadSafeCache()
stats_cache = ThreadSafeCache()


# ============================================
# OPTIMIZED API FUNCTIONS
# ============================================
def fetch_json(endpoint: str, params: dict = None, retries: int = 3,
               cache_ttl: int = 600) -> Optional[dict]:
    """Fetch JSON with caching and retry logic"""
    # Create cache key
    cache_key = f"{endpoint}:{json.dumps(params, sort_keys=True)}"

    # Check cache first
    cached = api_cache.get(cache_key, cache_ttl)
    if cached is not None:
        return cached

    url = f"{BASE_URL}{endpoint}"

    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                api_cache.set(cache_key, data)
                return data
            elif response.status_code == 429:
                wait_time = min(2 ** attempt, 8)  # Cap at 8 seconds
                if DEBUG_MODE:
                    print(f"   ⚠️ Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                if DEBUG_MODE and attempt == 0:
                    print(f"   ❌ Error {response.status_code}")
                return None

        except Exception as e:
            if DEBUG_MODE and attempt == retries - 1:
                print(f"   ⚠️ Request failed: {e}")
            if attempt < retries - 1:
                time.sleep(1)

    return None


def get_upcoming_games(days_ahead: int = 3) -> List[dict]:
    """Fetch upcoming games with caching"""
    params = {
        "league": LEAGUE_ID,
        "season": SEASON,
        "timezone": "America/Chicago"
    }

    data = fetch_json("/games", params=params, cache_ttl=1800)  # 30 min cache
    if not data or "response" not in data:
        return []

    from datetime import timezone
    now = datetime.datetime.now(timezone.utc)
    future_cutoff = now + datetime.timedelta(days=days_ahead)

    upcoming = []
    for game in data["response"]:
        try:
            game_date_str = game["date"].replace("Z", "+00:00")
            game_date = datetime.datetime.fromisoformat(game_date_str)

            if game_date.tzinfo is None:
                game_date = game_date.replace(tzinfo=timezone.utc)

            if now <= game_date <= future_cutoff:
                upcoming.append(game)
        except (ValueError, KeyError):
            continue

    return upcoming


def get_game_odds(game_id: int) -> Optional[dict]:
    """Fetch game odds with caching"""
    params = {
        "game": game_id,
        "bookmaker": BOOKMAKER_ID
    }

    data = fetch_json("/odds", params=params, cache_ttl=300)  # 5 min cache
    if not data or "response" not in data or len(data["response"]) == 0:
        return None

    return data["response"][0]


# ============================================
# PARALLEL API FETCHING
# ============================================
def batch_fetch_game_odds(game_ids: List[int], max_workers: int = MAX_WORKERS) -> Dict[int, dict]:
    """Fetch odds for multiple games in parallel"""
    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_game = {
            executor.submit(get_game_odds, game_id): game_id
            for game_id in game_ids
        }

        for future in as_completed(future_to_game):
            game_id = future_to_game[future]
            try:
                odds = future.result()
                if odds:
                    results[game_id] = odds
            except Exception as e:
                if DEBUG_MODE:
                    print(f"   ⚠️ Failed to fetch odds for game {game_id}: {e}")

    return results


def batch_fetch_team_stats(team_ids: List[int], max_workers: int = MAX_WORKERS) -> Dict[int, dict]:
    """Fetch team stats in parallel"""
    results = {}

    def fetch_team_stat(team_id):
        return team_id, get_team_stats(team_id)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_team_stat, tid) for tid in team_ids]

        for future in as_completed(futures):
            try:
                team_id, stats = future.result()
                results[team_id] = stats
            except Exception as e:
                if DEBUG_MODE:
                    print(f"   ⚠️ Failed to fetch team stats: {e}")

    return results


def batch_fetch_player_stats(player_names: List[str],
                             max_workers: int = MAX_WORKERS) -> Dict[str, pd.DataFrame]:
    """Fetch player stats in parallel"""
    results = {}

    def fetch_player_stat(player_name):
        return player_name, get_player_recent_stats(player_name, LOOKBACK_GAMES)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_player_stat, name) for name in player_names]

        for future in as_completed(futures):
            try:
                player_name, stats = future.result()
                results[player_name] = stats
            except Exception as e:
                if DEBUG_MODE:
                    print(f"   ⚠️ Failed to fetch player stats: {e}")

    return results


# ============================================
# OPTIMIZED PLAYER STATS
# ============================================
def get_player_recent_stats(player_name: str, num_games: int = LOOKBACK_GAMES) -> pd.DataFrame:
    """Fetch player stats with improved caching"""
    cache_key = f"{player_name}_{num_games}"

    # Check thread-safe cache
    cached = stats_cache.get(cache_key, 3600)
    if cached is not None:
        if DEBUG_MODE:
            print(f"      📦 Cache hit: {player_name} ({len(cached)} games)")
        return cached

    if DEBUG_MODE:
        print(f"      🔍 Searching API for: {player_name}")

    # Fetch player ID
    params = {"search": player_name}
    data = fetch_json("/players", params=params, cache_ttl=86400)  # 24hr cache for player IDs

    if not data or "response" not in data or len(data["response"]) == 0:
        if DEBUG_MODE:
            print(f"      ❌ Player not found: {player_name}")
        return pd.DataFrame()

    player_id = data["response"][0].get("id")

    if not player_id:
        if DEBUG_MODE:
            print(f"      ❌ No player ID for: {player_name}")
        return pd.DataFrame()

    if DEBUG_MODE:
        print(f"      ✓ Found player ID: {player_id}")

    # Pre-allocate list for better performance
    all_games_stats = []

    # Fetch from both seasons
    for season, season_name in [(SEASON, "current"), (STATS_SEASON, "last")]:
        if len(all_games_stats) >= num_games:
            break

        if DEBUG_MODE:
            print(f"      📅 Fetching stats from {season} season ({season_name})")

        params_season = {
            "season": season,
            "player": player_id
        }

        stats_data = fetch_json("/players/statistics", params=params_season, cache_ttl=1800)

        if stats_data and "response" in stats_data:
            if DEBUG_MODE:
                print(f"      📊 {season_name.capitalize()} season response: {len(stats_data['response'])} games")

            needed = num_games - len(all_games_stats)
            for game_entry in stats_data["response"][:needed]:
                if not isinstance(game_entry, dict):
                    continue

                stats = game_entry.get("statistics", game_entry)

                if isinstance(stats, list) and len(stats) > 0:
                    stats = stats[0]

                try:
                    all_games_stats.append({
                        "points": float(stats.get("points", 0) or 0),
                        "assists": float(stats.get("assists", 0) or 0),
                        "totReb": float(stats.get("totReb", 0) or 0),
                        "fgm": float(stats.get("tpm", 0) or 0)
                    })
                except (ValueError, TypeError):
                    continue

    # Create DataFrame in one operation
    df = pd.DataFrame(all_games_stats[:num_games])

    if DEBUG_MODE and len(df) > 0:
        print(f"      ✅ Parsed {len(df)} total games for {player_name}")
        # Use vectorized operations for averages
        print(f"         Avg: PTS={df['points'].mean():.1f}, AST={df['assists'].mean():.1f}, REB={df['totReb'].mean():.1f}")

    # Cache the result
    stats_cache.set(cache_key, df)

    return df


def get_team_stats(team_id: int) -> dict:
    """Fetch team stats with caching"""
    cache_key = f"team_{team_id}"

    cached = stats_cache.get(cache_key, 86400)  # 24hr cache
    if cached is not None:
        return cached

    params = {
        "league": LEAGUE_ID,
        "season": STATS_SEASON,
        "team": team_id
    }

    if DEBUG_MODE:
        print(f"      📅 Fetching team {team_id} stats from {STATS_SEASON}")

    data = fetch_json("/statistics", params=params, cache_ttl=86400)

    team_stats = {
        "pace": 1.0,
        "offensive_rating": 110.0,
        "defensive_rating": 110.0,
        "points_per_game": 110.0,
        "opp_points_per_game": 110.0,
        "games_played": 0
    }

    if data and "response" in data and len(data["response"]) > 0:
        response = data["response"]

        # Vectorized approach for aggregation
        points_for = []
        points_against = []

        for game_stat in response:
            try:
                team_points = float(game_stat.get("points", 0) or 0)
                opp_points = float(game_stat.get("opponent", {}).get("points", 0) or 0) if "opponent" in game_stat else 0

                if team_points > 0:
                    points_for.append(team_points)
                if opp_points > 0:
                    points_against.append(opp_points)
            except (ValueError, TypeError, KeyError):
                continue

        if points_for:
            ppg = np.mean(points_for)
            opp_ppg = np.mean(points_against) if points_against else 110.0
            estimated_pace = ((ppg + opp_ppg) / 2) / 110.0

            team_stats = {
                "pace": np.clip(estimated_pace, 0.85, 1.15),
                "offensive_rating": ppg,
                "defensive_rating": opp_ppg,
                "points_per_game": ppg,
                "opp_points_per_game": opp_ppg,
                "games_played": len(points_for)
            }

    stats_cache.set(cache_key, team_stats)

    return team_stats


def get_matchup_context(game_info: dict) -> dict:
    """Get matchup context with default values"""
    home_team_id = game_info.get("teams", {}).get("home", {}).get("id")
    away_team_id = game_info.get("teams", {}).get("away", {}).get("id")

    if not home_team_id or not away_team_id:
        return {"pace": 1.0, "offensive_adjustment": 1.0, "defensive_adjustment": 1.0}

    home_stats = get_team_stats(home_team_id)
    away_stats = get_team_stats(away_team_id)

    combined_pace = (home_stats["pace"] + away_stats["pace"]) * 0.5
    avg_def_rating = 110.0

    home_def_factor = avg_def_rating / max(home_stats["defensive_rating"], 90)
    away_def_factor = avg_def_rating / max(away_stats["defensive_rating"], 90)

    return {
        "pace": combined_pace,
        "offensive_adjustment": 1.0,
        "home_defensive_factor": home_def_factor,
        "away_defensive_factor": away_def_factor,
        "home_pace": home_stats["pace"],
        "away_pace": away_stats["pace"],
        "home_ppg": home_stats["points_per_game"],
        "away_ppg": away_stats["points_per_game"],
        "home_opp_ppg": home_stats["opp_points_per_game"],
        "away_opp_ppg": away_stats["opp_points_per_game"]
    }


# ============================================
# OPTIMIZED STATISTICAL CALCULATIONS
# ============================================
@lru_cache(maxsize=1000)
def american_to_decimal_cached(odds: int) -> float:
    """Cached odds conversion"""
    if odds > 0:
        return (odds / 100) + 1
    else:
        return (100 / abs(odds)) + 1


def american_to_decimal(odds) -> float:
    """Convert American odds to decimal"""
    if isinstance(odds, str):
        odds = float(odds)

    if isinstance(odds, float):
        if 1.0 <= odds <= 100.0:
            return odds

    return american_to_decimal_cached(int(odds))


# Pre-compute norm_cdf lookup table for common values
NORM_CDF_CACHE = {}
def norm_cdf(x):
    """Normal CDF with caching"""
    # Round to 3 decimals for cache key
    cache_key = round(x, 3)
    if cache_key in NORM_CDF_CACHE:
        return NORM_CDF_CACHE[cache_key]

    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    sign = 1 if x >= 0 else -1
    x_abs = abs(x) / np.sqrt(2.0)

    t = 1.0 / (1.0 + p * x_abs)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x_abs * x_abs)

    result = 0.5 * (1.0 + sign * y)

    # Cache result (limit cache size)
    if len(NORM_CDF_CACHE) < 10000:
        NORM_CDF_CACHE[cache_key] = result

    return result


def calculate_player_projection(player_stats: pd.DataFrame, prop_type: str,
                                team_context: dict, opponent_defense: float = 1.0) -> Tuple[float, float]:
    """Optimized player projection calculation"""
    if player_stats.empty:
        return 0.0, 0.0

    stat_col = stat_map.get(prop_type)
    if not stat_col or stat_col not in player_stats.columns:
        return 0.0, 0.0

    # Vectorized operations
    values = player_stats[stat_col].values.astype(float)
    n = len(values)

    # Pre-compute weights
    weights = np.exp(np.linspace(0, 1, n))
    base_projection = np.average(values, weights=weights)

    std_dev = np.std(values, ddof=1) if n > 1 else base_projection * 0.20

    pace_multiplier = team_context.get("pace", 1.0)

    # Vectorized adjustment calculation
    if prop_type == "points":
        adjustment = pace_multiplier * opponent_defense
    elif prop_type == "assists":
        adjustment = pace_multiplier * (0.7 + 0.3 * opponent_defense)
    elif prop_type == "rebounds":
        adjustment = (0.8 * pace_multiplier + 0.2)
    elif prop_type == "threes":
        adjustment = pace_multiplier * opponent_defense
    else:
        adjustment = 1.0

    adjusted_projection = base_projection * adjustment
    adjusted_std_dev = std_dev * (0.9 + 0.1 * pace_multiplier)

    return adjusted_projection, adjusted_std_dev


def calculate_win_probability(projection: float, line: float, std_dev: float,
                              pick: str = "over") -> float:
    """Optimized win probability calculation"""
    if std_dev == 0:
        std_dev = projection * 0.20

    z_score = (projection - line) / std_dev

    if pick == "over":
        win_prob = 1 - norm_cdf(z_score)
    else:
        win_prob = norm_cdf(z_score)

    return np.clip(win_prob, 0.25, 0.90)


def calculate_kelly_stake(win_prob: float, odds: int, bankroll: float,
                         fraction: float = KELLY_FRACTION) -> Tuple[float, float]:
    """Kelly criterion calculation"""
    decimal_odds = american_to_decimal(odds)
    b = decimal_odds - 1
    p = win_prob
    q = 1 - p

    kelly = (b * p - q) / b
    fractional_kelly = max(0, kelly * fraction)
    stake = bankroll * fractional_kelly

    return fractional_kelly * 100, stake


def calculate_expected_value(win_prob: float, odds: int, stake: float) -> float:
    """Expected value calculation"""
    decimal_odds = american_to_decimal(odds)
    profit = stake * (decimal_odds - 1)
    loss_prob = 1 - win_prob

    ev = (win_prob * profit) - (loss_prob * stake)
    return (ev / stake) * 100 if stake > 0 else 0


# ============================================
# ADAPTIVE LEARNING
# ============================================
def update_prop_weights(prop_id: str, actual_result: bool, predicted_prob: float):
    """Update weights based on results"""
    prop_results[prop_id].append({
        "result": actual_result,
        "predicted": predicted_prob,
        "timestamp": datetime.datetime.now()
    })

    results = prop_results[prop_id]
    if len(results) >= 5:
        correct = sum(1 for r in results[-20:] if r["result"])
        accuracy = correct / len(results[-20:])

        if accuracy > 0.60:
            prop_weights[prop_id] *= 1.05
        elif accuracy < 0.45:
            prop_weights[prop_id] *= 0.95

    save_data(WEIGHTS_FILE, dict(prop_weights))
    save_data(RESULTS_FILE, dict(prop_results))


def get_prop_confidence_multiplier(prop_id: str) -> float:
    """Get confidence multiplier for prop"""
    return prop_weights.get(prop_id, 1.0)


# ============================================
# PROP ANALYSIS - OPTIMIZED
# ============================================
def extract_props_from_odds(odds_data: dict, game_info: dict) -> List[dict]:
    """Extract props from odds data"""
    if not odds_data or "bookmakers" not in odds_data:
        return []

    bookmakers = odds_data.get("bookmakers", [])
    if not bookmakers:
        return []

    props = []
    bookmaker = bookmakers[0]

    # Use set for faster lookups
    ALLOWED_BET_TYPES = frozenset({
        "moneyline", "money line", "match winner",
        "spread", "point spread", "handicap",
        "totals", "total", "over/under",
        "points", "point", "player points",
        "assists", "assist", "player assists",
        "rebounds", "rebound", "player rebounds", "total rebounds",
        "threes", "three", "3-point", "3-pointers", "player threes"
    })

    game_id = game_info["id"]
    game_name = f"{game_info['teams']['home']['name']} vs {game_info['teams']['away']['name']}"
    game_date = game_info["date"]

    for bet in bookmaker.get("bets", []):
        bet_name = bet.get("name", "").lower()

        if not any(allowed in bet_name for allowed in ALLOWED_BET_TYPES):
            continue

        is_game_bet = any(x in bet_name for x in ["moneyline", "money line", "match winner", "spread", "point spread", "handicap", "totals", "total", "over/under"])

        if is_game_bet:
            bet_type = None
            if any(x in bet_name for x in ["moneyline", "money line", "match winner"]):
                bet_type = "moneyline"
            elif any(x in bet_name for x in ["spread", "point spread", "handicap"]):
                bet_type = "spread"
            elif any(x in bet_name for x in ["totals", "total", "over/under"]):
                bet_type = "game_total"

            if bet_type:
                for value in bet.get("values", []):
                    prop_text = value.get("value", "")
                    odds_value = value.get("odd", -110)

                    if isinstance(odds_value, str):
                        odds_value = float(odds_value)

                    if isinstance(odds_value, float) and 1.0 <= odds_value <= 100.0:
                        if odds_value >= 2.0:
                            odds = int((odds_value - 1) * 100)
                        else:
                            odds = int(-100 / (odds_value - 1))
                    else:
                        odds = int(odds_value)

                    prop_id = f"{game_id}_{bet_type}_{prop_text}".replace(" ", "_")

                    props.append({
                        "prop_id": prop_id,
                        "game_id": game_id,
                        "game": game_name,
                        "game_date": game_date,
                        "player": prop_text,
                        "prop_type": bet_type,
                        "line": 0,
                        "odds": odds,
                        "bookmaker": "DraftKings"
                    })
        else:
            prop_type = None
            if "point" in bet_name and "spread" not in bet_name:
                prop_type = "points"
            elif "assist" in bet_name:
                prop_type = "assists"
            elif "rebound" in bet_name:
                prop_type = "rebounds"
            elif "three" in bet_name or "3-point" in bet_name or "3-pointer" in bet_name:
                prop_type = "threes"

            if not prop_type:
                continue

            for value in bet.get("values", []):
                prop_text = value.get("value", "")
                parts = prop_text.split()

                if len(parts) < 2:
                    continue

                try:
                    line = float(parts[-1])
                    player_name = " ".join(parts[:-1])
                except ValueError:
                    continue

                odds_value = value.get("odd", -110)

                if isinstance(odds_value, str):
                    odds_value = float(odds_value)

                if isinstance(odds_value, float) and 1.0 <= odds_value <= 100.0:
                    if odds_value >= 2.0:
                        odds = int((odds_value - 1) * 100)
                    else:
                        odds = int(-100 / (odds_value - 1))
                else:
                    odds = int(odds_value)

                prop_id = f"{game_id}_{player_name}_{prop_type}".replace(" ", "_")

                props.append({
                    "prop_id": prop_id,
                    "game_id": game_id,
                    "game": game_name,
                    "game_date": game_date,
                    "player": player_name,
                    "prop_type": prop_type,
                    "line": line,
                    "odds": odds,
                    "bookmaker": "DraftKings"
                })

    return props


def analyze_prop(prop: dict, matchup_context: dict = None, player_stats_cache: dict = None) -> Optional[dict]:
    """Analyze a single prop - optimized with stats cache"""
    ALLOWED_PROPS = frozenset(["points", "assists", "rebounds", "threes", "moneyline", "spread", "game_total"])
    if prop["prop_type"] not in ALLOWED_PROPS:
        return None

    if matchup_context is None:
        matchup_context = {"pace": 1.0, "home_defensive_factor": 1.0, "away_defensive_factor": 1.0}

    # Handle game-level bets
    if prop["prop_type"] in ["moneyline", "spread", "game_total"]:
        odds = prop["odds"]
        if odds < 0:
            implied_prob = abs(odds) / (abs(odds) + 100)
        else:
            implied_prob = 100 / (odds + 100)

        adjusted_prob = implied_prob * 1.02

        if adjusted_prob < MIN_CONFIDENCE:
            return None

        kelly_pct, stake = calculate_kelly_stake(adjusted_prob, odds, BANKROLL)

        if stake < MIN_KELLY_STAKE:
            return None

        ev = calculate_expected_value(adjusted_prob, odds, stake)
        decimal_odds = american_to_decimal(odds)
        potential_profit = stake * (decimal_odds - 1)

        composite_score = (
            (adjusted_prob * 100 * 0.60) +
            (kelly_pct * 0.40)
        )

        prop.update({
            "projection": prop["line"],
            "std_dev": 0,
            "disparity": 0,
            "pick": prop.get("player", "N/A"),
            "edge": 0,
            "risk_adjusted": 0,
            "trend": 0,
            "win_prob": round(adjusted_prob * 100, 2),
            "kelly_pct": round(kelly_pct, 2),
            "stake": round(stake, 2),
            "potential_profit": round(potential_profit, 2),
            "ev": round(ev, 2),
            "confidence_mult": 1.0,
            "roi": round((potential_profit / stake) * 100, 2) if stake > 0 else 0,
            "composite_score": round(composite_score, 2),
            "games_analyzed": 0,
            "pace_factor": round(matchup_context.get("pace", 1.0), 3),
            "defense_factor": 1.0
        })

        return prop

    # Handle player props - use cache if provided
    if player_stats_cache and prop["player"] in player_stats_cache:
        player_stats = player_stats_cache[prop["player"]]
    else:
        player_stats = get_player_recent_stats(prop["player"], LOOKBACK_GAMES)

    if player_stats.empty or len(player_stats) < MIN_GAMES_REQUIRED:
        return None

    opponent_defense = (matchup_context.get("home_defensive_factor", 1.0) +
                       matchup_context.get("away_defensive_factor", 1.0)) / 2

    projection, std_dev = calculate_player_projection(
        player_stats,
        prop["prop_type"],
        matchup_context,
        opponent_defense
    )

    if projection == 0 or std_dev == 0:
        return None

    disparity = projection - prop["line"]
    edge = (disparity / prop["line"]) * 100
    pick = "over" if disparity > 0 else "under"
    risk_adjusted = disparity / std_dev if std_dev > 0 else 0
    win_prob = calculate_win_probability(projection, prop["line"], std_dev, pick)

    confidence_mult = get_prop_confidence_multiplier(prop["prop_id"])
    adjusted_prob = min(0.90, win_prob * confidence_mult)

    if adjusted_prob < MIN_CONFIDENCE:
        return None

    kelly_pct, stake = calculate_kelly_stake(adjusted_prob, prop["odds"], BANKROLL)

    if stake < MIN_KELLY_STAKE:
        return None

    ev = calculate_expected_value(adjusted_prob, prop["odds"], stake)
    decimal_odds = american_to_decimal(prop["odds"])
    potential_profit = stake * (decimal_odds - 1)

    # Calculate trend using vectorized operations
    stat_col = stat_map.get(prop["prop_type"])
    trend = 0
    if stat_col and stat_col in player_stats.columns:
        values = player_stats[stat_col].values
        if len(values) >= 7:
            avg_last3 = values[:3].mean()
            avg_last7 = values[:7].mean()
            trend = ((avg_last3 - avg_last7) / avg_last7) * 100 if avg_last7 > 0 else 0

    composite_score = (
        (adjusted_prob * 100 * 0.50) +
        (kelly_pct * 0.25) +
        (max(0, ev) * 0.15) +
        (max(0, risk_adjusted * 5) * 0.10)
    )

    prop.update({
        "projection": round(projection, 2),
        "std_dev": round(std_dev, 2),
        "disparity": round(disparity, 2),
        "pick": pick,
        "edge": round(edge, 2),
        "risk_adjusted": round(risk_adjusted, 2),
        "trend": round(trend, 2),
        "win_prob": round(adjusted_prob * 100, 2),
        "kelly_pct": round(kelly_pct, 2),
        "stake": round(stake, 2),
        "potential_profit": round(potential_profit, 2),
        "ev": round(ev, 2),
        "confidence_mult": round(confidence_mult, 3),
        "roi": round((potential_profit / stake) * 100, 2) if stake > 0 else 0,
        "composite_score": round(composite_score, 2),
        "games_analyzed": len(player_stats),
        "pace_factor": round(matchup_context.get("pace", 1.0), 3),
        "defense_factor": round(opponent_defense, 3)
    })

    return prop


# ============================================
# MAIN EXECUTION - OPTIMIZED
# ============================================
def run_analysis():
    """Run the analysis with optimizations"""
    print("=" * 70)
    print("NBA PROP ANALYZER - OPTIMIZED VERSION")
    print("=" * 70)
    print(f"Bankroll: ${BANKROLL:,.2f}")
    print(f"Kelly Fraction: {KELLY_FRACTION:.2%}")
    print(f"Min Confidence: {MIN_CONFIDENCE:.1%}")
    print(f"Min Kelly Stake: ${MIN_KELLY_STAKE}")
    print(f"Min Games Required: {MIN_GAMES_REQUIRED}")
    print(f"Max Parallel Workers: {MAX_WORKERS}")
    print(f"Debug Mode: {'ON' if DEBUG_MODE else 'OFF'}")
    print("=" * 70)
    print()

    print("📅 Fetching upcoming games (next 3 days)...")
    games = get_upcoming_games(days_ahead=3)
    print(f"   Found {len(games)} games in next 3 days\n")

    if not games:
        print("❌ No upcoming games found")
        return

    # Limit games to process
    games_to_process = games[:10]

    print("🎲 Fetching odds and props in parallel...")
    print("   ✅ ANALYZING: Points, Assists, Rebounds, 3PM, Moneyline, Spread, Totals")

    # OPTIMIZATION: Batch fetch all game odds in parallel
    game_ids = [game["id"] for game in games_to_process]
    odds_map = batch_fetch_game_odds(game_ids)

    # OPTIMIZATION: Batch fetch all team stats in parallel
    team_ids = set()
    for game in games_to_process:
        home_id = game.get("teams", {}).get("home", {}).get("id")
        away_id = game.get("teams", {}).get("away", {}).get("id")
        if home_id:
            team_ids.add(home_id)
        if away_id:
            team_ids.add(away_id)

    print(f"   📊 Fetching stats for {len(team_ids)} teams in parallel...")
    team_stats_map = batch_fetch_team_stats(list(team_ids))

    # Build matchup contexts and extract props
    all_props = []
    game_contexts = {}

    for game in games_to_process:
        matchup_context = get_matchup_context(game)
        game_contexts[game["id"]] = matchup_context

        game_odds = odds_map.get(game["id"])
        if game_odds:
            props = extract_props_from_odds(game_odds, game)
            all_props.extend(props)

            pace_info = f"Pace: {matchup_context['pace']:.2f}x" if matchup_context['pace'] != 1.0 else ""
            print(f"   ✓ {game['teams']['home']['name']} vs {game['teams']['away']['name']}: {len(props)} props {pace_info}")

    player_props = [p for p in all_props if p["prop_type"] in ["points", "assists", "rebounds", "threes"]]
    game_bets = [p for p in all_props if p["prop_type"] in ["moneyline", "spread", "game_total"]]

    print(f"\n   Total props extracted: {len(all_props)}")
    print(f"   Player props (PTS/AST/REB/3PM): {len(player_props)}")
    print(f"   Game bets (ML/Spread/Total): {len(game_bets)}")
    print(f"   Total to analyze: {len(player_props) + len(game_bets)}\n")

    # OPTIMIZATION: Batch fetch all player stats in parallel
    unique_players = list(set(p["player"] for p in player_props))
    print(f"🔍 Fetching stats for {len(unique_players)} unique players in parallel...")
    player_stats_map = batch_fetch_player_stats(unique_players)
    print(f"   ✅ Fetched stats for {len(player_stats_map)} players\n")

    print("🔍 Analyzing props with Kelly Criterion...")
    analyzed_props = []

    # Analyze game bets first (no player stats needed)
    for prop in game_bets:
        matchup_context = game_contexts.get(prop["game_id"], {})
        result = analyze_prop(prop, matchup_context)
        if result:
            analyzed_props.append(result)

    # Analyze player props with cached stats
    for prop in player_props:
        matchup_context = game_contexts.get(prop["game_id"], {})
        result = analyze_prop(prop, matchup_context, player_stats_map)
        if result:
            analyzed_props.append(result)

    print(f"\n   ✅ {len(analyzed_props)} props meet criteria (out of {len(all_props)})\n")

    if not analyzed_props:
        print("❌ No props met minimum thresholds")
        return

    # Sort and get top props
    analyzed_props.sort(key=lambda x: x["composite_score"], reverse=True)
    top_props = analyzed_props[:TOP_PROPS]

    print("=" * 70)
    print(f"TOP {len(top_props)} PROPS (Ranked by Composite Score)")
    print("=" * 70)
    print("Formula: Score = (WinProb×50%) + (Kelly×25%) + (EV×15%) + (RiskAdj×10%)")
    print("=" * 70)
    print()

    # Vectorized calculations for summary
    stakes = np.array([p["stake"] for p in top_props])
    profits = np.array([p["potential_profit"] for p in top_props])
    win_probs = np.array([p["win_prob"] for p in top_props])
    evs = np.array([p["ev"] for p in top_props])

    total_stake = stakes.sum()
    total_potential = profits.sum()
    avg_win_prob = win_probs.mean()
    avg_ev = evs.mean()

    for idx, prop in enumerate(top_props, 1):
        confidence_indicator = "🟢" if prop['win_prob'] >= 65 else "🟡" if prop['win_prob'] >= 55 else "🟠"

        print(f"{confidence_indicator} #{idx:2d} | {prop['player']:<25s} | {prop['prop_type'].upper():<8s} | Score: {prop['composite_score']:.2f}")
        print(f"     Game: {prop['game']}")
        print(f"     ⭐ WIN PROBABILITY: {prop['win_prob']:.1f}% | Confidence: {prop['confidence_mult']:.3f}x")
        print(f"     Line: {prop['line']:<6.1f} | Proj: {prop['projection']:<6.2f} | Disparity: {prop['disparity']:+.2f} | σ: {prop['std_dev']:.2f}")
        print(f"     Pick: {prop['pick'].upper():<6s} | Odds: {prop['odds']:+d}")
        print(f"     🏀 Pace: {prop.get('pace_factor', 1.0):.3f}x | 🛡️ Defense: {prop.get('defense_factor', 1.0):.3f}x")
        print(f"     Edge: {prop['edge']:+.1f}% | Risk-Adj: {prop['risk_adjusted']:+.2f} | Trend: {prop['trend']:+.1f}%")
        print(f"     Kelly: {prop['kelly_pct']:.2f}% | Stake: ${prop['stake']:.2f} | Profit: ${prop['potential_profit']:.2f}")
        print(f"     EV: {prop['ev']:+.2f}% | ROI: {prop['roi']:.1f}%")
        print()

    print("=" * 70)
    print("PORTFOLIO SUMMARY")
    print("=" * 70)
    print(f"Total Bets:           {len(top_props)}")
    print(f"Total Stake:          ${total_stake:.2f} ({total_stake/BANKROLL*100:.1f}% of bankroll)")
    print(f"Total Potential:      ${total_potential:.2f}")
    print(f"Expected Return:      ${total_potential * (avg_win_prob/100):.2f}")
    print(f"⭐ Avg Win Probability: {avg_win_prob:.1f}%")
    print(f"Avg Expected Value:   {avg_ev:+.2f}%")
    print(f"Risk Level:           {'LOW' if total_stake < BANKROLL * 0.15 else 'MODERATE' if total_stake < BANKROLL * 0.30 else 'HIGH'}")
    print()
    print("Confidence Levels:")
    high_conf = np.sum(win_probs >= 65)
    med_conf = np.sum((win_probs >= 55) & (win_probs < 65))
    low_conf = np.sum(win_probs < 55)
    print(f"  🟢 High (65%+):     {high_conf} bets")
    print(f"  🟡 Medium (55-65%): {med_conf} bets")
    print(f"  🟠 Lower (50-55%):  {low_conf} bets")
    print("=" * 70)

    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"prop_analysis_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "bankroll": BANKROLL,
            "kelly_fraction": KELLY_FRACTION,
            "top_props": top_props,
            "summary": {
                "total_bets": len(top_props),
                "total_stake": float(total_stake),
                "total_potential": float(total_potential),
                "avg_win_prob": float(avg_win_prob),
                "avg_ev": float(avg_ev)
            }
        }, f, indent=2)

    print(f"\n✅ Results saved to: {output_file}")

    # Clear expired cache entries
    api_cache.clear_expired(600)
    stats_cache.clear_expired(3600)


if __name__ == "__main__":
    try:
        run_analysis()
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
