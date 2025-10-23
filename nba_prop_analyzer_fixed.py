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

# ============================================
# API ENDPOINT STRUCTURE - CORRECTED FOR PRO PLAN
# ============================================
# Team Stats:       /statistics?league=12&season=2024-2025&team={team_id}
# Players Search:   /players?search={name}
# Player Stats:     /games/statistics/players?player={player_id}&season={season}  ← CORRECTED!
# Odds:             /odds?game={game_id}&bookmaker={bookmaker_id}
# Games:            /games?league={league_id}&season={season}
# NBA Team IDs: 132-161 (alphabetical)
# ============================================

# ============================================
# CONFIGURATION
# ============================================
API_KEY = "4979ac5e1f7ae10b1d6b58f1bba01140"  # ✅ Updated key
BASE_URL = "https://v1.basketball.api-sports.io"
HEADERS = {
    "x-apisports-key": API_KEY  # ✅ Corrected header format
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

# Odds filtering - only analyze reasonable odds
MIN_ODDS = -500  # Don't analyze heavy favorites (e.g., -999999)
MAX_ODDS = +500  # Don't analyze extreme longshots

# Data persistence
WEIGHTS_FILE = "prop_weights.pkl"
RESULTS_FILE = "prop_results.pkl"
CACHE_FILE = "player_cache.pkl"

# Stat mapping - UPDATED for correct API response structure
stat_map = {
    "points": "points",
    "assists": "assists",
    "rebounds": "rebounds",  # Changed from totReb to rebounds
    "threes": "threepoint_goals"  # Changed from fgm to threepoint_goals
}

# ============================================
# DATA PERSISTENCE
# ============================================
def load_data(filename, default=None):
    if os.path.exists(filename):
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except:
            return default if default is not None else {}
    return default if default is not None else {}


def save_data(filename, data):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


prop_weights = load_data(WEIGHTS_FILE, defaultdict(lambda: 1.0))
prop_results = load_data(RESULTS_FILE, defaultdict(list))
player_cache = load_data(CACHE_FILE, {})


# ============================================
# API FUNCTIONS
# ============================================
def fetch_json(endpoint: str, params: dict = None, retries: int = 3) -> Optional[dict]:
    url = f"{BASE_URL}{endpoint}"

    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=10)

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                wait_time = 2 ** attempt
                if DEBUG_MODE:
                    print(f"   ⚠️ Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                if DEBUG_MODE:
                    print(f"   ❌ Error {response.status_code}: {response.text[:200]}")
                return None

        except Exception as e:
            if DEBUG_MODE:
                print(f"   ⚠️ Request failed (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(1)

    return None


def get_upcoming_games(days_ahead: int = 3) -> List[dict]:
    params = {
        "league": LEAGUE_ID,
        "season": SEASON,
        "timezone": "America/Chicago"
    }

    data = fetch_json("/games", params=params)
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
        except (ValueError, KeyError) as e:
            if DEBUG_MODE:
                print(f"   ⚠️ Could not parse game date: {e}")
            continue

    return upcoming


def get_game_odds(game_id: int) -> Optional[dict]:
    params = {
        "game": game_id,
        "bookmaker": BOOKMAKER_ID
    }

    data = fetch_json("/odds", params=params)
    if not data or "response" not in data or len(data["response"]) == 0:
        return None

    return data["response"][0]


def get_player_recent_stats(player_name: str, num_games: int = LOOKBACK_GAMES) -> pd.DataFrame:
    """
    Fetch player stats using the CORRECT PRO plan endpoint:
    /games/statistics/players?player={player_id}&season={season}
    """
    cache_key = f"{player_name}_{num_games}"

    if cache_key in player_cache:
        cached_time, cached_data = player_cache[cache_key]
        if (datetime.datetime.now() - cached_time).seconds < 3600:
            if DEBUG_MODE:
                print(f"      📦 Cache hit: {player_name} ({len(cached_data)} games)")
            return cached_data

    if DEBUG_MODE:
        print(f"      🔍 Searching API for: {player_name}")

    # Step 1: Search for player to get ID
    # API stores names as "Last First" but odds use "First Last"
    # OPTIMIZED: Try reversed name first (most efficient)

    name_parts = player_name.strip().split()
    player = None
    data = None

    # Strategy 1: Try reversed name first (e.g., "LeBron James" → "James LeBron")
    if len(name_parts) >= 2:
        reversed_name = " ".join(name_parts[::-1])
        if DEBUG_MODE:
            print(f"      🔄 Trying reversed: {reversed_name}")

        params = {"search": reversed_name}
        data = fetch_json("/players", params=params)

        if data and "response" in data and len(data["response"]) > 0:
            player = data["response"][0]
            if DEBUG_MODE:
                print(f"      ✅ Found with reversed name: {player.get('name')} (ID: {player.get('id')})")

    # Strategy 2: Try original name as fallback
    if not player:
        if DEBUG_MODE:
            print(f"      🔄 Trying original: {player_name}")

        params = {"search": player_name}
        data = fetch_json("/players", params=params)

        if data and "response" in data and len(data["response"]) > 0:
            player = data["response"][0]
            if DEBUG_MODE:
                print(f"      ✅ Found with original name: {player.get('name')} (ID: {player.get('id')})")

    # Strategy 3: Final fallback - search by last name only
    if not player and len(name_parts) >= 2:
        last_name = name_parts[-1]
        if DEBUG_MODE:
            print(f"      🔄 Trying last name only: {last_name}")

        params = {"search": last_name}
        data = fetch_json("/players", params=params)

        if data and "response" in data and len(data["response"]) > 0:
            # Filter results to find best match
            best_match = None
            for result in data["response"]:
                api_name = result.get("name", "").lower()
                search_last = last_name.lower()

                if search_last in api_name:
                    best_match = result
                    break

            if not best_match:
                best_match = data["response"][0]

            player = best_match
            if DEBUG_MODE:
                print(f"      ✅ Matched by last name: {player.get('name')} (ID: {player.get('id')})")

    # If still not found, give up
    if not player:
        if DEBUG_MODE:
            print(f"      ❌ Player not found: {player_name}")
        return pd.DataFrame()
    player_id = player.get("id")

    if not player_id:
        if DEBUG_MODE:
            print(f"      ❌ No player ID for: {player_name}")
        return pd.DataFrame()

    if DEBUG_MODE:
        print(f"      ✓ Found player ID: {player_id}")

    # Step 2: Fetch player game stats using CORRECT endpoint
    all_games_stats = []

    # OPTIMIZED: Try LAST season first (2024-2025) since current season just started
    # This ensures we get recent, complete game data instead of sparse early-season games
    if DEBUG_MODE:
        print(f"      📅 Fetching stats from {STATS_SEASON} season (last/most data)")

    params_last = {
        "season": STATS_SEASON,
        "player": player_id
    }

    stats_data_last = fetch_json("/games/statistics/players", params=params_last)

    if stats_data_last and "response" in stats_data_last:
        if DEBUG_MODE:
            print(f"      📊 Last season response: {len(stats_data_last['response'])} games")

        for game_entry in stats_data_last["response"]:
            if isinstance(game_entry, dict):
                try:
                    # Parse the new API response structure
                    points = game_entry.get("points", 0) or 0
                    assists = game_entry.get("assists", 0) or 0

                    # Rebounds - check if it's a dict or int
                    rebounds_data = game_entry.get("rebounds", {})
                    if isinstance(rebounds_data, dict):
                        rebounds = rebounds_data.get("total", 0) or 0
                    else:
                        rebounds = rebounds_data or 0

                    # Three-pointers - get total made
                    threes_data = game_entry.get("threepoint_goals", {})
                    if isinstance(threes_data, dict):
                        threes = threes_data.get("total", 0) or 0
                    else:
                        threes = 0

                    all_games_stats.append({
                        "points": float(points),
                        "assists": float(assists),
                        "rebounds": float(rebounds),
                        "threes": float(threes),
                        "season": STATS_SEASON
                    })
                except (ValueError, TypeError) as e:
                    if DEBUG_MODE:
                        print(f"      ⚠️ Error parsing last season stat: {e}")
                    continue

    # If we need more games, try current season (2025-2026) to supplement
    if len(all_games_stats) < num_games:
        needed = num_games - len(all_games_stats)
        if DEBUG_MODE:
            print(f"      📅 Need {needed} more games, fetching from {SEASON} season (current)")

        params_current = {
            "season": SEASON,
            "player": player_id
        }

        stats_data_current = fetch_json("/games/statistics/players", params=params_current)

    if stats_data_current and "response" in stats_data_current:
        if DEBUG_MODE:
            print(f"      📊 Current season response: {len(stats_data_current['response'])} games")

        for game_entry in stats_data_current["response"]:
            if isinstance(game_entry, dict):
                try:
                    # Parse the new API response structure
                    points = game_entry.get("points", 0) or 0
                    assists = game_entry.get("assists", 0) or 0

                    # Rebounds - check if it's a dict or int
                    rebounds_data = game_entry.get("rebounds", {})
                    if isinstance(rebounds_data, dict):
                        rebounds = rebounds_data.get("total", 0) or 0
                    else:
                        rebounds = rebounds_data or 0

                    # Three-pointers - get total made
                    threes_data = game_entry.get("threepoint_goals", {})
                    if isinstance(threes_data, dict):
                        threes = threes_data.get("total", 0) or 0
                    else:
                        threes = 0

                    all_games_stats.append({
                        "points": float(points),
                        "assists": float(assists),
                        "rebounds": float(rebounds),
                        "threes": float(threes),
                        "season": SEASON
                    })
                except (ValueError, TypeError) as e:
                    if DEBUG_MODE:
                        print(f"      ⚠️ Error parsing current season stat: {e}")
                    continue

    # If we don't have enough games, fetch from last season (2024-2025)
    if len(all_games_stats) < num_games:
        needed = num_games - len(all_games_stats)
        if DEBUG_MODE:
            print(f"      📅 Need {needed} more games, fetching from {STATS_SEASON} season (last year)")

        params_last = {
            "season": STATS_SEASON,
            "player": player_id
        }

        stats_data_last = fetch_json("/games/statistics/players", params=params_last)

        if stats_data_last and "response" in stats_data_last:
            if DEBUG_MODE:
                print(f"      📊 Last season response: {len(stats_data_last['response'])} games")

            for game_entry in stats_data_last["response"][:needed]:
                if isinstance(game_entry, dict):
                    try:
                        points = game_entry.get("points", 0) or 0
                        assists = game_entry.get("assists", 0) or 0

                        rebounds_data = game_entry.get("rebounds", {})
                        if isinstance(rebounds_data, dict):
                            rebounds = rebounds_data.get("total", 0) or 0
                        else:
                            rebounds = rebounds_data or 0

                        threes_data = game_entry.get("threepoint_goals", {})
                        if isinstance(threes_data, dict):
                            threes = threes_data.get("total", 0) or 0
                        else:
                            threes = 0

                        all_games_stats.append({
                            "points": float(points),
                            "assists": float(assists),
                            "rebounds": float(rebounds),
                            "threes": float(threes),
                            "season": STATS_SEASON
                        })
                    except (ValueError, TypeError) as e:
                        if DEBUG_MODE:
                            print(f"      ⚠️ Error parsing last season stat: {e}")
                        continue

    # Take only the number of games we need
    all_games_stats = all_games_stats[:num_games]

    df = pd.DataFrame(all_games_stats)

    if DEBUG_MODE:
        print(f"      ✅ Parsed {len(df)} total games for {player_name}")
        if len(df) > 0:
            current_season_games = sum(1 for _, row in df.iterrows() if row.get('season') == SEASON)
            last_season_games = len(df) - current_season_games
            print(f"         {current_season_games} from {SEASON}, {last_season_games} from {STATS_SEASON}")
            print(f"         Avg: PTS={df['points'].mean():.1f}, AST={df['assists'].mean():.1f}, REB={df['rebounds'].mean():.1f}")

    # Remove season column before returning (not needed in calculations)
    if 'season' in df.columns:
        df = df.drop('season', axis=1)

    player_cache[cache_key] = (datetime.datetime.now(), df)
    save_data(CACHE_FILE, player_cache)

    return df


def get_team_stats(team_id: int) -> dict:
    cache_key = f"team_{team_id}"

    if cache_key in player_cache:
        cached_time, cached_data = player_cache[cache_key]
        if (datetime.datetime.now() - cached_time).seconds < 86400:
            return cached_data

    # Use last season's stats since current season just started
    params = {
        "league": LEAGUE_ID,
        "season": STATS_SEASON,  # 2024-2025
        "team": team_id
    }

    if DEBUG_MODE:
        print(f"      📅 Fetching team {team_id} stats from {STATS_SEASON}")

    data = fetch_json("/statistics", params=params)

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

        total_points_for = 0
        total_points_against = 0
        games_count = 0

        for game_stat in response:
            try:
                team_points = float(game_stat.get("points", 0) or 0)

                opp_points = 0
                if "opponent" in game_stat:
                    opp_points = float(game_stat["opponent"].get("points", 0) or 0)

                total_points_for += team_points
                total_points_against += opp_points
                games_count += 1
            except (ValueError, TypeError, KeyError):
                continue

        if games_count > 0:
            ppg = total_points_for / games_count
            opp_ppg = total_points_against / games_count if total_points_against > 0 else 110.0

            estimated_pace = ((ppg + opp_ppg) / 2) / 110.0

            team_stats = {
                "pace": max(0.85, min(1.15, estimated_pace)),
                "offensive_rating": ppg,
                "defensive_rating": opp_ppg,
                "points_per_game": ppg,
                "opp_points_per_game": opp_ppg,
                "games_played": games_count
            }

    player_cache[cache_key] = (datetime.datetime.now(), team_stats)
    save_data(CACHE_FILE, player_cache)

    return team_stats


def get_matchup_context(game_info: dict) -> dict:
    home_team_id = game_info.get("teams", {}).get("home", {}).get("id")
    away_team_id = game_info.get("teams", {}).get("away", {}).get("id")

    if not home_team_id or not away_team_id:
        return {"pace": 1.0, "offensive_adjustment": 1.0, "defensive_adjustment": 1.0}

    home_stats = get_team_stats(home_team_id)
    away_stats = get_team_stats(away_team_id)

    combined_pace = (home_stats["pace"] + away_stats["pace"]) / 2

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
# STATISTICAL CALCULATIONS
# ============================================
def calculate_player_projection(player_stats: pd.DataFrame, prop_type: str,
                                team_context: dict, opponent_defense: float = 1.0) -> Tuple[float, float]:
    if player_stats.empty:
        return 0.0, 0.0

    stat_col = stat_map.get(prop_type)
    if not stat_col or stat_col not in player_stats.columns:
        return 0.0, 0.0

    values = player_stats[stat_col].astype(float).values
    n = len(values)

    weights = np.exp(np.linspace(0, 1, n))
    base_projection = np.average(values, weights=weights)

    std_dev = np.std(values, ddof=1) if n > 1 else base_projection * 0.20

    pace_multiplier = team_context.get("pace", 1.0)
    defense_multiplier = opponent_defense

    if prop_type == "points":
        adjustment = pace_multiplier * defense_multiplier
    elif prop_type == "assists":
        adjustment = pace_multiplier * (0.7 + 0.3 * defense_multiplier)
    elif prop_type == "rebounds":
        adjustment = (0.8 * pace_multiplier + 0.2)
    elif prop_type == "threes":
        adjustment = pace_multiplier * defense_multiplier
    else:
        adjustment = 1.0

    adjusted_projection = base_projection * adjustment
    adjusted_std_dev = std_dev * (0.9 + 0.1 * pace_multiplier)

    return adjusted_projection, adjusted_std_dev


def american_to_decimal(odds) -> float:
    if isinstance(odds, str):
        odds = float(odds)

    if isinstance(odds, float):
        if 1.0 <= odds <= 100.0:
            return odds

    odds = int(odds)
    if odds > 0:
        return (odds / 100) + 1
    else:
        return (100 / abs(odds)) + 1


def calculate_win_probability(projection: float, line: float, std_dev: float,
                              pick: str = "over") -> float:
    if std_dev == 0:
        std_dev = projection * 0.20

    z_score = (projection - line) / std_dev

    def norm_cdf(x):
        a1 =  0.254829592
        a2 = -0.284496736
        a3 =  1.421413741
        a4 = -1.453152027
        a5 =  1.061405429
        p  =  0.3275911

        sign = 1 if x >= 0 else -1
        x = abs(x) / np.sqrt(2.0)

        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)

        return 0.5 * (1.0 + sign * y)

    if pick == "over":
        win_prob = 1 - norm_cdf(z_score)
    else:
        win_prob = norm_cdf(z_score)

    # Wider bounds: 25% to 90%
    return max(0.25, min(0.90, win_prob))


def calculate_kelly_stake(win_prob: float, odds: int, bankroll: float,
                         fraction: float = KELLY_FRACTION) -> Tuple[float, float]:
    decimal_odds = american_to_decimal(odds)
    b = decimal_odds - 1
    p = win_prob
    q = 1 - p

    kelly = (b * p - q) / b

    fractional_kelly = max(0, kelly * fraction)

    stake = bankroll * fractional_kelly

    return fractional_kelly * 100, stake


def calculate_expected_value(win_prob: float, odds: int, stake: float) -> float:
    decimal_odds = american_to_decimal(odds)
    profit = stake * (decimal_odds - 1)
    loss_prob = 1 - win_prob

    ev = (win_prob * profit) - (loss_prob * stake)

    return (ev / stake) * 100 if stake > 0 else 0


# ============================================
# ADAPTIVE LEARNING
# ============================================
def update_prop_weights(prop_id: str, actual_result: bool, predicted_prob: float):
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
    return prop_weights.get(prop_id, 1.0)


# ============================================
# PROP ANALYSIS
# ============================================
def extract_props_from_odds(odds_data: dict, game_info: dict) -> List[dict]:
    if not odds_data or "bookmakers" not in odds_data:
        return []

    bookmakers = odds_data.get("bookmakers", [])
    if not bookmakers:
        return []

    props = []
    bookmaker = bookmakers[0]

    ALLOWED_BET_TYPES = {
        "moneyline", "money line", "match winner",
        "spread", "point spread", "handicap",
        "totals", "total", "over/under",
        "points", "point", "player points",
        "assists", "assist", "player assists",
        "rebounds", "rebound", "player rebounds", "total rebounds",
        "threes", "three", "3-point", "3-pointers", "player threes"
    }

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

                    prop_id = f"{game_info['id']}_{bet_type}_{prop_text}".replace(" ", "_")

                    props.append({
                        "prop_id": prop_id,
                        "game_id": game_info["id"],
                        "game": f"{game_info['teams']['home']['name']} vs {game_info['teams']['away']['name']}",
                        "game_date": game_info["date"],
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

                prop_id = f"{game_info['id']}_{player_name}_{prop_type}".replace(" ", "_")

                props.append({
                    "prop_id": prop_id,
                    "game_id": game_info["id"],
                    "game": f"{game_info['teams']['home']['name']} vs {game_info['teams']['away']['name']}",
                    "game_date": game_info["date"],
                    "player": player_name,
                    "prop_type": prop_type,
                    "line": line,
                    "odds": odds,
                    "bookmaker": "DraftKings"
                })

    return props


def analyze_prop(prop: dict, matchup_context: dict = None) -> Optional[dict]:
    ALLOWED_PROPS = ["points", "assists", "rebounds", "threes", "moneyline", "spread", "game_total"]
    if prop["prop_type"] not in ALLOWED_PROPS:
        return None

    # Filter unreasonable odds (e.g., -999999 or extreme longshots)
    odds = prop.get("odds")
    if odds is not None and (odds < MIN_ODDS or odds > MAX_ODDS):
        if DEBUG_MODE:
            player_name = prop.get("player", prop.get("prop_type", "Unknown"))
            print(f"      ❌ {player_name} {prop['prop_type']} - Odds {odds:+d} outside profitable range [{MIN_ODDS}, {MAX_ODDS}]")
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
            if DEBUG_MODE:
                print(f"      ❌ {prop['prop_type']} - Win prob {adjusted_prob*100:.1f}% < {MIN_CONFIDENCE*100:.1f}%")
            return None

        kelly_pct, stake = calculate_kelly_stake(adjusted_prob, odds, BANKROLL)

        if stake < MIN_KELLY_STAKE:
            if DEBUG_MODE:
                print(f"      ❌ {prop['prop_type']} - Stake ${stake:.2f} < ${MIN_KELLY_STAKE}")
            return None

        ev = calculate_expected_value(adjusted_prob, odds, stake)

        decimal_odds = american_to_decimal(odds)
        potential_profit = stake * (decimal_odds - 1)

        composite_score = (
            (adjusted_prob * 100 * 0.60) +
            (kelly_pct * 0.40)
        )

        if DEBUG_MODE:
            print(f"      ✅ {prop['prop_type']} PASSED - Win: {adjusted_prob*100:.1f}%, Stake: ${stake:.2f}")

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

    # Handle player props
    if DEBUG_MODE:
        print(f"   🏀 Analyzing: {prop['player']} {prop['prop_type']} {prop['line']}")

    player_stats = get_player_recent_stats(prop["player"], LOOKBACK_GAMES)

    if player_stats.empty or len(player_stats) < MIN_GAMES_REQUIRED:
        if DEBUG_MODE:
            print(f"      ❌ Insufficient data: {len(player_stats)} games < {MIN_GAMES_REQUIRED} required")
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
        if DEBUG_MODE:
            print(f"      ❌ Invalid projection: proj={projection}, std={std_dev}")
        return None

    disparity = projection - prop["line"]
    edge = (disparity / prop["line"]) * 100

    pick = "over" if disparity > 0 else "under"

    risk_adjusted = disparity / std_dev if std_dev > 0 else 0

    win_prob = calculate_win_probability(projection, prop["line"], std_dev, pick)

    confidence_mult = get_prop_confidence_multiplier(prop["prop_id"])
    adjusted_prob = min(0.90, win_prob * confidence_mult)

    if adjusted_prob < MIN_CONFIDENCE:
        if DEBUG_MODE:
            print(f"      ❌ Low confidence: {adjusted_prob*100:.1f}% < {MIN_CONFIDENCE*100:.1f}%")
        return None

    kelly_pct, stake = calculate_kelly_stake(adjusted_prob, prop["odds"], BANKROLL)

    if stake < MIN_KELLY_STAKE:
        if DEBUG_MODE:
            print(f"      ❌ Small stake: ${stake:.2f} < ${MIN_KELLY_STAKE}")
        return None

    ev = calculate_expected_value(adjusted_prob, prop["odds"], stake)

    decimal_odds = american_to_decimal(prop["odds"])
    potential_profit = stake * (decimal_odds - 1)

    stat_col = stat_map.get(prop["prop_type"])
    if stat_col and stat_col in player_stats.columns:
        values = player_stats[stat_col].astype(float).values
        if len(values) >= 7:
            avg_last3 = np.mean(values[:3])
            avg_last7 = np.mean(values[:7])
            trend = ((avg_last3 - avg_last7) / avg_last7) * 100 if avg_last7 > 0 else 0
        else:
            trend = 0
    else:
        trend = 0

    composite_score = (
        (adjusted_prob * 100 * 0.50) +
        (kelly_pct * 0.25) +
        (max(0, ev) * 0.15) +
        (max(0, risk_adjusted * 5) * 0.10)
    )

    if DEBUG_MODE:
        print(f"      ✅ PASSED - Proj: {projection:.1f}, Line: {prop['line']}, Win: {adjusted_prob*100:.1f}%, Stake: ${stake:.2f}")

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
# MAIN EXECUTION
# ============================================
def run_analysis():
    print("=" * 70)
    print("NBA PROP ANALYZER - FIXED FOR PRO PLAN ✅")
    print("=" * 70)
    print(f"API Plan: PRO (Player Stats Enabled)")
    print(f"Games/Odds Season: {SEASON}")
    print(f"Player Stats Season: {STATS_SEASON} (most recent + current)")
    print(f"Bankroll: ${BANKROLL:,.2f}")
    print(f"Kelly Fraction: {KELLY_FRACTION:.2%}")
    print(f"Min Confidence: {MIN_CONFIDENCE:.1%}")
    print(f"Min Kelly Stake: ${MIN_KELLY_STAKE}")
    print(f"Min Games Required: {MIN_GAMES_REQUIRED}")
    print(f"Debug Mode: {'ON' if DEBUG_MODE else 'OFF'}")
    print("=" * 70)
    print()

    print("📅 Fetching upcoming games (next 3 days)...")
    games = get_upcoming_games(days_ahead=3)
    print(f"   Found {len(games)} games in next 3 days\n")

    if not games:
        print("❌ No upcoming games found")
        return

    print("🎲 Fetching odds and props...")
    print("   ✅ ANALYZING: Points, Assists, Rebounds, 3PM, Moneyline, Spread, Totals")
    print("   ❌ EXCLUDING: Blocks, Steals, Turnovers, Minutes, etc.")
    all_props = []
    game_contexts = {}

    for game in games[:10]:
        matchup_context = get_matchup_context(game)
        game_contexts[game["id"]] = matchup_context

        game_odds = get_game_odds(game["id"])
        if game_odds:
            props = extract_props_from_odds(game_odds, game)
            all_props.extend(props)

            pace_info = f"Pace: {matchup_context['pace']:.2f}x" if matchup_context['pace'] != 1.0 else ""
            print(f"   ✓ {game['teams']['home']['name']} vs {game['teams']['away']['name']}: {len(props)} props {pace_info}")

        time.sleep(0.5)

    player_props = [p for p in all_props if p["prop_type"] in ["points", "assists", "rebounds", "threes"]]
    game_bets = [p for p in all_props if p["prop_type"] in ["moneyline", "spread", "game_total"]]

    print(f"\n   Total props extracted: {len(all_props)}")
    print(f"   Player props (PTS/AST/REB/3PM): {len(player_props)}")

    # OPTIMIZATION: Limit to first 10 player props for faster testing
    if len(player_props) > 10:
        print(f"   ⚡ Limiting to first 10 player props for speed")
        player_props = player_props[:10]

    print(f"   Game bets (ML/Spread/Total): {len(game_bets)}")
    print(f"   Total to analyze: {len(player_props) + len(game_bets)}\n")

    print("🔍 Analyzing props with Kelly Criterion...")
    if DEBUG_MODE:
        print("   🐛 DEBUG MODE ON - Showing detailed analysis")
    analyzed_props = []

    all_to_analyze = player_props + game_bets

    for idx, prop in enumerate(all_to_analyze, 1):
        if not DEBUG_MODE and idx % 50 == 0:
            print(f"   Progress: {idx}/{len(all_to_analyze)} props analyzed...")

        matchup_context = game_contexts.get(prop["game_id"], {})

        result = analyze_prop(prop, matchup_context)
        if result:
            analyzed_props.append(result)

        if prop["prop_type"] in ["points", "assists", "rebounds", "threes"]:
            time.sleep(0.2)

    print(f"\n   ✅ {len(analyzed_props)} props meet criteria (out of {len(all_to_analyze)})\n")

    if not analyzed_props:
        print("❌ No props met minimum thresholds")
        print("\n💡 Troubleshooting:")
        print("   1. Check if player stats are being fetched (look for '📦 Cache hit' or '✅ Parsed' messages)")
        print("   2. Lower MIN_CONFIDENCE further (currently {:.1%})".format(MIN_CONFIDENCE))
        print("   3. Check API response structures match expected format")
        print("   4. Verify player names in odds match player search results")
        return

    analyzed_props.sort(key=lambda x: x["composite_score"], reverse=True)
    top_props = analyzed_props[:TOP_PROPS]

    print("=" * 70)
    print(f"TOP {len(top_props)} PROPS (Ranked by Composite Score)")
    print("=" * 70)
    print("Formula: Score = (WinProb×50%) + (Kelly×25%) + (EV×15%) + (RiskAdj×10%)")
    print("=" * 70)
    print()

    total_stake = sum(p["stake"] for p in top_props)
    total_potential = sum(p["potential_profit"] for p in top_props)
    avg_win_prob = sum(p["win_prob"] for p in top_props) / len(top_props)
    avg_ev = sum(p["ev"] for p in top_props) / len(top_props)

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
    high_conf = sum(1 for p in top_props if p['win_prob'] >= 65)
    med_conf = sum(1 for p in top_props if 55 <= p['win_prob'] < 65)
    low_conf = sum(1 for p in top_props if p['win_prob'] < 55)
    print(f"  🟢 High (65%+):     {high_conf} bets")
    print(f"  🟡 Medium (55-65%): {med_conf} bets")
    print(f"  🟠 Lower (50-55%):  {low_conf} bets")
    print("=" * 70)

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
                "total_stake": total_stake,
                "total_potential": total_potential,
                "avg_win_prob": avg_win_prob,
                "avg_ev": avg_ev
            }
        }, f, indent=2)

    print(f"\n✅ Results saved to: {output_file}")


if __name__ == "__main__":
    try:
        run_analysis()
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
