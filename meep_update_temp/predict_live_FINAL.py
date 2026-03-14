#!/usr/bin/env python3
"""
Research-Grade Live NBA Prediction System - OPTIMIZED FOR AGGREGATED DATA

Since you have aggregated_nba_data.csv.gzip with ALL features pre-computed:
- Load player's most recent game from aggregated data
- Update only real-time features (opponent, rest days, B2B)
- Use neural_hybrid.py for predictions with TabNet embeddings
- Fetch betting lines from The Odds API
- Identify +EV betting opportunities with Safe Mode protection

Usage:
    python predict_live.py --date 2025-11-09 --aggregated-data ./data/aggregated_nba_data.csv.gzip --betting
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
import os
import requests
import time
from scipy.stats import norm
warnings.filterwarnings('ignore')

# NBA API
try:
    from nba_api.stats.endpoints import ScoreboardV2, CommonTeamRoster
    HAS_NBA_API = True
except ImportError:
    HAS_NBA_API = False

# SHAP
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# Neural hybrid
import sys
sys.path.append(str(Path(__file__).parent))
from neural_hybrid import NeuralHybridPredictor


# ========== THE ODDS API CONFIGURATION ==========
THEODDS_API_KEY = os.getenv("THEODDS_API_KEY") or ""
THEODDS_BASE_URL = "https://api.the-odds-api.com/v4"
THEODDS_ENABLED = bool(THEODDS_API_KEY)
THEODDS_SPORT = "basketball_nba"
THEODDS_REGIONS = "us"
THEODDS_MARKETS = "player_points,player_rebounds,player_assists,player_threes"
THEODDS_BOOKMAKERS = "fanduel"
REQUEST_TIMEOUT = 10

# Safe Mode: Add extra margin to lines for conservative betting
SAFE_MODE = os.getenv("SAFE_MODE", "").lower() in ["true", "1", "yes"]
SAFE_MARGIN = float(os.getenv("SAFE_MARGIN", "1.0"))  # Extra buffer

# Minimum win probability filter (confidence threshold)
MIN_WIN_PROBABILITY = float(os.getenv("MIN_WIN_PROBABILITY", "0.56"))  # 56% default

# ELG gates by prop type
ELG_GATES = {
    "points": -0.005,
    "assists": -0.005,
    "rebounds": -0.005,
    "threes": -0.005,
}

DEBUG_MODE = False


# ========== BETTING HELPER FUNCTIONS ==========

def kelly_fraction(p: float, b: float) -> float:
    """
    Calculate Kelly Criterion fraction.

    Args:
        p: Win probability
        b: Decimal odds (payout multiplier minus 1)

    Returns:
        Optimal fraction of bankroll to bet
    """
    q = 1.0 - p
    f = (b * p - q) / max(1e-9, b)
    return max(0.0, f)


def american_to_decimal(odds: int) -> float:
    """Convert American odds to decimal odds."""
    if odds > 0:
        return (odds / 100.0) + 1.0
    else:
        return (100.0 / abs(odds)) + 1.0


def prop_win_probability(mu: float, sigma: float, line: float, pick: str) -> float:
    """
    Calculate win probability for a prop bet using normal distribution.

    Args:
        mu: Predicted mean
        sigma: Predicted standard deviation (uncertainty)
        line: Betting line
        pick: 'over' or 'under'

    Returns:
        Win probability [0, 1]
    """
    sigma = max(sigma, 1e-6)
    z = (mu - line) / sigma

    if pick == "over":
        p = 1.0 - norm.cdf((line - mu) / sigma)
    else:  # under
        p = norm.cdf((line - mu) / sigma)

    return min(1.0 - 1e-4, max(1e-4, p))


def calculate_ev(p: float, odds: int) -> float:
    """
    Calculate Expected Value (EV) for a bet.

    Args:
        p: Win probability
        odds: American odds

    Returns:
        EV per dollar bet
    """
    decimal_odds = american_to_decimal(odds)
    return (p * (decimal_odds - 1)) - (1 - p)


class LivePredictionEngine:
    """
    Optimized for pre-aggregated data.

    Strategy:
    1. Load aggregated_nba_data.csv.gzip (has all 150+ features)
    2. For each player, get their most recent game row
    3. Update only dynamic features (opponent, rest days)
    4. Predict using NeuralHybridPredictor
    """

    def __init__(self, models_dir: str = "./models",
                 aggregated_data_path: str = "./data/aggregated_nba_data.csv.gzip",
                 cache_dir: str = "./cache"):
        self.models_dir = Path(models_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        self.models = {}
        self.explainers = {}

        # Load aggregated data (has all features pre-computed!)
        print(f"\n📊 Loading aggregated data from {aggregated_data_path}...")
        self.aggregated_data = pd.read_csv(aggregated_data_path, compression='gzip')
        print(f"   Loaded {len(self.aggregated_data):,} player-games with {len(self.aggregated_data.columns)} features")

        # Convert date column
        if 'gameDate' in self.aggregated_data.columns:
            self.aggregated_data['gameDate'] = pd.to_datetime(self.aggregated_data['gameDate'])
        elif 'date' in self.aggregated_data.columns:
            self.aggregated_data['date'] = pd.to_datetime(self.aggregated_data['date'])

        # Load models
        self._load_models()

    def _load_models(self):
        """Load trained NeuralHybridPredictor models."""
        print("\n📦 Loading trained models...")

        props = ['minutes', 'points', 'rebounds', 'assists', 'threes']

        for prop in props:
            model_patterns = [
                f"{prop}_hybrid_*.pkl",
                f"{prop}_model.pkl",
                f"*{prop}*.pkl"
            ]

            model_path = None
            for pattern in model_patterns:
                matches = list(self.models_dir.glob(pattern))
                if matches:
                    model_path = matches[0]
                    break

            if model_path and model_path.exists():
                with open(model_path, 'rb') as f:
                    self.models[prop] = pickle.load(f)
                print(f"  ✓ Loaded {prop} model from {model_path.name}")
            else:
                print(f"  ✗ {prop} model not found")

        if not self.models:
            raise ValueError(f"No models found in {self.models_dir}")

        print(f"\n✅ Loaded {len(self.models)} models")

    def get_player_features(self, player_id: str, player_name: str,
                           opponent_team: str, is_home: bool,
                           game_date: str) -> Optional[pd.DataFrame]:
        """
        Get features for a player by loading their most recent game from aggregated data
        and updating dynamic features.

        Args:
            player_id: NBA player ID (as string)
            player_name: Player name (for fallback matching)
            opponent_team: Opponent team abbreviation
            is_home: True if home game
            game_date: Date of prediction (YYYY-MM-DD)

        Returns:
            DataFrame with single row of features, or None
        """
        # Try to find player in aggregated data
        player_data = None

        # Try by player ID first
        if 'personId' in self.aggregated_data.columns:
            player_data = self.aggregated_data[
                self.aggregated_data['personId'].astype(str) == str(player_id)
            ]

        # Fallback to name matching
        if (player_data is None or player_data.empty) and 'playerName' in self.aggregated_data.columns:
            player_data = self.aggregated_data[
                self.aggregated_data['playerName'].str.lower() == player_name.lower()
            ]

        if player_data is None or player_data.empty:
            print(f"      ⚠️  Player {player_name} not found in aggregated data")
            return None

        # Sort by date and get most recent game
        date_col = 'gameDate' if 'gameDate' in player_data.columns else 'date'
        player_data = player_data.sort_values(date_col, ascending=False)

        # Get most recent game (has all 150+ features already!)
        latest_game = player_data.iloc[0:1].copy()

        # ==============================================================
        # UPDATE ONLY DYNAMIC FEATURES FOR TODAY'S GAME
        # ==============================================================

        # Update home/away
        if 'is_home' in latest_game.columns:
            latest_game['is_home'] = 1.0 if is_home else 0.0

        # Update rest days
        if len(player_data) > 1:
            last_game_date = pd.to_datetime(player_data.iloc[0][date_col])
            pred_date = pd.to_datetime(game_date)
            days_rest = (pred_date - last_game_date).days

            if 'days_rest' in latest_game.columns:
                latest_game['days_rest'] = float(min(days_rest, 10))
            if 'player_b2b' in latest_game.columns:
                latest_game['player_b2b'] = 1.0 if days_rest <= 1 else 0.0

        # Update opponent (if opponent columns exist)
        # Note: Opponent stats would ideally come from recent opponent performance
        # For now, keep the last opponent's stats as proxy

        # Update season (if predicting future season)
        pred_date = pd.to_datetime(game_date)
        season_year = pred_date.year if pred_date.month < 8 else pred_date.year + 1
        if 'season_end_year' in latest_game.columns:
            latest_game['season_end_year'] = float(season_year)

        # Drop target columns if present (don't want to use them as features)
        target_cols = ['points', 'rebounds', 'assists', 'threes', 'minutes',
                       'threePointersMade', 'numMinutes']
        for col in target_cols:
            if col in latest_game.columns:
                latest_game = latest_game.drop(columns=[col])

        # Drop non-feature columns
        meta_cols = ['gameId', 'personId', 'playerName', 'gameDate', 'date',
                     'firstName', 'lastName', 'teamId']
        for col in meta_cols:
            if col in latest_game.columns:
                latest_game = latest_game.drop(columns=[col])

        return latest_game

    def predict_player_props(self, player_id: str, player_name: str,
                            team_abbr: str, opponent_abbr: str,
                            is_home: bool, game_date: str,
                            explain: bool = False) -> Dict:
        """
        Generate predictions using pre-aggregated features + neural hybrid model.
        """
        # Get features from aggregated data
        features = self.get_player_features(
            player_id, player_name, opponent_abbr, is_home, game_date
        )

        if features is None or features.empty:
            return {'error': 'Player not found in aggregated data'}

        predictions = {
            'player_id': player_id,
            'player_name': player_name,
            'team': team_abbr,
            'opponent': opponent_abbr,
            'is_home': is_home,
            'game_date': game_date,
        }

        # Make predictions for each prop
        for prop, model in self.models.items():
            try:
                # NeuralHybridPredictor handles TabNet embeddings + LightGBM internally
                if hasattr(model, 'predict'):
                    # Check if model has uncertainty
                    if hasattr(model, 'sigma_model') and model.sigma_model is not None:
                        pred, sigma = model.predict(features, return_uncertainty=True)
                        pred_val = float(pred[0]) if hasattr(pred, '__len__') else float(pred)
                        uncertainty = float(sigma[0]) if hasattr(sigma, '__len__') else float(sigma)
                    else:
                        pred = model.predict(features)
                        pred_val = float(pred[0]) if hasattr(pred, '__len__') else float(pred)
                        uncertainty = None

                    predictions[prop] = {
                        'prediction': round(pred_val, 2),
                        'uncertainty': round(uncertainty, 2) if uncertainty else None,
                        'lower_80': round(pred_val - 1.28 * uncertainty, 2) if uncertainty else None,
                        'upper_80': round(pred_val + 1.28 * uncertainty, 2) if uncertainty else None,
                        'lower_95': round(pred_val - 1.96 * uncertainty, 2) if uncertainty else None,
                        'upper_95': round(pred_val + 1.96 * uncertainty, 2) if uncertainty else None,
                    }

                    # SHAP explanations
                    if explain and HAS_SHAP:
                        if prop not in self.explainers:
                            # Initialize explainer for this prop
                            if hasattr(model, 'lgbm'):
                                self.explainers[prop] = shap.TreeExplainer(model.lgbm)

                        if prop in self.explainers:
                            try:
                                shap_values = self.explainers[prop].shap_values(features)

                                # Get top 5 features
                                feature_importance = pd.DataFrame({
                                    'feature': features.columns,
                                    'shap_value': shap_values[0] if len(shap_values.shape) > 1 else shap_values
                                }).sort_values('shap_value', key=abs, ascending=False).head(5)

                                predictions[prop]['explanation'] = feature_importance.to_dict('records')
                            except Exception as e:
                                print(f"      Warning: SHAP failed for {prop}: {e}")

            except Exception as e:
                print(f"      Error predicting {prop} for {player_name}: {e}")
                import traceback
                traceback.print_exc()
                predictions[prop] = {'error': str(e)}

        return predictions

    def get_todays_games(self, date: Optional[str] = None) -> pd.DataFrame:
        """Fetch today's games from nba_api."""
        if not HAS_NBA_API:
            raise RuntimeError("nba_api required. Install: pip install nba-api")

        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        print(f"\n📅 Fetching games for {date}...")

        try:
            scoreboard = ScoreboardV2(game_date=date)
            games_df = scoreboard.get_data_frames()[0]

            if games_df.empty:
                print(f"   No games scheduled for {date}")
                return pd.DataFrame()

            games = []
            for _, game in games_df.iterrows():
                games.append({
                    'game_id': game['GAME_ID'],
                    'home_team_id': game['HOME_TEAM_ID'],
                    'away_team_id': game['VISITOR_TEAM_ID'],
                    'home_team': game.get('HOME_TEAM_ABBREVIATION', ''),
                    'away_team': game.get('VISITOR_TEAM_ABBREVIATION', ''),
                    'game_time': game.get('GAME_STATUS_TEXT', 'TBD'),
                    'date': date
                })

            games_df = pd.DataFrame(games)
            print(f"   Found {len(games_df)} games")
            return games_df

        except Exception as e:
            print(f"   ❌ Error fetching games: {e}")
            return pd.DataFrame()

    def get_team_roster(self, team_id: int, season: str = "2025-26") -> pd.DataFrame:
        """Get team roster from nba_api."""
        try:
            roster = CommonTeamRoster(team_id=team_id, season=season)
            return roster.get_data_frames()[0]
        except Exception as e:
            print(f"   Warning: Could not fetch roster for team {team_id}: {e}")
            return pd.DataFrame()

    def predict_game(self, game_info: Dict, explain: bool = False) -> List[Dict]:
        """Predict all players in a game."""
        home_team_id = game_info['home_team_id']
        away_team_id = game_info['away_team_id']
        game_date = game_info['date']

        print(f"\n🏀 {game_info['away_team']} @ {game_info['home_team']}")

        all_predictions = []

        # Get rosters
        for team_id, team_abbr, is_home in [
            (home_team_id, game_info['home_team'], True),
            (away_team_id, game_info['away_team'], False)
        ]:
            opponent_abbr = game_info['away_team'] if is_home else game_info['home_team']
            roster = self.get_team_roster(team_id)

            if roster.empty:
                print(f"   ⚠️  No roster for {team_abbr}")
                continue

            print(f"   {team_abbr}: {len(roster)} players")

            for _, player in roster.iterrows():
                player_id = str(player['PLAYER_ID'])
                player_name = player['PLAYER']

                print(f"      Predicting {player_name}...")

                pred = self.predict_player_props(
                    player_id, player_name, team_abbr, opponent_abbr,
                    is_home, game_date, explain
                )

                if 'error' not in pred:
                    all_predictions.append(pred)

        return all_predictions

    def predict_all_games(self, date: Optional[str] = None, explain: bool = False) -> pd.DataFrame:
        """Predict all games for a date."""
        games = self.get_todays_games(date)

        if games.empty:
            return pd.DataFrame()

        all_predictions = []
        for _, game in games.iterrows():
            game_preds = self.predict_game(game.to_dict(), explain)
            all_predictions.extend(game_preds)

        return pd.DataFrame(all_predictions)

    # ========== BETTING INTEGRATION ==========

    def fetch_betting_lines(self, date: Optional[str] = None) -> List[Dict]:
        """
        Fetch player prop lines from The Odds API.

        Args:
            date: Date to fetch lines for (YYYY-MM-DD)

        Returns:
            List of props with betting lines
        """
        if not THEODDS_ENABLED:
            print("⚠️  The Odds API key not configured. Set THEODDS_API_KEY environment variable.")
            return []

        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        print(f"\n💰 Fetching betting lines from The Odds API...")

        try:
            # Get today's games first
            games = self.get_todays_games(date)
            if games.empty:
                print("   No games scheduled")
                return []

            # Build game mapping
            game_map = {}
            for _, game in games.iterrows():
                home = game['home_team'].lower()
                away = game['away_team'].lower()
                game_map[f"{away}_{home}"] = game
                game_map[f"{home}_{away}"] = game

            props = []

            # STEP 1: Fetch events to get event IDs
            events_url = f"{THEODDS_BASE_URL}/sports/{THEODDS_SPORT}/events"
            events_params = {"apiKey": THEODDS_API_KEY}

            events_resp = requests.get(events_url, params=events_params, timeout=REQUEST_TIMEOUT * 2)

            if DEBUG_MODE:
                print(f"   [TheOdds] GET {events_url} status={events_resp.status_code}")
                print(f"   [TheOdds] Remaining: {events_resp.headers.get('x-requests-remaining', 'unknown')}")

            if events_resp.status_code != 200:
                print(f"   ❌ Events fetch failed: {events_resp.status_code}")
                return []

            events = events_resp.json()
            print(f"   Found {len(events)} events")

            # STEP 2: For each event, fetch odds with player props
            for event in events:
                event_id = event.get("id")
                home_team = event.get("home_team", "").lower()
                away_team = event.get("away_team", "").lower()
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
                        print(f"   No match for {away_team} @ {home_team}")
                    continue

                game_id = game['game_id']
                game_label = f"{game['away_team']} at {game['home_team']}"
                game_date = event_commence_time or date

                # Fetch odds for this specific event
                event_odds_url = f"{THEODDS_BASE_URL}/sports/{THEODDS_SPORT}/events/{event_id}/odds"
                event_odds_params = {
                    "apiKey": THEODDS_API_KEY,
                    "regions": THEODDS_REGIONS,
                    "markets": THEODDS_MARKETS,
                    "oddsFormat": "american",
                }

                if THEODDS_BOOKMAKERS:
                    event_odds_params["bookmakers"] = THEODDS_BOOKMAKERS

                event_odds_resp = requests.get(event_odds_url, params=event_odds_params,
                                              timeout=REQUEST_TIMEOUT * 2)

                if event_odds_resp.status_code != 200:
                    if DEBUG_MODE:
                        print(f"   Event odds fetch failed: {event_odds_resp.status_code}")
                    continue

                event_data = event_odds_resp.json()
                bookmakers = event_data.get("bookmakers", [])

                for bookmaker in bookmakers:
                    bookmaker_name = bookmaker.get("title", "Unknown")
                    markets = bookmaker.get("markets", [])

                    for market in markets:
                        market_key = market.get("key", "")
                        outcomes = market.get("outcomes", [])

                        # Player Props only
                        if market_key.startswith("player_"):
                            prop_type_map = {
                                "player_points": "points",
                                "player_rebounds": "rebounds",
                                "player_assists": "assists",
                                "player_threes": "threes",
                            }
                            prop_type = prop_type_map.get(market_key)

                            if prop_type:
                                for outcome in outcomes:
                                    player_name = outcome.get("description")
                                    over_under = outcome.get("name", "").lower()
                                    point = outcome.get("point")
                                    odds = outcome.get("price")

                                    if player_name and point is not None and odds:
                                        prop_id = f"{game_id}_{player_name}_{prop_type}_{point}_{bookmaker_name}".replace(" ", "_")

                                        # Check if prop already exists (to combine over/under)
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
                                                "home_team": game['home_team'],
                                                "away_team": game['away_team'],
                                                "prop_type": prop_type,
                                                "line": float(point),
                                                "bookmaker": bookmaker_name,
                                                "source": "TheOddsAPI",
                                            }
                                            if over_under == "over":
                                                new_prop["odds_over"] = int(odds)
                                            else:
                                                new_prop["odds_under"] = int(odds)
                                            props.append(new_prop)

                # Rate limiting
                time.sleep(0.1)

            print(f"   ✅ Fetched {len(props)} player props")
            return props

        except Exception as e:
            print(f"   ❌ Error fetching lines: {e}")
            import traceback
            traceback.print_exc()
            return []

    def find_ev_opportunities(self, predictions: List[Dict], lines: List[Dict]) -> List[Dict]:
        """
        Compare predictions to betting lines and identify +EV opportunities.

        Args:
            predictions: List of prediction dicts from predict_all_games
            lines: List of betting line dicts from fetch_betting_lines

        Returns:
            List of +EV betting opportunities with analysis
        """
        opportunities = []

        # Create player prediction lookup
        pred_lookup = {}
        for pred in predictions:
            player = pred.get('player_name', '')
            pred_lookup[player.lower()] = pred

        print(f"\n🔍 Analyzing {len(lines)} betting lines for +EV opportunities...")

        for line in lines:
            player_name = line.get('player')
            prop_type = line.get('prop_type')
            betting_line = line.get('line')
            odds_over = line.get('odds_over')
            odds_under = line.get('odds_under')

            # Find matching prediction
            pred = pred_lookup.get(player_name.lower())
            if not pred:
                continue

            # Get prediction for this prop
            prop_pred = pred.get(prop_type)
            if not prop_pred or 'prediction' not in prop_pred:
                continue

            mu = prop_pred['prediction']
            sigma = prop_pred.get('uncertainty', 1.5)

            # Apply safe margin if enabled
            effective_line_over = betting_line - SAFE_MARGIN if SAFE_MODE else betting_line
            effective_line_under = betting_line + SAFE_MARGIN if SAFE_MODE else betting_line

            # Analyze OVER bet
            if odds_over:
                p_over = prop_win_probability(mu, sigma, effective_line_over, 'over')
                ev_over = calculate_ev(p_over, odds_over)

                if p_over >= MIN_WIN_PROBABILITY and ev_over >= ELG_GATES.get(prop_type, -0.005):
                    decimal_odds = american_to_decimal(odds_over)
                    kelly_frac = kelly_fraction(p_over, decimal_odds - 1.0)

                    opportunities.append({
                        'player': player_name,
                        'team': pred.get('team'),
                        'opponent': pred.get('opponent'),
                        'prop_type': prop_type,
                        'pick': 'OVER',
                        'line': betting_line,
                        'effective_line': effective_line_over,
                        'odds': odds_over,
                        'prediction': mu,
                        'uncertainty': sigma,
                        'win_probability': p_over,
                        'expected_value': ev_over,
                        'kelly_fraction': kelly_frac,
                        'bookmaker': line.get('bookmaker'),
                        'confidence': (mu - effective_line_over) / sigma,  # Z-score
                    })

            # Analyze UNDER bet
            if odds_under:
                p_under = prop_win_probability(mu, sigma, effective_line_under, 'under')
                ev_under = calculate_ev(p_under, odds_under)

                if p_under >= MIN_WIN_PROBABILITY and ev_under >= ELG_GATES.get(prop_type, -0.005):
                    decimal_odds = american_to_decimal(odds_under)
                    kelly_frac = kelly_fraction(p_under, decimal_odds - 1.0)

                    opportunities.append({
                        'player': player_name,
                        'team': pred.get('team'),
                        'opponent': pred.get('opponent'),
                        'prop_type': prop_type,
                        'pick': 'UNDER',
                        'line': betting_line,
                        'effective_line': effective_line_under,
                        'odds': odds_under,
                        'prediction': mu,
                        'uncertainty': sigma,
                        'win_probability': p_under,
                        'expected_value': ev_under,
                        'kelly_fraction': kelly_frac,
                        'bookmaker': line.get('bookmaker'),
                        'confidence': (effective_line_under - mu) / sigma,  # Z-score
                    })

        # Sort by expected value descending
        opportunities.sort(key=lambda x: x['expected_value'], reverse=True)

        print(f"   ✅ Found {len(opportunities)} +EV opportunities")

        return opportunities


def main():
    parser = argparse.ArgumentParser(description='Live NBA Predictions with Aggregated Data + Betting Integration')
    parser.add_argument('--date', type=str, default=None, help='Date (YYYY-MM-DD)')
    parser.add_argument('--aggregated-data', type=str,
                       default='./data/aggregated_nba_data.csv.gzip',
                       help='Path to aggregated data CSV')
    parser.add_argument('--team', type=str, default=None, help='Filter by team')
    parser.add_argument('--explain', action='store_true', help='Include SHAP')
    parser.add_argument('--betting', action='store_true', help='Fetch betting lines and find +EV opportunities')
    parser.add_argument('--output', type=str, default=None, help='Save predictions to CSV/JSON')
    parser.add_argument('--betting-output', type=str, default=None, help='Save +EV opportunities to CSV/JSON')
    parser.add_argument('--models-dir', type=str, default='./models')
    args = parser.parse_args()

    print("="*70)
    print("🏀 LIVE NBA PREDICTIONS - Aggregated Data + Neural Hybrid")
    if args.betting:
        print("💰 BETTING INTEGRATION: The Odds API")
    print("="*70)

    # Display configuration if betting enabled
    if args.betting:
        print("\n⚙️  Betting Configuration:")
        print(f"   Safe Mode: {'ON' if SAFE_MODE else 'OFF'}")
        if SAFE_MODE:
            print(f"   Safe Margin: {SAFE_MARGIN}")
        print(f"   Min Win Prob: {MIN_WIN_PROBABILITY:.1%}")
        print(f"   Bookmaker: {THEODDS_BOOKMAKERS}")

    engine = LivePredictionEngine(
        models_dir=args.models_dir,
        aggregated_data_path=args.aggregated_data
    )

    # Generate predictions
    predictions = engine.predict_all_games(date=args.date, explain=args.explain)

    if not predictions.empty:
        # Filter by team
        if args.team:
            team_upper = args.team.upper()
            predictions = predictions[
                (predictions['team'] == team_upper) |
                (predictions['opponent'] == team_upper)
            ]

        print(f"\n📊 Generated {len(predictions)} predictions")

        # Save predictions
        if args.output:
            if args.output.endswith('.csv'):
                predictions.to_csv(args.output, index=False)
            elif args.output.endswith('.json'):
                predictions.to_json(args.output, orient='records', indent=2)
            print(f"💾 Saved predictions to {args.output}")

        # Betting integration
        if args.betting:
            # Convert DataFrame to list of dicts
            pred_list = predictions.to_dict('records')

            # Fetch betting lines
            lines = engine.fetch_betting_lines(date=args.date)

            if lines:
                # Find +EV opportunities
                opportunities = engine.find_ev_opportunities(pred_list, lines)

                if opportunities:
                    print(f"\n{'='*70}")
                    print(f"💎 TOP +EV OPPORTUNITIES ({len(opportunities)} found)")
                    print(f"{'='*70}")

                    # Display top 10
                    for i, opp in enumerate(opportunities[:10], 1):
                        print(f"\n{i}. {opp['player']} ({opp['team']}) - {opp['prop_type'].upper()}")
                        print(f"   Pick: {opp['pick']} {opp['line']}")
                        print(f"   Odds: {opp['odds']:+d} @ {opp['bookmaker']}")
                        print(f"   Prediction: {opp['prediction']:.1f} ± {opp['uncertainty']:.1f}")
                        print(f"   Win Probability: {opp['win_probability']:.1%}")
                        print(f"   Expected Value: {opp['expected_value']:+.3f}")
                        print(f"   Kelly Fraction: {opp['kelly_fraction']:.2%}")
                        print(f"   Confidence: {opp['confidence']:.2f}σ")

                    # Save betting opportunities
                    if args.betting_output:
                        opp_df = pd.DataFrame(opportunities)
                        if args.betting_output.endswith('.csv'):
                            opp_df.to_csv(args.betting_output, index=False)
                        elif args.betting_output.endswith('.json'):
                            opp_df.to_json(args.betting_output, orient='records', indent=2)
                        print(f"\n💾 Saved {len(opportunities)} opportunities to {args.betting_output}")
                else:
                    print(f"\n⚠️  No +EV opportunities found with current filters")
                    print(f"   Try adjusting MIN_WIN_PROBABILITY or SAFE_MARGIN")

    print("\n" + "="*70)


if __name__ == '__main__':
    main()
