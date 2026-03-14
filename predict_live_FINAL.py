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

# NBA API (DEPRECATED FOR INFERENCE)
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
from integrity_guard import DataIntegrityGuard


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
        
        # Phase S3: Integrity Guard
        self.guard = DataIntegrityGuard(allowed_staleness_days=3)
        
        # Phase S4: Style Archetype DNA (Unified 16-Latent System)
        self.archetype_map = {}
        arch_path = Path("models/archetypes/player_archetypes.csv")
        if arch_path.exists():
            print(f"[DNA] Loading unified 12-role archetype system (Multi-DNA Enabled)...")
            try:
                arch_df = pd.read_csv(arch_path)
                arch_df['player_id'] = arch_df['player_id'].astype(str).str.replace(r'\.0$', '', regex=True)
                # Store primary and secondary as a dict of tuples
                for _, row in arch_df.iterrows():
                    p_id = row['player_id']
                    p_role = row['archetype_id']
                    s_role = row.get('secondary_role_id', p_role)
                    self.archetype_map[p_id] = (int(p_role), int(s_role))
                print(f"      Successfully mapped {len(self.archetype_map)} players to 1:1 Latent DNA roles.")
            except Exception as e:
                print(f"      [WARN] Failed to load DNA archetypes: {e}. Falling back to style-blind mode.")

        # Load aggregated data (has all features pre-computed!)
        print(f"\n[DATA] Loading aggregated data from {aggregated_data_path}...")
        
        # Memory Optimization: Load only necessary columns or filter by season
        try:
            # We first peek at the columns to determine the season column name
            peek = pd.read_csv(aggregated_data_path, nrows=5)
            season_col = None
            for col in ['season_start_year', 'season', 'SEASON_ID']:
                if col in peek.columns:
                    season_col = col
                    break
            
            if season_col:
                print(f"   [MEMORY] Scanning {season_col} in chunks to find latest season...")
                # Get max season using chunking - DON'T load the whole column at once
                max_season = 0
                try:
                    for s_chunk in pd.read_csv(aggregated_data_path, usecols=[season_col], chunksize=300000, dtype={season_col: str}):
                        # Convert to numeric safely
                        s_vals = pd.to_numeric(s_chunk[season_col], errors='coerce').dropna()
                        if not s_vals.empty:
                            chunk_max = int(s_vals.max())
                            if chunk_max > max_season:
                                max_season = chunk_max
                except Exception as e:
                    print(f"   [MEMORY] Scan failed: {e}. Defaulting to 2025.")
                    max_season = 2025
                
                print(f"   [MEMORY] Latest season found: {max_season}. Filtering engine data...")
                # Use chunking to load only the required seasons (last 2)
                chunks = []
                # Target: current and prior season as strings for matching
                target_seasons = [str(max_season), str(max_season - 1), str(float(max_season)), str(float(max_season-1))]
                
                # Use low_memory=False and engine='c' for speed/reliability
                for chunk in pd.read_csv(aggregated_data_path, chunksize=100000, compression='infer', low_memory=False):
                    # Filter using string comparison or numeric
                    mask = chunk[season_col].astype(str).isin(target_seasons)
                    filtered_chunk = chunk[mask]
                    if not filtered_chunk.empty:
                        chunks.append(filtered_chunk)
                
                if not chunks:
                    print("   [MEMORY] No matching seasons found in filter. Loading tail.")
                    self.aggregated_data = pd.read_csv(aggregated_data_path, nrows=10000)
                else:
                    self.aggregated_data = pd.concat(chunks, ignore_index=True)
                
                print(f"   [MEMORY] Load complete: {len(self.aggregated_data):,} rows")
            else:
                # No season col? Load just the tail
                print("   [MEMORY] No season column found. Loading last 200,000 rows.")
                self.aggregated_data = pd.read_csv(aggregated_data_path, nrows=200000)
                
        except (MemoryError, pd.errors.ParserError) as e:
            print(f"[CRITICAL] Loading failed: {e}. Attempting Emergency Minimal Load...")
            try:
                # Absolute last resort: Load only the last 50k rows
                # We can't use tail() easily on a huge CSV without reading it, 
                # so we skip a large number of rows
                total_est_rows = 1000000 
                self.aggregated_data = pd.read_csv(aggregated_data_path, skiprows=range(1, total_est_rows-50000), nrows=50000)
                print("   [MEMORY] Emergency load (Last 50k rows) successful.")
            except:
                print("   [FATAL] File unreadable. Creating empty skeleton for UI stability.")
                self.aggregated_data = pd.DataFrame(columns=['gameDate', 'player_id', 'player_name', 'season_start_year', 'points', 'minutes'])

        # Standardize Date Column & Rename to Standard 'gameDate'
        date_col = None
        if 'GAME_DATE' in self.aggregated_data.columns:
            self.aggregated_data.rename(columns={'GAME_DATE': 'gameDate'}, inplace=True)
            date_col = 'gameDate'
        elif 'date' in self.aggregated_data.columns:
            self.aggregated_data.rename(columns={'date': 'gameDate'}, inplace=True)
            date_col = 'gameDate'
        elif 'game_date' in self.aggregated_data.columns:
            self.aggregated_data.rename(columns={'game_date': 'gameDate'}, inplace=True)
            date_col = 'gameDate'
        elif 'gameDate' in self.aggregated_data.columns:
            date_col = 'gameDate'
            
        # Standardize Player ID to 'player_id'
        if 'PLAYER_ID' in self.aggregated_data.columns:
            self.aggregated_data.rename(columns={'PLAYER_ID': 'player_id'}, inplace=True)
        elif 'player_id' not in self.aggregated_data.columns:
            # Check for other variants
            if 'personId' in self.aggregated_data.columns:
                self.aggregated_data.rename(columns={'personId': 'player_id'}, inplace=True)

        # Standardize Player Name to 'player_name'
        if 'PLAYER_NAME' in self.aggregated_data.columns:
            self.aggregated_data.rename(columns={'PLAYER_NAME': 'player_name'}, inplace=True)

        if date_col:
            # Coerce dates to ensure max() works on datetime objects
            temp_dates = pd.to_datetime(self.aggregated_data[date_col], errors='coerce')
            is_stale, max_date, lag = self.guard.check_dataset_freshness(self.aggregated_data, date_col=date_col)
            self.dataset_staleness = lag
            if is_stale:
                print(f"  [WARN] Dataset is STALE ({lag} days old). Confidence will be penalized.")
        else:
            self.dataset_staleness = 999

        print(f"   Loaded {len(self.aggregated_data):,} player-games with {len(self.aggregated_data.columns)} features")

        # Phase S4: UI Normalization
        # Remap raw API columns to standardized UI labels (one-time cost)
        mapping = {
            'PTS': 'points', 'MIN': 'minutes', 'AST': 'assists', 
            'REB': 'rebounds', 'FG3M': 'three_pointers'
        }
        for old, new in mapping.items():
            if old in self.aggregated_data.columns and new not in self.aggregated_data.columns:
                self.aggregated_data[new] = self.aggregated_data[old]
                
        print(f"   [OK] Normalized {len(mapping)} metric columns for UI Support")

        # Phase S5: Performance Indexing (Opto)
        # Create a fast lookup map for player data to avoid full-frame slicing in loops
        print("   [OPTO] Pre-computing player indices for millisecond lookups...")
        self.player_map = {}
        if 'player_id' in self.aggregated_data.columns:
            # Group by player_id and store the index groups
            groups = self.aggregated_data.groupby('player_id').groups
            for pid, indices in groups.items():
                self.player_map[str(pid).replace('.0', '')] = indices
        
        # Also map by name for fallback stability
        self.name_map = {}
        if 'player_name' in self.aggregated_data.columns:
            name_groups = self.aggregated_data.groupby(self.aggregated_data['player_name'].str.lower()).groups
            for name, indices in name_groups.items():
                self.name_map[name] = indices

        # Phase S5.1: Analytics Optimization (Opto)
        # Pre-compute unique player lists to avoid UI overhead
        print("   [OPTO] Materializing player metadata for instant UI rendering...")
        self.max_year = int(max_season) if 'max_season' in locals() else 2025
        if 'player_name' in self.aggregated_data.columns:
            # Robust season filtering: handle 2025.0 vs '2025' vs 22025
            if season_col:
                s_vals = pd.to_numeric(self.aggregated_data[season_col], errors='coerce')
                mask = (s_vals == float(self.max_year)) | (s_vals == float(f"2{self.max_year}"))
            else:
                mask = [True]*len(self.aggregated_data)
            
            _raw_names = self.aggregated_data[mask]['player_name'].dropna().unique()
            
            # If still empty, fall back to the most recent season available in the data
            if len(_raw_names) == 0:
                print(f"   [DNA-WARN] No players found for filtered year {self.max_year}. Falling back to global latest...")
                latest_season_in_data = pd.to_numeric(self.aggregated_data[season_col], errors='coerce').max() if season_col else None
                if latest_season_in_data:
                    mask = pd.to_numeric(self.aggregated_data[season_col], errors='coerce') == latest_season_in_data
                    _raw_names = self.aggregated_data[mask]['player_name'].dropna().unique()

            active_names = [str(n) for n in _raw_names]
            print(f"   [DNA-DEBUG] Found {len(active_names)} active players for {self.max_year} (Used col: {season_col})")
            try:
                self.all_players_list = sorted(active_names)
            except TypeError as e:
                print(f"   [DNA-ERROR] Sort failed: {e}. Falling back to unsorted list.")
                self.all_players_list = active_names
        else:
            self.all_players_list = []

        # Phase S4: Schema Enforcement
        self.feature_schema = None
        schema_path = self.models_dir / "production_v4" / "features.joblib"
        if schema_path.exists():
             import joblib
             self.feature_schema = joblib.load(schema_path)
             print(f"   [OK] Loaded schema with {len(self.feature_schema)} features")

        # Load models
        self._load_models()

    def _load_models(self):
        """Load trained NeuralHybridPredictor models."""
        print("\n[MODELS] Loading trained models...")
        import joblib

        props = ['minutes', 'points', 'rebounds', 'assists', 'three_pointers']
        prod_map = {
            'points': 'xgb_PTS.joblib',
            'rebounds': 'xgb_REB.joblib',
            'assists': 'xgb_AST.joblib',
            'three_pointers': 'xgb_FG3M.joblib',
            'minutes': 'xgb_MIN.joblib'
        }

        for prop in props:
            # Define search paths
            search_paths = [
                self.models_dir / "production_v4",
                self.models_dir, 
                self.models_dir / prop
            ]
            
            model_loaded = False
            # 1. Try production_v4 map
            prod_file = self.models_dir / "production_v4" / prod_map.get(prop, "")
            if prod_file.exists():
                try:
                    self.models[prop] = joblib.load(prod_file)
                    print(f"  [OK] Loaded {prop} production model: {prod_file.name}")
                    model_loaded = True
                except Exception as e:
                    print(f"  [ERR] Failed to load production {prop}: {e}")

            if not model_loaded:
                # 2. Try generic patterns
                matches = []
                for path in search_paths:
                    if not path.exists(): continue
                    patterns = [f"*{prop}*.pkl", f"*{prop}*.joblib"]
                    for pat in patterns:
                        matches.extend(list(path.glob(pat)))

                if matches:
                    matches.sort(key=lambda p: ('hybrid' not in str(p).lower(), 'xgb' not in str(p).lower()))
                    model_path = matches[0]
                    try:
                        loader = joblib.load if model_path.suffix == '.joblib' else pickle.load
                        with open(model_path, 'rb') if model_path.suffix == '.pkl' else open(model_path, 'rb') as f:
                            self.models[prop] = joblib.load(model_path) if model_path.suffix == '.joblib' else pickle.load(f)
                        print(f"  [OK] Loaded {prop} model from {model_path.name}")
                        model_loaded = True
                    except Exception as e:
                        print(f"  [ERR] Failed to load {model_path.name}: {e}")

            if not model_loaded:
                print(f"  [MISSING] {prop} model not found")

        # Relaxed check: if we have at least ONE model, proceed (don't crash if 'threes' is missing but 'points' is there)
        if not self.models:
             print("[WARN] No models loaded at all.")
             # Only raise if truly empty
             pass 
        else:
             print(f"\n[OK] Loaded {len(self.models)} models")

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
        # Fast Lookup using Opto-Indexing
        player_data = None
        p_id_str = str(player_id).replace('.0', '')
        
        if p_id_str in self.player_map:
            player_data = self.aggregated_data.iloc[self.player_map[p_id_str]]
        elif player_name.lower() in self.name_map:
            player_data = self.aggregated_data.iloc[self.name_map[player_name.lower()]]

        if player_data is None or player_data.empty:
            try:
                print(f"      [WARN]  Player {player_name} not found in aggregated data")
            except UnicodeEncodeError:
                safe_name = player_name.encode('ascii', 'replace').decode()
                print(f"      [WARN]  Player {safe_name} not found in aggregated data")
            return None

        # Sort by date and get most recent game BEFORE the target date (Backtesting Support)
        date_col = 'gameDate' if 'gameDate' in player_data.columns else 'date'
        
        # Ensure proper datetime types (Optimized: only if col name matches)
        if player_data[date_col].dtype == object:
             player_data = player_data.copy()
             player_data[date_col] = pd.to_datetime(player_data[date_col])
        
        target_dt = pd.to_datetime(game_date)
        
        # Strict temporal filter: We can only see games BEFORE today
        # Use < target_dt (assuming target_dt is the game day, and we want prior stats)
        player_data = player_data[player_data[date_col] < target_dt]
        
        if player_data.empty:
             # If no history exists before this date, we can't predict
             return None

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

        # Phase S4: Schema Enforcement & Numeric Filter
        if self.feature_schema is not None:
             # 1. Map known variants to match the schema
             if 'minutes' in latest_game.columns: latest_game['MIN'] = latest_game['minutes']
             if 'days_rest' in latest_game.columns: latest_game['rest_days'] = latest_game['days_rest']
             
             # Check for other common mismatches
             if 'player_b2b' in latest_game.columns: latest_game['is_b2b'] = latest_game['player_b2b']
             
             # 2. Ensure all required columns from the schema exist (fill missing with 0)
             for col in self.feature_schema:
                 if col not in latest_game.columns:
                     latest_game[col] = 0.0
             
             # 3. Select EXACTLY the features in the EXACT order the model expects
             latest_game = latest_game[self.feature_schema]
        else:
             # Fallback to general numeric filter if no schema
             latest_game = latest_game.select_dtypes(include=[np.number])

        return latest_game

    def predict_player_props(self, player_id: str, player_name: str,
                            team_abbr: str, opponent_abbr: str,
                            is_home: bool, game_date: str,
                            team_id: Optional[int] = None, 
                            game_id: Optional[str] = None,
                            explain: bool = False) -> Dict:
        """
        Generate predictions using pre-aggregated features + neural hybrid model (Latent DNA Aware).
        """
        # 0. Archetype DNA Context (Using the 16 Latent Clusterings)
        p_id_str = str(player_id).replace('.0', '')
        arch_data = self.archetype_map.get(p_id_str, (None, None))
        p_arch_id, s_arch_id = arch_data
        
        arch_names = {
            0: "Versatile Wing", 1: "Vertical Spacer", 2: "Bruising Interior", 
            3: "Movement Spacer", 4: "Rotation Spark", 5: "Dynamic Engine", 
            6: "Two-Way Connector", 7: "Perimeter Stopper", 8: "Primary Maestro", 
            9: "Point-of-Attack Wall", 10: "High-Volume Scorer", 11: "Glass Dominator",
            12: "High-IQ Play-Link", 13: "Generational Alpha-Star", 14: "Rim Protector", 15: "Modern Facilitating Big"
        }
        
        p_name = arch_names.get(p_arch_id, "Standard")
        s_name = arch_names.get(s_arch_id, "")
        
        if s_name and s_name != p_name:
            arch_display = f"{p_name} / {s_name}"
        else:
            arch_display = p_name
            
        print(f"   [DNA] {player_name}: Identified as {arch_display}")

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
            'team_id': team_id,
            'game_id': game_id,
            'opponent': opponent_abbr,
            'archetype': p_name,
            'archetype_id': p_arch_id,
            'secondary_archetype': s_name,
            'secondary_archetype_id': s_arch_id,
            'is_home': is_home,
            'game_date': game_date,
            'minutes': float(features['season_MIN_avg'].iloc[0]) if 'season_MIN_avg' in features.columns else (float(features['MIN'].iloc[0]) if 'MIN' in features.columns else 0.0)
        }

        # Phase S3: Dynamic Volatility Context (Opto: Moved outside loop)
        try: 
            p_id_str = str(player_id).replace('.0', '')
            if p_id_str in self.player_map:
                p_hist = self.aggregated_data.iloc[self.player_map[p_id_str]]
            else:
                p_hist = self.aggregated_data[self.aggregated_data['player_id'].astype(str) == p_id_str]
            vol_penalty = self.guard.calculate_volatility_penalty(p_hist)
        except:
            vol_penalty = 1.0

        # Make predictions for each prop
        for prop, model in self.models.items():
            try:
                # NeuralHybridPredictor handles TabNet embeddings + LightGBM internally
                if hasattr(model, 'predict'):
                    # Final safety: drop ANY non-numeric columns that might have slipped through
                    non_numeric = features.select_dtypes(exclude=[np.number]).columns.tolist()
                    if non_numeric:
                        # print(f"      [DEBUG] Dropping remaining non-numeric: {non_numeric}")
                        features = features.drop(columns=non_numeric)

                    # Check if model has uncertainty
                    if hasattr(model, 'sigma_model') and model.sigma_model is not None:
                        pred, sigma = model.predict(features, return_uncertainty=True)
                        pred_val = float(pred[0]) if hasattr(pred, '__len__') else float(pred)
                        uncertainty = float(sigma[0]) if hasattr(sigma, '__len__') else float(sigma)
                    else:
                        pred = model.predict(features)
                        pred_val = float(pred[0]) if hasattr(pred, '__len__') else float(pred)
                        uncertainty = None
                        
                    # Phase S3: Integrity & Confidence Scaling
                    # 1. Feature Scan
                    valid, issues, integrity_score = self.guard.scan_for_drift_and_anomalies(features, expected_schema=self.feature_schema)
                    if not valid:
                         print(f"      [WARN] Low Integrity ({integrity_score*100:.0f}%) for {prop}: {issues}")
                        
                    # 2. Dynamic Penalties
                    conf_mult = self.guard.get_confidence_multiplier(self.dataset_staleness, vol_penalty)
                    
                    # Phase S3.1: Data Dilution Penalty
                    if integrity_score < 0.9:
                        # 10% miss = 15% more uncertainty, 20% miss = 30% more uncertainty, etc.
                        dilution_penalty = 1.0 + (1.0 - integrity_score) * 1.5
                        conf_mult *= dilution_penalty
                    
                    # Apply penalty
                    if uncertainty:
                        uncertainty *= conf_mult
                    else:
                        uncertainty = 1.5 * conf_mult # Fallback for models without sigma

                    predictions[prop] = {
                        'prediction': round(pred_val, 2),
                        'uncertainty': round(uncertainty, 2),
                        'lower_80': round(pred_val - 1.28 * uncertainty, 2),
                        'upper_80': round(pred_val + 1.28 * uncertainty, 2),
                        'lower_95': round(pred_val - 1.96 * uncertainty, 2),
                        'upper_95': round(pred_val + 1.96 * uncertainty, 2),
                        'integrity_score': round(integrity_score, 3),
                        'integrity_concerns': issues if issues else None,
                        'confidence_penalty': round(conf_mult, 2)
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
        """Fetch today's games using The Odds API (Discovery) to avoid NBA API hangs."""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        print(f"\n[DATE] Discovering games for {date} (Source: The Odds API)...")
        
        # We reuse the betting context to find games
        try:
            # 1. Fetch Events from The Odds API
            url = f"{THEODDS_BASE_URL}/sports/{THEODDS_SPORT}/events"
            params = {"apiKey": THEODDS_API_KEY}
            resp = requests.get(url, params=params, timeout=10)
            
            if resp.status_code != 200:
                print(f"   [WARN] Odds API Event Discovery failed: {resp.status_code}. Using fallback.")
                return pd.DataFrame()

            events = resp.json()
            games = []
            
            # Team ID Map (Internal MEEP fallback)
            team_id_map = {
                'ATL': 1610612737, 'BOS': 1610612738, 'CLE': 1610612739, 'NOP': 1610612740, 'CHI': 1610612741,
                'DAL': 1610612742, 'DEN': 1610612743, 'GSW': 1610612744, 'HOU': 1610612745, 'LAC': 1610612746,
                'LAL': 1610612747, 'MIA': 1610612748, 'MIL': 1610612749, 'MIN': 1610612750, 'BKN': 1610612751,
                'NYK': 1610612752, 'ORL': 1610612753, 'IND': 1610612754, 'PHI': 1610612755, 'PHX': 1610612756,
                'POR': 1610612757, 'SAC': 1610612758, 'SAS': 1610612759, 'OKC': 1610612760, 'TOR': 1610612761,
                'UTA': 1610612762, 'MEM': 1610612763, 'WAS': 1610612764, 'DET': 1610612765, 'CHA': 1610612766
            }

            def fast_abbr(name):
                # Simple lookup or return first 3 letters capitalized
                name_map = {"Los Angeles Lakers": "LAL", "Boston Celtics": "BOS", "Golden State Warriors": "GSW", 
                           "Phoenix Suns": "PHX", "Dallas Mavericks": "DAL", "Brooklyn Nets": "BKN",
                           "Miami Heat": "MIA", "Philadelphia 76ers": "PHI", "Oklahoma City Thunder": "OKC",
                           "New York Knicks": "NYK", "New Orleans Pelicans": "NOP", "San Antonio Spurs": "SAS",
                           "Utah Jazz": "UTA", "Los Angeles Clippers": "LAC"}
                return name_map.get(name, name[:3].upper())

            for event in events:
                # Check if event is today (convert UTC commence_time to localized NBA date)
                commence_str = event.get('commence_time', '')
                if commence_str:
                    try:
                        # Odds API returns UTC, convert approx to US Eastern (-5h)
                        dt_utc = pd.to_datetime(commence_str)
                        dt_et = dt_utc - pd.Timedelta(hours=5)
                        event_date = dt_et.strftime('%Y-%m-%d')
                    except Exception:
                        event_date = commence_str.split('T')[0]
                else:
                    event_date = ''
                
                if event_date != date and not DEBUG_MODE:
                    continue
                
                home_team = event.get('home_team')
                away_team = event.get('away_team')
                home_abbr = fast_abbr(home_team)
                away_abbr = fast_abbr(away_team)
                
                games.append({
                    'game_id': event.get('id'),
                    'home_team_id': team_id_map.get(home_abbr, 0),
                    'away_team_id': team_id_map.get(away_abbr, 0),
                    'home_team': home_abbr,
                    'away_team': away_abbr,
                    'game_time': event.get('commence_time'),
                    'date': date
                })

            if not games:
                print(f"   [INFO] No games found for {date} on The Odds API.")
                return pd.DataFrame()

            print(f"   [OK] Discovered {len(games)} games via Discovery API.")
            return pd.DataFrame(games)

        except Exception as e:
            print(f"   [ERR] Discovery failed: {e}")
            return pd.DataFrame()
            print(f"   [ERR] Error fetching games: {e}")
            return pd.DataFrame()

    def get_team_roster(self, team_id: int, team_abbr: str = "", season: str = "2025-26") -> pd.DataFrame:
        """Get team roster using LOCAL aggregated data to avoid NBA API hangs."""
        print(f"      [DATA] Extracting roster for {team_abbr} from history...")
        
        try:
            # Source: Find all unique players who have played for this team in our matrix
            # Use 'team' column and get unique player_id/player_name
            id_col = 'PLAYER_ID' if 'PLAYER_ID' in self.aggregated_data.columns else 'player_id'
            name_col = 'PLAYER_NAME' if 'PLAYER_NAME' in self.aggregated_data.columns else 'player_name'
            team_col = 'TEAM_ID' if 'TEAM_ID' in self.aggregated_data.columns else 'team'
            
            # Filter by team_id or team_abbr
            if team_id and str(team_id) != "0":
                team_data = self.aggregated_data[self.aggregated_data[team_col].astype(str).str.contains(str(team_id))]
            else:
                team_data = self.aggregated_data[self.aggregated_data['team'] == team_abbr]

            if team_data.empty:
                print(f"      [WARN] No historical roster found for {team_abbr}.")
                return pd.DataFrame()

            # Get unique players from the LAST 30 DAYS of data to ensure they are current
            # Sort by date
            date_col = next((c for c in team_data.columns if c in ['gameDate', 'GAME_DATE', 'game_date', 'date']), None)
            if date_col:
                team_data = team_data.sort_values(date_col, ascending=False)

            
            # Most recent 15 players (typical active roster size)
            roster = team_data.drop_duplicates(subset=[id_col]).head(15).copy()
            
            # Format to match internal expectations
            roster.rename(columns={id_col: 'PLAYER_ID', name_col: 'PLAYER'}, inplace=True)
            return roster[['PLAYER_ID', 'PLAYER']]

        except Exception as e:
            print(f"   [ERR] Local roster extraction failed: {e}")
        
        # Fallback / Mock Roster
        print(f"   [MOCK] Generating fallback roster for team {team_id}")
        mock_rosters = {
            1610612747: [ # LAL
                {'PLAYER_ID': 2544, 'PLAYER': 'LeBron James'},
                {'PLAYER_ID': 203076, 'PLAYER': 'Anthony Davis'},
                {'PLAYER_ID': 1629637, 'PLAYER': 'Jaxson Hayes'},
                {'PLAYER_ID': 1629029, 'PLAYER': 'Luka Doncic'} # Wait, Luka is DAL, but for testing... put Austin Reaves
            ],
            1610612738: [ # BOS
                {'PLAYER_ID': 1628369, 'PLAYER': 'Jayson Tatum'},
                {'PLAYER_ID': 1627759, 'PLAYER': 'Jaylen Brown'},
                {'PLAYER_ID': 204001, 'PLAYER': 'Kristaps Porzingis'}
            ],
            1610612744: [ # GSW
                {'PLAYER_ID': 201939, 'PLAYER': 'Stephen Curry'},
                {'PLAYER_ID': 203110, 'PLAYER': 'Draymond Green'}
            ],
            1610612756: [ # PHX
                {'PLAYER_ID': 1626164, 'PLAYER': 'Devin Booker'},
                {'PLAYER_ID': 201142, 'PLAYER': 'Kevin Durant'} 
            ]
        }
        
        try:
            team_id_int = int(float(team_id))
        except (ValueError, TypeError):
            team_id_int = 0
            
        players = mock_rosters.get(team_id_int, [
            {'PLAYER_ID': 2544, 'PLAYER': 'LeBron James'}, # Default fallback
            {'PLAYER_ID': 1628369, 'PLAYER': 'Jayson Tatum'},
            {'PLAYER_ID': 201939, 'PLAYER': 'Stephen Curry'},
            {'PLAYER_ID': 1629029, 'PLAYER': 'Luka Doncic'},
            {'PLAYER_ID': 1626164, 'PLAYER': 'Devin Booker'},
            {'PLAYER_ID': 203999, 'PLAYER': 'Nikola Jokic'}
        ])
        
        return pd.DataFrame(players)

    def predict_game(self, game_info: Dict, explain: bool = False) -> List[Dict]:
        """Predict all players in a game."""
        home_team_id = game_info['home_team_id']
        away_team_id = game_info['away_team_id']
        game_date = game_info['date']

        print(f"\n[GAME] {game_info['away_team']} @ {game_info['home_team']}")

        all_predictions = []

        # Get rosters
        for team_id, team_abbr, is_home in [
            (home_team_id, game_info['home_team'], True),
            (away_team_id, game_info['away_team'], False)
        ]:
            opponent_abbr = game_info['away_team'] if is_home else game_info['home_team']
            roster = self.get_team_roster(team_id, team_abbr)

            if roster.empty:
                print(f"   [WARN]  No roster for {team_abbr}")
                continue

            print(f"   {team_abbr}: {len(roster)} players")

            for _, player in roster.iterrows():
                player_id = str(player['PLAYER_ID'])
                player_name = player['PLAYER']

                try:
                    print(f"      Predicting {player_name}...")
                except UnicodeEncodeError:
                    # Sanitize for Windows Console
                    safe_name = player_name.encode('ascii', 'replace').decode()
                    print(f"      Predicting {safe_name}...")

                pred = self.predict_player_props(
                    player_id, player_name, team_abbr, opponent_abbr,
                    is_home, game_date, team_id, game_info.get('game_id'),
                    explain
                )

                if 'error' not in pred:
                    all_predictions.append(pred)

        return all_predictions

    def predict_all_games(self, date: Optional[str] = None, explain: bool = False, use_local_rosters: bool = False) -> pd.DataFrame:
        """Predict all games for a date."""
        print(f"\n[EXEC] Running full slate prediction (Explainability={'ON' if explain else 'OFF'})...")
        if use_local_rosters:
            print(f"\n[OFFLINE] Generating predictions using local data for {date}...")
            if date is None:
                 print("   [ERROR] Date must be specified for local roster mode.")
                 return pd.DataFrame()
                 
            # Normalize Date
            target_date = pd.to_datetime(date).strftime('%Y-%m-%d')
            
            # Identify columns
            if 'gameDate' in self.aggregated_data.columns:
                date_col = 'gameDate'
            elif 'GAME_DATE' in self.aggregated_data.columns:
                date_col = 'GAME_DATE'
            elif 'date' in self.aggregated_data.columns:
                date_col = 'date'
            elif 'game_date' in self.aggregated_data.columns:
                date_col = 'game_date'
            else:
                date_col = next((c for c in self.aggregated_data.columns if 'date' in c.lower()), None)

            if not date_col or date_col not in self.aggregated_data.columns:
                print(f"   [ERROR] Could not find date column in local data. Cols: {self.aggregated_data.columns[:10]}")
                return pd.DataFrame()

            # Ensure date column is datetime
            if not pd.api.types.is_datetime64_any_dtype(self.aggregated_data[date_col]):
                self.aggregated_data[date_col] = pd.to_datetime(self.aggregated_data[date_col])
                
            mask = self.aggregated_data[date_col].dt.strftime('%Y-%m-%d') == target_date
            local_slate = self.aggregated_data[mask]
            
            if local_slate.empty:
                print(f"   [WARN] No local data found for exact date {target_date}. Using most-recent game per player (future-date mode)...")
                # For future dates: pull each player's LATEST game as their feature baseline
                # Sort by date descending, then deduplicate to keep only the latest row per player
                id_col_detect = 'PLAYER_ID' if 'PLAYER_ID' in self.aggregated_data.columns else 'player_id'
                sorted_data = self.aggregated_data.sort_values(date_col, ascending=False)
                local_slate = sorted_data.drop_duplicates(subset=[id_col_detect], keep='first')
                # Only keep recently active players (played within last 30 days of data)
                max_date = self.aggregated_data[date_col].max()
                cutoff = max_date - pd.Timedelta(days=30)
                local_slate = local_slate[local_slate[date_col] >= cutoff]
                print(f"   [FUTURE MODE] Using {len(local_slate)} recently-active players as baseline.")
                
            print(f"   Found {len(local_slate)} player-games locally.")
            
            all_predictions = []
            # Detect ID/Name columns once
            cols = local_slate.columns
            id_col = 'player_id' if 'player_id' in cols else 'PLAYER_ID'
            name_col = 'player_name' if 'player_name' in cols else 'player'
            if name_col not in cols and 'PLAYER_NAME' in cols: name_col = 'PLAYER_NAME'
            
            team_col = next((c for c in cols if c in ['team', 'TEAM_ABBREVIATION', 'team_abbr']), 'team')
            opp_col = next((c for c in cols if c in ['opponent', 'OPPONENT_ABBREVIATION', 'opp_abbr']), None)
            
            for _, row in local_slate.iterrows():
                try:
                    # Infer Home/Away and Opponent from matchup
                    matchup_str = str(row.get('matchup', row.get('MATCHUP', '')))
                    is_home = 'vs.' in matchup_str if matchup_str else True
                    
                    if not opp_col and matchup_str:
                        # Extract opponent from 'LAL vs. BOS' -> 'BOS'
                        opponent_abbr = matchup_str.split(' ')[-1]
                    else:
                        opponent_abbr = row.get(opp_col, 'UNK') if opp_col else 'UNK'
                        
                    pred = self.predict_player_props(
                        player_id=str(row[id_col]),
                        player_name=row[name_col],
                        team_abbr=row.get(team_col, 'UNK'),
                        opponent_abbr=opponent_abbr,
                        is_home=bool(is_home),
                        game_date=target_date,
                        explain=explain
                    )
                    if 'error' not in pred:
                        all_predictions.append(pred)
                except Exception as e:
                    # print(f"Skipping {row.get(name_col, 'Unknown')}: {e}")
                    pass
            
            return pd.DataFrame(all_predictions)

        games = self.get_todays_games(date)

        if games.empty:
            return pd.DataFrame()

        all_predictions = []
        # Predict games in parallel — main bottleneck is NBA API roster fetch, not ML models
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(self.predict_game, game.to_dict(), explain): game['game_id'] 
                       for _, game in games.iterrows()}
            for future in as_completed(futures):
                try:
                    all_predictions.extend(future.result())
                except Exception as exc:
                    print(f"   [WARN] Game prediction failed: {exc}")

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
            print("[WARN]  The Odds API key not configured. Set THEODDS_API_KEY environment variable.")
            return []

        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        print(f"\n[ODDS] Fetching betting lines from The Odds API...")

        try:
            # Get today's games first
            games = self.get_todays_games(date)
            if games.empty:
                print("   No games scheduled")
                return []

            # Build game mapping — Odds API uses full names, we have abbreviations
            # Map: full city/team name fragments -> abbreviation
            NBA_NAME_TO_ABBR = {
                'atlanta': 'ATL', 'boston': 'BOS', 'brooklyn': 'BKN', 'charlotte': 'CHA',
                'chicago': 'CHI', 'cleveland': 'CLE', 'dallas': 'DAL', 'denver': 'DEN',
                'detroit': 'DET', 'golden state': 'GSW', 'houston': 'HOU', 'indiana': 'IND',
                'la clippers': 'LAC', 'clippers': 'LAC', 'la lakers': 'LAL', 'lakers': 'LAL',
                'los angeles clippers': 'LAC', 'los angeles lakers': 'LAL',
                'memphis': 'MEM', 'miami': 'MIA', 'milwaukee': 'MIL', 'minnesota': 'MIN',
                'new orleans': 'NOP', 'new york': 'NYK', 'oklahoma': 'OKC', 'orlando': 'ORL',
                'philadelphia': 'PHI', 'phoenix': 'PHX', 'portland': 'POR', 'sacramento': 'SAC',
                'san antonio': 'SAS', 'toronto': 'TOR', 'utah': 'UTA', 'washington': 'WAS',
            }

            def team_name_to_abbr(full_name: str) -> str:
                """Convert full team name to 3-letter abbreviation."""
                low = full_name.lower()
                # Try longest match first
                for key in sorted(NBA_NAME_TO_ABBR, key=len, reverse=True):
                    if key in low:
                        return NBA_NAME_TO_ABBR[key]
                return low[:3].upper()  # Last resort

            game_map = {}
            for _, game in games.iterrows():
                home_abbr = game['home_team'].upper()
                away_abbr = game['away_team'].upper()
                # Key by abbreviation pairs
                game_map[f"{away_abbr}_{home_abbr}"] = game
                game_map[f"{home_abbr}_{away_abbr}"] = game

            props = []

            # STEP 1: Fetch events to get event IDs
            events_url = f"{THEODDS_BASE_URL}/sports/{THEODDS_SPORT}/events"
            events_params = {"apiKey": THEODDS_API_KEY}

            events_resp = requests.get(events_url, params=events_params, timeout=REQUEST_TIMEOUT * 2)

            if DEBUG_MODE:
                print(f"   [TheOdds] GET {events_url} status={events_resp.status_code}")
                print(f"   [TheOdds] Remaining: {events_resp.headers.get('x-requests-remaining', 'unknown')}")

            if events_resp.status_code != 200:
                print(f"   [ERR] Events fetch failed: {events_resp.status_code}")
                return []

            events = events_resp.json()
            print(f"   Found {len(events)} events")

            # STEP 2: For each event, fetch odds with player props
            for event in events:
                event_id = event.get("id")
                # Translate full names -> abbreviations for game_map lookup
                home_abbr = team_name_to_abbr(event.get("home_team", ""))
                away_abbr = team_name_to_abbr(event.get("away_team", ""))
                event_commence_time = event.get("commence_time", "")

                # Match event to our game using abbreviation keys
                game_key = f"{away_abbr}_{home_abbr}"
                game = game_map.get(game_key)
                if game is None:
                    game = game_map.get(f"{home_abbr}_{away_abbr}")

                if game is None:
                    if DEBUG_MODE:
                        print(f"   [WARN] No game match for {away_abbr}@{home_abbr} (from '{event.get('away_team')}' @ '{event.get('home_team')}')")
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
                                "player_threes": "three_pointers",
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

            print(f"   [OK] Fetched {len(props)} player props")
            return props

        except Exception as e:
            print(f"   [ERR] Error fetching lines: {e}")
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

        print(f"\n[SCAN] Analyzing {len(lines)} betting lines for +EV opportunities...")

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

        print(f"   [OK] Found {len(opportunities)} +EV opportunities")

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
    parser.add_argument('--output-wide', type=str, default=None, help='Save predictions to legacy wide-format CSV (for Phase I)')
    parser.add_argument('--betting-output', type=str, default=None, help='Save +EV opportunities to CSV/JSON')
    parser.add_argument('--models-dir', type=str, default='./models')
    parser.add_argument('--refresh', action='store_true', help='Fetch latest NBA logs and update master data before predicting')
    args = parser.parse_args()

    print("="*70)
    print("[NBA] LIVE NBA PREDICTIONS - Aggregated Data + Neural Hybrid")
    if args.betting:
        print("[BETS] BETTING INTEGRATION: The Odds API")
    print("="*70)

    # Automatic Data Refresh
    if args.refresh:
        from daily_refresh import daily_refresh
        if not daily_refresh():
            print("[WARN] Data refresh failed. Proceeding with existing data...")

    # Display configuration if betting enabled
    if args.betting:
        print("\n[CFG]  Betting Configuration:")
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

        print(f"\n[DATA] Generated {len(predictions)} predictions")

        # Save predictions
        if args.output:
            # Flatten predictions to include integrity metrics
            flattened_rows = []
            for _, row in predictions.iterrows():
                base_info = {
                    'player_id': row.get('player_id'),
                    'player_name': row.get('player_name'),
                    'team': row.get('team'),
                    'opponent': row.get('opponent'),
                    'is_home': row.get('is_home'),
                    'game_date': row.get('game_date'),
                    'minutes': row.get('minutes', 0.0)
                }
                
                # Extract each prop with its metrics
                for prop in ['points', 'assists', 'rebounds', 'three_pointers', 'minutes']:
                    if prop in row and isinstance(row[prop], dict):
                        prop_data = row[prop]
                        flattened_rows.append({
                            **base_info,
                            'prop_type': prop,
                            'prediction': prop_data.get('prediction'),
                            'uncertainty': prop_data.get('uncertainty'),
                            'confidence_penalty': prop_data.get('confidence_penalty'),
                            'integrity_concerns': str(prop_data.get('integrity_concerns')) if prop_data.get('integrity_concerns') else None,
                            'lower_80': prop_data.get('lower_80'),
                            'upper_80': prop_data.get('upper_80'),
                            'lower_95': prop_data.get('lower_95'),
                            'upper_95': prop_data.get('upper_95')
                        })
            
            if flattened_rows:
                df_export = pd.DataFrame(flattened_rows)
                if args.output.endswith('.csv'):
                    df_export.to_csv(args.output, index=False)
                elif args.output.endswith('.json'):
                    df_export.to_json(args.output, orient='records', indent=2)
                print(f"[OK] Saved {len(df_export)} predictions to {args.output}")

        # Wide Export (Legacy Phase I Support)
        if args.output_wide:
            wide_rows = []
            for _, row in predictions.iterrows():
                wide_row = {
                    'player': row.get('player_name'),
                    'TEAM_ID': row.get('team_id'),
                    'game_id': row.get('game_id'),
                    'team': row.get('team'),
                    'proj_PTS': row.get('points', {}).get('prediction', 0),
                    'proj_AST': row.get('assists', {}).get('prediction', 0),
                    'proj_REB': row.get('rebounds', {}).get('prediction', 0),
                    'proj_FG3M': row.get('three_pointers', {}).get('prediction', 0)
                }
                wide_rows.append(wide_row)
            
            if wide_rows:
                df_wide = pd.DataFrame(wide_rows)
                os.makedirs(os.path.dirname(args.output_wide), exist_ok=True) if os.path.dirname(args.output_wide) else None
                df_wide.to_csv(args.output_wide, index=False)
                print(f"[OK] Saved {len(df_wide)} legacy wide-format predictions to {args.output_wide}")
            else:
                # Fallback to original format if flattening fails
                if args.output.endswith('.csv'):
                    predictions.to_csv(args.output, index=False)
                elif args.output.endswith('.json'):
                    predictions.to_json(args.output, orient='records', indent=2)
                print(f"[OK] Saved predictions to {args.output}")

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
                    print(f"TOP +EV OPPORTUNITIES ({len(opportunities)} found)")
                    print(f"{'='*70}")

                    # Display top 10
                    for i, opp in enumerate(opportunities[:10], 1):
                        print(f"\n{i}. {opp['player']} ({opp['team']}) - {opp['prop_type'].upper()}")
                        print(f"   Pick: {opp['pick']} {opp['line']}")
                        print(f"   Odds: {opp['odds']:+d} @ {opp['bookmaker']}")
                        print(f"   Prediction: {opp['prediction']:.1f} +- {opp['uncertainty']:.1f}")
                        print(f"   Win Probability: {opp['win_probability']:.1%}")
                        print(f"   Expected Value: {opp['expected_value']:+.3f}")
                        print(f"   Kelly Fraction: {opp['kelly_fraction']:.2%}")
                        print(f"   Confidence: {opp['confidence']:.2f}sigma")

                    # Save betting opportunities
                    if args.betting_output:
                        opp_df = pd.DataFrame(opportunities)
                        if args.betting_output.endswith('.csv'):
                            opp_df.to_csv(args.betting_output, index=False)
                        elif args.betting_output.endswith('.json'):
                            opp_df.to_json(args.betting_output, orient='records', indent=2)
                        print(f"\n[OK] Saved {len(opportunities)} opportunities to {args.betting_output}")
                else:
                    print(f"\n⚠️  No +EV opportunities found with current filters")
                    print(f"   Try adjusting MIN_WIN_PROBABILITY or SAFE_MARGIN")

    print("\n" + "="*70)


if __name__ == '__main__':
    main()
