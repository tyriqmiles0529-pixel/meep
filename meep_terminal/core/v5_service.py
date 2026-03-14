import os
# PHASE V5: Global Environment Hardening
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

# Local imports
from meep_terminal.core.monte_carlo import MonteCarloEngine
from services.market_service.odds_normalizer import american_to_implied
from services.market_service.edge_detector import detect_edge
from v6_engine.possession_flow_service import V6PossessionFlowService

class V5PredictionService:
    """
    V5 Evolution: Modular Prediction Service.
    Standardizes inference between V4 Ensemble and V5 Monte Carlo distributions.
    """
    def __init__(self, models_dir: str = "./models", 
                 cache_dir: str = "./cache"):
        self.models_dir = Path(models_dir)
        self.cache_dir = Path(cache_dir)
        
        # PHASE V5: Load Production V4 Models (Ensemble Hub)
        # We load XGB, LGB, CAT, and RIDGE for each stat target.
        self.models = self._load_production_models()
        
        # Initialize V5 Simulation Engine
        self.mc_engine = MonteCarloEngine(iterations=10000)
        
        # Initialize V6 Possession Flow Core
        self.v6_engine = V6PossessionFlowService(dna_dim=16)
        
        # Feature Schema Identification
        self.feature_cols = self._load_schema()
        
        # Data Persistence (V4/V5 Optimized Matrix)
        # Using the file updated by daily_refresh.py
        self.data_path = Path("final_feature_matrix_with_per_min_1997_onward.csv")
        self._load_data()
        
        # UI Alias for player_view.py / Deep Dive compatibility
        self.aggregated_data = self.df
        
    def reload_data(self):
        """High-performance data reload that preserves models and engines."""
        self._load_data()
        self.aggregated_data = self.df
        return True
        
    def _load_production_models(self) -> Dict:
        """Loads the verified 2025 production models from the models directory."""
        from catboost import CatBoostRegressor
        
        stat_models = {}
        targets = {
            'points': 'production_v4', 
            'assists': 'production_v4', 
            'rebounds': 'production_v4',
            'threes': 'production_v4'
        }
        
        for target, folder in targets.items():
            target_dir = self.models_dir / folder
            if not target_dir.exists():
                # Fallback to production_v4 if specific folder doesn't exist
                target_dir = self.models_dir / "production_v4"
                if not target_dir.exists():
                     print(f"[V5-WARN] Missing models directory: {target_dir}")
                     continue
                
            try:
                # Resolve filenames based on target
                fname_map = {
                    'points': 'PTS',
                    'assists': 'AST',
                    'rebounds': 'REB',
                    'threes': 'FG3M'
                }
                suffix = fname_map.get(target, target.upper())
                
                # Check for legacy (.pkl) or modern (.joblib)
                xgb_path = target_dir / f"xgb_{suffix}.joblib"
                if not xgb_path.exists(): xgb_path = target_dir / f"xgb_model_2025.pkl"
                
                lgb_path = target_dir / f"lgb_{suffix}.joblib"
                if not lgb_path.exists(): lgb_path = target_dir / f"lgb_model_2025.pkl"
                
                cat_path = target_dir / f"cat_{suffix}.cbm"
                if not cat_path.exists(): cat_path = target_dir / f"cat_model_2025.cbm"
                
                ridge_path = target_dir / f"ridge_{suffix}.joblib"
                if not ridge_path.exists(): ridge_path = target_dir / f"ridge_model_2025.pkl"

                stat_models[target] = {
                    'xgb': joblib.load(xgb_path) if xgb_path.exists() else None,
                    'lgb': joblib.load(lgb_path) if lgb_path.exists() else None,
                    'cat': CatBoostRegressor().load_model(str(cat_path)) if cat_path.exists() else None,
                    'ridge': joblib.load(ridge_path) if ridge_path.exists() else None
                }
                print(f"[V5-OK] Loaded {target} Production Ensemble.")
            except Exception as e:
                print(f"[V5-ERROR] Failed to load {target} models: {e}")
                
        return stat_models

    def _load_schema(self) -> List[str]:
        schema_path = self.models_dir / "production_v4" / "features.joblib"
        if schema_path.exists():
            return joblib.load(schema_path)
        return []

    def _load_data(self):
        """Loads and indexes the V5 feature matrix with Parquet shadow caching."""
        if not self.data_path.exists():
            print(f"[V5-WARN] Feature Matrix not found at {self.data_path}. Initializing empty.")
            self.df = pd.DataFrame()
            self.player_map = {}
            self.name_map = {}
            return

        parquet_path = self.data_path.with_suffix('.parquet')
        use_cache = False
        
        if parquet_path.exists():
            csv_mtime = self.data_path.stat().st_mtime
            pq_mtime = parquet_path.stat().st_mtime
            if pq_mtime > csv_mtime:
                use_cache = True
        
        if use_cache:
            print(f"[V5] Loading CACHED Feature Matrix (PARQUET): {parquet_path}")
            self.df = pd.read_parquet(parquet_path)
        else:
            print(f"[V5] Loading Feature Matrix (CSV) -> Generating Cache: {self.data_path}")
            self.df = pd.read_csv(self.data_path, low_memory=False)
            
            # Standardize columns for cross-version compatibility
            col_map = {
                'player_name': 'PLAYER_NAME',
                'player_id': 'PLAYER_ID',
                'gameDate': 'GAME_DATE',
                'matchup': 'MATCHUP',
                'MIN': 'minutes',
                'PTS': 'points',
                'AST': 'assists',
                'REB': 'reboundsTotal',
                'FG3M': 'three_pointers'
            }
            for c1, c2 in col_map.items():
                if c1 in self.df.columns and c2 not in self.df.columns: self.df[c2] = self.df[c1]
                elif c2 in self.df.columns and c1 not in self.df.columns: self.df[c1] = self.df[c2]
            
            # Optimized date conversion for the cache
            date_col = next((c for c in ['gameDate', 'GAME_DATE', 'date'] if c in self.df.columns), None)
            if date_col and not pd.api.types.is_datetime64_any_dtype(self.df[date_col]):
                self.df[date_col] = pd.to_datetime(self.df[date_col], errors='coerce')

            # Save to Parquet for future fast loads
            try:
                self.df.to_parquet(parquet_path, index=False)
                print(f"[V5-CACHE] Shadow cache updated: {parquet_path}")
            except Exception as e:
                print(f"[V5-WARN] Failed to write shadow cache: {e}")

        # Post-load indexing
        print(f"[V5-DATA] Matrix loaded. Shape: {self.df.shape}")
        
        # ID/Name Mapping setup
        id_col = 'PLAYER_ID' if 'PLAYER_ID' in self.df.columns else ('player_id' if 'player_id' in self.df.columns else None)
        name_col = 'PLAYER_NAME' if 'PLAYER_NAME' in self.df.columns else ('player_name' if 'player_name' in self.df.columns else None)
        
        if id_col and name_col:
            # Convert IDs to strings once for faster lookup
            if self.df[id_col].dtype != object:
                self.df[id_col] = self.df[id_col].astype(str).str.replace('.0', '', regex=False)
            
            self.player_map = self.df.groupby(id_col).groups
            self.name_map = self.df.groupby(self.df[name_col].str.lower()).groups
        else:
            self.player_map = {}
            self.name_map = {}

    def get_player_stats(self, player_name: str) -> Optional[pd.DataFrame]:
        """UI Helper: Get historical stats for the Pilot dashboard."""
        if player_name.lower() in self.name_map:
            return self.df.iloc[self.name_map[player_name.lower()]].sort_values('gameDate', ascending=False)
        return None

    def predict_prop_distribution(self, player_name: str, prop: str, line: float, context: Optional[dict] = None) -> Dict:
        """
        V5 Core: Monte Carlo Outcome Inference.
        Returns full probability map and win confidence.
        """
        # 1. Fetch latest features
        p_stats = self.get_player_stats(player_name)
        if p_stats is None or p_stats.empty:
            return {"error": "Player history unavailable."}
        
        # 2. V4 Ensemble Prediction (Point Estimate)
        target = prop.lower()
        if target in ['pts', 'points']: target = 'points'
        elif target in ['ast', 'assists']: target = 'assists'
        elif target in ['reb', 'rebounds']: target = 'rebounds'
        elif target in ['3pm', 'threes', 'fg3m']: target = 'threes'
        
        m_set = self.models.get(target)
        if not m_set:
            return {"error": f"Ensemble for '{target}' not loaded."}
            
        X_input = p_stats.iloc[0:1].copy()
        try:
            # Match feature schema of the production model (Defensive Reindexing)
            feat_cols = getattr(m_set['xgb'], 'feature_names_in_', [])
            X_feats = X_input.reindex(columns=feat_cols, fill_value=0)
            
            preds = []
            if m_set.get('xgb') is not None:
                preds.append(m_set['xgb'].predict(X_feats)[0])
            if m_set.get('lgb') is not None:
                preds.append(m_set['lgb'].predict(X_feats)[0])
            if m_set.get('cat') is not None:
                preds.append(m_set['cat'].predict(X_feats)[0])
            if m_set.get('ridge') is not None:
                preds.append(m_set['ridge'].predict(X_feats)[0])
            
            if preds:
                mu = sum(preds) / len(preds)
            else:
                # Absolute fallback if no models worked
                mu = p_stats.iloc[0]['PTS'] if 'PTS' in p_stats.columns else 20.0
        except Exception as e:
            # Fallback to simple mean if schema mismatch occurs
            print(f"[V5-WARN] Inference failed: {e}")
            mu = p_stats.iloc[0]['PTS'] if 'PTS' in p_stats.columns else 20.0
        
        # 3. Standard deviation derivation
        sigma_map = {'points': 4.5, 'assists': 1.8, 'rebounds': 2.0, 'threes': 0.8}
        sigma = sigma_map.get(target, 2.0)
        
        # 4. V5 Monte Carlo Simulation
        metrics = self.mc_engine.calculate_probabilities(mu, sigma, line, prop)
        
        # 5. Contextual Adjustments (Lineup, Availability)
        if context and context.get('is_star_out'):
            # V5 Logic: Boost usage DNA by 15% if primary co-star is out
            metrics['expected_value'] *= 1.15
            metrics['win_prob'] = min(0.99, metrics['win_prob'] * 1.05) # Conservative boost, capped at 99%
            
        return {
            "player": player_name,
            "prop": prop,
            "line": line,
            "projection": round(metrics['expected_value'], 2),
            "win_prob": round(metrics['win_prob'] * 100, 1),
            "side": metrics['side'],
            "distribution": metrics, # Full MC stats
            "model_version": "V5-Sim-Ensemble-v1"
        }

    def get_value_bets(self, slate: List[Dict]) -> List[Dict]:
        """
        V5.5: Market Intelligence Edge Detection.
        Filters slate for edges > 5%.
        """
        results = []
        for pick in slate:
            # Generate V5 simulation
            v5_res = self.predict_prop_distribution(
                pick['player'], 
                pick['prop'], 
                pick['line']
            )
            
            if 'error' in v5_res:
                continue
                
            model_prob = v5_res['win_prob'] / 100.0
            odds_raw = pick.get('odds', -110)
            market_prob = american_to_implied(odds_raw)
            
            edge_info = detect_edge(
                pick['player'], 
                pick['prop'], 
                pick['line'], 
                model_prob, 
                market_prob
            )
            
            # Filter for +5% edge
            if edge_info['edge'] >= 0.05:
                v5_res.update({
                    "market_odds": odds_raw,
                    "market_prob": round(market_prob * 100, 1),
                    "edge": round(edge_info['edge'] * 100, 1),
                    "edge_confidence": edge_info['confidence_level']
                })
                results.append(v5_res)
                
        # Sort by edge
        results = sorted(results, key=lambda x: x['edge'], reverse=True)
        
        # PERSIST ARTIFACTS
        if results:
            df_edges = pd.DataFrame(results)
            df_edges.to_csv("predictions/value_edges_today.csv", index=False)
            df_edges.to_json("predictions/value_edges_today.json", orient='records')
            
        return results

    def get_synergy_adjusted_picks(self, slate: List[Dict], synergy_score: float = 0.024) -> List[Dict]:
        """
        V6 Synergy Integration.
        Directly adjusts existing V5 projections in the slate to preserve market data.
        """
        results = []
        for pick in slate:
            # If we already have the V5 prediction in the dict, use it
            if 'win_prob' in pick:
                v5_res = pick.copy()
            else:
                v5_res = self.predict_prop_distribution(pick['player'], pick['prop'], pick['line'])
            
            if 'error' in v5_res: continue
            
            # Apply V6 Augmentation
            v6_adj = self.v6_engine.augment_v5_service(v5_res, synergy_score)
            
            v5_res.update({
                "raw_win_prob": v6_adj['raw_win_prob'],
                "win_prob": v6_adj['synergy_adjusted_prob'],
                "synergy_score": v6_adj['synergy_score'],
                "synergy_impact": v6_adj['synergy_impact'],
                "v6_active": True
            })
            results.append(v5_res)
            
        # Persistence requirement for V6
        if results:
            df_v6 = pd.DataFrame(results)
            df_v6.to_csv("predictions/v6_synergy_adjusted.csv", index=False)
        
        return results

    def batch_inference(self, slate: List[Dict]) -> List[Dict]:
        """Processes an entire game slate through the V5 pipeline."""
        results = []
        for pick in slate:
            res = self.predict_prop_distribution(
                pick['player'], 
                pick['prop'], 
                pick['line']
            )
            results.append(res)
        return results
