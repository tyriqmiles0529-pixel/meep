import os
import sys
import glob
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlalchemy
import subprocess
import csv
import re

# Path hack to import from parent
sys.path.append(str(Path(__file__).parent.parent.parent))

# PHASE V.5: MODERN PREDICTION SERVICE
from meep_terminal.core.v5_service import V5PredictionService as LivePredictionEngine
from meep_terminal.core.math_utils import BettingMath
from meep_terminal.data.models import DatabaseManager, ModelRun, Pick, User, Bet, BackgroundTask, SystemEvent
import itertools
from sqlalchemy import text
import sqlalchemy

class EventManager:
    """Manages the Real-Time Event Intelligence Stream."""
    @staticmethod
    def log_event(db_manager, message, level='info', category='system', metadata=None):
        session = db_manager.get_session()
        event = SystemEvent(
            message=message,
            level=level,
            category=category,
            metadata_json=metadata or {}
        )
        session.add(event)
        session.commit()
        session.close()

    @staticmethod
    def get_recent_events(db_manager, limit=10):
        session = db_manager.get_session()
        events = session.query(SystemEvent).order_by(SystemEvent.timestamp.desc()).limit(limit).all()
        session.close()
        return events
import threading
import time
import json
from concurrent.futures import ThreadPoolExecutor
from meep_terminal.data.models import BackgroundTask
from scipy.stats import norm, skewnorm

class BackgroundTaskManager:
    """Institutional-grade task scheduler for non-blocking engine operations."""
    _executor = ThreadPoolExecutor(max_workers=4)
    
    @classmethod
    def run_async(cls, db_manager, task_type, func, *args, **kwargs):
        session = db_manager.get_session()
        task = BackgroundTask(task_type=task_type, status='running', progress=0.1, message="Initializing background process...")
        session.add(task)
        session.commit()
        task_id = task.id
        session.close()
        
        def wrapper(tid, *a, **kw):
            s = db_manager.get_session()
            t = s.query(BackgroundTask).get(tid)
            try:
                # Update status
                t.message = f"Process {task_type} started..."
                s.commit()
                
                # Execute payload
                # Note: func should accept a 'progress_callback' if it wants to update progress
                result = func(*a, progress_callback=lambda p, m: cls._update_task(db_manager, tid, p, m), **kw)
                
                t.status = 'completed'
                t.progress = 1.0
                t.message = "Task finished successfully."
                t.result_json = result if isinstance(result, dict) else {"status": "success"}
                s.commit()
            except Exception as e:
                t.status = 'failed'
                t.message = f"Critical Error: {str(e)}"
                s.commit()
            finally:
                s.close()

        cls._executor.submit(wrapper, task_id, *args, **kwargs)
        return task_id

    @staticmethod
    def _update_task(db_manager, task_id, progress, message):
        s = db_manager.get_session()
        t = s.query(BackgroundTask).get(task_id)
        if t:
            t.progress = progress
            if message: t.message = message
            s.commit()
        s.close()

class CorrelationEngine:
    """Analyzes relational prop dependencies to quantify risk clusters."""
    @staticmethod
    def analyze_player_props(p_data):
        # Calculate correlation matrix for the player's last 20 games
        perf_cols = ['points', 'assists', 'reboundsTotal', 'three_pointers']
        subset = p_data[perf_cols].tail(20)
        corr = subset.corr().fillna(0).to_dict()
        
        # Identify high-risk clusters (highly correlated props)
        clusters = []
        for prop1 in perf_cols:
            for prop2 in perf_cols:
                if prop1 != prop2 and abs(corr[prop1][prop2]) > 0.6:
                    clusters.append({
                        "props": [prop1, prop2],
                        "r": corr[prop1][prop2],
                        "risk": "Amplification" if corr[prop1][prop2] > 0 else "Hedge"
                    })
        return {"matrix": corr, "clusters": clusters}

    @staticmethod
    def identify_arbitrage(picks):
        """Identifies large discrepancies between neural targets and market lines."""
        arbs = []
        for p in picks:
            if p.line and p.prediction:
                diff = abs(p.line - p.prediction)
                pct_diff = diff / p.line if p.line > 0 else 0
                if pct_diff > 0.25: # 25% discrepancy threshold
                    arbs.append({
                        "player": p.player_name,
                        "prop": p.prop_type,
                        "market": p.line,
                        "neural": p.prediction,
                        "edge": round(pct_diff * 100, 1)
                    })
        return arbs

class RiskManager:
    """Institutional risk engine for exposure control and correlation conflicts."""
    @staticmethod
    def analyze_portfolio_risk(db_manager, picks, mode='rookie'):
        # Support both ORM Pick objects and plain dicts from session_state
        def _get(p, attr, default=0):
            if isinstance(p, dict):
                return p.get(attr, default)
            return getattr(p, attr, default)

        exposure = sum([_get(p, 'ev', 0) * 10 for p in picks if (_get(p, 'ev', 0) or 0) > 0])
        risk_score = min(1.0, exposure / 1000.0)
        
        conflicts = []
        teams = [_get(p, 'team', '') for p in picks]
        for t in set(teams):
            if t and teams.count(t) > 2:
                conflicts.append(f"Over-exposure to {t} (3+ picks)")
        
        if mode == 'rookie':
            return {"status": "Safe" if risk_score < 0.3 else "Cautious", "score": risk_score}
        return {
            "exposure": exposure,
            "risk_score": risk_score,
            "conflicts": conflicts,
            "var": exposure * 0.15 # Mock VaR
        }

class AuditManager:
    """Handles persistent transaction logging for financial accountability."""
    @staticmethod
    def log_action(db_manager, username, action_type, mode, details, impact=0.0, meta=None):
        from meep_terminal.data.models import AuditLedger
        session = db_manager.get_session()
        entry = AuditLedger(
            username=username,
            action_type=action_type,
            mode=mode,
            details=details,
            impact_amount=impact,
            metadata_json=meta or {}
        )
        session.add(entry)
        session.commit()
        session.close()

class TelemetryManager:
    """Monitors platform performance and health telemetry."""
    @staticmethod
    def log_metric(db_manager, name, value, meta=None):
        from meep_terminal.data.models import Telemetry
        session = db_manager.get_session()
        m = Telemetry(metric_name=name, value=value, metadata_json=meta or {})
        session.add(m)
        session.commit()
        session.close()

class FeedbackManager:
    """Handles pilot user feedback collection."""
    @staticmethod
    def submit_feedback(db_manager, username, ftype, message):
        from meep_terminal.data.models import UserFeedback
        session = db_manager.get_session()
        fb = UserFeedback(username=username, feedback_type=ftype, message=message)
        session.add(fb)
        session.commit()
        session.close()

class ArbitrageEngine:
    """Institutional-grade Multi-Bookmaker Arbitrage Detection."""
    @staticmethod
    def scan_for_arbs(db_manager, current_picks, mode='rookie'):
        session = db_manager.get_session()
        try:
            from meep_terminal.data.models import MarketLine
            arbs = []
            
            # Group current picks — handle both ORM Pick objects and plain dicts
            def _p(pick, attr, default=None):
                if isinstance(pick, dict):
                    # dict keys from session_state use 'player'/'prop' not 'player_name'/'prop_type'
                    _map = {'player_name': 'player', 'prop_type': 'prop', 'prediction': 'mu'}
                    return pick.get(_map.get(attr, attr), pick.get(attr, default))
                return getattr(pick, attr, default)

            current_picks = current_picks or []
            pick_map = {f"{_p(p,'player_name')}_{_p(p,'prop_type')}": p for p in current_picks}
            
            # Fetch all cached market lines for today
            lines = session.query(MarketLine).all()
            
            # Group lines by player_prop
            market_map = {}
            for l in lines:
                key = f"{l.player_name}_{l.prop_type}"
                if key not in market_map: market_map[key] = []
                market_map[key].append(l)

            for key, pick in pick_map.items():
                m_lines = market_map.get(key, [])
                if not m_lines: continue
                
                # Identify best and worst lines
                best_over = min([l.line for l in m_lines]) # Lowest line for Over
                best_under = max([l.line for l in m_lines]) # Highest line for Under
                
                # Check for bookmaker discrepancies (Multi-Book Arbs)
                if best_under - best_over >= 1.0:
                    arbs.append({
                        "type": "Market Discrepancy",
                        "player": pick.player_name,
                        "prop": pick.prop_type,
                        "details": f"Line varied from {best_over} to {best_under} across books.",
                        "confidence": "High",
                        "tier": "rookie" if abs(best_under - best_over) > 1.5 else "all-star"
                    })

                # Check for Neural Arbitrage (Neural vs best market line)
                neural_diff = abs(pick.prediction - best_over)
                if neural_diff / best_over > 0.20:
                    arbs.append({
                        "type": "Neural Alpha",
                        "player": pick.player_name,
                        "prop": pick.prop_type,
                        "details": f"Neural Target {pick.prediction:.1f} vs Market {best_over}",
                        "confidence": "Institutional",
                        "tier": "superstar"
                    })
            
            # Filter based on mode
            if mode == 'rookie':
                return [a for a in arbs if a['tier'] == 'rookie']
            elif mode == 'all-star':
                return [a for a in arbs if a['tier'] in ['rookie', 'all-star']]
            return arbs # Superstar sees all

        finally:
            session.close()

class PortfolioManager:
    @staticmethod
    def generate_slate(picks, bankroll=1000.0, target_units=10.0):
        """
        Categorizes picks into Riq's Portfolio Structure using existing BettingStrategy:
        - Core (Daily Solids): Singles (Top 6)
        - Growth (Parlay Core): 2-Leg Parlays
        - Moonshot (Alpha RRs): 3+ Leg Parlays & Round Robins
        """
        if not picks:
            return {"core": [], "growth": [], "moonshot": []}
            
        # Convert DB picks to the dict format expected by BettingStrategy
        candidates_list = []
        for p in picks:
            candidates_list.append({
                'player': p.player_name,
                'player_name': p.player_name, # support both
                'team': str(p.team) if p.team else 'NBA',
                'target': str(p.prop_type) if p.prop_type else 'points',
                'market': str(p.prop_type) if p.prop_type else 'points', # support both
                'line': float(p.line) if p.line is not None else 0.0,
                'prediction': float(p.prediction) if p.prediction is not None else 0.5,
                'win_prob': float(p.win_prob) if p.win_prob is not None else 0.0,
                'odds': int(p.metadata_json.get('odds', -110)) if (p.metadata_json and p.metadata_json.get('odds')) else -110,
                'side': str(p.metadata_json.get('side', 'OVER')).upper() if (p.metadata_json and p.metadata_json.get('side')) else 'OVER',
                'confidence': (float(p.win_prob) * 100) if p.win_prob is not None else 0.0,
                'game_id': str(p.metadata_json.get('game_id', '00000000')) if p.metadata_json else '00000000'
            })
            
        import pandas as pd
        from betting_strategy import BettingStrategy
        bs = BettingStrategy(load_models=False)
        df_candidates = pd.DataFrame(candidates_list)
        
        # 1. Slate Quality Filter (Phase S2 Mirror)
        daily_units = target_units
        slate_status = "Standard Volume"
        if not df_candidates.empty:
            median_prob = df_candidates['win_prob'].median()
            if median_prob < 0.56:
                daily_units = target_units * 0.5 # 5.0 if 10.0
                slate_status = "REDUCED VOLUME (Low Confidence Slate)"

        # 2. Mirror run_phase_i.py logic
        slate_results = bs.generate_optimal_targeted_parlays(
            df_candidates,
            bankroll=bankroll,
            target_units=daily_units
        )
        
        # Generate Lotto Slips
        lotto_slips = bs.generate_lotto_parlays(df_candidates, n=3)
        
        # Classification for UI display
        core = slate_results.get('singles', [])
        growth = [p for p in slate_results.get('traditional', []) if len(p['legs']) == 2]
        moonshot = [p for p in slate_results.get('traditional', []) if len(p['legs']) > 2] + slate_results.get('rr', [])
        
        # Update DB picks with tiers
        single_names = [s['name'].replace('Single: ', '') for s in core]
        for p in picks:
            if p.player_name in single_names:
                p.confidence_tier = 'A'
            else:
                p.confidence_tier = 'C'
            
        return {
            "core": core,
            "growth": growth,
            "moonshot": moonshot,
            "lotto": lotto_slips,
            "slate_status": slate_status,
            "target_units": daily_units
        }

    @staticmethod
    def get_ai_parlay_suggestions(db_manager, mode='rookie'):
        """Generates AI-guided optimal parlay slips based on user tier."""
        session = db_manager.get_session()
        try:
            from meep_terminal.data.models import Pick
            # Fetch latest picks
            picks = session.query(Pick).filter(Pick.win_prob > 0.6).order_by(Pick.ev.desc()).limit(10).all()
            if not picks: return []

            if mode == 'rookie':
                # Suggest a 2-leg high-confidence solid
                return [{
                    "name": "🛡️ The Daily Solid",
                    "legs": [picks[0], picks[1]],
                    "odds": "+260",
                    "win_prob": "42%",
                    "logic": "Combines our two highest-confidence neural projections for today's slate."
                }]
            elif mode == 'all-star':
                # Suggest a 3-leg growth slip
                return [{
                    "name": "📈 Growth Accelerator",
                    "legs": [picks[0], picks[1], picks[2]],
                    "odds": "+600",
                    "win_prob": "28%",
                    "logic": "Aggressive EV targeting across correlated usage spikes."
                }]
            else:
                # Superstar: High Alpha Moonshot
                return [{
                    "name": "🚀 Neural Alpha RR",
                    "legs": [picks[0], picks[1], picks[2], picks[3]],
                    "odds": "+1200",
                    "win_prob": "15%",
                    "logic": "Maximum edge exploitation using outlier volatility clusters."
                }]
        finally:
            session.close()

class PaperSimulator:
    def __init__(self, engine):
        self.engine = engine
    def resolve_picks(self, picks):
        results = []
        df = self.engine.engine.aggregated_data
        
        # Standardize date column in df
        date_col = next((c for c in ['date', 'GAME_DATE', 'gameDate'] if c in df.columns), None)
        if not date_col: return []
        
        # Ensure date_col is datetime (Optimized: Check if already converted)
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
             df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Mapping for resolution
        prop_map = {
            'points': 'PTS',
            'assists': 'AST',
            'rebounds': 'REB',
            'three_pointers': 'FG3M'
        }
        
        for p in picks:
            # Fast Lookup using player_map
            p_id_str = str(p.player_id).replace('.0', '')
            if p_id_str in self.engine.engine.player_map:
                player_history = df.iloc[self.engine.engine.player_map[p_id_str]]
                
                # Find the specific game by date (Date matching optimization)
                p_date = pd.to_datetime(p.game_date).date()
                match = player_history[player_history[date_col].dt.date == p_date]
                
                if not match.empty:
                    actual_col = prop_map.get(p.prop_type, p.prop_type)
                    actual_val = match.iloc[0].get(actual_col)
                    
                    if actual_val is not None:
                        win = float(actual_val) > float(p.line) if p.line else False
                        results.append({
                            "pick_id": p.id,
                            "player": p.player_name,
                            "prop": p.prop_type,
                            "line": p.line,
                            "actual": actual_val,
                            "win": win,
                            "ev": p.ev,
                            "prob": p.win_prob,
                            "game_date": p_date
                        })
        return results

    def run_full_sim(self, days=30):
        """Runs a portfolio sim over the last X days of picks."""
        session = self.db.get_session()
        try:
            cutoff = datetime.now() - pd.Timedelta(days=days)
            picks = session.query(Pick).filter(Pick.game_date >= cutoff).all()
            
            if not picks:
                return {"roi": 0, "win_rate": 0, "profit": 0, "equity_curve": [], "total_bets": 0}
                
            outcomes = self.resolve_picks(picks)
            if not outcomes:
                return {"roi": 0, "win_rate": 0, "profit": 0, "equity_curve": [], "total_bets": 0}
            
            df_res = pd.DataFrame(outcomes)
            
            total_staked = 0
            total_returned = 0
            equity = 1000.0
            curve = []
            
            # Daily aggregation
            for date, group in df_res.groupby('game_date'):
                daily_profit = 0
                for _, row in group.iterrows():
                    stake = 10.0
                    total_staked += stake
                    if row['win']:
                        ret = stake * (100/110 + 1)
                        daily_profit += (ret - stake)
                        total_returned += ret
                    else:
                        daily_profit -= stake
                
                equity += daily_profit
                curve.append({"date": date, "equity": equity})
                
            roi = (total_returned - total_staked) / total_staked if total_staked > 0 else 0
            win_rate = df_res['win'].mean()
            
            return {
                "roi": round(roi * 100, 1),
                "win_rate": round(win_rate * 100, 1),
                "profit": round(total_returned - total_staked, 2),
                "equity_curve": curve,
                "total_bets": len(df_res)
            }
        finally:
            session.close()

class TerminalEngine:
    def __init__(self, models_dir="./models", data_path="final_feature_matrix_with_per_min_1997_onward.csv"):
        self.models_dir = models_dir
        self.data_path = data_path
        self.db = DatabaseManager()
        self.engine = LivePredictionEngine(
            models_dir=models_dir
        )
        self._cached_stats = None
        self._last_stats_time = None
        
        # AUTOMATED STARTUP PULSE
        try:
             # Progress: Sync the ledger immediately (FAST)
             self.run_local_sync()
             
             # Proactive Freshness Check (If data is > 24h old, start background heal)
             stats = self.get_stats()
             if stats.get('data_date'):
                 diff = datetime.now() - stats['data_date']
                 if diff.days >= 1:
                     EventManager.log_event(self.db, "Stale data detected (>24h). Triggering auto-refresh...", level='warning')
                     self.start_background_refresh()
        except Exception as e:
             EventManager.log_event(self.db, f"Startup Auto-Sync Failed: {e}", level='error')
             
        EventManager.log_event(self.db, "MEEP Platform Core Initialized", level='success', category='system')

    def get_events(self, limit=15):
        return EventManager.get_recent_events(self.db, limit)

    def get_user_preferences(self, username="admin"):
        session = self.db.get_session()
        user = session.query(User).filter(User.username == username).first()
        if not user:
            user = User(username=username, role='admin', preferences={"mode": "casual", "theme": "dark"})
            session.add(user)
            session.commit()
        prefs = user.preferences
        session.close()
        return prefs

    def update_user_preferences(self, preferences, username="admin"):
        session = self.db.get_session()
        user = session.query(User).filter(User.username == username).first()
        if user:
            # Merge preferences
            current = dict(user.preferences)
            current.update(preferences)
            user.preferences = current
            session.commit()
            # Invalidate stats cache so UI updates immediately
            self._cached_stats = None
            self._last_stats_time = None
            EventManager.log_event(self.db, f"Preferences updated for {username}", level='info', category='system')
        session.close()

    def get_arbitrage(self, current_picks, mode='rookie'):
        # If None passed, use session_state picks or fall back to empty list
        if current_picks is None:
            current_picks = []
        return ArbitrageEngine.scan_for_arbs(self.db, current_picks, mode=mode)

    def get_ai_parlays(self, mode='rookie'):
        return PortfolioManager.get_ai_parlay_suggestions(self.db, mode=mode)

    def get_stats(self):
        """High-performance status retrieval with micro-caching (5 minute TTL)."""
        now = datetime.now()
        if self._cached_stats and self._last_stats_time and (now - self._last_stats_time).seconds < 300:
            return self._cached_stats

        # Fetch user-set bankroll if exists (Priority: Manual > Calculated)
        prefs = self.get_user_preferences("admin")
        manual_bankroll = prefs.get('bankroll')
        current_bankroll = float(manual_bankroll) if manual_bankroll is not None else None

        session = self.db.get_session()
        try:
            # Priority: Try to find the latest successful ModelRun (reflects the 8am Task)
            last_run = session.query(ModelRun).filter(ModelRun.status == 'success').order_by(ModelRun.timestamp.desc()).first()
            last_bet = session.query(Bet).order_by(Bet.placed_at.desc()).first()
            
            # Efficient Date Detection (Scan once and memoize)
            date_col = next((c for c in ['gameDate', 'GAME_DATE', 'date'] if c in self.engine.aggregated_data.columns), None)
            
            # Use ModelRun freshness date if available, otherwise look at data
            if last_run and last_run.data_freshness_date:
                data_date = last_run.data_freshness_date
            elif date_col:
                # Get max but avoid full conversion if possible
                data_date = pd.to_datetime(self.engine.aggregated_data[date_col].dropna().tail(5000)).max()
            else:
                data_date = now - timedelta(days=1)
            
            # Performance Aggregation
            total_staked = session.query(sqlalchemy.func.sum(Bet.stake_amount)).scalar() or 0
            
            # Simple ROI calculation from ledger if available (mocking real for now but removing hard placeholder)
            if current_bankroll is None:
                current_bankroll = 1000.0 + (total_staked * 0.05) # Assume 5% edge as baseline until fully audited
            
            self._cached_stats = {
                "data_date": data_date,
                "last_run": last_run.timestamp if last_run else None,
                "last_run_status": last_run.status if last_run else "N/A",
                "total_bets": session.query(Bet).count(),
                "last_bet_date": last_bet.placed_at if last_bet else None,
                "bankroll": current_bankroll,
                "daily_roi": "Calculated via Ledger" 
            }
            self._last_stats_time = now
            return self._cached_stats
        except Exception as e:
            # Fallback to manual or default if engine fails
            return {
                "data_date": now, "last_run": None, "last_run_status": "ERROR",
                "total_bets": 0, "last_bet_date": None, 
                "bankroll": current_bankroll if current_bankroll is not None else 1000.0
            }
        finally:
            session.close()

    def get_player_correlation(self, player_name):
        """Quantify relational prop dependencies for a specific player."""
        p_name_lower = player_name.lower()
        name_map = getattr(self.engine, 'name_map', {})
        if p_name_lower in name_map:
            p_data = self.engine.aggregated_data.iloc[name_map[p_name_lower]]
            return CorrelationEngine.analyze_player_props(p_data)
        return None

    def start_background_refresh(self):
        """Trigger a full data refresh cycle (scrape + aggregate + archetypes)."""
        return BackgroundTaskManager.run_async(
            self.db, 
            'data_refresh', 
            self.run_background_refresh
        )

    def run_background_refresh(self, progress_callback=None):
        """Execute the daily_refresh.py logic in a background process."""
        if progress_callback: progress_callback(0.1, "Starting Daily Refresh Pipeline...")
        try:
            # We run the script as a subprocess to maintain independence
            cmd = [sys.executable, "daily_refresh.py"]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            if progress_callback: progress_callback(0.3, "Fetching NBA logs and updating matrix...")
            
            # Monitor output for progress hints if possible, or just wait
            # For now, let's just wait and update progress periodically
            time.sleep(10) 
            if progress_callback: progress_callback(0.6, "Data aggregation in progress...")
            
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                if progress_callback: progress_callback(1.0, "Refresh Complete.")
                EventManager.log_event(self.db, "Background Data Refresh Success", category='system', level='success')
                return {"status": "success", "output": stdout}
            else:
                EventManager.log_event(self.db, f"Background Data Refresh Failed: {stderr}", category='system', level='error')
                raise Exception(stderr)
        except Exception as e:
            if progress_callback: progress_callback(1.0, f"Error: {str(e)}")
            raise e

    def start_inference(self):
        """Trigger the production inference pipeline (Predictions + Picks)."""
        return BackgroundTaskManager.run_async(
            self.db,
            'inference_engine',
            self.run_inference
        )

    def run_inference(self, progress_callback=None):
        """Run predict_live_FINAL.py and run_phase_i.py."""
        if progress_callback: progress_callback(0.1, "Initializing Neural Inference...")
        try:
            # 1. Prediction Step
            if progress_callback: progress_callback(0.3, "Executing Model Ensembles...")
            master_csv = "final_feature_matrix_with_per_min_1997_onward.csv"
            pred_cmd = [
                sys.executable, "predict_live_FINAL.py", 
                "--betting", 
                "--output", "predictions/live_ensemble_2025.csv", 
                "--aggregated-data", master_csv
            ]
            process1 = subprocess.run(pred_cmd, capture_output=True, text=True)
            if process1.returncode != 0:
                raise Exception(f"Prediction Phase Failed: {process1.stderr}")

            # 2. Pick Generation Step
            if progress_callback: progress_callback(0.7, "Generating Strategy Slates (Phase I)...")
            pick_cmd = [sys.executable, "run_phase_i.py"]
            process2 = subprocess.run(pick_cmd, capture_output=True, text=True)
            if process2.returncode != 0:
                raise Exception(f"Pick Generation Failed: {process2.stderr}")

            if progress_callback: progress_callback(0.95, "Syncing workspace database...")
            self.run_local_sync(progress_callback=progress_callback)
            
            if progress_callback: progress_callback(1.0, "Inference and Sync Complete.")
            EventManager.log_event(self.db, "Production Inference and Sync Complete", category='system', level='success')
            return {"status": "success"}
        except Exception as e:
            if progress_callback: progress_callback(1.0, f"Error: {str(e)}")
            raise e

    def start_local_sync(self):
        """Trigger a non-blocking sync with the local ledger CSV."""
        return BackgroundTaskManager.run_async(
            self.db, 
            'local_data_sync', 
            self.run_local_sync
        )

    def get_task_status(self, task_id):
        session = self.db.get_session()
        task = session.query(BackgroundTask).get(task_id)
        res = {
            "status": task.status if task else "unknown",
            "progress": task.progress if task else 0,
            "message": task.message if task else "Task not found"
        }
        session.close()
        return res

    def _find_latest_file(self, pattern: str, default: str) -> str:
        """Helper to find the most recently modified file matching a pattern."""
        files = glob.glob(pattern)
        if not files:
            return default
        # Return the file with the most recent modification time
        return max(files, key=os.path.getmtime)

    def run_local_sync(self, progress_callback=None):
        """
        Ingests the latest local CSV data (ledger and picks) into the database.
        Zero API calls.
        """
        session = self.db.get_session()
        
        if progress_callback: progress_callback(0.1, "Initializing dynamic file discovery...")
        
        # 1. Discover relevant files
        ledger_path = self._find_latest_file("betting_ledger*.csv", "betting_ledger.csv")
        projection_path = self._find_latest_file("predictions/projections_*.csv", "predictions/live_ensemble_2025.csv")
        
        # 2. Start Audit Record (Tail optimized)
        # OPTIMIZATION: Only use tail(100) for freshness date to avoid converting 70k rows just for the max
        _date_col = next((c for c in ['gameDate', 'GAME_DATE', 'date'] if c in self.engine.aggregated_data.columns), None)
        if _date_col:
            dt_series = self.engine.aggregated_data[_date_col]
            # If already converted, just get max. If string, convert only the tail.
            if pd.api.types.is_datetime64_any_dtype(dt_series):
                _freshness = dt_series.max()
            else:
                _freshness = pd.to_datetime(dt_series.tail(100)).max()
        else:
            _freshness = datetime.now()
        
        run = ModelRun(
            timestamp=datetime.now(),
            status="running",
            data_freshness_date=_freshness
        )
        session.add(run)
        session.commit()

        try:
            if progress_callback: progress_callback(0.3, "Performing High-Speed Data Refresh...")
            # OPTIMIZATION: Use specialized reload_data() instead of re-instantiating heavy models/engines
            self.engine.reload_data()
            
            if progress_callback: progress_callback(0.6, "Cleaning up stale picks for today...")
            # 2. Aggressive Cleanup of any picks for today to ensure we only have the ledger's latest
            today_str = datetime.now().strftime('%Y-%m-%d')
            # Using raw SQL via text for simple/reliable multi-table deletion
            session.execute(text("DELETE FROM bets WHERE pick_id IN (SELECT id FROM picks WHERE game_date >= :t)"), {"t": today_str})
            session.execute(text("DELETE FROM picks WHERE game_date >= :t"), {"t": today_str})
            session.commit()

            if progress_callback: progress_callback(0.8, f"Ingesting ledger: {ledger_path}...")
            # 3. Ingest Ledger into DB
            # This captures the output of the scheduled task's Phase I / Grader run
            self._ingest_ledger_to_db(session, run.id, ledger_path=ledger_path)
            
            run.status = "success"
            # Refresh 'Engine Freshness' calculation (Tail optimized)
            _date_col = next((c for c in ['gameDate', 'GAME_DATE', 'date'] if c in self.engine.aggregated_data.columns), None)
            if _date_col:
                dt_series = self.engine.aggregated_data[_date_col]
                if pd.api.types.is_datetime64_any_dtype(dt_series):
                    run.data_freshness_date = dt_series.max()
                else:
                    run.data_freshness_date = pd.to_datetime(dt_series.tail(100)).max()

            session.commit()
            
            # Invalidate stats cache so UI updates 'Engine Freshness' immediately
            self._cached_stats = None
            self._last_stats_time = None

            EventManager.log_event(self.db, "Manual Workspace Sync Complete", category='system', level='success')
            return True

        except Exception as e:
            run.status = "failed"
            run.metadata_json = {"error": str(e)}
            session.commit()
            EventManager.log_event(self.db, f"Workspace Sync Failed: {str(e)}", category='system', level='error')
            raise e
        finally:
            session.close()

    def _ingest_ledger_to_db(self, session, run_id, ledger_path="betting_ledger.csv"):
        """Parses the betting_ledger.csv and populates Pick and Bet tables."""
        if not os.path.exists(ledger_path):
            EventManager.log_event(self.db, "Ledger file not found for ingestion", category='system', level='error')
            return

        from meep_terminal.data.models import Pick, Bet
        
        # Load recent ledger rows
        df = pd.read_csv(ledger_path)
        # Convert 'Run Date' (MM.DD.YY) to datetime
        df['game_date_dt'] = pd.to_datetime(df['Run Date'], format='%m.%d.%y')
        
        # Filter for recent entries (last 48 hours to be safe)
        cutoff = datetime.now() - timedelta(days=2)
        recent = df[df['game_date_dt'] >= cutoff]

        for _, row in recent.iterrows():
            # Check if this row is a parlay/multi-bet
            is_parlay = "Parlay" in str(row['Player']) or "Round Robin" in str(row['Player']) or "RR (" in str(row['Player']) or "Lotto" in str(row['Player'])
            side_raw = str(row['Side'])
            
            if is_parlay:
                leg_strings = [leg.strip() for leg in side_raw.split(' | ')]
            else:
                leg_strings = [side_raw]
                
            for leg_str in leg_strings:
                # Robust Parsing for this leg
                player_raw = str(row['Player']) if not is_parlay else "N/A"
                market_raw = str(row['Market']) if not is_parlay else "Prop"
                line_raw = row['Line'] if not is_parlay else 0.0
                odds_raw = row['Odds'] if not is_parlay else -110

                # Extract Player and Team more robustly
                p_name = player_raw
                team_abbr = str(row['Team']) if (pd.notna(row['Team']) and str(row['Team']) != "N/A" and not is_parlay) else "N/A"
                prop_type = market_raw.lower()
                
                # Default side extraction
                side = "OVER"
                if not is_parlay and pd.notna(row['Side']) and row['Side'] in ['Over', 'Under', 'OVER', 'UNDER']:
                    side = str(row['Side']).upper()
                    
                line = float(line_raw) if pd.notna(line_raw) else 0.0
                odds = int(float(odds_raw)) if pd.notna(odds_raw) else -110

                # Handle the "Description in Side" format
                if (" - " in leg_str or "@" in leg_str) and "(" in leg_str:
                    # Format 1: "Josh Giddey (CHI) - rebounds Over 6.5 (-146.0)"
                    match1 = re.search(r'^(.*?)\s*\(([A-Z]{2,4})\)\s*-\s*([\w\s]+)\s+(Over|Under)\s+([\d.]+)\s+\((.*?)\)', leg_str, re.I)
                    # Format 2: "Player (TEAM) (Market Side @ Odds)" - common in complex parlays
                    match2 = re.search(r'^(.*?)\s*\(([A-Z]{2,4})\)\s*\((.*?)\s+(Over|Under)\s+@\s+(.*?)\)', leg_str, re.I) if not match1 else None
                    
                    final_match = match1 or match2
                    if final_match:
                        p_name = final_match.group(1).strip()
                        team_abbr = final_match.group(2).strip()
                        
                        if match1:
                            prop_details = final_match.group(3).strip().lower()
                            side = final_match.group(4).upper()
                            line = float(final_match.group(5))
                            odds_str = final_match.group(6)
                        else:
                            # Extract from match2
                            prop_details = final_match.group(3).strip().lower()
                            side = final_match.group(4).upper()
                            odds_str = final_match.group(5)
                            line = 0.5 # default

                        # Extract line from prop_details if it ends in a number (Format: "points 10.5")
                        if prop_details:
                            line_m = re.search(r'([\d.]+)$', prop_details)
                            if line_m:
                                line = float(line_m.group(1))
                                prop_details = prop_details[:line_m.start()].strip()
                        
                        prop_type = prop_details
                        
                        # Clean odds
                        if (odds == 0 or odds == -110) and odds_str:
                            try:
                                val = odds_str.replace('+', '').replace('@', '').strip()
                                odds = int(float(val))
                            except: pass
                    else:
                        print(f"[DEBUG] Regex failed to match description: {leg_str}")

                if p_name == "N/A" or not p_name or p_name == "nan":
                    continue

                # Clean name/team
                if "(" in p_name and team_abbr == "N/A":
                    match = re.search(r'^(.*?)\s*\(([A-Z]{3})\)', p_name)
                    if match:
                        p_name = match.group(1).strip()
                        team_abbr = match.group(2).strip()

                # Mapping for prop types to match internal engine keys
                if 'three' in prop_type or '3pt' in prop_type:
                    prop_type = 'three_pointers'
                elif 'reb' in prop_type:
                    prop_type = 'rebounds'

                opponent = "N/A"
                game_id = "N/A"
                try:
                    _date_col = next((c for c in ['gameDate', 'GAME_DATE', 'date'] if c in self.engine.aggregated_data.columns), None)
                    if _date_col:
                        mask = (self.engine.aggregated_data['player_name'].str.lower() == p_name.lower())
                        p_hist = self.engine.aggregated_data[mask]
                        if not p_hist.empty:
                            target_date = row['game_date_dt'].date()
                            match = p_hist[pd.to_datetime(p_hist[_date_col]).dt.date == target_date]
                            if not match.empty:
                                game_id = str(match.iloc[0].get('game_id', match.iloc[0].get('GAME_ID', 'N/A')))
                                if 'opponent' in match.columns:
                                    opponent = str(match.iloc[0]['opponent'])
                                elif 'matchup' in match.columns:
                                    matchup = str(match.iloc[0]['matchup'])
                                    player_team = str(match.iloc[0].get('team', match.iloc[0].get('TEAM_ABBREVIATION', '')))
                                    parts = re.split(r'\s+vs\.\s+|\s+@\s+', matchup, flags=re.I)
                                    if len(parts) > 1:
                                        opponent = parts[1] if player_team in parts[0] else parts[0]
                except: pass

                if opponent == "N/A": opponent = "NBA"
                if game_id == "N/A": game_id = "00000000"

                existing = session.query(Pick).filter(
                    Pick.player_name == p_name,
                    Pick.prop_type == prop_type,
                    Pick.game_date == row['game_date_dt']
                ).first()

                if not existing:
                    model_prob = float(row['Model Prob']) if (pd.notna(row['Model Prob']) and not is_parlay) else 0.50
                    ev_val = float(row.get('EV', 0.0)) if (pd.notna(row.get('EV', 0.0)) and not is_parlay) else 0.0
                    stake_size = float(row['Stake Size']) if (pd.notna(row['Stake Size']) and not is_parlay) else 0.0
                    
                    pick = Pick(
                        run_id=run_id,
                        game_date=row['game_date_dt'],
                        player_name=p_name,
                        team=team_abbr,
                        opponent=opponent,
                        prop_type=prop_type,
                        line=line,
                        prediction=model_prob, 
                        win_prob=model_prob,
                        ev=ev_val,
                        confidence_tier='A' if ev_val > 0.4 else 'B',
                        explanation=f"Unified Core Import. EV: {ev_val}",
                        metadata_json={"side": side, "odds": odds, "game_id": game_id}
                    )
                    session.add(pick)
                    session.flush()
                    
                    bet = Bet(
                        pick_id=pick.id,
                        status="placed" if stake_size > 0 else "considered",
                        stake_amount=stake_size,
                        actual_line=line,
                        actual_odds=odds,
                        placed_at=datetime.now(),
                        journal_entry=f"Auto-ingest: {row['Timestamp_ET']}"
                    )
                    session.add(bet)

    def run_portfolio_simulation(self, days=30):
        sim = PaperSimulator(self)
        return sim.run_full_sim(days=days)

    def get_portfolio_slate(self, date_str=None):
        """Retrieves and structures the saved portfolio for a specific day."""
        if not date_str:
            date_str = datetime.now().strftime("%Y-%m-%d")
        
        session = self.db.get_session()
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d")
            picks = session.query(Pick).filter(Pick.game_date == target_date).all()
            
            if not picks:
                return None
                
            # Retrieve structured portfolio (Singles, Parlays, RRs)
            slate = PortfolioManager.generate_slate(picks)
            
            # --- OVERRIDE WITH ACTUAL LEDGER PARLAYS ---
            # To sync perfectly with the MD file, bypass dynamic generator for parlays
            try:
                import os
                import pandas as pd
                ledger_path = "betting_ledger.csv"
                if os.path.exists(ledger_path):
                    ledger_df = pd.read_csv(ledger_path)
                    ledger_date_str = target_date.strftime("%m.%d.%y")
                    day_ledger = ledger_df[ledger_df['Run Date'] == ledger_date_str]
                    
                    parlays = day_ledger[day_ledger['Market'].isin(['Parlay', 'Round Robin', 'MULTI'])]
                    
                    if not parlays.empty:
                        # Clear dynamic generation except for singles
                        new_core = [s for s in slate['core'] if str(s.get('name', '')).startswith('Single')]
                        new_growth = []
                        new_moonshot = []
                        new_lotto = []
                        
                        # Use the most recent block of parlays for the day
                        # A user might have refreshed multiple times, so we take the last unique instances
                        parlays_latest = parlays.drop_duplicates(subset=['Player', 'Market'], keep='last')
                        
                        total_units = 0.0
                        for _, row in parlays_latest.iterrows():
                            p_name = str(row['Player'])
                            legs_raw = [lg.strip() for lg in str(row['Side']).split('|')]
                            
                            stk = float(row['Stake Size']) if pd.notna(row['Stake Size']) else 0.0
                            total_units += stk
                            
                            pl_dict = {
                                'name': p_name,
                                'legs': legs_raw,
                                'combined_odds': int(float(row['Odds'])) if pd.notna(row['Odds']) else 0,
                                'prob': float(row['Model Prob']) if pd.notna(row['Model Prob']) else 0.0,
                                'combined_prob': float(row['Model Prob']) if pd.notna(row['Model Prob']) else 0.0,
                                'ev': float(row.get('EV', 0.0)) if 'EV' in row and pd.notna(row.get('EV', 0.0)) else 0.0,
                                'stake_amt': stk,
                                'units': stk,
                                'stake_pct': (stk / 100.0) if stk > 0 else 0.05
                            }
                            
                            if 'Moonshot' in p_name or 'RR (' in p_name:
                                new_moonshot.append(pl_dict)
                            elif 'Lotto' in p_name:
                                new_lotto.append(pl_dict)
                            else:
                                # By default, 2-to-3 leg standard parlays are Growth portfolios in the new setup.
                                # Even if they are named "Parlay (Core X)", they belong in the Growth UI bracket.
                                new_growth.append(pl_dict)
                            
                        slate['core'] = new_core
                        slate['growth'] = new_growth
                        slate['moonshot'] = new_moonshot
                        slate['lotto'] = new_lotto
                        slate['slate_status'] = "Active Deployment (Synced from Ledger)"
                        if total_units > 0:
                            slate['target_units'] = float(total_units)
            except Exception as e:
                import logging
                logging.error(f"Ledger parlay override failed: {e}")

            session.commit() # Persist tiers (A/B/C)
            
            # Detach for UI safety
            for p in picks:
                session.refresh(p)
                session.expunge(p)
                
            return {
                "date": target_date.strftime("%Y-%m-%d"),
                "is_fallback": False,
                "core": slate['core'],
                "growth": slate['growth'],
                "moonshot": slate['moonshot'],
                "lotto": slate['lotto'],
                "slate_status": slate['slate_status'],
                "target_units": slate['target_units'],
                "raw_picks": picks
            }
        finally:
            session.close()

    def get_portfolio_risk(self, active_picks, mode='rookie'):
        return RiskManager.analyze_portfolio_risk(self.db, active_picks, mode=mode)

    def log_audit_action(self, action_type, details, impact=0.0):
        AuditManager.log_action(self.db, "admin", action_type, "superstar", details, impact)

    def get_audit_trail(self, limit=50):
        from meep_terminal.data.models import AuditLedger
        session = self.db.get_session()
        trail = session.query(AuditLedger).order_by(AuditLedger.timestamp.desc()).limit(limit).all()
        session.close()
        return trail

    def log_telemetry(self, name, value, meta=None):
        TelemetryManager.log_metric(self.db, name, value, meta)

    def submit_feedback(self, ftype, message, username="admin"):
        FeedbackManager.submit_feedback(self.db, username, ftype, message)

    def get_market_opportunities(self, picks, lines):
        """Compare projections to live lines to find +EV."""
        opportunities = []
        for pick in picks:
            pass
        return opportunities

    def get_ledger_history(self, limit=10):
        """Reads the actual production betting ledger for the AI Analyst."""
        ledger_path = "betting_ledger.csv"
        if not os.path.exists(ledger_path):
            return {"error": "Production ledger not found."}
        
        try:
            df = pd.read_csv(ledger_path)
            # Get last N rows
            tail = df.tail(limit).to_dict('records')
            return tail
        except Exception as e:
            return {"error": str(e)}

    def get_performance_metrics(self):
        """Calculates real-world performance from betting_ledger.csv."""
        ledger_path = "betting_ledger.csv"
        if not os.path.exists(ledger_path):
            return {"win_rate": 0, "profit": 0, "total_bets": 0, "roi": 0, "equity_curve": []}
            
        try:
            df = pd.read_csv(ledger_path)
            # Filter for graded bets
            graded = df[df['Outcome'].isin(['Win', 'Loss'])].copy()
            
            if graded.empty:
                return {"win_rate": 0, "profit": 0, "total_bets": 0, "roi": 0, "equity_curve": []}

            wins = len(graded[graded['Outcome'] == 'Win'])
            total = len(graded)
            win_rate = (wins / total) * 100 if total > 0 else 0
            
            # Calculate Profit
            profit = 0
            equity = 1000.0 # Starting baseline
            curve = []
            
            # Group by Run Date for curve
            graded['Run Date DT'] = pd.to_datetime(graded['Run Date'], format='%m.%d.%y')
            daily = graded.groupby('Run Date DT')
            
            for date, group in daily:
                day_pnl = 0
                for _, row in group.iterrows():
                    stake = row['Stake Size']
                    odds = row['Odds']
                    if row['Outcome'] == 'Win':
                        if odds > 0: day_pnl += stake * (odds / 100)
                        else: day_pnl += stake * (100 / abs(odds))
                    else:
                        day_pnl -= stake
                profit += day_pnl
                equity += day_pnl
                curve.append({"date": date.strftime("%Y-%m-%d"), "equity": equity})
            
            total_staked = graded['Stake Size'].sum()
            roi = (profit / total_staked) * 100 if total_staked > 0 else 0
            
            return {
                "win_rate": round(win_rate, 1),
                "profit": round(profit, 2),
                "total_bets": total,
                "roi": round(roi, 1),
                "equity_curve": curve
            }
        except Exception as e:
            print(f"Metrics Error: {e}")
            return {"win_rate": 0, "profit": 0, "total_bets": 0, "roi": 0, "equity_curve": []}
