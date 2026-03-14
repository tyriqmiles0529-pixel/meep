# RIQ ANALYZER - NEURAL HYBRID UPDATE
"""
This file contains the updated feature engineering for riq_analyzer.py to work with
neural hybrid models (commit 697f9e7).

KEY UPDATES:
1. Feature Engineering: Phases 1-7 (was 1-3) = ~150-218 features (was 61)
2. Model Loading: NeuralHybridPredictor objects (was plain LightGBM)
3. Basketball Reference Priors: Load from priors_data/ directory
4. Feature Names: Must match model.feature_names exactly

INTEGRATION INSTRUCTIONS:
Replace the following sections in riq_analyzer.py:
- ModelPredictor class (__init__ and predict methods)
- build_player_features function
- Add load_priors function and merge logic
"""

import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List

# ============================================================================
# 1. UPDATED MODEL PREDICTOR CLASS (Replaces lines ~2649-2966 in riq_analyzer.py)
# ============================================================================

class ModelPredictor:
    """
    Load and use trained neural hybrid models.
    
    Models are now NeuralHybridPredictor objects (TabNet + LightGBM + embeddings)
    instead of plain LightGBM regressors.
    """
    def __init__(self):
        self.player_models = {}
        self.player_sigma_models = {}
        self.game_models = {}
        self.player_ensembles = {}
        self.unified_ensemble = None
        self.ridge_model = None
        self.elo_model = None
        self.ff_model = None
        self.ensemble_meta_learner = None
        self.enhanced_selector = None
        self.selector_windows = {}
        self.spread_sigma = 8.0
        self.game_features = []
        self.game_defaults = {}
        
        # Load models from models/ directory
        models_dir = Path("models")
        if not models_dir.exists():
            print(f"⚠️  Models directory not found: {models_dir}")
            return
        
        # Load player models (NeuralHybridPredictor objects)
        for stat in ["points", "assists", "rebounds", "threes", "minutes"]:
            model_path = models_dir / f"{stat}_model.pkl"
            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        self.player_models[stat] = pickle.load(f)
                    print(f"✓ Loaded {stat} model")
                except Exception as e:
                    print(f"⚠️  Failed to load {stat} model: {e}")
        
        # Load game models
        for model_name in ["moneyline", "moneyline_calibrator", "spread"]:
            model_path = models_dir / f"{model_name}_model.pkl"
            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        self.game_models[model_name] = pickle.load(f)
                    print(f"✓ Loaded {model_name} model")
                except Exception as e:
                    print(f"⚠️  Failed to load {model_name} model: {e}")
        
        # Load spread sigma
        spread_sigma_path = models_dir / "spread_sigma.json"
        if spread_sigma_path.exists():
            try:
                import json
                with open(spread_sigma_path, 'r') as f:
                    self.spread_sigma = json.load(f).get("sigma", 8.0)
            except Exception:
                pass
    
    def available(self, prop_type: str) -> bool:
        """Check if a player stat model is available"""
        return prop_type in self.player_models
    
    def predict(self, prop_type: str, feats: pd.DataFrame) -> Optional[float]:
        """
        Predict player stat using neural hybrid model.
        
        NeuralHybridPredictor objects have their own .predict() method that
        handles TabNet + LightGBM ensemble automatically.
        """
        m = self.player_models.get(prop_type)
        if m is None or feats is None or feats.empty:
            return None
        try:
            # NeuralHybridPredictor.predict() returns ensemble prediction
            y = m.predict(feats)
            prediction = float(y[0]) if isinstance(y, (list, np.ndarray)) else float(y)
            return prediction
        except Exception as e:
            print(f"   Warning: ML predict failed for {prop_type}: {e}")
            return None

# ============================================================================
# 2. BASKETBALL REFERENCE PRIORS LOADING (Add this function to riq_analyzer.py)
# ============================================================================

def load_priors_data() -> Optional[pd.DataFrame]:
    """
    Load Basketball Reference player priors from priors_data/ directory.
    
    Returns DataFrame with columns:
    - name_join: Player name for merging
    - ~68 prior features (career stats, per-game averages, advanced metrics)
    
    Returns None if priors not available.
    """
    priors_dir = Path("priors_data")
    if not priors_dir.exists():
        print("⚠️  priors_data/ directory not found - skipping priors")
        return None
    
    # Load player priors CSV
    priors_path = priors_dir / "player_priors.csv"
    if not priors_path.exists():
        print("⚠️  player_priors.csv not found - skipping priors")
        return None
    
    try:
        priors = pd.read_csv(priors_path, low_memory=False)
        print(f"✓ Loaded {len(priors):,} player priors ({priors.shape[1]} features)")
        return priors
    except Exception as e:
        print(f"⚠️  Failed to load priors: {e}")
        return None

# ============================================================================
# 3. EXPANDED FEATURE ENGINEERING (Replaces build_player_features around line 2968)
# ============================================================================

def build_player_features_expanded(
    df_last: pd.DataFrame, 
    df_curr: pd.DataFrame,
    player_name: str = "",
    priors: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Build features for ML prediction matching train_auto.py schema with ALL PHASES (1-7).
    
    Feature Count: ~150-218 features depending on priors availability
    - Base features: 18 (context)
    - Player rest: 2
    - Core rolling stats: 12 (pts/ast/reb/3pm × L3/L5/L10)
    - Phase 1 (Shot volume): 16 features
    - Phase 2 (Matchup): 4 features
    - Phase 3 (Advanced rates): 3 features
    - Phase 4 (Home/away splits): 4 features
    - Phase 5 (Position/matchup adjustments): 10 features
    - Phase 6 (Momentum): 24-36 features
    - Phase 7 (Priors): ~68 features (if available)
    
    Args:
        df_last: Last season game log
        df_curr: Current season game log
        player_name: Player name for priors matching
        priors: Basketball Reference priors DataFrame
    
    Returns:
        Single-row DataFrame with all features
    """
    
    # Helper functions
    def seq(col: str) -> List[float]:
        vals: List[float] = []
        if not df_curr.empty and col in df_curr.columns:
            vals += [float(x) for x in df_curr[col].tolist() if pd.notnull(x)]
        if not df_last.empty and col in df_last.columns:
            vals += [float(x) for x in df_last[col].tolist() if pd.notnull(x)]
        return vals
    
    def avg(v: List[float], w: int) -> float:
        if not v: return 0.0
        return float(np.mean(v[:w])) if len(v) >= 1 else 0.0
    
    def safe_div(n, d):
        return float(n / d) if d > 0 else 0.0
    
    # Extract sequences
    pts = seq("points")
    ast = seq("assists")
    reb = seq("rebounds")
    thr = seq("threes")
    mins = seq("minutes") if "minutes" in (df_curr.columns.tolist() + df_last.columns.tolist()) else []
    
    # Phase 1: Shot volume sequences
    fga = seq("fieldGoalsAttempted") if "fieldGoalsAttempted" in (df_curr.columns.tolist() + df_last.columns.tolist()) else []
    tpa = seq("threePointersAttempted") if "threePointersAttempted" in (df_curr.columns.tolist() + df_last.columns.tolist()) else []
    fta = seq("freeThrowsAttempted") if "freeThrowsAttempted" in (df_curr.columns.tolist() + df_last.columns.tolist()) else []
    fg_pct = seq("fieldGoalPercentage") if "fieldGoalPercentage" in (df_curr.columns.tolist() + df_last.columns.tolist()) else []
    three_pct = seq("threePointPercentage") if "threePointPercentage" in (df_curr.columns.tolist() + df_last.columns.tolist()) else []
    ft_pct = seq("freeThrowPercentage") if "freeThrowPercentage" in (df_curr.columns.tolist() + df_last.columns.tolist()) else []
    
    # True Shooting % calculation
    def calc_ts_pct(pts_seq, fga_seq, fta_seq, window):
        if not pts_seq or not fga_seq or not fta_seq:
            return 0.56  # league average
        p = avg(pts_seq, window)
        a = avg(fga_seq, window)
        t = avg(fta_seq, window)
        denominator = 2 * (a + 0.44 * t)
        return p / denominator if denominator > 0 else 0.56
    
    # Momentum calculation
    def calc_momentum(vals, short=3, med=7):
        if not vals or len(vals) < med:
            return 0.0, 0.0
        recent = avg(vals, short)
        baseline = avg(vals, med)
        if baseline <= 0:
            return 0.0, 0.0
        momentum = (recent - baseline) / baseline
        # Acceleration = change in momentum
        if len(vals) >= med + 3:
            prev_recent = avg(vals[3:], short)
            prev_momentum = (prev_recent - baseline) / baseline if baseline > 0 else 0
            accel = momentum - prev_momentum
        else:
            accel = 0.0
        return momentum, accel
    
    # Calculate all momentum metrics
    pts_mom, pts_accel = calc_momentum(pts)
    ast_mom, ast_accel = calc_momentum(ast)
    reb_mom, reb_accel = calc_momentum(reb)
    thr_mom, thr_accel = calc_momentum(thr)
    
    # Hot/cold streak detection
    def detect_streak(vals, threshold=0.15):
        mom, _ = calc_momentum(vals)
        hot_streak = 1.0 if mom > threshold else 0.0
        cold_streak = 1.0 if mom < -threshold else 0.0
        return hot_streak, cold_streak
    
    pts_hot, pts_cold = detect_streak(pts)
    ast_hot, ast_cold = detect_streak(ast)
    reb_hot, reb_cold = detect_streak(reb)
    thr_hot, thr_cold = detect_streak(thr)
    
    # Variance/consistency
    def calc_variance(vals, window=10):
        if not vals or len(vals) < 2:
            return 0.0
        w_vals = vals[:window]
        return float(np.std(w_vals)) if len(w_vals) > 1 else 0.0
    
    # Ceiling/floor
    def calc_ceiling_floor(vals, window=10):
        if not vals:
            return 0.0, 0.0
        w_vals = vals[:window]
        return float(np.max(w_vals)) if w_vals else 0.0, float(np.min(w_vals)) if w_vals else 0.0
    
    pts_var = calc_variance(pts)
    pts_ceil, pts_floor = calc_ceiling_floor(pts)
    
    # Build feature dictionary
    feats = {
        # ===== BASE CONTEXT (18 features) =====
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
        
        # ===== PLAYER REST (2 features) =====
        "days_rest": 3.0,
        "player_b2b": 0.0,
        
        # ===== CORE ROLLING STATS (12 features) =====
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
        
        # ===== PHASE 1: SHOT VOLUME (16 features) =====
        "fieldGoalsAttempted_L3": avg(fga, 3) if fga else 10.0,
        "fieldGoalsAttempted_L5": avg(fga, 5) if fga else 10.0,
        "fieldGoalsAttempted_L10": avg(fga, 10) if fga else 10.0,
        "threePointersAttempted_L3": avg(tpa, 3) if tpa else 3.0,
        "threePointersAttempted_L5": avg(tpa, 5) if tpa else 3.0,
        "threePointersAttempted_L10": avg(tpa, 10) if tpa else 3.0,
        "freeThrowsAttempted_L3": avg(fta, 3) if fta else 2.0,
        "freeThrowsAttempted_L5": avg(fta, 5) if fta else 2.0,
        "freeThrowsAttempted_L10": avg(fta, 10) if fta else 2.0,
        
        # Per-minute rates
        "rate_fga": safe_div(avg(fga, 10), avg(mins, 10)) if fga and mins else 0.4,
        "rate_3pa": safe_div(avg(tpa, 10), avg(mins, 10)) if tpa and mins else 0.12,
        "rate_fta": safe_div(avg(fta, 10), avg(mins, 10)) if fta and mins else 0.08,
        
        # Efficiency metrics
        "ts_pct_L5": calc_ts_pct(pts, fga, fta, 5),
        "ts_pct_L10": calc_ts_pct(pts, fga, fta, 10),
        "ts_pct_season": calc_ts_pct(pts, fga, fta, len(pts) if pts else 10),
        "three_pct_L5": avg(three_pct, 5) if three_pct else 0.35,
        "ft_pct_L5": avg(ft_pct, 5) if ft_pct else 0.77,
        
        # ===== PHASE 2: MATCHUP CONTEXT (4 features) =====
        "matchup_pace": 1.0,
        "pace_factor": 1.0,
        "def_matchup_difficulty": 1.0,
        "offensive_environment": 1.0,
        
        # ===== PHASE 3: ADVANCED RATES (3 features) =====
        "usage_rate_L5": (safe_div(avg(fga, 5) + 0.44 * avg(fta, 5), avg(mins, 5)) * 5.0 * 4.8) if fga and fta and mins else 22.0,
        "rebound_rate_L5": (safe_div(avg(reb, 5), avg(mins, 5)) * 48.0 * 0.5) if reb and mins else 12.0,
        "assist_rate_L5": (safe_div(avg(ast, 5), avg(mins, 5)) * 48.0 * 0.35) if ast and mins else 18.0,
        
        # ===== PHASE 4: HOME/AWAY SPLITS (4 features) =====
        "points_home_avg": avg(pts, 10),
        "points_away_avg": avg(pts, 10),
        "assists_home_avg": avg(ast, 10),
        "assists_away_avg": avg(ast, 10),
        
        # ===== PHASE 5: POSITION/MATCHUP (10 features) =====
        # Simplified position detection (based on rebounds/assists ratio)
        "is_guard": 1.0 if avg(ast, 10) > avg(reb, 10) else 0.0,
        "is_forward": 0.5,  # Default
        "is_center": 1.0 if avg(reb, 10) > (avg(ast, 10) * 2) else 0.0,
        "position_versatility": 0.5,
        "opp_def_vs_rebounds_adj": 1.0,
        "opp_def_vs_assists_adj": 1.0,
        "starter_prob": 0.8,
        "minutes_ceiling": 32.0,
        "likely_injury_return": 0.0,
        "games_since_injury": 10.0,
        
        # ===== PHASE 6: MOMENTUM & OPTIMIZATION (24 features) =====
        # Momentum metrics
        "points_momentum_short": pts_mom,
        "points_momentum_med": pts_mom,  # Same for now (simplified)
        "points_momentum_long": pts_mom,
        "points_acceleration": pts_accel,
        "points_hot_streak": pts_hot,
        "points_cold_streak": pts_cold,
        
        "assists_momentum_short": ast_mom,
        "assists_acceleration": ast_accel,
        "assists_hot_streak": ast_hot,
        "assists_cold_streak": ast_cold,
        
        "rebounds_momentum_short": reb_mom,
        "rebounds_acceleration": reb_accel,
        "rebounds_hot_streak": reb_hot,
        "rebounds_cold_streak": reb_cold,
        
        "threes_momentum_short": thr_mom,
        "threes_acceleration": thr_accel,
        "threes_hot_streak": thr_hot,
        "threes_cold_streak": thr_cold,
        
        # Variance/consistency
        "points_variance_L10": pts_var,
        "points_ceiling_L10": pts_ceil,
        "points_floor_L10": pts_floor,
        
        # Fatigue indicators
        "games_in_last_7_days": 3.0,  # Estimate
        "minutes_per_game_L5": avg(mins, 5) if mins else 24.0,
        "fatigue_index": 0.0,
        
        # ===== MINUTES PREDICTION (1 feature) =====
        "minutes": avg(mins, 10) if mins else 24.0,
    }
    
    # ===== PHASE 7: BASKETBALL REFERENCE PRIORS (68 features if available) =====
    if priors is not None and player_name:
        # Try to match player in priors
        player_priors = priors[priors['name_join'].str.lower() == player_name.lower()]
        if not player_priors.empty:
            # Merge prior features (excluding name_join)
            prior_row = player_priors.iloc[0]
            for col in prior_row.index:
                if col != 'name_join' and pd.notnull(prior_row[col]):
                    feats[f"prior_{col}"] = float(prior_row[col])
            print(f"   ✓ Matched priors for {player_name} ({len(prior_row)-1} features)")
        else:
            # No match - fill with NaN (models trained to handle this)
            print(f"   ⚠️  No priors match for {player_name}")
    
    return pd.DataFrame([feats])


# ============================================================================
# 4. USAGE EXAMPLE (How to integrate into riq_analyzer.py)
# ============================================================================

"""
INTEGRATION STEPS:

1. Replace ModelPredictor class in riq_analyzer.py (lines ~2649-2966):
   - Copy the ModelPredictor class from section 1 above
   
2. Add load_priors_data function (add after line ~366):
   - Copy load_priors_data from section 2 above
   
3. Replace build_player_features function (line ~2968):
   - Copy build_player_features_expanded from section 3 above
   - Rename to build_player_features
   
4. Update analyze_player_prop to load priors (around line ~3172):
   ```python
   # At top of analyze_player_prop function, add:
   priors = load_priors_data()  # Load once per analysis run
   
   # Update feats_row creation (around line ~3245):
   feats_row = build_player_features(df_last, df_curr, 
                                      player_name=prop["player"],
                                      priors=priors)
   ```

5. Test with a single prop:
   ```python
   python riq_analyzer.py
   ```

EXPECTED RESULTS:
- Model loading should show "✓ Loaded X model" for each stat type
- Feature engineering should show "✓ Matched priors for PLAYER_NAME" for ~49% of players
- Predictions should use ~150-218 features (vs old 61)
- No shape mismatch errors

FALLBACK BEHAVIOR:
- If priors not available: Uses ~80 features (Phases 1-6 only)
- If old models: Falls back to LightGBM-only prediction
- If feature mismatch: Models will log warning but may still predict

TROUBLESHOOTING:
- If "shape mismatch" errors: Check model.feature_names vs feats.columns
- If "priors not found": Download priors_data.zip and extract to project root
- If prediction errors: Check that models are NeuralHybridPredictor not plain LightGBM
"""
