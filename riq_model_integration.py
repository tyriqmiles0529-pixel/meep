from __future__ import annotations
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

STAT_NAME_MAP = {
    "points": "points",
    "assists": "assists",
    "rebounds": "rebounds",
    "threes": "threepoint_goals",  # analyzer uses 'threes'; model uses 'threepoint_goals'
}

class ModelPredictor:
    def __init__(self, models_dir: str | Path = "models"):
        self.models_dir = Path(models_dir)
        self.models: Dict[str, dict] = {}
        self._load_all()

    def _load_all(self):
        # Load each stat model if present
        for stat_key in ["points", "assists", "rebounds", "threepoint_goals"]:
            pkl = self.models_dir / f"{stat_key}_model.pkl"
            if pkl.exists():
                with open(pkl, "rb") as f:
                    self.models[stat_key] = pickle.load(f)

    def available(self) -> Dict[str, bool]:
        return {k: (k in self.models) for k in ["points", "assists", "rebounds", "threepoint_goals"]}

    @staticmethod
    def _rolling_features(series: pd.Series) -> dict:
        # Expect a chronological series of past game values
        s = pd.to_numeric(series, errors="coerce")
        s_shift = s.shift(1)
        def rmean(w): return s_shift.rolling(w, min_periods=1).mean()
        def rstd(w):  return s_shift.rolling(w, min_periods=1).std()
        def rmax(w):  return s_shift.rolling(w, min_periods=1).max()

        feats = {
            "avg_3g": rmean(3).iloc[-1],
            "avg_5g": rmean(5).iloc[-1],
            "avg_10g": rmean(10).iloc[-1],
            "std_3g": rstd(3).iloc[-1],
            "std_5g": rstd(5).iloc[-1],
            "max_3g": rmax(3).iloc[-1],
        }
        # Trend: (3g - 10g) / |10g|
        if pd.notna(feats["avg_3g"]) and pd.notna(feats["avg_10g"]):
            denom = abs(feats["avg_10g"]) + 1e-6
            feats["trend"] = (feats["avg_3g"] - feats["avg_10g"]) / denom
        else:
            feats["trend"] = np.nan
        return feats

    def predict(self, stat_name: str, history_df: pd.DataFrame, is_home: Optional[int] = None, game_num: Optional[int] = None) -> Tuple[Optional[float], Optional[float]]:
        # Map analyzer stat to model stat name
        model_stat = STAT_NAME_MAP.get(stat_name)
        if model_stat is None:
            return None, None
        model_bundle = self.models.get(model_stat)
        if not model_bundle:
            return None, None

        # Expect a column named like the model's target
        if model_stat not in history_df.columns or len(history_df) == 0:
            return None, None

        # Minimal features to match training
        feats = self._rolling_features(history_df[model_stat])
        X = {
            f"{model_stat}_avg_3g": feats["avg_3g"],
            f"{model_stat}_avg_5g": feats["avg_5g"],
            f"{model_stat}_avg_10g": feats["avg_10g"],
            f"{model_stat}_std_3g": feats["std_3g"],
            f"{model_stat}_std_5g": feats["std_5g"],
            f"{model_stat}_max_3g": feats["max_3g"],
            f"{model_stat}_trend": feats["trend"],
            "is_home": np.nan if is_home is None else int(is_home),
            "game_num": np.nan if game_num is None else int(game_num),
        }

        # Align to model feature list (LightGBM handles NaN)
        feature_cols = model_bundle["features"]
        xrow = pd.DataFrame([{c: X.get(c, np.nan) for c in feature_cols}])

        model = model_bundle["model"]
        yhat = float(model.predict(xrow)[0])

        # Uncertainty: blend model RMSE with recent variability
        rmse = float(model_bundle["metrics"].get("test_rmse", 0.0))
        recent = pd.to_numeric(history_df[model_stat].tail(10), errors="coerce").dropna()
        recent_std = float(recent.std(ddof=1)) if len(recent) >= 2 else rmse
        sigma = 0.5 * rmse + 0.5 * recent_std
        # Guardrails per market (keep reasonable floors)
        floors = {"points": 0.8, "assists": 0.6, "rebounds": 0.8, "threepoint_goals": 0.5}
        sigma = max(floors.get(model_stat, 0.7), sigma)

        return yhat, sigma