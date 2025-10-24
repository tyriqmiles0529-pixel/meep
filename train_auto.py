#!/usr/bin/env python3
"""
End-to-end NBA training pipeline (always fetches from Kaggle).

What it trains
1) Game-level models (TeamStatistics)
   - Moneyline classifier: P(home wins), with isotonic probability calibration.
   - Spread regressor: expected margin (home - away), plus residual sigma for cover probabilities.
   - Time-safe OOF predictions to feed into player models (learn-from-each-other without leakage).

2) Player-level models (PlayerStatistics)
   - Minutes model (regression).
   - Points, Rebounds, Assists, 3PM models (regression).
   - Every player row includes BOTH team and opponent context, matchup edges, OOF game signals, and player rolling rates/minutes trends.
   - Works even if teamId/opponentTeamId are missing — uses 'home' flag to pick the correct side.

Key safety and robustness
- Leakage-safe team context: all rolling stats are shifted by 1 (pre-game only).
- Missing dates: never dropped; per-team order falls back to numeric gameId.
- Clean console output with grouped sections and rounded metrics.
- LightGBM noise reduced (force_col_wise=True, verbosity=-1); sklearn fallback available.

Practical era handling
- Season cutoffs for data inclusion (player and game).
- Season features in models (season_end_year, season_decade).
- Time-decay sample weights with lockout downweight (1999, 2012).

Run (PowerShell)
- pip install kagglehub
- python .\\train_auto.py --dataset "eoinamoore/historical-nba-data-and-player-box-scores" --verbose --skip-rest --fresh --lgb-log-period 50
"""

from __future__ import annotations

import sys
import os
import re
import json
import math
import argparse
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, log_loss, brier_score_loss
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.calibration import CalibratedClassifierCV

# Prefer LightGBM; fallback to sklearn HistGBM
_HAS_LGB = False
try:
    import lightgbm as lgb  # type: ignore
    _HAS_LGB = True
    try:
        lgb.set_config(verbosity=-1)
    except Exception:
        pass
except Exception:
    pass

try:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
except Exception:  # pragma: no cover
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

# Kaggle (required)
try:
    import kagglehub
except Exception:
    kagglehub = None

# Silence upcoming sklearn FutureWarning for CalibratedClassifierCV(cv='prefit')
warnings.filterwarnings(
    "ignore",
    message="The `cv='prefit'` option is deprecated",
    category=FutureWarning,
)

# ---------------- Kaggle credentials (hardcoded for venv compatibility) ----------------
KAGGLE_KEY = "bcb440122af5ae76181e68d48ca728e6"
KAGGLE_USERNAME = "tyriqmiles"

if KAGGLE_KEY and KAGGLE_KEY != "YOUR_KEY_HERE":
    os.environ['KAGGLE_KEY'] = KAGGLE_KEY
    if KAGGLE_USERNAME:
        os.environ['KAGGLE_USERNAME'] = KAGGLE_USERNAME

# ---------------- Pretty printing helpers ----------------

def _line(char: str = "─", n: int = 60) -> str:
    return char * n

def _sec(title: str) -> str:
    return f"\\n{_line()} \\n{title}\\n{_line()}"

def _fmt(n: float, nd: int = 3) -> str:
    try:
        return f"{n:.{nd}f}"
    except Exception:
        return str(n)

def log(msg: str, verbose: bool):
    if verbose:
        print(msg, flush=True)

# ---------------- Era helpers ----------------

LOCKOUT_SEASONS = {1999, 2012}

def _season_from_date(dt: pd.Series) -> pd.Series:
    """
    Convert a UTC-naive datetime to NBA season end-year.
    Season end-year = year if month <= 7; else year+1 (Aug..Dec map to next year's season).
    """
    d = pd.to_datetime(dt, errors="coerce", utc=False)
    y = d.dt.year
    m = d.dt.month
    return np.where(m >= 8, y + 1, y)

def _decade_from_season(season_end_year: pd.Series) -> pd.Series:
    s = pd.to_numeric(season_end_year, errors="coerce")
    return (s // 10) * 10

def _parse_season_cutoff(arg_val: str, kind: str) -> int:
    """
    kind='game' presets: 'classic'->1997, 'balanced'->2002
    kind='player' presets: 'min'->1998, 'balanced'->2002, 'modern'->2005
    Also accepts integer strings, e.g., '2001', '2005'.
    """
    v = str(arg_val).strip().lower()
    if kind == "game":
        presets = {"classic": 1997, "balanced": 2002}
    else:
        presets = {"min": 1998, "balanced": 2002, "modern": 2005}
    if v in presets:
        return presets[v]
    try:
        return int(v)
    except Exception:
        # Fallback to balanced defaults
        return presets["balanced"]

def _compute_sample_weights(seasons: np.ndarray, decay: float, min_weight: float, lockout_weight: float) -> np.ndarray:
    seasons = seasons.astype("float64")
    valid = np.isfinite(seasons)
    if not valid.any():
        return np.ones_like(seasons, dtype="float64")
    max_season = int(np.nanmax(seasons[valid]))
    base = np.ones_like(seasons, dtype="float64")
    base[valid] = decay ** (max_season - seasons[valid])
    base = np.maximum(base, min_weight)
    # Lockouts
    for lo in LOCKOUT_SEASONS:
        base = np.where(seasons == lo, base * lockout_weight, base)
    # For NaN seasons, keep weight at 1.0
    base = np.where(valid, base, 1.0)
    return base.astype("float64")

# ---------------- Game feature schema ----------------

GAME_FEATURES: List[str] = [
    "home_advantage", "neutral_site",
    "home_recent_pace", "away_recent_pace",
    "home_off_strength", "home_def_strength",
    "away_off_strength", "away_def_strength",
    "home_recent_winrate", "away_recent_winrate",
    # matchup features
    "match_off_edge", "match_def_edge", "match_pace_sum", "winrate_diff",
    # schedule/injury (often constants if --skip-rest)
    "home_days_rest", "away_days_rest",
    "home_b2b", "away_b2b",
    "home_injury_impact", "away_injury_impact",
    # era features
    "season_end_year", "season_decade",
    # betting market features (optional - added when --odds-dataset provided)
    "market_implied_home", "market_implied_away",
    "market_spread", "spread_move",
    "market_total", "total_move",
    # team priors from Basketball Reference (optional - added when --priors-dataset provided)
    "home_o_rtg_prior", "home_d_rtg_prior", "home_pace_prior",
    "away_o_rtg_prior", "away_d_rtg_prior", "away_pace_prior",
    "home_srs_prior", "away_srs_prior",
]

GAME_DEFAULTS: Dict[str, float] = {
    "home_advantage": 1.0,
    "neutral_site": 0.0,
    "home_recent_pace": 1.0,
    "away_recent_pace": 1.0,
    "home_off_strength": 1.0,
    "home_def_strength": 1.0,
    "away_off_strength": 1.0,
    "away_def_strength": 1.0,
    "home_recent_winrate": 0.5,
    "away_recent_winrate": 0.5,
    "match_off_edge": 0.0,
    "match_def_edge": 0.0,
    "match_pace_sum": 2.0,
    "winrate_diff": 0.0,
    "home_days_rest": 2.0,
    "away_days_rest": 2.0,
    "home_b2b": 0.0,
    "away_b2b": 0.0,
    "home_injury_impact": 0.0,
    "away_injury_impact": 0.0,
    # era defaults (will be filled from actual dates when present)
    "season_end_year": 2002.0,
    "season_decade": 2000.0,
    # betting market defaults (neutral priors)
    "market_implied_home": 0.5,
    "market_implied_away": 0.5,
    "market_spread": 0.0,
    "spread_move": 0.0,
    "market_total": 210.0,  # typical NBA total
    "total_move": 0.0,
    # team priors defaults (league-average baseline)
    "home_o_rtg_prior": 110.0,
    "home_d_rtg_prior": 110.0,
    "home_pace_prior": 100.0,
    "away_o_rtg_prior": 110.0,
    "away_d_rtg_prior": 110.0,
    "away_pace_prior": 100.0,
    "home_srs_prior": 0.0,
    "away_srs_prior": 0.0,
}

# ---------------- Utilities ----------------

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def _id_to_str(s: pd.Series) -> pd.Series:
    s_num = pd.to_numeric(s, errors="coerce")
    out = pd.Series(index=s.index, dtype="object")
    num_mask = s_num.notna()
    out.loc[num_mask] = s_num.loc[num_mask].astype("Int64").astype(str)
    out.loc[~num_mask] = s.astype(str)
    return out.astype(str).str.strip()

def _find_dataset_files(ds_root: Path) -> Tuple[Optional[Path], Optional[Path]]:
    teams_path: Optional[Path] = None
    players_path: Optional[Path] = None
    for p in ds_root.glob("*.csv"):
        n = p.name.lower()
        if "teamstatistics" in n or "team_statistics" in n:
            teams_path = p
        if "playerstatistics" in n or "player_stats" in n or "playerstats" in n:
            players_path = p
    # Fallback: scan names that start with Team/Player
    if teams_path is None:
        cand = [p for p in ds_root.glob("*.csv") if "team" in p.name.lower()]
        if cand:
            teams_path = cand[0]
    if players_path is None:
        cand = [p for p in ds_root.glob("*.csv") if "player" in p.name.lower()]
        if cand:
            players_path = cand[0]
    return teams_path, players_path

def _fresh_run_dir(base: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def _copy_if_exists(src: Optional[Path], dst_dir: Path) -> Optional[Path]:
    if src and src.exists():
        out = dst_dir / src.name
        out.write_bytes(src.read_bytes())
        return out
    return None

# ---------------- Build games from TeamStatistics ----------------

def build_games_from_teamstats(teams_path: Path, verbose: bool, skip_rest: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    header_cols = list(pd.read_csv(teams_path, nrows=0).columns)
    usecols = [c for c in [
        "gameId", "gameDate", "teamId", "opponentTeamId",
        "home", "teamScore", "opponentScore",
    ] if c in header_cols]
    ts = pd.read_csv(teams_path, low_memory=False, usecols=usecols)

    for c in ["gameId", "teamId", "opponentTeamId"]:
        if c in ts.columns:
            ts[c] = _id_to_str(ts[c])

    # home flag to 0/1
    if "home" in ts.columns:
        hv = ts["home"]
        hvn = pd.to_numeric(hv, errors="coerce")
        ts["home_flag"] = np.where(hvn.notna(), (hvn.fillna(0) != 0).astype(int),
                                   hv.astype(str).str.strip().str.lower().isin(["1","true","t","home","h","yes","y"]).astype(int))
    else:
        ts["home_flag"] = 1

    # date (keep NaT)
    ts["gameDate"] = pd.to_datetime(ts["gameDate"], errors="coerce", utc=True).dt.tz_convert(None)

    # scores
    ts["teamScore"] = pd.to_numeric(ts["teamScore"], errors="coerce")
    ts["opponentScore"] = pd.to_numeric(ts["opponentScore"], errors="coerce")
    ts = ts.dropna(subset=["gameId", "teamId", "opponentTeamId", "teamScore", "opponentScore"]).copy()

    # pair home/away
    ts_sorted = ts.sort_values(["gameId", "home_flag"], ascending=[True, False])
    home_rows = ts_sorted[ts_sorted["home_flag"] == 1].drop_duplicates("gameId", keep="first")
    away_rows = ts_sorted[ts_sorted["home_flag"] == 0].drop_duplicates("gameId", keep="first")

    g = home_rows.merge(
        away_rows[["gameId", "teamId", "teamScore", "gameDate"]].rename(columns={
            "teamId": "away_tid", "teamScore": "away_score", "gameDate": "date_check"
        }),
        on="gameId", how="left"
    )

    # fill away from opponent fields if missing
    need_away = g["away_tid"].isna()
    if need_away.any():
        tss = ts.set_index("gameId")
        idx = g.loc[need_away, "gameId"]
        g.loc[need_away, "away_tid"] = _id_to_str(tss.loc[idx, "opponentTeamId"])
        g.loc[need_away, "away_score"] = pd.to_numeric(tss.loc[idx, "opponentScore"], errors="coerce")

    g = g.rename(columns={
        "gameId": "gid", "teamId": "home_tid", "teamScore": "home_score", "gameDate": "date"
    })[["gid", "date", "home_tid", "away_tid", "home_score", "away_score"]]

    g = g.dropna(subset=["home_tid", "away_tid", "home_score", "away_score"]).copy()
    for c in ["gid", "home_tid", "away_tid"]:
        g[c] = _id_to_str(g[c])
    g["date"] = pd.to_datetime(g["date"], errors="coerce", utc=True).dt.tz_convert(None)

    # season features
    g["season_end_year"] = _season_from_date(g["date"]).astype("float32")
    g["season_decade"]   = _decade_from_season(g["season_end_year"]).astype("float32")

    # long view for rolling context
    long_home = g[["gid", "date", "season_end_year", "home_tid", "home_score", "away_score"]].rename(
        columns={"home_tid": "tid", "home_score": "team_pts", "away_score": "opp_pts"}
    )
    long_away = g[["gid", "date", "season_end_year", "away_tid", "away_score", "home_score"]].rename(
        columns={"away_tid": "tid", "away_score": "team_pts", "home_score": "opp_pts"}
    )
    teams_long = pd.concat(
        [long_home[["gid", "date", "season_end_year", "tid", "team_pts", "opp_pts"]],
         long_away[["gid", "date", "season_end_year", "tid", "team_pts", "opp_pts"]]],
        ignore_index=True
    )

    teams_long["gid_num"] = pd.to_numeric(teams_long["gid"], errors="coerce")
    teams_long = teams_long.sort_values(["tid", "date", "gid_num"], ascending=[True, True, True], na_position="last")

    # leakage-safe rolling with min_periods=1
    teams_long["off_pts_10"] = teams_long.groupby("tid")["team_pts"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    teams_long["def_pts_10"] = teams_long.groupby("tid")["opp_pts"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    teams_long["tot_pts"]    = teams_long["team_pts"] + teams_long["opp_pts"]
    teams_long["pace_10"]    = teams_long.groupby("tid")["tot_pts"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())

    teams_long["win"] = (teams_long["team_pts"] > teams_long["opp_pts"]).astype(int)
    teams_long["wins_prev"]  = teams_long.groupby("tid")["win"].transform(lambda x: x.shift(1).cumsum())
    teams_long["games_prev"] = teams_long.groupby("tid").cumcount()
    teams_long["winrate_prev"] = np.where(
        teams_long["games_prev"] > 0,
        teams_long["wins_prev"] / teams_long["games_prev"].clip(lower=1),
        0.5
    )

    # per-season medians for normalization (era-aware)
    def _per_season_median(col: str, fallback: float) -> pd.Series:
        s = teams_long.groupby("season_end_year")[col].transform("median")
        s = s.fillna(fallback)
        z = s.replace(0, np.nan)
        z = z.fillna(fallback if fallback != 0 else 1.0)
        return z

    off_med_g  = np.nanmedian(teams_long["off_pts_10"].values)
    def_med_g  = np.nanmedian(teams_long["def_pts_10"].values)
    pace_med_g = np.nanmedian(teams_long["pace_10"].values)
    off_med_s  = _per_season_median("off_pts_10", off_med_g if np.isfinite(off_med_g) and off_med_g > 0 else 1.0)
    def_med_s  = _per_season_median("def_pts_10", def_med_g if np.isfinite(def_med_g) and def_med_g > 0 else 1.0)
    pace_med_s = _per_season_median("pace_10", pace_med_g if np.isfinite(pace_med_g) and pace_med_g > 0 else 1.0)

    teams_long["off_norm"]  = teams_long["off_pts_10"] / off_med_s
    teams_long["def_norm"]  = def_med_s / teams_long["def_pts_10"]
    teams_long["pace_norm"] = teams_long["pace_10"] / pace_med_s

    # split to home/away and merge
    hstats = teams_long.rename(columns={
        "tid": "home_tid",
        "off_norm": "home_off_strength",
        "def_norm": "home_def_strength",
        "pace_norm": "home_recent_pace",
        "winrate_prev": "home_recent_winrate"
    })[["gid", "season_end_year", "home_tid", "home_off_strength", "home_def_strength", "home_recent_pace", "home_recent_winrate"]].drop_duplicates(["gid", "home_tid"])
    astats = teams_long.rename(columns={
        "tid": "away_tid",
        "off_norm": "away_off_strength",
        "def_norm": "away_def_strength",
        "pace_norm": "away_recent_pace",
        "winrate_prev": "away_recent_winrate"
    })[["gid", "season_end_year", "away_tid", "away_off_strength", "away_def_strength", "away_recent_pace", "away_recent_winrate"]].drop_duplicates(["gid", "away_tid"])

    for c in ["gid", "home_tid"]:
        hstats[c] = _id_to_str(hstats[c])
    for c in ["gid", "away_tid"]:
        astats[c] = _id_to_str(astats[c])

    g = g.merge(hstats, on=["gid", "home_tid", "season_end_year"], how="left").merge(astats, on=["gid", "away_tid", "season_end_year"], how="left")

    # matchup features
    g["match_off_edge"] = g["home_off_strength"] - g["away_def_strength"]
    g["match_def_edge"] = g["home_def_strength"] - g["away_off_strength"]
    g["match_pace_sum"] = g["home_recent_pace"] + g["away_recent_pace"]
    g["winrate_diff"]   = g["home_recent_winrate"] - g["away_recent_winrate"]

    # optional rest/b2b
    if not skip_rest:
        sched_home = g[["home_tid", "date"]].rename(columns={"home_tid": "tid"})
        sched_away = g[["away_tid", "date"]].rename(columns={"away_tid": "tid"})
        sched = pd.concat([sched_home, sched_away], ignore_index=True)
        sched["tid"]  = _id_to_str(sched["tid"])
        sched["date"] = pd.to_datetime(sched["date"], errors="coerce", utc=True).dt.tz_convert(None)
        sched = sched.dropna(subset=["tid", "date"]).drop_duplicates(["tid", "date"]).sort_values(["tid", "date"])

        sched["prev_date"] = sched.groupby("tid")["date"].shift(1)
        delta = sched["date"] - sched["prev_date"]
        sched["days_rest"] = pd.to_numeric(delta.dt.days, errors="coerce").fillna(2.0).astype("float32")
        sched["b2b"] = (sched["days_rest"] <= 1).astype("int8")

        hr = sched.rename(columns={"tid": "home_tid"})[["home_tid", "date", "days_rest", "b2b"]].rename(
            columns={"days_rest": "home_days_rest", "b2b": "home_b2b"}
        )
        ar = sched.rename(columns={"tid": "away_tid"})[["away_tid", "date", "days_rest", "b2b"]].rename(
            columns={"days_rest": "away_days_rest", "b2b": "away_b2b"}
        )

        for df_, tid_col in ((hr, "home_tid"), (ar, "away_tid")):
            df_[tid_col] = _id_to_str(df_[tid_col])
            df_["date"]  = pd.to_datetime(df_["date"], errors="coerce", utc=True).dt.tz_convert(None)
            df_.sort_values([tid_col, "date"], inplace=True)
            df_.drop_duplicates([tid_col, "date"], keep="last", inplace=True)

        g.sort_values(["gid", "date"], inplace=True)
        g.drop_duplicates(subset=["gid", "home_tid", "away_tid", "date"], keep="last", inplace=True)
        g = g.merge(hr, on=["home_tid", "date"], how="left")
        g = g.merge(ar, on=["away_tid", "date"], how="left")
    else:
        g["home_days_rest"] = GAME_DEFAULTS["home_days_rest"]
        g["away_days_rest"] = GAME_DEFAULTS["away_days_rest"]
        g["home_b2b"] = GAME_DEFAULTS["home_b2b"]
        g["away_b2b"] = GAME_DEFAULTS["away_b2b"]

    # era features (decade again in case of merges)
    g["season_decade"] = _decade_from_season(g["season_end_year"]).astype("float32")

    # fill defaults
    for col, val in GAME_DEFAULTS.items():
        if col not in g.columns:
            g[col] = val
        g[col] = pd.to_numeric(g[col], errors="coerce").fillna(val).astype("float32")

    g["home_advantage"] = np.float32(1.0)
    g["neutral_site"]   = np.float32(0.0)

    # FIX: Don't duplicate season_end_year/season_decade when building games_df
    base_cols = ["gid", "date", "season_end_year", "season_decade", "home_tid", "away_tid", "home_score", "away_score"]
    feature_cols = [f for f in GAME_FEATURES if f not in base_cols]
    games_df = g[base_cols + feature_cols].copy()

    games_df["date"] = pd.to_datetime(games_df["date"], errors="coerce", utc=True).dt.tz_convert(None)
    games_df = games_df.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)

    # context for players (both sides)
    context_cols = [
        "gid", "date", "season_end_year", "season_decade", "home_tid", "away_tid",
        "home_recent_pace", "home_off_strength", "home_def_strength", "home_recent_winrate",
        "away_recent_pace", "away_off_strength", "away_def_strength", "away_recent_winrate",
        "match_off_edge", "match_def_edge", "match_pace_sum", "winrate_diff"
    ]
    context_map = g[context_cols].drop_duplicates(["gid", "home_tid", "away_tid"]).copy()

    log(f"Built TeamStatistics games frame: {len(games_df):,} rows", verbose)
    return games_df, context_map

# ---------------- Train game models + OOF ----------------

def _fit_game_models(
    games_df: pd.DataFrame,
    seed: int,
    verbose: bool,
    folds: int = 5,
    lgb_log_period: int = 0,
    sample_weights: Optional[np.ndarray] = None,
) -> Tuple[object, Optional[CalibratedClassifierCV], object, float, pd.DataFrame, Dict[str, float]]:
    X_full = games_df[GAME_FEATURES].apply(pd.to_numeric, errors="coerce").replace([np.inf,-np.inf], np.nan).astype("float32")
    y_ml = (games_df["home_score"].values > games_df["away_score"].values).astype(int)
    y_sp = (games_df["home_score"].values - games_df["away_score"].values).astype(float)

    # chronological order (NaT last)
    order = pd.to_datetime(games_df["date"], errors="coerce")
    idx_sorted = order.sort_values(na_position="last", kind="mergesort").index
    X_sorted = X_full.loc[idx_sorted]
    y_ml_sorted = y_ml[idx_sorted]
    y_sp_sorted = y_sp[idx_sorted]
    gid_sorted = games_df.loc[idx_sorted, "gid"].astype(str).values
    if sample_weights is None or len(sample_weights) != len(games_df):
        w_sorted = np.ones(len(X_sorted), dtype=np.float64)
    else:
        w_sorted = sample_weights[idx_sorted]

    n = len(X_sorted)
    if n <= 1:
        ml_dummy = DummyClassifier(strategy="prior").fit(X_sorted.iloc[:1], y_ml_sorted[:1])
        sp_dummy = DummyRegressor(strategy="mean").fit(X_sorted.iloc[:1], y_sp_sorted[:1])
        return ml_dummy, None, sp_dummy, float("nan"), pd.DataFrame(), {
            "train_size": int(n), "val_size": 0,
            "ml_logloss": float("nan"), "ml_brier": float("nan"),
            "sp_rmse": float("nan"), "sp_mae": float("nan"),
        }

    # Time-based folds (equal chunks in sorted order)
    fold_sizes = [n // folds] * folds
    for i in range(n % folds):
        fold_sizes[i] += 1
    splits = []
    start = 0
    for fs in fold_sizes:
        end = start + fs
        splits.append((start, end))
        start = end

    oof_ml = np.full(n, np.nan, dtype=np.float32)
    oof_sp = np.full(n, np.nan, dtype=np.float32)

    # Common LGB params (quiet + col-wise)
    clf_params = dict(
        objective="binary", learning_rate=0.05, num_leaves=31, max_depth=-1,
        colsample_bytree=0.9, subsample=0.8, subsample_freq=5,
        n_estimators=800, random_state=seed, n_jobs=-1,
        force_col_wise=True, verbosity=-1
    )
    reg_params = dict(
        objective="regression", learning_rate=0.05, num_leaves=31, max_depth=-1,
        colsample_bytree=0.9, subsample=0.8, subsample_freq=5,
        n_estimators=800, random_state=seed, n_jobs=-1,
        force_col_wise=True, verbosity=-1
    )

    # Train per fold: train on [0:val_start], predict on [val_start:val_end]
    for k, (val_start, val_end) in enumerate(splits):
        if val_start == 0:
            continue
        tr_slice = slice(0, val_start)
        vl_slice = slice(val_start, val_end)

        X_tr, y_ml_tr, y_sp_tr = X_sorted.iloc[tr_slice, :], y_ml_sorted[tr_slice], y_sp_sorted[tr_slice]
        X_vl = X_sorted.iloc[vl_slice, :]
        w_tr = w_sorted[tr_slice]

        if _HAS_LGB:
            clf_k = lgb.LGBMClassifier(**{**clf_params, "random_state": seed + k})
            clf_k.fit(X_tr, y_ml_tr, sample_weight=w_tr)
            oof_ml[val_start:val_end] = clf_k.predict_proba(X_vl)[:, 1]

            reg_k = lgb.LGBMRegressor(**{**reg_params, "random_state": seed + k})
            reg_k.fit(X_tr, y_sp_tr, sample_weight=w_tr)
            oof_sp[val_start:val_end] = reg_k.predict(X_vl)
        else:
            clf_k = HistGradientBoostingClassifier(
                learning_rate=0.06, max_depth=None, max_iter=400, random_state=seed + k,
                validation_fraction=0.2, early_stopping=True
            )
            clf_k.fit(X_tr, y_ml_tr, sample_weight=w_tr)
            oof_ml[val_start:val_end] = clf_k.predict_proba(X_vl)[:, 1]

            reg_k = HistGradientBoostingRegressor(
                learning_rate=0.06, max_depth=None, max_iter=400, random_state=seed + k,
                validation_fraction=0.2, early_stopping=True
            )
            reg_k.fit(X_tr, y_sp_tr, sample_weight=w_tr)
            oof_sp[val_start:val_end] = reg_k.predict(X_vl)

    # Final train/val split (last 20%) for metrics/calibration
    split = max(1, min(int(n * 0.8), n - 1))
    X_tr, X_val = X_sorted.iloc[:split, :], X_sorted.iloc[split:, :]
    y_ml_tr, y_ml_val = y_ml_sorted[:split], y_ml_sorted[split:]
    y_sp_tr, y_sp_val = y_sp_sorted[:split], y_sp_sorted[split:]
    w_tr, w_val = w_sorted[:split], w_sorted[split:]

    # LightGBM callbacks for iteration logging (if requested)
    lgb_callbacks = []
    if _HAS_LGB and lgb_log_period and lgb_log_period > 0:
        lgb_callbacks.append(lgb.log_evaluation(period=int(lgb_log_period)))

    if _HAS_LGB:
        clf_final = lgb.LGBMClassifier(**{**clf_params, "n_estimators": 1000, "random_state": seed})
        clf_final.fit(
            X_tr, y_ml_tr, sample_weight=w_tr,
            eval_set=[(X_val, y_ml_val)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(80, verbose=bool(lgb_callbacks))] + lgb_callbacks
        )
        p_val_raw = clf_final.predict_proba(X_val)[:, 1]

        calibrator = CalibratedClassifierCV(clf_final, cv="prefit", method="isotonic")
        # Isotonic supports sample_weight
        calibrator.fit(X_val, y_ml_val, sample_weight=w_val)
        p_val_cal = calibrator.predict_proba(X_val)[:, 1]

        reg_final = lgb.LGBMRegressor(**{**reg_params, "n_estimators": 1000, "random_state": seed})
        reg_final.fit(
            X_tr, y_sp_tr, sample_weight=w_tr,
            eval_set=[(X_val, y_sp_val)],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(80, verbose=bool(lgb_callbacks))] + lgb_callbacks
        )
        y_pred_val = reg_final.predict(X_val)
    else:
        clf_final = HistGradientBoostingClassifier(
            learning_rate=0.06, max_depth=None, max_iter=400, random_state=seed,
            validation_fraction=0.2, early_stopping=True
        )
        clf_final.fit(X_tr, y_ml_tr, sample_weight=w_tr)
        p_val_raw = clf_final.predict_proba(X_val)[:, 1]
        calibrator = None
        p_val_cal = p_val_raw

        reg_final = HistGradientBoostingRegressor(
            learning_rate=0.06, max_depth=None, max_iter=400, random_state=seed,
            validation_fraction=0.2, early_stopping=True
        )
        reg_final.fit(X_tr, y_sp_tr, sample_weight=w_tr)
        y_pred_val = reg_final.predict(X_val)

    # Metrics
    ml_logloss = float(log_loss(y_ml_val, p_val_cal)) if len(np.unique(y_ml_val)) == 2 else float("nan")
    ml_brier = float(brier_score_loss(y_ml_val, p_val_cal))
    sp_rmse = float(math.sqrt(mean_squared_error(y_sp_val, y_pred_val)))
    sp_mae = float(mean_absolute_error(y_sp_val, y_pred_val))

    # Spread sigma from validation residuals
    spread_sigma = float(np.std(y_sp_val - y_pred_val, ddof=1))

    # Compose OOF frame
    oof_df = pd.DataFrame({
        "gid": gid_sorted,
        "oof_ml_prob": oof_ml,
        "oof_spread_pred": oof_sp,
    })

    # Pretty summary
    print(_sec("Game model metrics (validation)"))
    print(f"- Moneyline: logloss={_fmt(ml_logloss)}, Brier={_fmt(ml_brier)}")
    print(f"- Spread:    RMSE={_fmt(sp_rmse)}, MAE={_fmt(sp_mae)}, sigma={_fmt(spread_sigma)}")

    metrics = {
        "train_size": int(len(X_tr)),
        "val_size": int(len(X_val)),
        "ml_logloss": ml_logloss,
        "ml_brier": ml_brier,
        "sp_rmse": sp_rmse,
        "sp_mae": sp_mae,
        "spread_sigma": spread_sigma,
    }
    return clf_final, calibrator, reg_final, spread_sigma, oof_df, metrics

# ---------------- Player dataset (with opponent context) ----------------

def build_players_from_playerstats(
    player_path: Path,
    games_context: pd.DataFrame,
    oof_games: pd.DataFrame,
    verbose: bool
) -> Dict[str, pd.DataFrame]:
    """
    Build training frames for player models from PlayerStatistics.csv.
    Supports files without teamId/opponentTeamId by using the 'home' flag to select side context.
    Includes opponent context columns (opp_*).
    """
    # Read header first (no usecols) so we can detect what exists
    hdr = list(pd.read_csv(player_path, nrows=0).columns)

    # Helpers to resolve columns (case-insensitive, many aliases)
    def resolve_any(groups: list[list[str]]) -> Optional[str]:
        norm_map = {_norm(c): c for c in hdr}
        for group in groups:
            for cand in group:
                nc = _norm(cand)
                if nc in norm_map:
                    return norm_map[nc]
        return None

    # Column groups (include common names + your exact names)
    gid_col   = resolve_any([["gameId", "GAME_ID", "game_id", "gid"]])
    date_col  = resolve_any([["gameDate", "date", "GAME_DATE", "game_date"]])
    pid_col   = resolve_any([["personId", "playerId", "PLAYER_ID", "player_id", "person_id"]])
    name_col  = resolve_any([["playerName", "PLAYER_NAME", "player_name", "firstName", "lastName"]])  # optional
    tid_col   = resolve_any([["teamId", "TEAM_ID", "team_id", "tid"]])  # may be None
    home_col  = resolve_any([["home", "isHome", "HOME"]])  # present in your file

    min_col   = resolve_any([["numMinutes", "minutes", "mins", "min", "MIN"]])  # numMinutes in your file
    pts_col   = resolve_any([["points", "pts", "POINTS", "PTS"]])
    reb_col   = resolve_any([["reboundsTotal", "rebounds", "REB", "REBOUNDS"]])  # reboundsTotal in your file
    ast_col   = resolve_any([["assists", "ast", "ASSISTS", "AST"]])
    tpm_col   = resolve_any([["threePointersMade", "3pm", "FG3M", "threes", "three_pm"]])
    starter_col = resolve_any([["starter", "isStarter", "started", "STARTER", "IS_STARTER"]])  # likely missing

    # Read all detected columns (avoid usecols mismatch)
    want_cols = [gid_col, date_col, pid_col, name_col, tid_col, home_col,
                 min_col, pts_col, reb_col, ast_col, tpm_col, starter_col]
    usecols = [c for c in want_cols if c is not None]
    ps = pd.read_csv(player_path, low_memory=False, usecols=sorted(set(usecols)) if usecols else None)

    # Show what we detected (easier to debug)
    print(_sec("Detected player columns"))
    print(f"- gid: {gid_col}  date: {date_col}  pid: {pid_col}  name: {name_col}")
    print(f"- teamId: {tid_col}  home_flag: {home_col}")
    print(f"- minutes: {min_col}  points: {pts_col}  rebounds: {reb_col}  assists: {ast_col}  threes: {tpm_col}")

    # IDs to string (where present)
    for c in [gid_col, pid_col, tid_col]:
        if c and c in ps.columns:
            ps[c] = _id_to_str(ps[c])

    # Ensure player identifier
    if not pid_col or pid_col not in ps.columns:
        if name_col and name_col in ps.columns:
            keys = ps[name_col].astype(str).fillna("unknown")
            pid_codes, _ = pd.factorize(keys, sort=True)
            ps["__player_id__"] = pd.Series(pid_codes, index=ps.index).astype("Int64").astype(str)
            pid_col = "__player_id__"
        else:
            raise KeyError("No player identifier found (tried personId/playerId/player_name).")

    # Date parse
    if date_col and date_col in ps.columns:
        ps[date_col] = pd.to_datetime(ps[date_col], errors="coerce", utc=True).dt.tz_convert(None)
    else:
        ps["__no_date__"] = pd.NaT
        date_col = "__no_date__"

    # Era features from date
    ps["season_end_year"] = _season_from_date(ps[date_col]).astype("float32")
    ps["season_decade"]   = _decade_from_season(ps["season_end_year"]).astype("float32")

    # Numeric conversions
    for stat_col in [min_col, pts_col, reb_col, ast_col, tpm_col]:
        if stat_col and stat_col in ps.columns:
            ps[stat_col] = pd.to_numeric(ps[stat_col], errors="coerce")

    # Starter flag (optional)
    if starter_col and starter_col in ps.columns:
        s = ps[starter_col]
        num = pd.to_numeric(s, errors="coerce")
        ps["starter_flag"] = np.where(num.notna(), (num.fillna(0) != 0).astype(int),
                                      s.astype(str).str.strip().str.lower().isin(["1", "true", "t", "yes", "y", "starter", "start"]).astype(int))
    else:
        ps["starter_flag"] = 0

    # Home/away flag from player CSV (strong signal if teamId is missing)
    if home_col and home_col in ps.columns:
        h = ps[home_col]
        hnum = pd.to_numeric(h, errors="coerce")
        ps["is_home"] = np.where(
            hnum.notna(),
            (hnum.fillna(0) != 0).astype(int),
            h.astype(str).str.strip().str.lower().isin(["1", "true", "t", "home", "h", "yes", "y"]).astype(int)
        )
    else:
        ps["is_home"] = np.nan  # will fallback later

    # Sort per player by date then gid numeric
    ps["gid_num"] = pd.to_numeric(ps[gid_col], errors="coerce")
    ps = ps.sort_values([pid_col, date_col, "gid_num"], ascending=[True, True, True], na_position="last")

    # Rolling minutes trend
    if min_col and min_col in ps.columns:
        ps["min_prev_mean5"]  = ps.groupby(pid_col)[min_col].transform(lambda x: x.shift(1).rolling(5,  min_periods=1).mean())
        ps["min_prev_mean10"] = ps.groupby(pid_col)[min_col].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        ps["min_prev_last1"]  = ps.groupby(pid_col)[min_col].transform(lambda x: x.shift(1))
    else:
        ps["min_prev_mean5"] = 20.0
        ps["min_prev_mean10"] = 20.0
        ps["min_prev_last1"] = 20.0

    # Per-minute rates (shifted)
    def rate(prev_count_col: Optional[str]) -> pd.Series:
        if not (min_col and prev_count_col and prev_count_col in ps.columns and min_col in ps.columns):
            return pd.Series(index=ps.index, dtype="float32")
        minutes_shift = ps.groupby(pid_col)[min_col].shift(1)
        stat_shift = ps.groupby(pid_col)[prev_count_col].shift(1)
        r = stat_shift / minutes_shift.replace(0, np.nan)
        return r.fillna(r.median(skipna=True)).astype("float32")

    ps["rate_pts"] = rate(pts_col).fillna(0.5).astype("float32")
    ps["rate_reb"] = rate(reb_col).fillna(0.2).astype("float32")
    ps["rate_ast"] = rate(ast_col).fillna(0.2).astype("float32")
    ps["rate_3pm"] = rate(tpm_col).fillna(0.1).astype("float32")

    # Build side-aware context keyed by (gid, is_home), including opponent context
    ctx = games_context.copy()

    # Home-side row: team_* from home_*, opp_* from away_*
    home_side = ctx[[
        "gid", "date", "season_end_year", "season_decade", "home_tid", "away_tid",
        "home_recent_pace", "home_off_strength", "home_def_strength", "home_recent_winrate",
        "away_recent_pace", "away_off_strength", "away_def_strength", "away_recent_winrate",
        "match_off_edge", "match_def_edge", "match_pace_sum", "winrate_diff"
    ]].rename(columns={
        "home_tid": "tid", "away_tid": "opp_tid",
        "home_recent_pace": "team_recent_pace",
        "home_off_strength": "team_off_strength",
        "home_def_strength": "team_def_strength",
        "home_recent_winrate": "team_recent_winrate",
        "away_recent_pace": "opp_recent_pace",
        "away_off_strength": "opp_off_strength",
        "away_def_strength": "opp_def_strength",
        "away_recent_winrate": "opp_recent_winrate",
    })
    home_side["is_home"] = 1

    # Away-side row: team_* from away_*, opp_* from home_*
    away_side = ctx[[
        "gid", "date", "season_end_year", "season_decade", "home_tid", "away_tid",
        "away_recent_pace", "away_off_strength", "away_def_strength", "away_recent_winrate",
        "home_recent_pace", "home_off_strength", "home_def_strength", "home_recent_winrate",
        "match_off_edge", "match_def_edge", "match_pace_sum", "winrate_diff"
    ]].rename(columns={
        "away_tid": "tid", "home_tid": "opp_tid",
        "away_recent_pace": "team_recent_pace",
        "away_off_strength": "team_off_strength",
        "away_def_strength": "team_def_strength",
        "away_recent_winrate": "team_recent_winrate",
        "home_recent_pace": "opp_recent_pace",
        "home_off_strength": "opp_off_strength",
        "home_def_strength": "opp_def_strength",
        "home_recent_winrate": "opp_recent_winrate",
    })
    away_side["is_home"] = 0

    side_ctx_tid = pd.concat([home_side, away_side], ignore_index=True)
    side_ctx_flag = side_ctx_tid.drop(columns=["tid", "opp_tid"]).drop_duplicates(["gid", "is_home"])

    # Drop season columns from ps before merge (they come from game context to avoid conflicts)
    ps = ps.drop(columns=["season_end_year", "season_decade"], errors="ignore")

    # Join context to players:
    # 1) If teamId exists: join by (gid, tid)
    # 2) Else if home flag exists: join by (gid, is_home)
    # 3) Else: fallback to per-game average context (team_* and opp_*), plus matchup
    if tid_col and tid_col in ps.columns:
        ps_join = ps.merge(
            side_ctx_tid,
            left_on=[gid_col, tid_col],
            right_on=["gid", "tid"],
            how="left",
            validate="many_to_one"
        )
    elif "is_home" in ps.columns and ps["is_home"].notna().any():
        ps_join = ps.merge(
            side_ctx_flag,
            left_on=[gid_col, "is_home"],
            right_on=["gid", "is_home"],
            how="left",
            validate="many_to_one"
        )
    else:
        avg_cols = [
            "season_end_year", "season_decade",
            "team_recent_pace", "team_off_strength", "team_def_strength", "team_recent_winrate",
            "opp_recent_pace", "opp_off_strength", "opp_def_strength", "opp_recent_winrate",
            "match_off_edge", "match_def_edge", "match_pace_sum", "winrate_diff"
        ]
        avg_ctx = side_ctx_tid.groupby("gid", as_index=False)[avg_cols].mean()
        ps_join = ps.merge(avg_ctx, left_on=gid_col, right_on="gid", how="left")

    # Add OOF game predictions
    oof = oof_games.copy()
    ps_join = ps_join.merge(oof[["gid", "oof_ml_prob", "oof_spread_pred"]], on="gid", how="left")

    # Build frames
    frames: Dict[str, pd.DataFrame] = {}

    base_ctx_cols = [
        "is_home",
        "season_end_year", "season_decade",
        "team_recent_pace", "team_off_strength", "team_def_strength", "team_recent_winrate",
        "opp_recent_pace",  "opp_off_strength",  "opp_def_strength",  "opp_recent_winrate",
        "match_off_edge", "match_def_edge", "match_pace_sum", "winrate_diff",
        "oof_ml_prob", "oof_spread_pred", "starter_flag",
    ]

    # Minutes
    if min_col and min_col in ps_join.columns:
        minutes_df = ps_join[[gid_col, pid_col] + base_ctx_cols + [
            "min_prev_mean5", "min_prev_mean10", "min_prev_last1", min_col
        ]].copy()
        minutes_df = minutes_df.dropna(subset=[min_col]).reset_index(drop=True)
        frames["minutes"] = minutes_df
    else:
        frames["minutes"] = pd.DataFrame()

    # Helper builder
    def build_stat_frame(stat_col: Optional[str], rate_col: str) -> pd.DataFrame:
        if not stat_col or stat_col not in ps_join.columns:
            return pd.DataFrame()
        cols = [gid_col, pid_col] + base_ctx_cols + [rate_col]
        df = ps_join[cols].copy()
        if min_col and min_col in ps_join.columns:
            df["minutes"] = ps_join[min_col]
        df["label"] = pd.to_numeric(ps_join[stat_col], errors="coerce")
        df = df.dropna(subset=["label"]).reset_index(drop=True)
        return df

    frames["points"]   = build_stat_frame(pts_col, "rate_pts")
    frames["rebounds"] = build_stat_frame(reb_col, "rate_reb")
    frames["assists"]  = build_stat_frame(ast_col, "rate_ast")
    frames["threes"]   = build_stat_frame(tpm_col, "rate_3pm")

    # Counts
    print(_sec("Player training frames"))
    for k, v in frames.items():
        print(f"- {k}: {len(v):,} rows")

    return frames

# ---------------- Fit player models ----------------

def _fit_minutes_model(df: pd.DataFrame, seed: int, verbose: bool) -> Tuple[object, Dict[str, float]]:
    if df.empty:
        mdl = DummyRegressor(strategy="mean").fit([[0]], [24.0])
        return mdl, {"rows": 0, "rmse": float("nan"), "mae": float("nan")}
    features = [
        # team + opponent context
        "is_home",
        "season_end_year", "season_decade",
        "team_recent_pace", "team_off_strength", "team_def_strength", "team_recent_winrate",
        "opp_recent_pace",  "opp_off_strength",  "opp_def_strength",  "opp_recent_winrate",
        # matchup + game signals
        "match_off_edge", "match_def_edge", "match_pace_sum", "winrate_diff",
        "oof_ml_prob", "oof_spread_pred", "starter_flag",
        # minutes trends
        "min_prev_mean5", "min_prev_mean10", "min_prev_last1"
    ]
    features = [f for f in features if f in df.columns]
    X = df[features].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32")
    y = pd.to_numeric(df.iloc[:, -1], errors="coerce").astype(float).values  # minutes column is last
    w = pd.to_numeric(df.get("sample_weight", pd.Series(np.ones(len(df)))), errors="coerce").fillna(1.0).values

    n = len(X)
    split = max(1, min(int(n * 0.8), n - 1))
    X_tr, X_val = X.iloc[:split, :], X.iloc[split:, :]
    y_tr, y_val = y[:split], y[split:]
    w_tr, w_val = w[:split], w[split:]

    if _HAS_LGB:
        reg = lgb.LGBMRegressor(
            objective="regression",
            learning_rate=0.05, num_leaves=31, max_depth=-1,
            colsample_bytree=0.9, subsample=0.8, subsample_freq=5,
            n_estimators=800, random_state=seed, n_jobs=-1,
            force_col_wise=True, verbosity=-1
        )
    else:
        reg = HistGradientBoostingRegressor(
            learning_rate=0.06, max_depth=None, max_iter=400, random_state=seed,
            validation_fraction=0.2, early_stopping=True
        )
    reg.fit(X_tr, y_tr, sample_weight=w_tr)
    y_pred = reg.predict(X_val) if len(X_val) else np.array([])
    rmse = float(math.sqrt(mean_squared_error(y_val, y_pred))) if len(y_pred) else float("nan")
    mae = float(mean_absolute_error(y_val, y_pred)) if len(y_pred) else float("nan")

    print(_sec("Minutes model metrics (validation)"))
    print(f"- RMSE={_fmt(rmse)}, MAE={_fmt(mae)}")
    return reg, {"rows": int(n), "rmse": rmse, "mae": mae}

def _fit_stat_model(df: pd.DataFrame, seed: int, verbose: bool, name: str) -> Tuple[object, Dict[str, float]]:
    if df.empty:
        mdl = DummyRegressor(strategy="mean").fit([[0]], [0.0])
        return mdl, {"rows": 0, "rmse": float("nan"), "mae": float("nan")}
    features = [
        # team + opponent context
        "is_home",
        "season_end_year", "season_decade",
        "team_recent_pace", "team_off_strength", "team_def_strength", "team_recent_winrate",
        "opp_recent_pace",  "opp_off_strength",  "opp_def_strength",  "opp_recent_winrate",
        # matchup + game signals
        "match_off_edge", "match_def_edge", "match_pace_sum", "winrate_diff",
        "oof_ml_prob", "oof_spread_pred", "starter_flag",
        # player rates
        "rate_pts", "rate_reb", "rate_ast", "rate_3pm",
    ]
    if "minutes" in df.columns:
        features.append("minutes")
    features = [f for f in features if f in df.columns]

    X = df[features].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32")
    y = pd.to_numeric(df["label"], errors="coerce").astype(float).values
    w = pd.to_numeric(df.get("sample_weight", pd.Series(np.ones(len(df)))), errors="coerce").fillna(1.0).values

    n = len(X)
    split = max(1, min(int(n * 0.8), n - 1))
    X_tr, X_val = X.iloc[:split, :], X.iloc[split:, :]
    y_tr, y_val = y[:split], y[split:]
    w_tr, w_val = w[:split], w[split:]

    if _HAS_LGB:
        reg = lgb.LGBMRegressor(
            objective="regression",
            learning_rate=0.05, num_leaves=31, max_depth=-1,
            colsample_bytree=0.9, subsample=0.8, subsample_freq=5,
            n_estimators=800, random_state=seed, n_jobs=-1,
            force_col_wise=True, verbosity=-1
        )
    else:
        reg = HistGradientBoostingRegressor(
            learning_rate=0.06, max_depth=None, max_iter=400, random_state=seed,
            validation_fraction=0.2, early_stopping=True
        )
    reg.fit(X_tr, y_tr, sample_weight=w_tr)
    y_pred = reg.predict(X_val) if len(X_val) else np.array([])
    rmse = float(math.sqrt(mean_squared_error(y_val, y_pred))) if len(y_pred) else float("nan")
    mae = float(mean_absolute_error(y_val, y_pred)) if len(y_pred) else float("nan")

    print(_sec(f"{name.capitalize()} model metrics (validation)"))
    print(f"- RMSE={_fmt(rmse)}, MAE={_fmt(mae)}")
    return reg, {"rows": int(n), "rmse": rmse, "mae": mae}

# ---------------- Betting Odds and Priors Loaders ----------------

def load_team_abbrev_map(priors_root: Path, verbose: bool) -> Dict[Tuple[int, str], str]:
    """
    Load Team Abbrev.csv and return dict: (season, team_name) -> abbreviation
    Handles renames across seasons (NOH->NOP, CHH/CHA, SEA->OKC, etc.)
    """
    abbrev_path = priors_root / "Team Abbrev.csv"
    if not abbrev_path.exists():
        log(f"Warning: Team Abbrev.csv not found at {abbrev_path}, returning empty map", verbose)
        return {}

    df = pd.read_csv(abbrev_path, low_memory=False)
    # Filter NBA non-playoff
    if "lg" in df.columns:
        df = df[df["lg"] == "NBA"]
    if "playoffs" in df.columns:
        df = df[df["playoffs"] == False]

    abbrev_map: Dict[Tuple[int, str], str] = {}
    for _, row in df.iterrows():
        season = int(row["season"]) if "season" in row and pd.notna(row["season"]) else 0
        team = str(row["team"]).strip() if "team" in row else ""
        abbr = str(row["abbreviation"]).strip() if "abbreviation" in row else ""
        if season > 0 and team and abbr:
            abbrev_map[(season, _norm(team))] = abbr

    log(f"Loaded {len(abbrev_map)} team abbreviation mappings from {abbrev_path.name}", verbose)
    return abbrev_map

def resolve_team_abbrev(team_name: str, season: int, abbrev_map: Dict[Tuple[int, str], str]) -> Optional[str]:
    """Resolve team name to abbreviation for a given season"""
    key = (season, _norm(team_name))
    if key in abbrev_map:
        return abbrev_map[key]
    # Fallback: try finding any season with same team name
    for (s, tn), abbr in abbrev_map.items():
        if tn == _norm(team_name):
            return abbr
    return None

def load_betting_odds(odds_path: Path, abbrev_map: Dict[Tuple[int, str], str], verbose: bool) -> pd.DataFrame:
    """
    Load betting odds dataset and normalize to canonical schema.
    Returns DataFrame with columns: gid, game_date_utc, season_end_year, home_abbrev, away_abbrev,
    market_home_ml, market_away_ml, market_spread, spread_move, market_total, total_move,
    market_implied_home, market_implied_away
    """
    if not odds_path.exists():
        log(f"Betting odds file not found: {odds_path}", verbose)
        return pd.DataFrame()

    log(_sec("Loading betting odds dataset"), verbose)
    df = pd.read_csv(odds_path, low_memory=False)
    log(f"Loaded {len(df):,} odds rows from {odds_path.name}", verbose)

    # Detect and normalize columns (flexible mapping)
    def find_col(patterns: List[str]) -> Optional[str]:
        for pat in patterns:
            matches = [c for c in df.columns if pat.lower() in c.lower()]
            if matches:
                return matches[0]
        return None

    gid_col = find_col(["gameid", "game_id", "gid"])
    date_col = find_col(["date", "game_date", "gamedate"])
    home_col = find_col(["hometeam", "home_team", "home"])
    away_col = find_col(["awayteam", "away_team", "away"])
    book_col = find_col(["book", "bookmaker", "sportsbook"])

    # Moneyline
    home_ml_open_col = find_col(["home_ml_open", "homemlopen", "home_moneyline_open"])
    home_ml_close_col = find_col(["home_ml_close", "homemlclose", "home_moneyline_close", "home_ml"])
    away_ml_open_col = find_col(["away_ml_open", "awaymlopen", "away_moneyline_open"])
    away_ml_close_col = find_col(["away_ml_close", "awaymlclose", "away_moneyline_close", "away_ml"])

    # Spread
    spread_open_col = find_col(["spread_open", "spreadopen", "line_open", "handicap_open"])
    spread_close_col = find_col(["spread_close", "spreadclose", "line_close", "handicap_close", "spread", "line"])

    # Total
    total_open_col = find_col(["total_open", "totalopen", "ou_open", "over_under_open"])
    total_close_col = find_col(["total_close", "totalclose", "ou_close", "over_under_close", "total", "ou"])

    # Build canonical df
    canonical = pd.DataFrame()

    if gid_col:
        canonical["gid"] = _id_to_str(df[gid_col])

    if date_col:
        canonical["game_date_utc"] = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_convert(None)
        canonical["season_end_year"] = _season_from_date(canonical["game_date_utc"]).astype("float32")

    if home_col:
        canonical["home_team"] = df[home_col].astype(str).str.strip()
    if away_col:
        canonical["away_team"] = df[away_col].astype(str).str.strip()

    if book_col:
        canonical["book"] = df[book_col].astype(str).str.strip()

    # Odds columns (numeric)
    for src, dst in [
        (home_ml_open_col, "home_ml_open"), (home_ml_close_col, "home_ml_close"),
        (away_ml_open_col, "away_ml_open"), (away_ml_close_col, "away_ml_close"),
        (spread_open_col, "spread_open"), (spread_close_col, "spread_close"),
        (total_open_col, "total_open"), (total_close_col, "total_close"),
    ]:
        if src:
            canonical[dst] = pd.to_numeric(df[src], errors="coerce")

    # Resolve team abbreviations
    if "home_team" in canonical.columns and "season_end_year" in canonical.columns:
        canonical["home_abbrev"] = canonical.apply(
            lambda r: resolve_team_abbrev(r["home_team"], int(r["season_end_year"]), abbrev_map) if pd.notna(r["season_end_year"]) else None,
            axis=1
        )
        canonical["away_abbrev"] = canonical.apply(
            lambda r: resolve_team_abbrev(r["away_team"], int(r["season_end_year"]), abbrev_map) if pd.notna(r["season_end_year"]) else None,
            axis=1
        )

    # Compute consensus closing values per game (median across books)
    if "gid" not in canonical.columns and "game_date_utc" in canonical.columns and "home_abbrev" in canonical.columns:
        # Generate synthetic gid from date + teams
        canonical["gid"] = (
            canonical["game_date_utc"].dt.strftime("%Y%m%d") + "_" +
            canonical["home_abbrev"].fillna("UNK") + "_" +
            canonical["away_abbrev"].fillna("UNK")
        )

    # Aggregate to one row per game (consensus medians)
    agg_dict = {}
    for col in ["season_end_year", "home_ml_close", "away_ml_close", "spread_close", "total_close",
                "home_ml_open", "away_ml_open", "spread_open", "total_open"]:
        if col in canonical.columns:
            agg_dict[col] = "median"

    # Keep first non-null for identifiers
    for col in ["game_date_utc", "home_team", "away_team", "home_abbrev", "away_abbrev"]:
        if col in canonical.columns:
            agg_dict[col] = "first"

    if not agg_dict:
        log("Warning: No odds columns found to aggregate", verbose)
        return pd.DataFrame()

    game_odds = canonical.groupby("gid", as_index=False).agg(agg_dict)

    # Compute market priors and movement
    def american_to_prob(ml: float) -> float:
        """Convert American odds to implied probability"""
        if pd.isna(ml):
            return np.nan
        if ml < 0:
            return -ml / (-ml + 100)
        else:
            return 100 / (ml + 100)

    if "home_ml_close" in game_odds.columns:
        game_odds["market_home_ml"] = game_odds["home_ml_close"]
        game_odds["market_implied_home"] = game_odds["home_ml_close"].apply(american_to_prob)
    if "away_ml_close" in game_odds.columns:
        game_odds["market_away_ml"] = game_odds["away_ml_close"]
        game_odds["market_implied_away"] = game_odds["away_ml_close"].apply(american_to_prob)

    # No-vig normalization
    if "market_implied_home" in game_odds.columns and "market_implied_away" in game_odds.columns:
        total_vig = game_odds["market_implied_home"] + game_odds["market_implied_away"]
        game_odds["market_implied_home"] = game_odds["market_implied_home"] / total_vig
        game_odds["market_implied_away"] = game_odds["market_implied_away"] / total_vig

    if "spread_close" in game_odds.columns:
        game_odds["market_spread"] = game_odds["spread_close"]
        if "spread_open" in game_odds.columns:
            game_odds["spread_move"] = game_odds["spread_close"] - game_odds["spread_open"]

    if "total_close" in game_odds.columns:
        game_odds["market_total"] = game_odds["total_close"]
        if "total_open" in game_odds.columns:
            game_odds["total_move"] = game_odds["total_close"] - game_odds["total_open"]

    # Keep only needed columns
    keep_cols = ["gid", "game_date_utc", "season_end_year", "home_abbrev", "away_abbrev",
                 "market_home_ml", "market_away_ml", "market_spread", "spread_move",
                 "market_total", "total_move", "market_implied_home", "market_implied_away"]
    keep_cols = [c for c in keep_cols if c in game_odds.columns]
    game_odds = game_odds[keep_cols]

    log(f"Processed {len(game_odds):,} unique games with market odds", verbose)
    return game_odds

def load_basketball_reference_priors(priors_root: Path, verbose: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Basketball Reference priors bundle (player + team CSVs).
    Returns: (priors_players, priors_teams) with season_for_game = season + 1 (shifted for leakage safety)
    """
    log(_sec("Loading Basketball Reference priors"), verbose)

    # Team priors
    team_summaries_path = priors_root / "Team Summaries.csv"
    team_per100_path = priors_root / "Team Stats Per 100 Poss.csv"

    priors_teams = pd.DataFrame()

    if team_summaries_path.exists():
        ts = pd.read_csv(team_summaries_path, low_memory=False)
        # Filter NBA non-playoff
        if "lg" in ts.columns:
            ts = ts[ts["lg"] == "NBA"]
        if "playoffs" in ts.columns:
            ts = ts[ts["playoffs"] == False]

        # Keep key columns
        team_cols = ["season", "abbreviation", "w", "l", "mov", "sos", "srs",
                     "o_rtg", "d_rtg", "n_rtg", "pace", "f_tr", "x3p_ar", "ts_percent",
                     "e_fg_percent", "tov_percent", "orb_percent", "ft_fga",
                     "opp_e_fg_percent", "opp_tov_percent", "drb_percent", "opp_ft_fga"]
        team_cols = ["season", "abbreviation"] + [c for c in team_cols if c in ts.columns]
        priors_teams = ts[team_cols].copy()

        # Normalize percents (0-100 -> 0-1)
        for col in priors_teams.columns:
            if "percent" in col.lower() or col in ["f_tr", "x3p_ar", "ft_fga", "opp_ft_fga"]:
                vals = pd.to_numeric(priors_teams[col], errors="coerce")
                # If max > 1.5, assume 0-100 scale
                if vals.max() > 1.5:
                    priors_teams[col] = vals / 100.0

        # Shift: season S priors are used in season S+1
        if "season" in priors_teams.columns:
            season_vals = pd.to_numeric(priors_teams["season"], errors="coerce")
            priors_teams = priors_teams.drop(columns=["season"])
            priors_teams.insert(0, "season_for_game", season_vals + 1)
        else:
            log("Warning: No 'season' column found in Team Summaries", verbose)

        log(f"Loaded {len(priors_teams):,} team-season priors from Team Summaries", verbose)

    # Player priors (simplified - just Per 100 Poss for now)
    per100_path = priors_root / "Per 100 Poss.csv"
    priors_players = pd.DataFrame()

    if per100_path.exists():
        pp = pd.read_csv(per100_path, low_memory=False)
        # Filter NBA
        if "lg" in pp.columns:
            pp = pp[pp["lg"] == "NBA"]

        # Prefer TOT rows for multi-team seasons; else keep all
        if "team" in pp.columns:
            tot_rows = pp[pp["team"] == "TOT"]
            non_tot = pp[pp["team"] != "TOT"]
            # For players with TOT, keep only TOT; else keep their single-team row
            has_tot = set(tot_rows["player_id"]) if "player_id" in tot_rows.columns else set()
            non_tot = non_tot[~non_tot["player_id"].isin(has_tot)] if "player_id" in non_tot.columns else non_tot
            pp = pd.concat([tot_rows, non_tot], ignore_index=True)

        # Keep key rate columns
        player_cols = ["season", "player_id", "player", "age", "pos", "g", "mp",
                       "pts_per_100_poss", "trb_per_100_poss", "ast_per_100_poss",
                       "stl_per_100_poss", "blk_per_100_poss", "tov_per_100_poss",
                       "fg_per_100_poss", "fga_per_100_poss", "fg_percent",
                       "x3p_per_100_poss", "x3pa_per_100_poss", "x3p_percent",
                       "ft_per_100_poss", "fta_per_100_poss", "ft_percent",
                       "o_rtg", "d_rtg"]
        player_cols = [c for c in player_cols if c in pp.columns]
        priors_players = pp[player_cols].copy()

        # Normalize percents
        for col in priors_players.columns:
            if "percent" in col.lower():
                vals = pd.to_numeric(priors_players[col], errors="coerce")
                if vals.max() > 1.5:
                    priors_players[col] = vals / 100.0

        # Shift to next season
        if "season" in priors_players.columns:
            season_vals = pd.to_numeric(priors_players["season"], errors="coerce")
            priors_players = priors_players.drop(columns=["season"])
            priors_players.insert(0, "season_for_game", season_vals + 1)
        else:
            log("Warning: No 'season' column found in Per 100 Poss", verbose)

        log(f"Loaded {len(priors_players):,} player-season priors from Per 100 Poss", verbose)

    return priors_players, priors_teams

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(description="Train NBA game and player models (always fetch from Kaggle).")
    ap.add_argument("--dataset", type=str, default="eoinamoore/historical-nba-data-and-player-box-scores", help="Kaggle dataset ref")
    ap.add_argument("--models-dir", type=str, default="models", help="Output dir for models")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging")
    ap.add_argument("--skip-rest", action="store_true", help="Skip rest/b2b features (useful when dates missing)")
    ap.add_argument("--fresh", action="store_true", help="Copy CSVs into a new per-run folder to avoid any stale artifacts")
    ap.add_argument("--lgb-log-period", type=int, default=0, help="Print LightGBM eval metrics every N iterations (0 = silent)")

    # Era controls
    ap.add_argument("--game-season-cutoff", type=str, default="2002",
                    help="Minimum season_end_year to include for game models. Presets: classic=1997, balanced=2002. Also accepts an int like 1997/2001/2004.")
    ap.add_argument("--player-season-cutoff", type=str, default="2002",
                    help="Minimum season_end_year to include for player models. Presets: min=1998, balanced=2002, modern=2005. Also accepts an int.")
    ap.add_argument("--decay", type=float, default=0.97, help="Time-decay factor for sample weights (0.90-0.99 typical).")
    ap.add_argument("--min-weight", type=float, default=0.30, help="Minimum sample weight after decay.")
    ap.add_argument("--lockout-weight", type=float, default=0.90, help="Extra multiplier applied to lockout seasons (1999, 2012).")

    # Additional datasets (now enabled by default with caching)
    ap.add_argument("--odds-dataset", type=str, default="cviaxmiwnptr/nba-betting-data-october-2007-to-june-2024",
                    help="Path or Kaggle dataset ref for betting odds (default: cviaxmiwnptr/nba-betting-data-october-2007-to-june-2024)")
    ap.add_argument("--priors-dataset", type=str, default="sumitrodatta/nba-aba-baa-stats",
                    help="Path or Kaggle dataset ref for Basketball Reference priors bundle (default: sumitrodatta/nba-aba-baa-stats)")

    args = ap.parse_args()

    verbose = args.verbose
    seed = args.seed

    print(_sec("Training configuration"))
    print(f"- dataset: {args.dataset}")
    print(f"- models_dir: {args.models_dir}")
    print(f"- seed: {seed}")
    print(f"- skip_rest: {args.skip_rest}")
    print(f"- fresh_run_copy: {args.fresh}")
    print(f"- lgb_log_period: {args.lgb_log_period}")
    print(f"- game_season_cutoff: {args.game_season_cutoff}  player_season_cutoff: {args.player_season_cutoff}")
    print(f"- decay: {args.decay}  min_weight: {args.min_weight}  lockout_weight: {args.lockout_weight}")

    if kagglehub is None:
        raise RuntimeError("kagglehub is required. Install with: pip install kagglehub")

    # Download Kaggle dataset (always)
    print(_sec("Fetching latest from Kaggle"))
    ds_root = Path(kagglehub.dataset_download(args.dataset))
    print(f"- Downloaded to cache: {ds_root}")

    # Optionally copy to a per-run folder (ensures clean, fresh artifacts)
    ds_use_root = ds_root
    if args.fresh:
        run_dir = _fresh_run_dir(Path(".kaggle_runs"))
        teams_src, players_src = _find_dataset_files(ds_root)
        teams_dst = _copy_if_exists(teams_src, run_dir)
        players_dst = _copy_if_exists(players_src, run_dir)
        ds_use_root = run_dir
        print(f"- Copied CSVs to: {run_dir}")
        if not teams_dst:
            raise FileNotFoundError("TeamStatistics CSV not found in Kaggle dataset.")
        teams_path, players_path = teams_dst, players_dst
    else:
        teams_path, players_path = _find_dataset_files(ds_root)
        if not teams_path:
            raise FileNotFoundError("TeamStatistics CSV not found in Kaggle dataset.")

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Build games and team context
    print(_sec("Building game dataset"))
    games_df, context_map = build_games_from_teamstats(teams_path, verbose=verbose, skip_rest=args.skip_rest)

    # Era filter for games
    game_cut = _parse_season_cutoff(args.game_season_cutoff, kind="game")
    if "season_end_year" in games_df.columns:
        before_len = len(games_df)
        games_df = games_df[(games_df["season_end_year"].fillna(game_cut)) >= game_cut].reset_index(drop=True)
        print(f"- Games filtered by season >= {game_cut}: {before_len:,} -> {len(games_df):,} rows")

    # Compute sample weights for games
    if "season_end_year" in games_df.columns:
        game_weights = _compute_sample_weights(
            games_df["season_end_year"].to_numpy(dtype="float64"),
            decay=args.decay, min_weight=args.min_weight, lockout_weight=args.lockout_weight
        )
    else:
        game_weights = np.ones(len(games_df), dtype="float64")

    print(f"- Games: {len(games_df):,} rows")

    # Load and merge betting odds (if provided)
    if args.odds_dataset:
        odds_path_str = args.odds_dataset
        # Check if it's a Kaggle dataset ref or local path
        if "/" in odds_path_str and not os.path.exists(odds_path_str):
            # Assume Kaggle dataset
            if not kagglehub:
                log("Warning: kagglehub not available, skipping odds dataset", verbose)
                odds_df = pd.DataFrame()
            else:
                try:
                    log(f"Downloading odds dataset from Kaggle: {odds_path_str}", verbose)
                    odds_root = Path(kagglehub.dataset_download(odds_path_str))
                    # Find CSV file in downloaded dataset
                    odds_csvs = list(odds_root.glob("*.csv"))
                    if odds_csvs:
                        odds_path = odds_csvs[0]  # Use first CSV found
                        # Load team abbrev map if priors dataset is available (for team name resolution)
                        abbrev_map = {}
                        if args.priors_dataset:
                            priors_root_str = args.priors_dataset
                            if "/" in priors_root_str and not os.path.exists(priors_root_str):
                                priors_root = Path(kagglehub.dataset_download(priors_root_str))
                            else:
                                priors_root = Path(priors_root_str)
                            abbrev_map = load_team_abbrev_map(priors_root, verbose)
                        odds_df = load_betting_odds(odds_path, abbrev_map, verbose)
                    else:
                        log(f"Warning: No CSV files found in {odds_root}", verbose)
                        odds_df = pd.DataFrame()
                except Exception as e:
                    log(f"Warning: Failed to load odds dataset: {e}", verbose)
                    odds_df = pd.DataFrame()
        else:
            odds_path = Path(odds_path_str)
            abbrev_map = {}
            if args.priors_dataset:
                priors_root = Path(args.priors_dataset)
                abbrev_map = load_team_abbrev_map(priors_root, verbose)
            odds_df = load_betting_odds(odds_path, abbrev_map, verbose)

        # Merge odds into games_df
        if not odds_df.empty:
            # Create abbreviations for games_df home/away teams (for matching)
            if "home_tid" in games_df.columns and "away_tid" in games_df.columns:
                # Try to match by gid first if present in both
                if "gid" in games_df.columns and "gid" in odds_df.columns:
                    games_df = games_df.merge(
                        odds_df.drop(columns=["game_date_utc", "season_end_year"], errors="ignore"),
                        on="gid", how="left", suffixes=("", "_odds")
                    )
                    log(f"- Merged {odds_df['gid'].nunique()} odds games by gid", verbose)
                # Fallback: match by date + team abbrev
                elif "date" in games_df.columns and "home_abbrev" in odds_df.columns:
                    # This would need team ID -> abbrev mapping from priors
                    # For now, just log that we couldn't match
                    log("Warning: Could not match odds to games (no gid overlap, need team abbrev mapping)", verbose)

            # Fill NaN odds with defaults
            for col in ["market_implied_home", "market_implied_away", "market_spread", "spread_move", "market_total", "total_move"]:
                if col in games_df.columns:
                    games_df[col] = games_df[col].fillna(GAME_DEFAULTS.get(col, 0.0))

    # Load and merge Basketball Reference priors (if provided)
    if args.priors_dataset:
        priors_path_str = args.priors_dataset
        # Check if Kaggle dataset or local path
        if "/" in priors_path_str and not os.path.exists(priors_path_str):
            if not kagglehub:
                log("Warning: kagglehub not available, skipping priors dataset", verbose)
                priors_players, priors_teams = pd.DataFrame(), pd.DataFrame()
            else:
                try:
                    log(f"Downloading priors dataset from Kaggle: {priors_path_str}", verbose)
                    priors_root = Path(kagglehub.dataset_download(priors_path_str))
                    priors_players, priors_teams = load_basketball_reference_priors(priors_root, verbose)
                except Exception as e:
                    log(f"Warning: Failed to load priors dataset: {e}", verbose)
                    priors_players, priors_teams = pd.DataFrame(), pd.DataFrame()
        else:
            priors_root = Path(priors_path_str)
            priors_players, priors_teams = load_basketball_reference_priors(priors_root, verbose)

        # Merge team priors into games_df
        if not priors_teams.empty and "abbreviation" in priors_teams.columns and "season_end_year" in games_df.columns:
            # TODO: Need to map home_tid/away_tid to abbreviations
            # For now, if games_df already has home_abbrev/away_abbrev from odds, use those
            # Otherwise, we'd need a tid -> abbrev map from Team Abbrev.csv
            if "home_abbrev" in games_df.columns and "away_abbrev" in games_df.columns:
                # Merge home team priors
                home_priors = priors_teams.rename(columns={
                    "abbreviation": "home_abbrev",
                    "season_for_game": "season_end_year",
                    "o_rtg": "home_o_rtg_prior",
                    "d_rtg": "home_d_rtg_prior",
                    "pace": "home_pace_prior",
                    "srs": "home_srs_prior"
                })
                games_df = games_df.merge(
                    home_priors[["home_abbrev", "season_end_year", "home_o_rtg_prior", "home_d_rtg_prior", "home_pace_prior", "home_srs_prior"]],
                    on=["home_abbrev", "season_end_year"], how="left"
                )

                # Merge away team priors
                away_priors = priors_teams.rename(columns={
                    "abbreviation": "away_abbrev",
                    "season_for_game": "season_end_year",
                    "o_rtg": "away_o_rtg_prior",
                    "d_rtg": "away_d_rtg_prior",
                    "pace": "away_pace_prior",
                    "srs": "away_srs_prior"
                })
                games_df = games_df.merge(
                    away_priors[["away_abbrev", "season_end_year", "away_o_rtg_prior", "away_d_rtg_prior", "away_pace_prior", "away_srs_prior"]],
                    on=["away_abbrev", "season_end_year"], how="left"
                )
                log(f"- Merged team priors for {len(priors_teams):,} team-seasons", verbose)

            # Fill NaN priors with defaults
            for col in ["home_o_rtg_prior", "home_d_rtg_prior", "home_pace_prior", "home_srs_prior",
                       "away_o_rtg_prior", "away_d_rtg_prior", "away_pace_prior", "away_srs_prior"]:
                if col in games_df.columns:
                    games_df[col] = games_df[col].fillna(GAME_DEFAULTS.get(col, 0.0))

    # Train game models + OOF
    print(_sec("Training game models"))
    clf_final, calibrator, reg_final, spread_sigma, oof_games, game_metrics = _fit_game_models(
        games_df, seed=seed, verbose=verbose, folds=5, lgb_log_period=args.lgb_log_period, sample_weights=game_weights
    )

    # Save game models
    import pickle
    with open(models_dir / "moneyline_model.pkl", "wb") as f:
        pickle.dump(clf_final, f)
    if calibrator is not None:
        with open(models_dir / "moneyline_calibrator.pkl", "wb") as f:
            pickle.dump(calibrator, f)
    with open(models_dir / "spread_model.pkl", "wb") as f:
        pickle.dump(reg_final, f)
    with open(models_dir / "spread_sigma.json", "w", encoding="utf-8") as f:
        json.dump({"spread_sigma": spread_sigma}, f)

    # Player models
    player_metrics: Dict[str, Dict[str, float]] = {}
    if players_path and players_path.exists():
        print(_sec("Building player datasets"))
        frames = build_players_from_playerstats(players_path, context_map, oof_games, verbose=verbose)

        # Era filter + weights per frame
        player_cut = _parse_season_cutoff(args.player_season_cutoff, kind="player")
        for k, df in list(frames.items()):
            if df is None or df.empty:
                continue
            if "season_end_year" in df.columns:
                before = len(df)
                df = df[(df["season_end_year"].fillna(player_cut)) >= player_cut].reset_index(drop=True)
                frames[k] = df
                print(f"- {k}: filtered by season >= {player_cut}: {before:,} -> {len(df):,}")
                # weights
                df["sample_weight"] = _compute_sample_weights(
                    df["season_end_year"].to_numpy(dtype="float64"),
                    decay=args.decay, min_weight=args.min_weight, lockout_weight=args.lockout_weight
                )
                frames[k] = df
            else:
                df["sample_weight"] = 1.0
                frames[k] = df

        print(_sec("Training player models"))
        # Minutes
        minutes_model, m_metrics = _fit_minutes_model(frames.get("minutes", pd.DataFrame()), seed=seed + 10, verbose=verbose)
        with open(models_dir / "minutes_model.pkl", "wb") as f:
            pickle.dump(minutes_model, f)
        player_metrics["minutes"] = m_metrics

        # If minutes missing in stat frames, inject a simple proxy for training convenience
        for key in ["points", "rebounds", "assists", "threes"]:
            df = frames.get(key, pd.DataFrame())
            if not df.empty and "minutes" not in df.columns and not frames["minutes"].empty:
                df["minutes"] = frames["minutes"]["min_prev_mean10"].median() if "min_prev_mean10" in frames["minutes"].columns else 24.0
                frames[key] = df

        # Points
        points_model, p_metrics = _fit_stat_model(frames.get("points", pd.DataFrame()), seed=seed + 20, verbose=verbose, name="points")
        with open(models_dir / "points_model.pkl", "wb") as f:
            pickle.dump(points_model, f)
        player_metrics["points"] = p_metrics

        # Rebounds
        rebounds_model, r_metrics = _fit_stat_model(frames.get("rebounds", pd.DataFrame()), seed=seed + 21, verbose=verbose, name="rebounds")
        with open(models_dir / "rebounds_model.pkl", "wb") as f:
            pickle.dump(rebounds_model, f)
        player_metrics["rebounds"] = r_metrics

        # Assists
        assists_model, a_metrics = _fit_stat_model(frames.get("assists", pd.DataFrame()), seed=seed + 22, verbose=verbose, name="assists")
        with open(models_dir / "assists_model.pkl", "wb") as f:
            pickle.dump(assists_model, f)
        player_metrics["assists"] = a_metrics

        # Threes
        threes_model, t_metrics = _fit_stat_model(frames.get("threes", pd.DataFrame()), seed=seed + 23, verbose=verbose, name="threes")
        with open(models_dir / "threes_model.pkl", "wb") as f:
            pickle.dump(threes_model, f)
        player_metrics["threes"] = t_metrics
    else:
        print(_sec("Player models"))
        print("- Skipped (PlayerStatistics.csv not found in Kaggle dataset).")

    # Save metadata
    meta = {
        "game_features": GAME_FEATURES,
        "game_defaults": GAME_DEFAULTS,
        "game_metrics": game_metrics,
        "player_metrics": player_metrics,
        "era": {
            "game_season_cutoff": _parse_season_cutoff(args.game_season_cutoff, "game"),
            "player_season_cutoff": _parse_season_cutoff(args.player_season_cutoff, "player"),
            "decay": args.decay,
            "min_weight": args.min_weight,
            "lockout_weight": args.lockout_weight,
        },
        "versions": {
            "lightgbm": getattr(lgb, "__version__", None) if _HAS_LGB else None,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        }
    }
    with open(models_dir / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Final report
    print(_sec("Saved artifacts"))
    print(f"- { (models_dir / 'moneyline_model.pkl').as_posix() }")
    if os.path.exists(models_dir / "moneyline_calibrator.pkl"):
        print(f"- { (models_dir / 'moneyline_calibrator.pkl').as_posix() }")
        print(f"- { (models_dir / 'spread_model.pkl').as_posix() }")
    else:
        print(f"- { (models_dir / 'spread_model.pkl').as_posix() }")
    print(f"- { (models_dir / 'spread_sigma.json').as_posix() }")
    if player_metrics:
        print(f"- { (models_dir / 'minutes_model.pkl').as_posix() }")
        print(f"- { (models_dir / 'points_model.pkl').as_posix() }")
        print(f"- { (models_dir / 'rebounds_model.pkl').as_posix() }")
        print(f"- { (models_dir / 'assists_model.pkl').as_posix() }")
        print(f"- { (models_dir / 'threes_model.pkl').as_posix() }")

    print(_sec("Summary"))
    print(f"- Games: {len(games_df):,}")
    print(f"- Moneyline: logloss={_fmt(game_metrics['ml_logloss'])}, Brier={_fmt(game_metrics['ml_brier'])}")
    print(f"- Spread:    RMSE={_fmt(game_metrics['sp_rmse'])}, MAE={_fmt(game_metrics['sp_mae'])}, sigma={_fmt(game_metrics['spread_sigma'])}")
    if player_metrics:
        for k, mm in player_metrics.items():
            print(f"- {k.capitalize()}: rows={mm.get('rows')}, RMSE={_fmt(mm.get('rmse'))}, MAE={_fmt(mm.get('mae'))}")

if __name__ == "__main__":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    main()
