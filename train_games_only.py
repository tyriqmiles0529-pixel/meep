#!/usr/bin/env python3
"""
Train ONLY the NBA game models (moneyline win probability + spread margin).

This version prefers TeamStatistics.csv (if available) to build game rows:
- Reliable team/opponent IDs and home flag
- Avoids dependence on Games.csv date quirks

Key features (leakage-safe, team-context included)
- Rolling L10 offensive strength (normalized ~1.0)
- Rolling L10 defensive strength (normalized ~1.0; lower pts allowed => higher strength)
- Rolling L10 pace proxy (normalized ~1.0)
- Recent pre-game win rate (shifted by 1 game)
- Matchup features:
  - match_off_edge = home_off_strength - away_def_strength
  - match_def_edge = home_def_strength - away_off_strength
  - match_pace_sum = home_recent_pace + away_recent_pace
  - winrate_diff   = home_recent_winrate - away_recent_winrate

Other
- Optional rest/b2b (skip with --skip-rest; needs reliable dates)
- LightGBM if available, else sklearn HistGradientBoosting
"""

from __future__ import annotations

import sys
import math
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, log_loss, brier_score_loss
from sklearn.dummy import DummyClassifier, DummyRegressor

_HAS_LGB = False
try:
    import lightgbm as lgb  # type: ignore
    _HAS_LGB = True
except Exception:
    pass

try:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
except Exception:  # pragma: no cover
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

try:
    import kagglehub
except Exception:
    kagglehub = None

# Model feature schema
GAME_DEFAULT_FEATURES: List[str] = [
    "home_advantage", "neutral_site",
    "home_recent_pace", "away_recent_pace",
    "home_off_strength", "home_def_strength",
    "away_off_strength", "away_def_strength",
    "home_recent_winrate", "away_recent_winrate",
    # matchup features
    "match_off_edge", "match_def_edge", "match_pace_sum", "winrate_diff",
    # optional (often constant when --skip-rest)
    "home_days_rest", "away_days_rest",
    "home_b2b", "away_b2b",
    "home_injury_impact", "away_injury_impact",
]

GAME_NEUTRAL_DEFAULTS: Dict[str, float] = {
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
    # matchup neutral priors
    "match_off_edge": 0.0,
    "match_def_edge": 0.0,
    "match_pace_sum": 2.0,   # 1.0 + 1.0
    "winrate_diff": 0.0,
    # schedule/injury defaults
    "home_days_rest": 2.0,
    "away_days_rest": 2.0,
    "home_b2b": 0.0,
    "away_b2b": 0.0,
    "home_injury_impact": 0.0,
    "away_injury_impact": 0.0,
}


def log(msg: str, verbose: bool):
    if verbose:
        print(msg, flush=True)


def _id_to_str(s: pd.Series) -> pd.Series:
    s_num = pd.to_numeric(s, errors="coerce")
    out = pd.Series(index=s.index, dtype="object")
    num_mask = s_num.notna()
    out.loc[num_mask] = s_num.loc[num_mask].astype("Int64").astype(str)
    out.loc[~num_mask] = s.astype(str)
    return out.astype(str).str.strip()


def load_games_csv(path: Path, verbose: bool) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Games CSV not found: {path}")
    df = pd.read_csv(path, low_memory=False)

    for req in ["home_score", "away_score"]:
        if req not in df.columns:
            raise ValueError("CSV must include columns: home_score, away_score (and optionally 'date' + features).")

    for col, val in GAME_NEUTRAL_DEFAULTS.items():
        if col not in df.columns:
            df[col] = val
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(val).astype("float32")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_convert(None)
    else:
        df["date"] = pd.NaT

    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    log(f"Loaded games: {len(df):,} rows (dropped {before - len(df)} with NaN scores)", verbose)
    return df


def _find_dataset_files(ds_root: Path) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    games_path: Optional[Path] = None
    teams_path: Optional[Path] = None
    players_path: Optional[Path] = None
    for p in ds_root.glob("*.csv"):
        n = p.name.lower()
        stem = n.split(".")[0]
        if stem == "games" or n == "games.csv":
            games_path = p
        elif "teamstatistics" in stem or "team_statistics" in stem:
            teams_path = p
        elif "playerstatistics" in stem or "player_stats" in stem or "playerstats" in stem:
            players_path = p
    if games_path is None:
        cand = [p for p in ds_root.glob("*.csv") if p.name.lower().startswith("games")]
        if cand:
            games_path = cand[0]
    return games_path, teams_path, players_path


def _build_games_from_teamstats(teams_path: Path, verbose: bool, skip_rest: bool) -> pd.DataFrame:
    header_cols = list(pd.read_csv(teams_path, nrows=0).columns)
    usecols = [c for c in [
        "gameId", "gameDate", "teamId", "opponentTeamId",
        "home", "teamScore", "opponentScore",
    ] if c in header_cols]
    ts = pd.read_csv(teams_path, low_memory=False, usecols=usecols)

    for c in ["gameId", "teamId", "opponentTeamId"]:
        if c in ts.columns:
            ts[c] = _id_to_str(ts[c])

    if "home" in ts.columns:
        home_vals = ts["home"]
        num = pd.to_numeric(home_vals, errors="coerce")
        ts["home_flag"] = np.where(
            num.notna(), (num.fillna(0) != 0).astype(int),
            home_vals.astype(str).str.lower().isin(["1", "true", "t", "home", "h", "yes", "y"]).astype(int),
        )
    else:
        ts["home_flag"] = 1

    # Dates may be NaT for many rows; do not drop on date
    ts["gameDate"] = pd.to_datetime(ts["gameDate"], errors="coerce", utc=True).dt.tz_convert(None)

    ts["teamScore"] = pd.to_numeric(ts["teamScore"], errors="coerce")
    ts["opponentScore"] = pd.to_numeric(ts["opponentScore"], errors="coerce")
    ts = ts.dropna(subset=["gameId", "teamId", "opponentTeamId", "teamScore", "opponentScore"]).copy()

    ts_sorted = ts.sort_values(["gameId", "home_flag"], ascending=[True, False])
    home_rows = ts_sorted[ts_sorted["home_flag"] == 1].drop_duplicates("gameId", keep="first")
    away_rows = ts_sorted[ts_sorted["home_flag"] == 0].drop_duplicates("gameId", keep="first")

    g = home_rows.merge(
        away_rows[["gameId", "teamId", "teamScore", "gameDate"]].rename(columns={
            "teamId": "away_tid", "teamScore": "away_score", "gameDate": "date_check"
        }),
        on="gameId", how="left"
    )

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

    # Long view to compute leakage-safe rolling context
    long_home = g[["gid", "date", "home_tid", "home_score", "away_score"]].rename(
        columns={"home_tid": "tid", "home_score": "team_pts", "away_score": "opp_pts"}
    )
    long_away = g[["gid", "date", "away_tid", "away_score", "home_score"]].rename(
        columns={"away_tid": "tid", "away_score": "team_pts", "home_score": "opp_pts"}
    )
    teams_long = pd.concat(
        [long_home[["gid", "date", "tid", "team_pts", "opp_pts"]],
         long_away[["gid", "date", "tid", "team_pts", "opp_pts"]]],
        ignore_index=True
    )

    # Order by date (NaT last) then numeric gid as fallback
    teams_long["gid_num"] = pd.to_numeric(teams_long["gid"], errors="coerce")
    teams_long = teams_long.sort_values(["tid", "date", "gid_num"], ascending=[True, True, True], na_position="last")

    # Rolling with min_periods=1 and shift(1) to avoid leakage
    teams_long["off_pts_10"] = teams_long.groupby("tid")["team_pts"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    teams_long["def_pts_10"] = teams_long.groupby("tid")["opp_pts"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    teams_long["tot_pts"] = teams_long["team_pts"] + teams_long["opp_pts"]
    teams_long["pace_10"] = teams_long.groupby("tid")["tot_pts"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())

    teams_long["win"] = (teams_long["team_pts"] > teams_long["opp_pts"]).astype(int)
    teams_long["wins_prev"] = teams_long.groupby("tid")["win"].transform(lambda x: x.shift(1).cumsum())
    teams_long["games_prev"] = teams_long.groupby("tid").cumcount()
    teams_long["winrate_prev"] = np.where(
        teams_long["games_prev"] > 0,
        teams_long["wins_prev"] / teams_long["games_prev"].clip(lower=1),
        0.5
    )

    # Normalize around ~1.0 (or 0.5 for winrate)
    off_med = np.nanmedian(teams_long["off_pts_10"].values); off_med = off_med if np.isfinite(off_med) and off_med > 0 else 1.0
    def_med = np.nanmedian(teams_long["def_pts_10"].values); def_med = def_med if np.isfinite(def_med) and def_med > 0 else 1.0
    pace_med = np.nanmedian(teams_long["pace_10"].values); pace_med = pace_med if np.isfinite(pace_med) and pace_med > 0 else 1.0

    teams_long["off_norm"] = teams_long["off_pts_10"] / off_med
    teams_long["def_norm"] = def_med / teams_long["def_pts_10"]
    teams_long["pace_norm"] = teams_long["pace_10"] / pace_med

    hstats = teams_long.rename(columns={
        "tid": "home_tid",
        "off_norm": "home_off_strength",
        "def_norm": "home_def_strength",
        "pace_norm": "home_recent_pace",
        "winrate_prev": "home_recent_winrate"
    })[["gid", "home_tid", "home_off_strength", "home_def_strength", "home_recent_pace", "home_recent_winrate"]].drop_duplicates(["gid", "home_tid"])

    astats = teams_long.rename(columns={
        "tid": "away_tid",
        "off_norm": "away_off_strength",
        "def_norm": "away_def_strength",
        "pace_norm": "away_recent_pace",
        "winrate_prev": "away_recent_winrate"
    })[["gid", "away_tid", "away_off_strength", "away_def_strength", "away_recent_pace", "away_recent_winrate"]].drop_duplicates(["gid", "away_tid"])

    for c in ["gid", "home_tid"]:
        hstats[c] = _id_to_str(hstats[c])
    for c in ["gid", "away_tid"]:
        astats[c] = _id_to_str(astats[c])

    g = g.merge(hstats, on=["gid", "home_tid"], how="left").merge(astats, on=["gid", "away_tid"], how="left")

    # Matchup features
    g["match_off_edge"] = g["home_off_strength"] - g["away_def_strength"]
    g["match_def_edge"] = g["home_def_strength"] - g["away_off_strength"]
    g["match_pace_sum"] = g["home_recent_pace"] + g["away_recent_pace"]
    g["winrate_diff"] = g["home_recent_winrate"] - g["away_recent_winrate"]

    # Optional rest/b2b (dates mostly NaT in some files, so usually skip)
    if not skip_rest:
        sched_home = g[["home_tid", "date"]].rename(columns={"home_tid": "tid"})
        sched_away = g[["away_tid", "date"]].rename(columns={"away_tid": "tid"})
        sched = pd.concat([sched_home, sched_away], ignore_index=True)
        sched["tid"] = _id_to_str(sched["tid"])
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
            df_["date"] = pd.to_datetime(df_["date"], errors="coerce", utc=True).dt.tz_convert(None)
            df_.sort_values([tid_col, "date"], inplace=True)
            df_.drop_duplicates([tid_col, "date"], keep="last", inplace=True)

        g.sort_values(["gid", "date"], inplace=True)
        g.drop_duplicates(subset=["gid", "home_tid", "away_tid", "date"], keep="last", inplace=True)
        g = g.merge(hr, on=["home_tid", "date"], how="left", validate="many_to_one")
        g = g.merge(ar, on=["away_tid", "date"], how="left", validate="many_to_one")

    # Fill defaults and finalize
    for col, val in GAME_NEUTRAL_DEFAULTS.items():
        if col not in g.columns:
            g[col] = val
        g[col] = pd.to_numeric(g[col], errors="coerce").fillna(val).astype("float32")

    g["home_advantage"] = np.float32(1.0)
    g["neutral_site"] = np.float32(0.0)

    out = g[["date", "home_score", "away_score"] + GAME_DEFAULT_FEATURES].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=True).dt.tz_convert(None)
    out = out.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)
    log(f"Built TeamStatistics games frame: {len(out):,} rows", verbose)
    return out


def build_games_training_from_kaggle(dataset_ref: str, verbose: bool, skip_rest: bool) -> pd.DataFrame:
    if kagglehub is None:
        raise RuntimeError("kagglehub is not installed. pip install kagglehub")
    log(f"📦 Downloading Kaggle dataset: {dataset_ref}", verbose)
    ds_root = Path(kagglehub.dataset_download(dataset_ref))
    _, teams_path, _ = _find_dataset_files(ds_root)
    if teams_path and teams_path.exists():
        log("Using TeamStatistics.csv as primary source", verbose)
        return _build_games_from_teamstats(teams_path, verbose=verbose, skip_rest=skip_rest)
    raise FileNotFoundError("TeamStatistics.csv not found in dataset.")


def train_game_models(df: pd.DataFrame, seed: int, verbose: bool):
    # Keep as DataFrame to preserve feature names (silences sklearn LightGBM warnings)
    X_full: pd.DataFrame = df[GAME_DEFAULT_FEATURES].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
    y_ml_full = (df["home_score"].values > df["away_score"].values).astype(int)
    y_sp_full = (df["home_score"].values - df["away_score"].values).astype(float)

    # Chronological split by date (NaT last)
    order = pd.to_datetime(df["date"], errors="coerce") if "date" in df.columns else pd.Series(index=df.index, dtype="datetime64[ns]")
    sorted_idx = order.sort_values(na_position="last", kind="mergesort").index

    # Row validity (finite features)
    valid_mask = np.isfinite(X_full.to_numpy(dtype=np.float32)).all(axis=1)
    valid_series = pd.Series(valid_mask, index=df.index)
    keep_idx = [i for i in sorted_idx if bool(valid_series.loc[i])]

    # Slice in order
    df_sorted = df.loc[keep_idx]
    X_sorted = X_full.loc[keep_idx]
    y_ml_sorted = (df_sorted["home_score"].values > df_sorted["away_score"].values).astype(int)
    y_sp_sorted = (df_sorted["home_score"].values - df_sorted["away_score"].values).astype(float)

    n = len(X_sorted)
    log(f"Game rows after cleaning: {n}", verbose)
    if n <= 1:
        log("⚠️ Not enough rows to split; training dummy models.", verbose)
        ml_dummy = DummyClassifier(strategy="prior")
        sp_dummy = DummyRegressor(strategy="mean")
        # fit on a single neutral row
        ml_dummy.fit(pd.DataFrame([GAME_NEUTRAL_DEFAULTS], columns=GAME_DEFAULT_FEATURES), np.array([int(np.mean(y_ml_full)) if n else 0]))
        sp_dummy.fit(pd.DataFrame([GAME_NEUTRAL_DEFAULTS], columns=GAME_DEFAULT_FEATURES), np.array([float(np.mean(y_sp_full)) if n else 0.0]))
        return ml_dummy, sp_dummy, {
            "train_size": int(n), "val_size": 0,
            "moneyline_logloss": float("nan"), "moneyline_brier": float("nan"),
            "spread_rmse": float("nan"), "spread_mae": float("nan"),
        }

    split = max(1, min(int(n * 0.8), n - 1))
    X_tr, X_val = X_sorted.iloc[:split, :], X_sorted.iloc[split:, :]
    y_ml_tr, y_ml_val = y_ml_sorted[:split], y_ml_sorted[split:]
    y_sp_tr, y_sp_val = y_sp_sorted[:split], y_sp_sorted[split:]

    log(f"Split sizes — train: {len(X_tr)}, val: {len(X_val)}", verbose)

    # Moneyline model (use canonical params to avoid alias warnings)
    ml_logloss = float("nan"); ml_brier = float("nan")
    if _HAS_LGB:
        clf = lgb.LGBMClassifier(
            objective="binary",
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            colsample_bytree=0.9,  # instead of feature_fraction
            subsample=0.8,         # instead of bagging_fraction
            subsample_freq=5,      # instead of bagging_freq
            n_estimators=1000,
            random_state=seed,
            n_jobs=-1,
        )
        clf.fit(
            X_tr, y_ml_tr,
            eval_set=[(X_val, y_ml_val)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(80, verbose=False)]
        )
        if len(X_val) > 0:
            p_val = clf.predict_proba(X_val)[:, 1]
            if len(np.unique(y_ml_val)) == 2:
                ml_logloss = float(log_loss(y_ml_val, p_val))
            ml_brier = float(brier_score_loss(y_ml_val, p_val))
    else:
        clf = HistGradientBoostingClassifier(
            learning_rate=0.06, max_depth=None, max_iter=400, random_state=seed,
            validation_fraction=0.2, early_stopping=True
        )
        clf.fit(X_tr, y_ml_tr)
        if len(X_val) > 0:
            p_val = clf.predict_proba(X_val)[:, 1]
            if len(np.unique(y_ml_val)) == 2:
                ml_logloss = float(log_loss(y_ml_val, p_val))
            ml_brier = float(brier_score_loss(y_ml_val, p_val))

    # Spread model
    if _HAS_LGB:
        reg = lgb.LGBMRegressor(
            objective="regression",
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            colsample_bytree=0.9,
            subsample=0.8,
            subsample_freq=5,
            n_estimators=1000,
            random_state=seed,
            n_jobs=-1,
        )
        reg.fit(
            X_tr, y_sp_tr,
            eval_set=[(X_val, y_sp_val)],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(80, verbose=False)]
        )
    else:
        reg = HistGradientBoostingRegressor(
            learning_rate=0.06, max_depth=None, max_iter=400, random_state=seed,
            validation_fraction=0.2, early_stopping=True
        )
        reg.fit(X_tr, y_sp_tr)

    sp_rmse = float("nan"); sp_mae = float("nan")
    if len(X_val) >= 1:
        try:
            y_pred_val = reg.predict(X_val)
            sp_rmse = float(math.sqrt(mean_squared_error(y_sp_val, y_pred_val)))
            sp_mae = float(mean_absolute_error(y_sp_val, y_pred_val))
        except Exception as e:
            log(f"Spread metrics fallback: {e}", verbose)

    return clf, reg, {
        "train_size": int(len(X_tr)),
        "val_size": int(len(X_val)),
        "moneyline_logloss": ml_logloss,
        "moneyline_brier": ml_brier,
        "spread_rmse": sp_rmse,
        "spread_mae": sp_mae,
    }


def main():
    ap = argparse.ArgumentParser(description="Train only game models (moneyline + spread).")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--games-csv", type=str, help="Path to games CSV (must contain home_score, away_score; date optional).")
    mode.add_argument("--source", choices=["kaggle"], help="Build games frame from a Kaggle dataset.")
    mode.add_argument("--teamstats-csv", type=str, help="Path to TeamStatistics CSV to build training data locally.")
    ap.add_argument("--dataset", type=str, default="eoinamoore/historical-nba-data-and-player-box-scores", help="Kaggle dataset ref (when --source kaggle)")
    ap.add_argument("--models-dir", type=str, default="models", help="Output dir for .pkl files (default: models)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging")
    ap.add_argument("--skip-rest", action="store_true", help="Skip rest/b2b merges to maximize retained rows")
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    if args.games_csv:
        df = load_games_csv(Path(args.games_csv), verbose=args.verbose)
    elif args.teamstats_csv:
        df = _build_games_from_teamstats(Path(args.teamstats_csv), verbose=args.verbose, skip_rest=args.skip_rest)
        for col, val in GAME_NEUTRAL_DEFAULTS.items():
            if col not in df.columns:
                df[col] = val
    else:
        df = build_games_training_from_kaggle(args.dataset, verbose=args.verbose, skip_rest=args.skip_rest)
        for col, val in GAME_NEUTRAL_DEFAULTS.items():
            if col not in df.columns:
                df[col] = val

    ml_model, sp_model, metrics = train_game_models(df, seed=args.seed, verbose=args.verbose)

    import pickle
    with open(models_dir / "moneyline_winprob_model.pkl", "wb") as f:
        pickle.dump(ml_model, f)
    with open(models_dir / "spread_margin_model.pkl", "wb") as f:
        pickle.dump(sp_model, f)

    print("\nSaved game models:")
    print(f" - {(models_dir / 'moneyline_winprob_model.pkl').as_posix()}")
    print(f" - {(models_dir / 'spread_margin_model.pkl').as_posix()}")

    print("\nMetrics (val set):")
    print(f" - moneyline: logloss={metrics['moneyline_logloss']}, brier={metrics['moneyline_brier']}")
    print(f" - spread:    RMSE={metrics['spread_rmse']}, MAE={metrics['spread_mae']}")
    print(f"Sizes: train={metrics['train_size']}, val={metrics['val_size']}")


if __name__ == "__main__":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    main()