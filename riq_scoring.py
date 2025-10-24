# Riq Meeping Machine — ELG, Dynamic Kelly, Exposure Caps, and Odds utils
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import math, random

try:
    import numpy as np
except Exception:
    np = None

def american_to_decimal(american: int | float) -> float:
    if isinstance(american, str):
        american = float(american)
    if american >= 100:
        return 1.0 + (american / 100.0)
    elif american <= -100:
        return 1.0 + (100.0 / abs(american))
    return 2.0

def break_even_p(american: int | float) -> Tuple[float, float]:
    dec = american_to_decimal(american)
    b = dec - 1.0
    return 1.0 / (1.0 + b), b

def implied_prob_from_american(american: int | float) -> float:
    if isinstance(american, str):
        american = float(american)
    return 100.0 / (american + 100.0) if american >= 100 else abs(american) / (abs(american) + 100.0)

def sample_beta_posterior(p_hat: float, n_eff: float, n_samples: int = 600, seed: int = 42) -> List[float]:
    p_hat = min(max(1e-4, p_hat), 1.0 - 1e-4)
    alpha = 1.0 + p_hat * max(1e-6, n_eff)
    beta = 1.0 + (1.0 - p_hat) * max(1e-6, n_eff)
    rng = random.Random(seed)
    out = []
    for _ in range(n_samples):
        x = rng.gammavariate(alpha, 1.0)
        y = rng.gammavariate(beta, 1.0)
        out.append(x / (x + y + 1e-12))
    return out

def kelly_fraction(p: float, b: float) -> float:
    q = 1.0 - p
    f = (b * p - q) / max(1e-9, b)
    return max(0.0, f)

@dataclass
class KellyConfig:
    q_conservative: float = 0.30
    fk_low: float = 0.25
    fk_high: float = 0.50
    dd_scale: float = 1.0

def dynamic_fractional_kelly(p_samples: List[float], b: float, cfg: KellyConfig) -> Tuple[float, float, float, float]:
    if not p_samples:
        return 0.0, 0.0, 0.0, 0.0
    if np is not None:
        arr = np.array(p_samples)
        p_c = float(np.quantile(arr, cfg.q_conservative))
        p_mean = float(arr.mean())
    else:
        a = sorted(p_samples)
        idx = max(0, min(len(a) - 1, int(cfg.q_conservative * (len(a) - 1))))
        p_c = a[idx]
        p_mean = sum(a) / len(a)
    p_be = 1.0 / (1.0 + b)
    if p_c <= p_be:
        return 0.0, p_c, 0.0, p_mean
    denom = max(1e-9, p_mean - p_c)
    conf = (p_mean - p_be) / denom
    conf = max(0.0, min(2.0, conf)) / 2.0
    frac_k = (cfg.fk_low + (cfg.fk_high - cfg.fk_low) * conf) * cfg.dd_scale
    f_star = kelly_fraction(p_c, b)
    f = frac_k * f_star
    return f, p_c, frac_k, p_mean

def expected_log_growth(p: float, b: float, f: float) -> float:
    p = min(max(1e-6, p), 1.0 - 1e-6)
    f = min(max(0.0, f), 0.999999)
    return p * math.log1p(f * b) + (1.0 - p) * math.log1p(-f)

def risk_adjusted_elg(p_samples: List[float], b: float, f: float) -> float:
    if not p_samples or f <= 0.0:
        return -1e9
    if np is not None:
        ps = np.clip(np.array(p_samples), 1e-6, 1.0 - 1e-6)
        return float(np.mean(ps * math.log1p(f * b) + (1.0 - ps) * math.log1p(-f)))
    acc = 0.0
    for p in p_samples:
        p = min(max(1e-6, p), 1.0 - 1e-6)
        acc += p * math.log(1.0 + f * b) + (1.0 - p) * math.log(1.0 - f)
    return acc / len(p_samples)

@dataclass
class ExposureCaps:
    max_per_game: float = 0.20
    max_per_player: float = 0.12
    max_per_team: float = 0.20
    max_props_per_player: int = 2

@dataclass
class Candidate:
    id: str
    player_id: str
    team_id: str
    game_id: str
    market: str
    american: int
    p_samples: List[float]
    meta: Optional[Dict] = None

@dataclass
class ScoredPick:
    cand: Candidate
    p_c: float
    p_mean: float
    frac_kelly: float
    kelly_f: float
    elg: float

def score_candidate(cand: Candidate, kcfg: KellyConfig) -> Optional[ScoredPick]:
    dec = american_to_decimal(cand.american); b = dec - 1.0
    f, p_c, frac_k, p_mean = dynamic_fractional_kelly(cand.p_samples, b, kcfg)
    if f <= 0.0:
        return None
    p_be = 1.0 / (1.0 + b)
    if p_c <= p_be:
        return None
    elg = risk_adjusted_elg(cand.p_samples, b, f)
    if elg <= 0.0:
        return None
    return ScoredPick(cand=cand, p_c=p_c, p_mean=p_mean, frac_kelly=frac_k, kelly_f=f, elg=elg)

def select_portfolio(scored: List[ScoredPick], bankroll: float, caps: ExposureCaps) -> List[Tuple[ScoredPick, float]]:
    picked: List[Tuple[ScoredPick, float]] = []
    per_game: Dict[str, float] = {}
    per_player: Dict[str, float] = {}
    per_team: Dict[str, float] = {}
    props_count: Dict[str, int] = {}
    for sp in sorted(scored, key=lambda x: x.elg, reverse=True):
        stake = max(0.0, sp.kelly_f * bankroll)
        gid = sp.cand.game_id; pid = sp.cand.player_id; tid = sp.cand.team_id
        if props_count.get(pid, 0) >= caps.max_props_per_player: continue
        if per_game.get(gid, 0.0) + (stake / bankroll) > caps.max_per_game: continue
        if per_player.get(pid, 0.0) + (stake / bankroll) > caps.max_per_player: continue
        if per_team.get(tid, 0.0) + (stake / bankroll) > caps.max_per_team: continue
        picked.append((sp, stake))
        props_count[pid] = props_count.get(pid, 0) + 1
        per_game[gid] = per_game.get(gid, 0.0) + (stake / bankroll)
        per_player[pid] = per_player.get(pid, 0.0) + (stake / bankroll)
        per_team[tid] = per_team.get(tid, 0.0) + (stake / bankroll)
    return picked

def drawdown_scale(equity_curve: List[float], floor: float = 0.6, window: int = 14) -> float:
    if not equity_curve or len(equity_curve) < 2:
        return 1.0
    recent = equity_curve[-window:] if len(equity_curve) >= window else equity_curve
    peak = max(recent); curr = recent[-1]
    if peak <= 0: return 1.0
    dd = (peak - curr) / peak
    if dd <= 0: return 1.0
    if dd >= 0.30: return floor
    return 1.0 - (1.0 - floor) * (dd / 0.30)