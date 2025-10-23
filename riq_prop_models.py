# Prop-specific projections and win probability models
from __future__ import annotations
from typing import Tuple, List
import math

try:
    import numpy as np
except Exception:
    np = None

def _safe_array(a: List[float]):
    return np.array(a, dtype=float) if np is not None else [float(x) for x in a]

def _ewma(values: List[float], half_life: float = 5.0) -> float:
    if not values: return 0.0
    if np is None:
        n = len(values); weights = [2 ** (i / (n / half_life + 1e-9)) for i in range(n)]
        s = sum(w * v for w, v in zip(weights, values)); return s / (sum(weights) + 1e-9)
    v = np.asarray(values, dtype=float); n = len(v); idx = np.arange(n, dtype=float)
    lam = 0.5 ** (1.0 / max(1.0, half_life)); w = lam ** (n - 1 - idx); w /= w.sum()
    return float((v * w).sum())

def _robust_sigma(values: List[float], mu_ref: float) -> float:
    if not values: return 0.0
    if np is None:
        diffs = sorted(abs(x - mu_ref) for x in values); med = diffs[len(diffs)//2]
        return max(1e-6, med / 0.6745 if med > 0 else 0.0)
    v = np.asarray(values, dtype=float); mad = float(np.median(np.abs(v - mu_ref)))
    return max(1e-6, mad / 0.6745)

def _normal_cdf(x: float) -> float:
    a1,a2,a3,a4,a5,p = 0.254829592,-0.284496736,1.421413741,-1.453152027,1.061405429,0.3275911
    sign = 1 if x >= 0 else -1; x = abs(x) / math.sqrt(2.0)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    return 0.5 * (1.0 + sign * y)

def _poisson_cdf(k: int, lam: float) -> float:
    total = 0.0; term = math.exp(-lam); total += term
    for i in range(1, max(1, k + 1)):
        term *= lam / i; total += term
        if i >= k: break
    return min(1.0, max(0.0, total))

def _fit_nb_params(values: List[float]) -> Tuple[float, float, float]:
    v = _safe_array(values)
    if np is None:
        m = sum(v) / max(1, len(v))
        var = sum((x - m) ** 2 for x in v) / max(1, len(v) - 1)
    else:
        m = float(np.mean(v)); var = float(np.var(v, ddof=1)) if len(v) > 1 else max(1e-6, m)
    if var <= m + 1e-9: return m, float("inf"), 0.0
    r = (m * m) / (var - m + 1e-9); p = m / (m + r)
    return max(1e-6, r), min(max(1e-9, p), 1 - 1e-9), m

def _nb_cdf(k: int, r: float, p: float) -> float:
    if r == float("inf"): return 0.0
    total = 0.0; prob = (1 - p) ** r; total += prob
    for i in range(1, max(1, k + 1)):
        prob *= ((i - 1 + r) / i) * p; total += prob
        if i >= k: break
    return min(1.0, max(0.0, total))

def project_stat(values: List[float], prop_type: str, pace_multiplier: float, defense_factor: float) -> Tuple[float, float]:
    vals = _safe_array(values)
    if len(vals) == 0: return 0.0, 0.0
    mu_base = _ewma(list(vals), half_life=5.0)
    if np is None:
        last3 = list(vals)[:3] if len(vals) >= 3 else list(vals); last8 = list(vals)[:8] if len(vals) >= 8 else list(vals)
        m3 = sum(last3)/max(1,len(last3)); m8 = sum(last8)/max(1,len(last8))
    else:
        m3 = float(np.mean(vals[:3])) if len(vals) >= 3 else float(np.mean(vals))
        m8 = float(np.mean(vals[:8])) if len(vals) >= 8 else float(np.mean(vals))
    trend = 0.0 if m8 <= 1e-9 else (m3 - m8) / m8
    trend_boost = max(-0.05, min(0.05, 0.25 * trend))
    mu = mu_base * (1.0 + trend_boost)
    mu *= pace_multiplier
    if prop_type == "assists":
        mu *= (0.7 + 0.3 * defense_factor)
    elif prop_type in ("points", "threes"):
        mu *= defense_factor
    elif prop_type == "rebounds":
        mu *= (0.8 * pace_multiplier + 0.2)
    raw_sigma = _robust_sigma(list(vals), mu_base)
    if prop_type == "points": sigma = max(0.8, raw_sigma * 1.10)
    elif prop_type == "assists": sigma = max(0.6, raw_sigma * 1.20)
    elif prop_type == "rebounds": sigma = max(0.8, raw_sigma * 1.15)
    elif prop_type == "threes": sigma = max(0.5, raw_sigma * 1.30)
    else: sigma = max(0.8, raw_sigma * 1.10)
    return float(mu), float(sigma)

def prop_win_probability(prop_type: str, values: List[float], line: float, pick: str, mu: float, sigma: float) -> Tuple[float, float]:
    if prop_type != "threes":
        sigma = max(sigma, 1e-6); z = (mu - line) / sigma
        p = 1.0 - _normal_cdf((line - mu) / sigma) if pick == "over" else _normal_cdf((line - mu) / sigma)
        return min(1.0 - 1e-4, max(1e-4, p)), z
    ints = [max(0, int(round(v))) for v in values if v is not None]
    if len(ints) == 0:
        sigma = max(sigma, 1e-6); z = (mu - line) / sigma
        p = 1.0 - _normal_cdf((line - mu) / sigma) if pick == "over" else _normal_cdf((line - mu) / sigma)
        return min(1.0 - 1e-4, max(1e-4, p)), z
    mean, r, p_like = _fit_nb_params(ints)
    if r == float("inf"):
        lam = mean
        if pick == "over":
            k = math.ceil(line); p = 1.0 - _poisson_cdf(k - 1, lam)
        else:
            k = math.floor(line); p = _poisson_cdf(k, lam)
        z_like = (mu - line) / max(1e-6, sigma)
        return min(1.0 - 1e-4, max(1e-4, p)), z_like
    p_nb = mean / (mean + r)
    if pick == "over":
        k = math.ceil(line); p = 1.0 - _nb_cdf(k - 1, r, p_nb)
    else:
        k = math.floor(line); p = _nb_cdf(k, r, p_nb)
    z_like = (mu - line) / max(1e-6, sigma)
    return min(1.0 - 1e-4, max(1e-4, p)), z_like
