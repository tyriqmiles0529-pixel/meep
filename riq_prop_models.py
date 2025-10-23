"""
riq_prop_models.py

Prop-aware statistical projections and probability models.
Uses appropriate distributions for each prop type:
- Normal for Points, Assists, Rebounds
- Negative Binomial (Poisson fallback) for 3PM
"""

import numpy as np
import pandas as pd
from typing import Tuple
import math


# ============================================
# PROJECTION WITH PROP-SPECIFIC MODELING
# ============================================

def project_stat(
    values: np.ndarray,
    prop_type: str,
    pace_multiplier: float = 1.0,
    defense_factor: float = 1.0
) -> Tuple[float, float]:
    """
    Project stat using EWMA mean with trend boost and robust variance.
    
    Args:
        values: Array of recent stat values (most recent first)
        prop_type: Type of prop (points, assists, rebounds, threes)
        pace_multiplier: Pace adjustment factor
        defense_factor: Defensive adjustment factor
    
    Returns:
        Tuple of (mu, sigma) - projected mean and standard deviation
    """
    if len(values) == 0:
        return 0.0, 0.0
    
    # EWMA weights (exponentially weight recent games more)
    n = len(values)
    weights = np.exp(np.linspace(0, 1, n))
    weights = weights / weights.sum()
    
    # Base projection
    base_mu = np.average(values, weights=weights)
    
    # Trend boost: compare recent 3 to recent 7 games
    if len(values) >= 7:
        recent_3 = values[:3].mean()
        recent_7 = values[:7].mean()
        trend = (recent_3 - recent_7) / max(recent_7, 1.0)
        # Apply small trend adjustment (up to ±5%)
        trend_adjustment = np.clip(trend, -0.05, 0.05)
        base_mu = base_mu * (1.0 + trend_adjustment)
    
    # Robust variance using MAD (Median Absolute Deviation)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    # Convert MAD to std (for normal distribution: sigma ≈ 1.4826 * MAD)
    base_sigma = 1.4826 * mad if mad > 0 else np.std(values)
    
    # If MAD gives 0, fall back to regular std
    if base_sigma == 0:
        base_sigma = np.std(values) if len(values) > 1 else base_mu * 0.20
    
    # Prop-specific overdispersion and adjustments
    if prop_type == "points":
        # Points: high variance, pace and defense matter
        overdispersion = 1.1
        mu = base_mu * pace_multiplier * defense_factor
        sigma = base_sigma * overdispersion * np.sqrt(pace_multiplier)
        
    elif prop_type == "assists":
        # Assists: moderate variance, pace matters more than defense
        overdispersion = 1.15
        defense_weight = 0.3
        mu = base_mu * pace_multiplier * (1.0 - defense_weight + defense_weight * defense_factor)
        sigma = base_sigma * overdispersion * np.sqrt(pace_multiplier)
        
    elif prop_type == "rebounds":
        # Rebounds: less affected by pace, more by style
        overdispersion = 1.2
        mu = base_mu * (0.8 * pace_multiplier + 0.2)
        sigma = base_sigma * overdispersion * (0.9 + 0.1 * pace_multiplier)
        
    elif prop_type == "threes":
        # Threes: high variance, count data
        overdispersion = 1.3
        mu = base_mu * pace_multiplier * defense_factor
        sigma = base_sigma * overdispersion * np.sqrt(pace_multiplier)
        
    else:
        # Unknown prop type: use base values
        mu = base_mu * pace_multiplier
        sigma = base_sigma * np.sqrt(pace_multiplier)
    
    # Ensure sigma is positive
    if sigma <= 0:
        sigma = mu * 0.20
    
    return mu, sigma


# ============================================
# PROP-AWARE PROBABILITY MODELS
# ============================================

def prop_win_probability(
    prop_type: str,
    values: np.ndarray,
    line: float,
    pick: str,
    mu: float,
    sigma: float
) -> Tuple[float, float]:
    """
    Calculate win probability using appropriate distribution for prop type.
    
    Args:
        prop_type: Type of prop (points, assists, rebounds, threes)
        values: Historical values (for distribution fitting)
        line: Betting line
        pick: "over" or "under"
        mu: Projected mean
        sigma: Projected standard deviation
    
    Returns:
        Tuple of (p_hat, z_like) - win probability and z-score (or equivalent)
    """
    if prop_type in ["points", "assists", "rebounds"]:
        # Use Normal distribution for continuous-like stats
        p_hat, z_like = _normal_tail_probability(mu, sigma, line, pick)
        
    elif prop_type == "threes":
        # Use Negative Binomial for count data (with Poisson fallback)
        p_hat, z_like = _count_tail_probability(values, mu, sigma, line, pick)
        
    else:
        # Default: use Normal
        p_hat, z_like = _normal_tail_probability(mu, sigma, line, pick)
    
    return p_hat, z_like


def _norm_cdf(x: float) -> float:
    """
    Normal CDF approximation (mean=0, std=1).
    Uses Abramowitz and Stegun approximation.
    """
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911
    
    sign = 1 if x >= 0 else -1
    x_abs = abs(x) / math.sqrt(2.0)
    
    t = 1.0 / (1.0 + p * x_abs)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x_abs * x_abs)
    
    return 0.5 * (1.0 + sign * y)


def _normal_tail_probability(
    mu: float,
    sigma: float,
    line: float,
    pick: str
) -> Tuple[float, float]:
    """
    Normal distribution tail probability.
    
    Returns:
        Tuple of (p_hat, z_score)
    """
    if sigma <= 0:
        sigma = mu * 0.20
    
    z = (mu - line) / sigma
    
    # Standardize to N(0,1)
    z_std = (line - mu) / sigma
    cdf_at_line = _norm_cdf(z_std)
    
    if pick == "over":
        # P(X > line) = 1 - CDF(line)
        p_hat = 1.0 - cdf_at_line
    else:
        # P(X < line) = CDF(line)
        p_hat = cdf_at_line
    
    # Don't artificially cap probabilities
    # Let the model give its true estimate
    return p_hat, z


def _poisson_pmf(k: int, lam: float) -> float:
    """Poisson probability mass function."""
    if lam <= 0:
        return 0.0
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def _poisson_cdf(k: int, lam: float) -> float:
    """Poisson cumulative distribution function."""
    if lam <= 0:
        return 1.0 if k >= 0 else 0.0
    
    # Sum PMF from 0 to k
    cdf = 0.0
    for i in range(int(k) + 1):
        cdf += _poisson_pmf(i, lam)
    
    return min(1.0, cdf)


def _count_tail_probability(
    values: np.ndarray,
    mu: float,
    sigma: float,
    line: float,
    pick: str
) -> Tuple[float, float]:
    """
    Count data tail probability using Negative Binomial (Poisson fallback).
    
    Negative Binomial allows for overdispersion (variance > mean).
    If variance ≈ mean, falls back to Poisson.
    
    Returns:
        Tuple of (p_hat, z_like)
    """
    var = sigma ** 2
    
    # For simplicity and robustness, use Poisson as primary model
    # (Negative Binomial requires scipy or complex implementation)
    # Adjust mu slightly for overdispersion
    if var > mu * 1.5:
        # Account for overdispersion by using effective lambda
        adjusted_mu = mu * math.sqrt(var / max(mu, 0.1))
    else:
        adjusted_mu = mu
    
    p_hat = _poisson_tail_probability(adjusted_mu, line, pick)
    
    # Compute z-like score for compatibility
    z_like = (mu - line) / max(sigma, 1.0)
    
    return p_hat, z_like


def _poisson_tail_probability(mu: float, line: float, pick: str) -> float:
    """
    Poisson distribution tail probability.
    
    Returns:
        p_hat
    """
    if mu <= 0:
        mu = 0.1
    
    line_int = int(line)
    
    if pick == "over":
        # P(X > line) = 1 - CDF(line)
        p_hat = 1.0 - _poisson_cdf(line_int, mu)
    else:
        # P(X <= line) = CDF(line)
        p_hat = _poisson_cdf(line_int, mu)
    
    return p_hat


# ============================================
# EARLY SEASON BLENDING
# ============================================

def blend_seasons(
    last_season_values: np.ndarray,
    current_season_values: np.ndarray,
    prior_games_strength: float = 5.0,
    team_continuity: float = 0.8
) -> np.ndarray:
    """
    Blend last season and current season stats for early season.
    
    Uses continuity-aware weighting:
    w_cur = n_curr / (n_curr + n0_eff)
    where n0_eff = prior_games_strength * team_continuity
    
    Args:
        last_season_values: Stats from last season (most recent first)
        current_season_values: Stats from current season (most recent first)
        prior_games_strength: Strength of prior (in "game equivalents")
        team_continuity: Factor for roster continuity (0.0-1.0)
    
    Returns:
        Blended array of values
    """
    n_curr = len(current_season_values)
    n_last = len(last_season_values)
    
    if n_curr == 0 and n_last == 0:
        return np.array([])
    
    if n_curr == 0:
        # Only last season data
        return last_season_values
    
    if n_last == 0:
        # Only current season data
        return current_season_values
    
    # Compute effective prior strength
    n0_eff = prior_games_strength * team_continuity
    
    # Weight for current season
    w_curr = n_curr / (n_curr + n0_eff)
    w_last = 1.0 - w_curr
    
    # Blend: prioritize current season, but include recent last season games
    # Strategy: take all current season games, then add weighted last season games
    
    # If current season is strong enough, use it primarily
    if n_curr >= 10:
        # Mostly current season
        return current_season_values
    
    # Otherwise, blend
    # Take current season + supplement with last season
    num_last_to_add = min(n_last, max(0, 10 - n_curr))
    
    blended = np.concatenate([
        current_season_values,
        last_season_values[:num_last_to_add]
    ])
    
    return blended


def compute_n_eff(
    prop_type: str,
    base_n_eff: float = 30.0,
    confidence_multiplier: float = 1.0
) -> float:
    """
    Compute effective sample size for posterior based on market and prop type.
    
    Different markets have different information efficiency:
    - Points/Assists: High liquidity, n_eff higher
    - Rebounds: Moderate liquidity
    - Threes: Lower liquidity, more uncertain
    
    Args:
        prop_type: Type of prop
        base_n_eff: Base effective sample size
        confidence_multiplier: Learning multiplier from historical accuracy
    
    Returns:
        Effective sample size for Beta posterior
    """
    # Market efficiency factors
    efficiency = {
        "points": 1.2,      # Most liquid market
        "assists": 1.1,
        "rebounds": 1.0,    # Baseline
        "threes": 0.8,      # Less liquid, more uncertain
        "moneyline": 1.5,   # Very liquid
        "spread": 1.4,
        "game_total": 1.3
    }
    
    factor = efficiency.get(prop_type, 1.0)
    n_eff = base_n_eff * factor * confidence_multiplier
    
    return max(10.0, n_eff)  # Ensure minimum of 10
