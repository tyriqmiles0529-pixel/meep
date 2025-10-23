"""
riq_scoring.py

Expected Log Growth (ELG) scoring and Kelly betting framework.
Provides dynamic fractional Kelly sizing, risk-adjusted ELG computation,
and exposure-constrained portfolio selection.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


# ============================================
# ODDS AND PROBABILITY UTILITIES
# ============================================

def american_to_decimal(odds: int) -> float:
    """Convert American odds to decimal odds."""
    if odds > 0:
        return (odds / 100.0) + 1.0
    else:
        return (100.0 / abs(odds)) + 1.0


def break_even_p(odds: int) -> float:
    """Calculate break-even probability from American odds."""
    decimal = american_to_decimal(odds)
    return 1.0 / decimal


def implied_prob_from_american(odds: int) -> float:
    """Calculate implied probability from American odds (no vig removal)."""
    return break_even_p(odds)


def sample_beta_posterior(p_hat: float, n_eff: float, num_samples: int = 1000) -> np.ndarray:
    """
    Sample from Beta posterior distribution.
    
    Args:
        p_hat: Point estimate of win probability
        n_eff: Effective sample size (market information strength)
        num_samples: Number of samples to draw
    
    Returns:
        Array of probability samples from posterior
    """
    # Beta parameters from Bayesian update
    # Assume uniform prior Beta(1, 1), then posterior is Beta(s+1, f+1)
    # where s = successes, f = failures
    # With p_hat and n_eff: s ≈ p_hat * n_eff, f ≈ (1 - p_hat) * n_eff
    alpha = max(1.0, p_hat * n_eff + 1.0)
    beta = max(1.0, (1.0 - p_hat) * n_eff + 1.0)
    
    samples = np.random.beta(alpha, beta, size=num_samples)
    return samples


# ============================================
# KELLY CRITERION AND ELG
# ============================================

@dataclass
class KellyConfig:
    """Configuration for Kelly betting and ELG calculation."""
    min_kelly_stake: float = 0.01  # Minimum stake required
    max_kelly_fraction: float = 0.25  # Maximum Kelly fraction to use
    conservative_quantile: float = 0.25  # Use 25th percentile for conservative sizing
    elg_samples: int = 1000  # Number of samples for ELG Monte Carlo
    

def dynamic_fractional_kelly(
    p_samples: np.ndarray,
    odds: int,
    bankroll: float,
    config: KellyConfig
) -> Tuple[float, float, float]:
    """
    Compute dynamic fractional Kelly stake using conservative probability.
    
    Args:
        p_samples: Posterior probability samples
        odds: American odds
        bankroll: Current bankroll
        config: Kelly configuration
    
    Returns:
        Tuple of (kelly_fraction, stake, conservative_p)
    """
    # Use conservative quantile (e.g., 25th percentile)
    p_conservative = np.percentile(p_samples, config.conservative_quantile * 100)
    
    # Break-even probability
    p_be = break_even_p(odds)
    
    # If conservative p doesn't beat break-even, no bet
    if p_conservative <= p_be:
        return 0.0, 0.0, p_conservative
    
    # Kelly formula: f = (b*p - q) / b, where b = decimal_odds - 1
    decimal_odds = american_to_decimal(odds)
    b = decimal_odds - 1.0
    q = 1.0 - p_conservative
    
    kelly_fraction = (b * p_conservative - q) / b
    kelly_fraction = max(0.0, kelly_fraction)
    
    # Apply fractional Kelly cap
    kelly_fraction = min(kelly_fraction, config.max_kelly_fraction)
    
    # Calculate stake
    stake = bankroll * kelly_fraction
    
    return kelly_fraction, stake, p_conservative


def risk_adjusted_elg(
    p_samples: np.ndarray,
    odds: int,
    kelly_fraction: float,
    config: KellyConfig
) -> float:
    """
    Compute Expected Log Growth (ELG) using Monte Carlo simulation.
    
    ELG = E[log(1 + f * (b * X - (1-X)))]
    where X ~ Bernoulli(p), f = Kelly fraction, b = decimal_odds - 1
    
    Args:
        p_samples: Posterior probability samples
        odds: American odds
        kelly_fraction: Fraction of bankroll to bet
        config: Kelly configuration
    
    Returns:
        Expected log growth
    """
    if kelly_fraction <= 0:
        return 0.0
    
    decimal_odds = american_to_decimal(odds)
    b = decimal_odds - 1.0
    
    # Monte Carlo: sample outcomes and compute log growth
    # For each posterior p sample, simulate outcome and compute log growth
    log_growths = []
    
    for p in p_samples[:config.elg_samples]:
        # Simulate outcome: win with probability p
        outcome = np.random.random() < p
        
        if outcome:
            # Win: gain b * stake
            growth = 1.0 + kelly_fraction * b
        else:
            # Loss: lose stake
            growth = 1.0 - kelly_fraction
        
        # Ensure growth is positive (avoid log of negative)
        if growth > 0:
            log_growths.append(np.log(growth))
        else:
            # Rare edge case: bet too large, would bankrupt
            log_growths.append(-10.0)  # Large negative
    
    elg = np.mean(log_growths)
    return elg


# ============================================
# EXPOSURE CAPS AND PORTFOLIO ASSEMBLY
# ============================================

@dataclass
class ExposureCaps:
    """Exposure limits for portfolio risk management."""
    max_per_game: float = 0.15  # Max 15% of bankroll per game
    max_per_player: float = 0.10  # Max 10% per player
    max_per_team: float = 0.20  # Max 20% per team
    max_total: float = 0.50  # Max 50% total bankroll exposure


@dataclass
class Candidate:
    """A candidate bet for portfolio selection."""
    prop_id: str
    player: str
    prop_type: str
    game_id: int
    team: str = ""  # Team identifier (optional)
    stake: float = 0.0
    elg: float = 0.0
    win_prob: float = 0.0
    kelly_pct: float = 0.0
    # Metadata for output
    metadata: Dict = field(default_factory=dict)


@dataclass
class ScoredPick:
    """A scored pick with all relevant information."""
    candidate: Candidate
    elg_score: float
    composite_score: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for output."""
        return {
            **self.candidate.metadata,
            "prop_id": self.candidate.prop_id,
            "player": self.candidate.player,
            "prop_type": self.candidate.prop_type,
            "game_id": self.candidate.game_id,
            "team": self.candidate.team,
            "stake": self.candidate.stake,
            "elg": self.elg_score,
            "composite_score": self.composite_score,
            "win_prob": self.candidate.win_prob,
            "kelly_pct": self.candidate.kelly_pct
        }


def score_candidate(
    prop: Dict,
    p_samples: np.ndarray,
    odds: int,
    bankroll: float,
    config: KellyConfig
) -> Optional[ScoredPick]:
    """
    Score a candidate bet using ELG framework.
    
    Args:
        prop: Prop dictionary with metadata
        p_samples: Posterior probability samples
        odds: American odds
        bankroll: Current bankroll
        config: Kelly configuration
    
    Returns:
        ScoredPick if bet meets criteria, None otherwise
    """
    # Compute Kelly sizing
    kelly_fraction, stake, p_conservative = dynamic_fractional_kelly(
        p_samples, odds, bankroll, config
    )
    
    # Gate 1: Minimum stake
    if stake < config.min_kelly_stake:
        return None
    
    # Gate 2: Conservative edge (p_c > p_be)
    p_be = break_even_p(odds)
    if p_conservative <= p_be:
        return None
    
    # Compute ELG
    elg = risk_adjusted_elg(p_samples, odds, kelly_fraction, config)
    
    # Gate 3: Positive ELG
    if elg <= 0:
        return None
    
    # Compute composite score (for backwards compatibility)
    # Use mean probability for composite
    p_mean = np.mean(p_samples)
    composite_score = (
        (p_mean * 100 * 0.50) +
        (kelly_fraction * 100 * 0.25) +
        (max(0, elg * 100) * 0.15) +
        (max(0, (p_conservative - p_be) * 100) * 0.10)
    )
    
    # Create candidate
    candidate = Candidate(
        prop_id=prop.get("prop_id", ""),
        player=prop.get("player", ""),
        prop_type=prop.get("prop_type", ""),
        game_id=prop.get("game_id", 0),
        team=prop.get("team", ""),
        stake=stake,
        elg=elg,
        win_prob=p_mean * 100,
        kelly_pct=kelly_fraction * 100,
        metadata=prop  # Store full prop for output
    )
    
    return ScoredPick(
        candidate=candidate,
        elg_score=elg,
        composite_score=composite_score
    )


def select_portfolio(
    scored_picks: List[ScoredPick],
    bankroll: float,
    caps: ExposureCaps,
    sort_by_elg: bool = True
) -> List[ScoredPick]:
    """
    Select portfolio of bets respecting exposure caps.
    
    Args:
        scored_picks: List of scored picks
        bankroll: Current bankroll
        caps: Exposure caps
        sort_by_elg: If True, rank by ELG; else by composite_score
    
    Returns:
        List of selected picks respecting constraints
    """
    # Sort by score
    if sort_by_elg:
        picks = sorted(scored_picks, key=lambda x: x.elg_score, reverse=True)
    else:
        picks = sorted(scored_picks, key=lambda x: x.composite_score, reverse=True)
    
    # Track exposure by category
    game_exposure: Dict[int, float] = {}
    player_exposure: Dict[str, float] = {}
    team_exposure: Dict[str, float] = {}
    total_exposure = 0.0
    
    selected = []
    
    for pick in picks:
        cand = pick.candidate
        stake = cand.stake
        
        # Check exposure limits
        new_game_exp = game_exposure.get(cand.game_id, 0.0) + stake
        new_player_exp = player_exposure.get(cand.player, 0.0) + stake
        new_team_exp = team_exposure.get(cand.team, 0.0) + stake if cand.team else 0.0
        new_total_exp = total_exposure + stake
        
        # Check if adding this bet would exceed caps
        if new_game_exp > caps.max_per_game * bankroll:
            continue
        if new_player_exp > caps.max_per_player * bankroll:
            continue
        if cand.team and new_team_exp > caps.max_per_team * bankroll:
            continue
        if new_total_exp > caps.max_total * bankroll:
            continue
        
        # Accept bet
        selected.append(pick)
        game_exposure[cand.game_id] = new_game_exp
        player_exposure[cand.player] = new_player_exp
        if cand.team:
            team_exposure[cand.team] = new_team_exp
        total_exposure = new_total_exp
    
    return selected


def drawdown_scale(current_bankroll: float, initial_bankroll: float) -> float:
    """
    Optional: scale down Kelly fractions during drawdown.
    This is a risk governor that can be applied if bankroll is tracked.
    
    Args:
        current_bankroll: Current bankroll
        initial_bankroll: Initial bankroll
    
    Returns:
        Scale factor (1.0 = no scaling, < 1.0 = reduce bets)
    """
    if current_bankroll >= initial_bankroll:
        return 1.0
    
    # Scale down linearly if in drawdown
    ratio = current_bankroll / initial_bankroll
    
    # Don't scale below 0.5
    return max(0.5, ratio)
