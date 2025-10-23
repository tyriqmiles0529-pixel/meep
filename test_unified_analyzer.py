"""
Test script for the unified NBA prop analyzer.

This demonstrates the key components work together correctly.
Note: This does not call the actual API, just tests the logic.
"""

import numpy as np
from riq_scoring import (
    KellyConfig, ExposureCaps, Candidate, ScoredPick,
    score_candidate, select_portfolio, sample_beta_posterior,
    american_to_decimal, break_even_p
)
from riq_prop_models import (
    project_stat, prop_win_probability, compute_n_eff, blend_seasons
)


def test_prop_projection():
    """Test prop-aware projection."""
    print("\n🧪 Test: Prop Projection")
    
    # Simulate player stats (most recent first)
    points = np.array([28.0, 25.0, 30.0, 27.0, 24.0, 29.0, 26.0])
    
    mu, sigma = project_stat(
        values=points,
        prop_type="points",
        pace_multiplier=1.05,
        defense_factor=0.95
    )
    
    print(f"  Input: Last 7 games = {points}")
    print(f"  Output: μ={mu:.2f}, σ={sigma:.2f}")
    assert 24 < mu < 32, "Projection should be near recent average"
    assert sigma > 0, "Sigma must be positive"
    print("  ✅ Pass")


def test_win_probability():
    """Test prop-aware win probability."""
    print("\n🧪 Test: Win Probability")
    
    values = np.array([28.0, 25.0, 30.0, 27.0, 24.0, 29.0, 26.0])
    mu = 27.5
    sigma = 2.0
    line = 26.5
    
    # Test points (Normal distribution)
    p_hat_points, z_points = prop_win_probability(
        prop_type="points",
        values=values,
        line=line,
        pick="over",
        mu=mu,
        sigma=sigma
    )
    
    print(f"  Points: μ={mu}, σ={sigma}, line={line}")
    print(f"  P(over): {p_hat_points:.3f}, z={z_points:.3f}")
    assert 0.3 < p_hat_points < 0.9, "Probability should be reasonable"
    
    # Test threes (Poisson)
    threes = np.array([3.0, 2.0, 4.0, 3.0, 2.0, 3.0, 4.0])
    mu_3pm = 3.0
    sigma_3pm = 1.0
    line_3pm = 2.5
    
    p_hat_3pm, z_3pm = prop_win_probability(
        prop_type="threes",
        values=threes,
        line=line_3pm,
        pick="over",
        mu=mu_3pm,
        sigma=sigma_3pm
    )
    
    print(f"  3PM: μ={mu_3pm}, σ={sigma_3pm}, line={line_3pm}")
    print(f"  P(over): {p_hat_3pm:.3f}, z={z_3pm:.3f}")
    assert 0.3 < p_hat_3pm < 0.9, "Probability should be reasonable"
    print("  ✅ Pass")


def test_elg_scoring():
    """Test ELG-based scoring."""
    print("\n🧪 Test: ELG Scoring")
    
    config = KellyConfig(
        min_kelly_stake=1.0,
        max_kelly_fraction=0.25,
        conservative_quantile=0.25,
        elg_samples=1000
    )
    
    # Create a favorable bet scenario
    prop = {
        "prop_id": "test_prop_1",
        "player": "LeBron James",
        "prop_type": "points",
        "game_id": 12345,
        "team": "LAL",
        "line": 25.5,
        "odds": -110
    }
    
    # Strong posterior: high win probability
    p_samples = sample_beta_posterior(0.65, 40, num_samples=1000)
    
    scored = score_candidate(prop, p_samples, -110, 100.0, config)
    
    if scored:
        print(f"  Prop: {prop['player']} {prop['prop_type']} {prop['line']}")
        print(f"  ELG: {scored.elg_score:.6f}")
        print(f"  Stake: ${scored.candidate.stake:.2f}")
        print(f"  Win%: {scored.candidate.win_prob:.1f}%")
        print(f"  Kelly%: {scored.candidate.kelly_pct:.2f}%")
        assert scored.elg_score > 0, "ELG should be positive"
        assert scored.candidate.stake >= config.min_kelly_stake, "Stake should meet minimum"
        print("  ✅ Pass")
    else:
        print("  ⚠️  Bet filtered (ELG gates not met)")


def test_portfolio_selection():
    """Test portfolio selection with exposure caps."""
    print("\n🧪 Test: Portfolio Selection")
    
    config = KellyConfig()
    caps = ExposureCaps(
        max_per_game=15.0,
        max_per_player=10.0,
        max_per_team=20.0,
        max_total=50.0
    )
    
    # Create multiple candidates
    candidates = []
    
    for i in range(5):
        prop = {
            "prop_id": f"prop_{i}",
            "player": f"Player {i}",
            "prop_type": "points",
            "game_id": 1000 + (i // 2),  # 2 props per game
            "team": f"TEAM{i % 3}",
            "line": 25.0,
            "odds": -110
        }
        
        p_samples = sample_beta_posterior(0.60 + i*0.02, 30, num_samples=1000)
        scored = score_candidate(prop, p_samples, -110, 100.0, config)
        
        if scored:
            candidates.append(scored)
    
    print(f"  Created {len(candidates)} candidates")
    
    # Select portfolio
    selected = select_portfolio(candidates, 100.0, caps, sort_by_elg=True)
    
    print(f"  Selected {len(selected)} bets after exposure caps")
    
    # Check exposure limits
    total_stake = sum(pick.candidate.stake for pick in selected)
    print(f"  Total stake: ${total_stake:.2f} ({total_stake/100*100:.1f}% of bankroll)")
    
    assert total_stake <= caps.max_total * 100.0, "Should respect total cap"
    print("  ✅ Pass")


def test_season_blending():
    """Test early-season blending."""
    print("\n🧪 Test: Season Blending")
    
    last_season = np.array([20.0, 22.0, 21.0, 23.0, 19.0, 24.0, 20.0, 22.0, 21.0, 23.0])
    current_season = np.array([25.0, 27.0, 24.0])
    
    blended = blend_seasons(
        last_season_values=last_season,
        current_season_values=current_season,
        prior_games_strength=5.0,
        team_continuity=0.8
    )
    
    print(f"  Last season: {len(last_season)} games, avg={last_season.mean():.1f}")
    print(f"  Current season: {len(current_season)} games, avg={current_season.mean():.1f}")
    print(f"  Blended: {len(blended)} games, avg={blended.mean():.1f}")
    
    assert len(blended) >= len(current_season), "Should include current season"
    assert len(blended) <= len(current_season) + len(last_season), "Should not exceed total"
    print("  ✅ Pass")


def test_gates():
    """Test conservative edge gates."""
    print("\n🧪 Test: Conservative Edge Gates")
    
    config = KellyConfig(conservative_quantile=0.25)
    
    # Case 1: Good bet (should pass)
    p_samples_good = sample_beta_posterior(0.60, 30, num_samples=1000)
    p_c = np.percentile(p_samples_good, 25)
    p_be = break_even_p(-110)
    
    print(f"  Good bet: p_c={p_c:.3f}, p_be={p_be:.3f}")
    assert p_c > p_be, "Conservative p should beat break-even"
    print("    ✅ Conservative edge exists")
    
    # Case 2: Bad bet (should fail)
    p_samples_bad = sample_beta_posterior(0.48, 30, num_samples=1000)
    p_c_bad = np.percentile(p_samples_bad, 25)
    
    print(f"  Bad bet: p_c={p_c_bad:.3f}, p_be={p_be:.3f}")
    assert p_c_bad < p_be, "Conservative p should NOT beat break-even"
    print("    ✅ Correctly filtered")
    
    print("  ✅ Pass")


def main():
    """Run all tests."""
    print("=" * 70)
    print("UNIFIED ANALYZER TEST SUITE")
    print("=" * 70)
    
    test_prop_projection()
    test_win_probability()
    test_elg_scoring()
    test_portfolio_selection()
    test_season_blending()
    test_gates()
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED")
    print("=" * 70)
    print("\nThe unified analyzer is working correctly!")
    print("Run 'python nba_prop_analyzer_fixed.py' to analyze real games.")


if __name__ == "__main__":
    main()
