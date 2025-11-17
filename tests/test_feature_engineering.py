"""
Unit tests for NBA prediction feature engineering and core functions.

Tests cover:
- Odds conversion (American to decimal)
- EWMA projection calculations
- Win probability computation
- Kelly criterion sizing
- Beta posterior sampling
- NaN handling in features
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestOddsConversion:
    """Test American to decimal odds conversion"""

    def test_positive_odds(self):
        from riq_analyzer import american_to_decimal
        # +150 means bet $100 to win $150, decimal = 2.50
        assert american_to_decimal(150) == 2.5
        assert american_to_decimal(200) == 3.0
        assert american_to_decimal(100) == 2.0

    def test_negative_odds(self):
        from riq_analyzer import american_to_decimal
        # -150 means bet $150 to win $100, decimal = 1.667
        assert abs(american_to_decimal(-150) - 1.6667) < 0.01
        assert abs(american_to_decimal(-110) - 1.909) < 0.01
        assert american_to_decimal(-100) == 2.0

    def test_string_input(self):
        from riq_analyzer import american_to_decimal
        assert american_to_decimal("150") == 2.5
        assert abs(american_to_decimal("-110") - 1.909) < 0.01

    def test_break_even_probability(self):
        from riq_analyzer import break_even_p
        # -110 odds require 52.38% to break even
        p_be, b = break_even_p(-110)
        assert abs(p_be - 0.5238) < 0.01

        # +150 odds require 40% to break even
        p_be, b = break_even_p(150)
        assert abs(p_be - 0.40) < 0.01


class TestProjection:
    """Test EWMA and stat projection"""

    def test_ewma_recent_weighted(self):
        from riq_analyzer import _ewma
        # Recent values should be weighted more heavily
        values = [10.0, 15.0, 20.0, 25.0, 30.0]
        result = _ewma(values, half_life=5.0)
        # Should be closer to 30 than 10
        assert result > 20.0
        assert result < 30.0

    def test_ewma_empty_list(self):
        from riq_analyzer import _ewma
        assert _ewma([]) == 0.0

    def test_ewma_single_value(self):
        from riq_analyzer import _ewma
        assert _ewma([25.0]) == 25.0

    def test_project_stat_with_pace(self):
        from riq_analyzer import project_stat
        values = [20.0] * 10  # Consistent 20 points
        mu, sigma = project_stat(values, "points", pace_multiplier=1.1, defense_factor=1.0)
        # 1.1x pace should increase projection
        assert abs(mu - 22.0) < 1.0  # ~10% increase

    def test_project_stat_with_defense(self):
        from riq_analyzer import project_stat
        values = [25.0] * 10
        mu, sigma = project_stat(values, "points", pace_multiplier=1.0, defense_factor=0.9)
        # Tough defense (0.9) should decrease projection
        assert mu < 25.0

    def test_project_stat_different_props(self):
        from riq_analyzer import project_stat
        values = [10.0] * 10

        # Different prop types have different adjustments
        mu_pts, _ = project_stat(values, "points", 1.0, 0.95)
        mu_ast, _ = project_stat(values, "assists", 1.0, 0.95)
        mu_reb, _ = project_stat(values, "rebounds", 1.0, 0.95)

        # Assists should be less affected by defense
        # Points fully affected: 10 * 0.95 = 9.5
        # Assists partially: 10 * (0.7 + 0.3*0.95) = 9.85
        assert mu_ast > mu_pts

    def test_project_stat_sigma_minimums(self):
        from riq_analyzer import project_stat
        # Even with consistent data, sigma has minimums
        values = [20.0] * 20  # Very consistent
        _, sigma = project_stat(values, "points", 1.0, 1.0)
        assert sigma >= 0.8  # Minimum for points

        _, sigma = project_stat(values, "threes", 1.0, 1.0)
        assert sigma >= 0.5  # Minimum for threes

    def test_project_stat_empty(self):
        from riq_analyzer import project_stat
        mu, sigma = project_stat([], "points", 1.0, 1.0)
        assert mu == 0.0
        assert sigma == 0.0


class TestWinProbability:
    """Test win probability calculations"""

    def test_over_favorable(self):
        from riq_analyzer import prop_win_probability
        # Projection 28, Line 24.5 -> clearly OVER
        values = [28.0] * 10
        p, z = prop_win_probability("points", values, 24.5, "over", 28.0, 4.0)
        assert p > 0.75  # Should be very confident

    def test_under_favorable(self):
        from riq_analyzer import prop_win_probability
        # Projection 20, Line 24.5 -> clearly UNDER
        values = [20.0] * 10
        p, z = prop_win_probability("points", values, 24.5, "under", 20.0, 4.0)
        assert p > 0.75

    def test_close_to_line(self):
        from riq_analyzer import prop_win_probability
        # Projection close to line -> ~50%
        values = [24.0] * 10
        p, z = prop_win_probability("points", values, 24.5, "over", 24.0, 4.0)
        assert 0.35 < p < 0.65  # Near 50%

    def test_threes_uses_poisson(self):
        from riq_analyzer import prop_win_probability
        # 3-pointers use Poisson/NegBinom distribution
        values = [3.0, 4.0, 2.0, 5.0, 3.0, 4.0, 3.0, 2.0]
        p, z = prop_win_probability("threes", values, 3.5, "over", 3.5, 1.0)
        # Should be around 0.40-0.50 for over
        assert 0.30 < p < 0.60

    def test_probability_bounds(self):
        from riq_analyzer import prop_win_probability
        # Probabilities should be bounded (1e-4, 1-1e-4)
        values = [100.0] * 10  # Extreme case
        p, z = prop_win_probability("points", values, 10.0, "over", 100.0, 5.0)
        assert p <= 1.0 - 1e-4
        assert p >= 1e-4


class TestKellyCriterion:
    """Test Kelly criterion bet sizing"""

    def test_positive_edge(self):
        from riq_analyzer import kelly_fraction
        # p=0.6, b=1.0 (even money) -> f = (0.6*1.0 - 0.4)/1.0 = 0.2
        f = kelly_fraction(0.6, 1.0)
        assert abs(f - 0.2) < 0.01

    def test_no_edge(self):
        from riq_analyzer import kelly_fraction
        # p=0.5, b=1.0 -> f = 0 (no edge)
        f = kelly_fraction(0.5, 1.0)
        assert f == 0.0

    def test_negative_edge(self):
        from riq_analyzer import kelly_fraction
        # p=0.4, b=1.0 -> negative Kelly, should return 0
        f = kelly_fraction(0.4, 1.0)
        assert f == 0.0

    def test_higher_odds_need_less_edge(self):
        from riq_analyzer import kelly_fraction
        # +200 odds (b=2.0) with p=0.4
        # f = (2.0*0.4 - 0.6)/2.0 = 0.1
        f = kelly_fraction(0.4, 2.0)
        assert abs(f - 0.1) < 0.01


class TestBetaPosterior:
    """Test uncertainty quantification"""

    def test_sample_count(self):
        from riq_analyzer import sample_beta_posterior
        samples = sample_beta_posterior(0.6, 50.0, n_samples=600)
        assert len(samples) == 600

    def test_mean_near_p_hat(self):
        from riq_analyzer import sample_beta_posterior
        p_hat = 0.65
        samples = sample_beta_posterior(p_hat, 100.0, n_samples=1000)
        mean_samples = np.mean(samples)
        # With high n_eff, mean should be very close to p_hat
        assert abs(mean_samples - p_hat) < 0.05

    def test_higher_n_eff_tighter_distribution(self):
        from riq_analyzer import sample_beta_posterior
        samples_low = sample_beta_posterior(0.6, 10.0, n_samples=1000)
        samples_high = sample_beta_posterior(0.6, 100.0, n_samples=1000)

        std_low = np.std(samples_low)
        std_high = np.std(samples_high)

        # Higher n_eff should give tighter (lower std) distribution
        assert std_high < std_low

    def test_bounds(self):
        from riq_analyzer import sample_beta_posterior
        samples = sample_beta_posterior(0.6, 50.0, n_samples=100)
        # All samples should be between 0 and 1
        assert all(0 < s < 1 for s in samples)


class TestELGScoring:
    """Test Expected Log Growth calculations"""

    def test_positive_elg_with_edge(self):
        from riq_analyzer import risk_adjusted_elg
        # With positive edge, ELG should be positive
        p_samples = [0.65] * 100  # 65% win rate
        b = 1.0  # Even money
        f = 0.10  # 10% of bankroll

        elg = risk_adjusted_elg(p_samples, b, f)
        assert elg > 0

    def test_zero_stake_elg(self):
        from riq_analyzer import risk_adjusted_elg
        # Zero stake should give very negative ELG
        p_samples = [0.65] * 100
        elg = risk_adjusted_elg(p_samples, 1.0, 0.0)
        assert elg < -1e6

    def test_empty_samples(self):
        from riq_analyzer import risk_adjusted_elg
        elg = risk_adjusted_elg([], 1.0, 0.1)
        assert elg < -1e6


class TestDynamicKelly:
    """Test dynamic fractional Kelly sizing"""

    def test_conservative_quantile(self):
        from riq_analyzer import dynamic_fractional_kelly, KellyConfig
        # High variance samples
        p_samples = list(np.random.beta(6, 4, 600))  # Mean ~0.6, some variance
        cfg = KellyConfig(q_conservative=0.35)
        f, p_c, frac_k, p_mean = dynamic_fractional_kelly(p_samples, 1.0, cfg)

        # Conservative quantile should be lower than mean
        assert p_c < p_mean

    def test_no_bet_when_under_break_even(self):
        from riq_analyzer import dynamic_fractional_kelly, KellyConfig
        # Low probability samples
        p_samples = [0.35] * 600  # Below 50% break-even for even money
        cfg = KellyConfig()
        f, p_c, frac_k, p_mean = dynamic_fractional_kelly(p_samples, 1.0, cfg)

        # Should not bet
        assert f == 0.0

    def test_fractional_kelly_scales_down(self):
        from riq_analyzer import dynamic_fractional_kelly, KellyConfig
        p_samples = [0.65] * 600
        cfg = KellyConfig(fk_low=0.25, fk_high=0.50)
        f, _, frac_k, _ = dynamic_fractional_kelly(p_samples, 1.0, cfg)

        # Full Kelly for p=0.65, b=1.0 is 0.30
        # Fractional Kelly (0.25-0.50) should scale it down
        assert f < 0.30  # Should be fraction of full Kelly


class TestDrawdownScale:
    """Test drawdown risk adjustment"""

    def test_no_drawdown(self):
        from riq_analyzer import drawdown_scale
        equity = [100, 105, 110, 115, 120]  # Increasing equity
        scale = drawdown_scale(equity)
        assert scale == 1.0  # No scaling

    def test_moderate_drawdown(self):
        from riq_analyzer import drawdown_scale
        equity = [120, 115, 110, 105]  # 12.5% drawdown
        scale = drawdown_scale(equity, floor=0.6)
        assert 0.6 < scale < 1.0

    def test_severe_drawdown(self):
        from riq_analyzer import drawdown_scale
        equity = [120, 100, 90, 84]  # 30% drawdown
        scale = drawdown_scale(equity, floor=0.6)
        assert scale == 0.6  # At floor

    def test_empty_equity(self):
        from riq_analyzer import drawdown_scale
        scale = drawdown_scale([])
        assert scale == 1.0


class TestNaNHandling:
    """Test NaN value handling in features"""

    def test_fill_nan_with_median(self):
        # Simulate NaN handling from train_auto.py
        X = np.array([
            [1.0, 2.0, np.nan],
            [2.0, np.nan, 4.0],
            [3.0, 4.0, 5.0]
        ])

        # Fill NaN with column medians
        col_medians = np.nanmedian(X, axis=0)
        nan_mask = np.isnan(X)
        for col_idx in range(X.shape[1]):
            col_nan_mask = nan_mask[:, col_idx]
            if col_nan_mask.any():
                X[col_nan_mask, col_idx] = col_medians[col_idx]

        # Check no NaN remains
        assert not np.any(np.isnan(X))
        # Check median fill: col 1 median = 3.0, col 2 median = 4.5
        assert X[1, 1] == 3.0  # median of [2, 4]
        assert X[0, 2] == 4.5  # median of [4, 5]

    def test_all_nan_column_fallback(self):
        X = np.array([
            [1.0, np.nan],
            [2.0, np.nan],
            [3.0, np.nan]
        ])

        # Fill with column median (which is nan for all-nan column)
        col_medians = np.nanmedian(X, axis=0)
        nan_mask = np.isnan(X)
        for col_idx in range(X.shape[1]):
            col_nan_mask = nan_mask[:, col_idx]
            if col_nan_mask.any():
                X[col_nan_mask, col_idx] = col_medians[col_idx]

        # Fallback to 0 for remaining NaN
        X = np.nan_to_num(X, nan=0.0)

        assert not np.any(np.isnan(X))
        assert X[0, 1] == 0.0  # Fallback to 0


class TestIntegration:
    """Integration tests for full prediction pipeline"""

    def test_full_analysis_flow(self):
        from riq_analyzer import (
            project_stat,
            prop_win_probability,
            sample_beta_posterior,
            dynamic_fractional_kelly,
            risk_adjusted_elg,
            american_to_decimal,
            KellyConfig
        )

        # Simulate LeBron's points analysis
        recent_games = [28, 32, 25, 30, 27, 29, 31, 26, 28, 30]
        line = 26.5
        odds = -115

        # 1. Project stat
        mu, sigma = project_stat(recent_games, "points", pace_multiplier=1.02, defense_factor=0.98)
        assert 25 < mu < 35

        # 2. Calculate win probability
        p_win, z = prop_win_probability("points", recent_games, line, "over", mu, sigma)
        assert 0.5 < p_win < 0.9  # Should favor OVER

        # 3. Sample posterior
        p_samples = sample_beta_posterior(p_win, 80.0, n_samples=600)
        assert len(p_samples) == 600

        # 4. Calculate Kelly
        dec = american_to_decimal(odds)
        b = dec - 1.0
        cfg = KellyConfig()
        f, p_c, frac_k, p_mean = dynamic_fractional_kelly(p_samples, b, cfg)

        # Should recommend a bet
        if p_c > 1.0 / (1.0 + b):
            assert f > 0

        # 5. Calculate ELG
        if f > 0:
            elg = risk_adjusted_elg(p_samples, b, f)
            assert elg > -1.0  # Should be reasonable


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
