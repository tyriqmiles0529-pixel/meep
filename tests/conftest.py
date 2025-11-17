"""
Pytest configuration and fixtures for NBA prediction tests.
"""
import pytest
import sys
import os

# Add parent directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_game_data():
    """Sample game statistics for testing."""
    return {
        'points': [25, 28, 22, 30, 27, 29, 24, 31, 26, 28],
        'assists': [8, 10, 7, 9, 11, 8, 10, 9, 7, 8],
        'rebounds': [7, 9, 8, 10, 7, 8, 9, 11, 8, 9],
        'threes': [3, 4, 2, 5, 3, 4, 3, 2, 4, 3],
        'minutes': [36, 38, 34, 40, 37, 38, 35, 39, 36, 37]
    }


@pytest.fixture
def sample_odds():
    """Sample betting odds for testing."""
    return {
        'favorite': -150,
        'underdog': +130,
        'even': -110,
        'heavy_favorite': -250,
        'long_shot': +350
    }


@pytest.fixture
def kelly_config():
    """Standard Kelly configuration for testing."""
    from riq_analyzer import KellyConfig
    return KellyConfig(
        q_conservative=0.35,
        fk_low=0.25,
        fk_high=0.50,
        dd_scale=1.0
    )
