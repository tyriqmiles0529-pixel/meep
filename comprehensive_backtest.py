"""
Comprehensive Backtesting Suite for NBA Prediction Models

Tests both game predictions (moneyline, spread, totals) and player props
against historical data to identify model weaknesses and improvement areas.

Usage:
    python comprehensive_backtest.py --start-date 2024-01-01 --end-date 2024-06-30 --output backtest_results.json
"""

import os
import sys
import argparse
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

# ============================================================================
# CONFIGURATION
# ============================================================================

MODELS_DIR = Path("models")
MODEL_CACHE_DIR = Path("model_cache")
DATA_CACHE_DIR = Path("data")

# Player models
PLAYER_MODELS = {
    'points': MODELS_DIR / "points_model.pkl",
    'assists': MODELS_DIR / "assists_model.pkl",
    'rebounds': MODELS_DIR / "rebounds_model.pkl",
    'threes': MODELS_DIR / "threes_model.pkl",
    'minutes': MODELS_DIR / "minutes_model.pkl",
}

PLAYER_SIGMA_MODELS = {
    'points': MODELS_DIR / "points_sigma_model.pkl",
    'assists': MODELS_DIR / "assists_sigma_model.pkl",
    'rebounds': MODELS_DIR / "rebounds_sigma_model.pkl",
    'threes': MODELS_DIR / "threes_sigma_model.pkl",
}

# Game models
GAME_MODELS = {
    'moneyline': MODELS_DIR / "moneyline_model.pkl",
    'spread': MODELS_DIR / "spread_model.pkl",
}

# Metadata
TRAINING_METADATA = MODELS_DIR / "training_metadata.json"
SPREAD_SIGMA = MODELS_DIR / "spread_sigma.json"

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_models() -> Dict:
    """Load all trained models."""
    models = {
        'player': {},
        'player_sigma': {},
        'game': {},
        'metadata': {}
    }

    # Load player models
    for stat, path in PLAYER_MODELS.items():
        if path.exists():
            with open(path, 'rb') as f:
                models['player'][stat] = pickle.load(f)
            print(f"✓ Loaded {stat} model")
        else:
            print(f"⚠ Missing {stat} model at {path}")

    # Load player sigma models
    for stat, path in PLAYER_SIGMA_MODELS.items():
        if path.exists():
            with open(path, 'rb') as f:
                models['player_sigma'][stat] = pickle.load(f)
            print(f"✓ Loaded {stat} sigma model")

    # Load game models
    for model_type, path in GAME_MODELS.items():
        if path.exists():
            with open(path, 'rb') as f:
                models['game'][model_type] = pickle.load(f)
            print(f"✓ Loaded {model_type} model")

    # Load metadata
    if TRAINING_METADATA.exists():
        with open(TRAINING_METADATA, 'r') as f:
            models['metadata'] = json.load(f)
        print(f"✓ Loaded training metadata")

    if SPREAD_SIGMA.exists():
        with open(SPREAD_SIGMA, 'r') as f:
            models['spread_sigma'] = json.load(f)
        print(f"✓ Loaded spread sigma")

    return models


# ============================================================================
# BACKTESTING METRICS
# ============================================================================

class BacktestMetrics:
    """Calculate comprehensive backtesting metrics."""

    @staticmethod
    def player_prop_metrics(predictions: np.ndarray, actuals: np.ndarray,
                           market_lines: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate player prop prediction metrics.

        Args:
            predictions: Model predictions
            actuals: Actual values
            market_lines: Optional market lines for over/under accuracy

        Returns:
            Dict with RMSE, MAE, bias, accuracy, etc.
        """
        valid_mask = ~(np.isnan(predictions) | np.isnan(actuals))
        pred_valid = predictions[valid_mask]
        actual_valid = actuals[valid_mask]

        if len(pred_valid) == 0:
            return {'error': 'No valid predictions'}

        errors = pred_valid - actual_valid

        metrics = {
            'n_samples': len(pred_valid),
            'rmse': np.sqrt(np.mean(errors ** 2)),
            'mae': np.mean(np.abs(errors)),
            'bias': np.mean(errors),
            'median_error': np.median(errors),
            'std_error': np.std(errors),
            'max_underpredict': np.min(errors),
            'max_overpredict': np.max(errors),
            'r_squared': 1 - (np.sum(errors ** 2) / np.sum((actual_valid - np.mean(actual_valid)) ** 2))
        }

        # Over/Under accuracy if market lines provided
        if market_lines is not None:
            market_valid = market_lines[valid_mask]
            valid_market = ~np.isnan(market_valid)

            if np.any(valid_market):
                pred_over = pred_valid[valid_market] > market_valid[valid_market]
                actual_over = actual_valid[valid_market] > market_valid[valid_market]

                metrics['over_under_accuracy'] = np.mean(pred_over == actual_over)
                metrics['n_market_samples'] = np.sum(valid_market)

        return metrics

    @staticmethod
    def game_outcome_metrics(predicted_probs: np.ndarray, actuals: np.ndarray,
                            market_probs: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate game outcome prediction metrics.

        Args:
            predicted_probs: Model win probabilities
            actuals: Actual outcomes (0 or 1)
            market_probs: Optional market implied probabilities

        Returns:
            Dict with log loss, Brier score, accuracy, ROI, etc.
        """
        valid_mask = ~(np.isnan(predicted_probs) | np.isnan(actuals))
        pred_valid = predicted_probs[valid_mask]
        actual_valid = actuals[valid_mask]

        if len(pred_valid) == 0:
            return {'error': 'No valid predictions'}

        # Clip probabilities to avoid log(0)
        pred_clipped = np.clip(pred_valid, 1e-10, 1 - 1e-10)

        metrics = {
            'n_samples': len(pred_valid),
            'log_loss': -np.mean(actual_valid * np.log(pred_clipped) +
                                (1 - actual_valid) * np.log(1 - pred_clipped)),
            'brier_score': np.mean((pred_valid - actual_valid) ** 2),
            'accuracy': np.mean((pred_valid > 0.5) == actual_valid),
            'mean_confidence': np.mean(np.abs(pred_valid - 0.5)),
            'calibration_error': np.mean(pred_valid - actual_valid)
        }

        # Kelly Criterion ROI simulation if market probs provided
        if market_probs is not None:
            market_valid = market_probs[valid_mask]
            valid_market = ~np.isnan(market_valid)

            if np.any(valid_market):
                pred_m = pred_valid[valid_market]
                actual_m = actual_valid[valid_market]
                market_m = market_valid[valid_market]

                # Find profitable edges (pred > market by 5%+)
                edge_mask = (pred_m - market_m) > 0.05

                if np.any(edge_mask):
                    # Kelly bet sizing
                    edge = pred_m[edge_mask] - market_m[edge_mask]
                    odds = 1 / market_m[edge_mask]
                    kelly_fraction = edge / (odds - 1)
                    kelly_fraction = np.clip(kelly_fraction, 0, 0.25)  # Max 25% bankroll

                    # Simulate unit bets
                    wins = actual_m[edge_mask] == 1
                    unit_returns = np.where(wins, odds - 1, -1) * kelly_fraction

                    metrics['edge_bets'] = np.sum(edge_mask)
                    metrics['edge_accuracy'] = np.mean(actual_m[edge_mask])
                    metrics['roi_percent'] = np.sum(unit_returns) / np.sum(kelly_fraction) * 100
                    metrics['total_return_units'] = np.sum(unit_returns)

        return metrics

    @staticmethod
    def spread_metrics(predicted_spreads: np.ndarray, actual_margins: np.ndarray,
                      market_spreads: Optional[np.ndarray] = None) -> Dict:
        """Calculate spread prediction metrics."""
        valid_mask = ~(np.isnan(predicted_spreads) | np.isnan(actual_margins))
        pred_valid = predicted_spreads[valid_mask]
        actual_valid = actual_margins[valid_mask]

        if len(pred_valid) == 0:
            return {'error': 'No valid predictions'}

        errors = pred_valid - actual_valid

        metrics = {
            'n_samples': len(pred_valid),
            'rmse': np.sqrt(np.mean(errors ** 2)),
            'mae': np.mean(np.abs(errors)),
            'bias': np.mean(errors),
            'median_error': np.median(errors)
        }

        # ATS (against the spread) accuracy if market spreads provided
        if market_spreads is not None:
            market_valid = market_spreads[valid_mask]
            valid_market = ~np.isnan(market_valid)

            if np.any(valid_market):
                pred_ats = pred_valid[valid_market] > market_valid[valid_market]
                actual_ats = actual_valid[valid_market] > market_valid[valid_market]

                metrics['ats_accuracy'] = np.mean(pred_ats == actual_ats)
                metrics['n_market_samples'] = np.sum(valid_market)

        return metrics


# ============================================================================
# DATA LOADING
# ============================================================================

def load_historical_games(start_season: int, end_season: int) -> pd.DataFrame:
    """
    Load historical game data from Kaggle dataset.

    Args:
        start_season: Starting season end year (e.g., 2020 for 2019-20 season)
        end_season: Ending season end year (e.g., 2024 for 2023-24 season)

    Returns DataFrame with columns:
    - gameId, date, season_end_year
    - home_team, away_team
    - home_score, away_score
    - home_win (0/1)
    - margin (home_score - away_score)
    """
    print(f"\nLoading historical games for seasons {start_season-1}-{start_season} to {end_season-1}-{end_season}...")

    # Try to load from cached Kaggle data
    kaggle_cache = Path.home() / ".cache" / "kagglehub" / "datasets" / "eoinamoore" / \
                   "historical-nba-data-and-player-box-scores" / "versions" / "258"

    games_path = kaggle_cache / "Games.csv"
    team_stats_path = kaggle_cache / "TeamStatistics.csv"

    if not games_path.exists() or not team_stats_path.exists():
        print(f"⚠ Kaggle data not found at {kaggle_cache}")
        print("Run train_auto.py first to download data")
        return pd.DataFrame()

    # Load Games.csv
    games = pd.read_csv(games_path, low_memory=False)

    # Extract season from gameId (first 3 digits = season, e.g., 224 = 2024-25)
    games['gameId_str'] = games['gameId'].astype(str)
    games['season_prefix'] = games['gameId_str'].str[:3].astype(int)
    games['season_end_year'] = 2000 + (games['season_prefix'] % 100)

    # Filter by season range
    games_filtered = games[
        (games['season_end_year'] >= start_season) &
        (games['season_end_year'] <= end_season)
    ].copy()

    # Add derived columns
    games_filtered['home_win'] = (games_filtered['homeScore'] > games_filtered['awayScore']).astype(int)
    games_filtered['margin'] = games_filtered['homeScore'] - games_filtered['awayScore']

    print(f"Found {len(games_filtered):,} games across {games_filtered['season_end_year'].nunique()} seasons")
    print(f"Season range: {games_filtered['season_end_year'].min()}-{games_filtered['season_end_year'].max()}")

    return games_filtered


def load_historical_player_stats(start_season: int, end_season: int) -> pd.DataFrame:
    """
    Load historical player game stats.

    Args:
        start_season: Starting season end year (e.g., 2020)
        end_season: Ending season end year (e.g., 2024)

    Returns DataFrame with columns:
    - gameId, date, playerId, playerName
    - points, rebounds, assists, threes, minutes
    """
    print(f"\nLoading historical player stats for seasons {start_season-1}-{start_season} to {end_season-1}-{end_season}...")

    kaggle_cache = Path.home() / ".cache" / "kagglehub" / "datasets" / "eoinamoore" / \
                   "historical-nba-data-and-player-box-scores" / "versions" / "258"

    player_stats_path = kaggle_cache / "PlayerStatistics.csv"

    if not player_stats_path.exists():
        print(f"⚠ Player stats not found at {kaggle_cache}")
        return pd.DataFrame()

    # Load PlayerStatistics.csv
    player_stats = pd.read_csv(player_stats_path, low_memory=False)

    # Extract season from gameId
    if 'gameId' in player_stats.columns:
        player_stats['gameId_str'] = player_stats['gameId'].astype(str)
        player_stats['season_prefix'] = player_stats['gameId_str'].str[:3].astype(int)
        player_stats['season_end_year'] = 2000 + (player_stats['season_prefix'] % 100)

        player_stats_filtered = player_stats[
            (player_stats['season_end_year'] >= start_season) &
            (player_stats['season_end_year'] <= end_season)
        ].copy()

        print(f"Found {len(player_stats_filtered):,} player-game records across {player_stats_filtered['season_end_year'].nunique()} seasons")

        return player_stats_filtered
    else:
        print("⚠ No gameId column found in PlayerStatistics.csv")
        return pd.DataFrame()


# ============================================================================
# BACKTESTING FUNCTIONS
# ============================================================================

def backtest_player_props(models: Dict, player_stats: pd.DataFrame,
                         verbose: bool = True) -> Dict:
    """
    Backtest player prop predictions using simple rolling average baseline.

    Since full feature engineering from train_auto.py is complex,
    we use rolling averages as a baseline comparison.

    Returns dict with metrics for each stat type.
    """
    print("\n" + "="*70)
    print("BACKTESTING PLAYER PROPS (Rolling Average Baseline)")
    print("="*70)

    results = {}

    # Column mapping
    stat_col_map = {
        'points': 'points',
        'rebounds': 'reboundsTotal',
        'assists': 'assists',
        'threes': 'threePointersMade',
        'minutes': 'numMinutes'
    }

    for stat_name in ['points', 'rebounds', 'assists', 'threes', 'minutes']:
        if stat_name not in models['player']:
            print(f"⚠ No model for {stat_name}, skipping")
            continue

        print(f"\nBacktesting {stat_name.upper()}...")

        actual_col = stat_col_map.get(stat_name)
        if actual_col not in player_stats.columns:
            print(f"⚠ Column {actual_col} not found, skipping {stat_name}")
            continue

        # Sort by player and date for rolling calculations
        df = player_stats.copy()
        df = df.sort_values(['personId', 'gameDate'])

        # Calculate rolling average (last 10 games)
        df['rolling_avg'] = df.groupby('personId')[actual_col].transform(
            lambda x: x.shift(1).rolling(window=10, min_periods=3).mean()
        )

        # Also calculate season average up to current game
        df['season_avg'] = df.groupby(['personId', 'season_end_year'])[actual_col].transform(
            lambda x: x.shift(1).expanding(min_periods=3).mean()
        )

        # Use whichever is available (prefer rolling, fallback to season avg)
        df['prediction'] = df['rolling_avg'].fillna(df['season_avg'])

        # Get valid predictions
        valid_mask = df['prediction'].notna() & df[actual_col].notna()
        predictions = df.loc[valid_mask, 'prediction'].values
        actuals = df.loc[valid_mask, actual_col].values

        if len(predictions) == 0:
            print(f"⚠ No valid predictions for {stat_name}")
            continue

        # Calculate metrics
        metrics = BacktestMetrics.player_prop_metrics(predictions, actuals)

        # Add baseline metadata
        metrics['method'] = 'rolling_10game_average'
        metrics['min_games'] = 3

        results[stat_name] = metrics

        if verbose:
            print(f"\n{stat_name.upper()} Metrics (Rolling Avg Baseline):")
            print(f"  Samples: {metrics.get('n_samples', 0):,}")
            print(f"  RMSE: {metrics.get('rmse', 0):.2f}")
            print(f"  MAE: {metrics.get('mae', 0):.2f}")
            print(f"  Bias: {metrics.get('bias', 0):.2f} ({'over' if metrics.get('bias', 0) > 0 else 'under'}-prediction)")
            print(f"  R²: {metrics.get('r_squared', 0):.3f}")
            print(f"  Std Error: {metrics.get('std_error', 0):.2f}")

            # Expected model performance from training metadata
            if 'metadata' in models and 'player' in models['metadata']:
                model_meta = models['metadata']['player'].get(stat_name, {})
                expected_rmse = model_meta.get('rmse')
                if expected_rmse:
                    improvement = ((metrics.get('rmse', 0) - expected_rmse) / metrics.get('rmse', 1)) * 100
                    print(f"  Expected model RMSE: {expected_rmse:.2f} ({improvement:+.1f}% vs baseline)")

    return results


def backtest_game_predictions(models: Dict, games: pd.DataFrame,
                              verbose: bool = True) -> Dict:
    """
    Backtest game outcome predictions using rolling team stats baseline.

    Simpler approach:
    - Team win % last 10 games
    - Home/away adjustment
    - Point differential trends

    Returns dict with moneyline and spread metrics.
    """
    print("\n" + "="*70)
    print("BACKTESTING GAME PREDICTIONS (Team Win % Baseline)")
    print("="*70)

    results = {}

    if games.empty:
        print("⚠ No games data")
        return results

    # Ensure required columns exist
    required_cols = ['hometeamId', 'awayteamId', 'homeScore', 'awayScore', 'home_win']
    missing = [c for c in required_cols if c not in games.columns]
    if missing:
        print(f"⚠ Missing columns: {missing}")
        return results

    # Sort by date
    if 'gameDate' in games.columns:
        games = games.sort_values('gameDate').reset_index(drop=True)

    # Calculate rolling team stats
    print("\nCalculating rolling team statistics...")

    team_stats = defaultdict(lambda: {
        'wins': [],
        'point_diffs': [],
        'home_wins': [],
        'away_wins': []
    })

    predictions = []
    actuals = []
    predicted_spreads = []
    actual_spreads = []

    for idx, game in games.iterrows():
        home_team = game['hometeamId']
        away_team = game['awayteamId']

        # Get team stats (last 10 games)
        home_recent = team_stats[home_team]
        away_recent = team_stats[away_team]

        # Calculate win rates
        home_winrate = np.mean(home_recent['wins'][-10:]) if home_recent['wins'] else 0.5
        away_winrate = np.mean(away_recent['wins'][-10:]) if away_recent['wins'] else 0.5

        # Home court advantage (~5% boost)
        home_boost = 0.05

        # Predicted home win probability (simple logistic function)
        winrate_diff = home_winrate - away_winrate + home_boost
        home_win_prob = 1 / (1 + np.exp(-5 * winrate_diff))  # Scale to sigmoid

        # Predicted spread (point differential)
        home_avg_diff = np.mean(home_recent['point_diffs'][-10:]) if home_recent['point_diffs'] else 0
        away_avg_diff = np.mean(away_recent['point_diffs'][-10:]) if away_recent['point_diffs'] else 0
        predicted_spread = home_avg_diff - away_avg_diff + 3  # +3 home advantage

        # Actual outcome
        actual_home_win = game['home_win']
        actual_spread = game['margin']

        predictions.append(home_win_prob)
        actuals.append(actual_home_win)
        predicted_spreads.append(predicted_spread)
        actual_spreads.append(actual_spread)

        # Update team stats after game
        home_won = game['homeScore'] > game['awayScore']
        away_won = game['awayScore'] > game['homeScore']
        point_diff = game['homeScore'] - game['awayScore']

        team_stats[home_team]['wins'].append(1 if home_won else 0)
        team_stats[home_team]['point_diffs'].append(point_diff)
        team_stats[home_team]['home_wins'].append(1 if home_won else 0)

        team_stats[away_team]['wins'].append(1 if away_won else 0)
        team_stats[away_team]['point_diffs'].append(-point_diff)
        team_stats[away_team]['away_wins'].append(1 if away_won else 0)

    # Calculate moneyline metrics
    print("\n" + "="*70)
    print("MONEYLINE METRICS (Baseline)")
    print("="*70)

    ml_metrics = BacktestMetrics.game_outcome_metrics(
        np.array(predictions),
        np.array(actuals)
    )

    if verbose:
        print(f"  Samples: {ml_metrics.get('n_samples', 0):,}")
        print(f"  Accuracy: {ml_metrics.get('accuracy', 0):.3f}")
        print(f"  Log Loss: {ml_metrics.get('log_loss', 0):.4f}")
        print(f"  Brier Score: {ml_metrics.get('brier_score', 0):.4f}")

    results['moneyline'] = ml_metrics

    # Calculate spread metrics
    print("\n" + "="*70)
    print("SPREAD METRICS (Baseline)")
    print("="*70)

    spread_metrics = BacktestMetrics.spread_metrics(
        np.array(predicted_spreads),
        np.array(actual_spreads)
    )

    if verbose:
        print(f"  Samples: {spread_metrics.get('n_samples', 0):,}")
        print(f"  RMSE: {spread_metrics.get('rmse', 0):.2f} points")
        print(f"  MAE: {spread_metrics.get('mae', 0):.2f} points")
        print(f"  Bias: {spread_metrics.get('bias', 0):.2f} points")

    results['spread'] = spread_metrics

    # Compare to model expectations
    if 'metadata' in models and 'game_metrics' in models['metadata']:
        game_meta = models['metadata']['game_metrics']

        print("\n" + "="*70)
        print("COMPARISON TO TRAINED MODELS")
        print("="*70)

        # Moneyline
        expected_acc = game_meta.get('ensemble', {}).get('final_accuracy')
        expected_logloss = game_meta.get('ensemble', {}).get('final_logloss')

        if expected_acc:
            print(f"\nMoneyline:")
            print(f"  Baseline accuracy: {ml_metrics.get('accuracy', 0):.3f}")
            print(f"  Model accuracy: {expected_acc:.3f}")
            improvement = (expected_acc - ml_metrics.get('accuracy', 0)) / ml_metrics.get('accuracy', 1) * 100
            print(f"  Improvement: {improvement:+.1f}%")

        if expected_logloss:
            print(f"\n  Baseline log loss: {ml_metrics.get('log_loss', 0):.4f}")
            print(f"  Model log loss: {expected_logloss:.4f}")

        # Spread
        expected_rmse = game_meta.get('sp_rmse')
        expected_mae = game_meta.get('sp_mae')

        if expected_rmse:
            print(f"\nSpread:")
            print(f"  Baseline RMSE: {spread_metrics.get('rmse', 0):.2f}")
            print(f"  Model RMSE: {expected_rmse:.2f}")
            improvement = (spread_metrics.get('rmse', 0) - expected_rmse) / spread_metrics.get('rmse', 1) * 100
            print(f"  Improvement: {improvement:+.1f}%")

    return results


# ============================================================================
# ANALYSIS & RECOMMENDATIONS
# ============================================================================

def analyze_weaknesses(backtest_results: Dict) -> Dict:
    """Analyze backtesting results to identify model weaknesses."""
    print("\n" + "="*70)
    print("WEAKNESS ANALYSIS")
    print("="*70)

    weaknesses = []
    recommendations = []

    # Analyze player props
    if 'player_props' in backtest_results:
        for stat, metrics in backtest_results['player_props'].items():
            if 'error' in metrics:
                weaknesses.append(f"{stat}: {metrics['error']}")
                continue

            # High RMSE
            if metrics.get('rmse', 0) > 5.0:
                weaknesses.append(f"{stat}: High RMSE ({metrics['rmse']:.2f})")
                recommendations.append(f"Consider adding more features for {stat} prediction")

            # High bias (systematic over/under prediction)
            if abs(metrics.get('bias', 0)) > 1.0:
                direction = "over" if metrics['bias'] > 0 else "under"
                weaknesses.append(f"{stat}: Systematic {direction}-prediction (bias={metrics['bias']:.2f})")
                recommendations.append(f"Calibrate {stat} model to reduce bias")

            # Low R²
            if metrics.get('r_squared', 0) < 0.5:
                weaknesses.append(f"{stat}: Low predictive power (R²={metrics['r_squared']:.3f})")
                recommendations.append(f"Re-engineer features for {stat} to improve correlation")

    # Analyze game predictions
    if 'game_predictions' in backtest_results:
        # TODO: Add game prediction weakness analysis
        pass

    return {
        'weaknesses': weaknesses,
        'recommendations': recommendations
    }


def generate_report(backtest_results: Dict, analysis: Dict, output_path: str):
    """Generate comprehensive backtest report."""
    report = {
        'timestamp': datetime.now().isoformat(),
        'backtest_results': backtest_results,
        'weakness_analysis': analysis,
        'summary': {
            'n_weaknesses': len(analysis['weaknesses']),
            'n_recommendations': len(analysis['recommendations'])
        }
    }

    # Save JSON report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Report saved to {output_path}")

    # Print summary
    print("\n" + "="*70)
    print("BACKTEST SUMMARY")
    print("="*70)

    print(f"\n🔍 Identified {len(analysis['weaknesses'])} weaknesses:")
    for weakness in analysis['weaknesses']:
        print(f"  • {weakness}")

    print(f"\n💡 {len(analysis['recommendations'])} recommendations:")
    for rec in analysis['recommendations']:
        print(f"  • {rec}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Comprehensive NBA model backtesting")
    parser.add_argument("--start-season", type=int, default=2020,
                       help="Start season end year (e.g., 2020 for 2019-20 season)")
    parser.add_argument("--end-season", type=int, default=2024,
                       help="End season end year (e.g., 2024 for 2023-24 season)")
    parser.add_argument("--output", type=str, default="backtest_results.json",
                       help="Output file for results")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")

    args = parser.parse_args()

    print("="*70)
    print("NBA PREDICTION MODEL BACKTESTING")
    print("="*70)
    print(f"Season range: {args.start_season-1}-{args.start_season} to {args.end_season-1}-{args.end_season}")
    print(f"Output: {args.output}")

    # Load models
    print("\n" + "="*70)
    print("LOADING MODELS")
    print("="*70)
    models = load_models()

    # Load historical data
    games = load_historical_games(args.start_season, args.end_season)
    player_stats = load_historical_player_stats(args.start_season, args.end_season)

    if games.empty and player_stats.empty:
        print("\n❌ No historical data found. Run train_auto.py first.")
        return

    # Run backtests
    backtest_results = {}

    if not player_stats.empty:
        backtest_results['player_props'] = backtest_player_props(
            models, player_stats, verbose=args.verbose
        )

    if not games.empty:
        backtest_results['game_predictions'] = backtest_game_predictions(
            models, games, verbose=args.verbose
        )

    # Analyze weaknesses
    analysis = analyze_weaknesses(backtest_results)

    # Generate report
    generate_report(backtest_results, analysis, args.output)

    print("\n✅ Backtesting complete!")


if __name__ == "__main__":
    main()
