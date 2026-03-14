#!/usr/bin/env python3
"""
Live NBA Predictions for Today's Games

Fetches today's games and generates predictions for all player props.

Usage:
    python predict_today.py
    python predict_today.py --date 2025-11-07
    python predict_today.py --team LAL
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from nba_api.stats.endpoints import ScoreboardV2
from nba_api.stats.static import teams
import warnings
warnings.filterwarnings('ignore')

def load_models(models_dir='./models'):
    """Load all trained models."""
    models_dir = Path(models_dir)
    models = {}

    for prop in ['minutes', 'points', 'rebounds', 'assists', 'threes']:
        model_path = models_dir / f"{prop}_model.pkl"
        if model_path.exists():
            with open(model_path, 'rb') as f:
                models[prop] = pickle.load(f)
            print(f"✅ Loaded {prop} model")
        else:
            print(f"⚠️  {prop} model not found at {model_path}")

    return models

def get_todays_games(date=None):
    """Fetch today's NBA games using nba_api."""
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')

    print(f"\n📅 Fetching games for {date}...")

    try:
        scoreboard = ScoreboardV2(game_date=date)
        games_df = scoreboard.get_data_frames()[0]

        if games_df.empty:
            print(f"   No games scheduled for {date}")
            return pd.DataFrame()

        # Extract relevant info
        games_info = []
        for _, game in games_df.iterrows():
            games_info.append({
                'game_id': game['GAME_ID'],
                'home_team': game['HOME_TEAM_ID'],
                'away_team': game['VISITOR_TEAM_ID'],
                'home_team_abbr': game.get('HOME_TEAM_ABBREVIATION', ''),
                'away_team_abbr': game.get('VISITOR_TEAM_ABBREVIATION', ''),
                'game_time': game.get('GAME_STATUS_TEXT', 'TBD')
            })

        games_df_clean = pd.DataFrame(games_info)
        print(f"   Found {len(games_df_clean)} games")

        return games_df_clean

    except Exception as e:
        print(f"❌ Error fetching games: {e}")
        return pd.DataFrame()

def get_team_roster(team_id):
    """Get current roster for a team (placeholder - needs implementation)."""
    # This would need to fetch roster from nba_api
    # For now, return empty to show structure
    print(f"   ⚠️  Roster fetching not yet implemented for team {team_id}")
    return []

def prepare_features_for_player(player_name, team_abbr, opponent_abbr, is_home):
    """
    Prepare feature vector for a single player prediction.

    This is a PLACEHOLDER - you need to implement full feature engineering.

    In production, you'd:
    1. Load player's recent game history
    2. Calculate rolling stats (L3, L5, L10)
    3. Get team context features
    4. Get opponent matchup features
    5. Load Basketball Reference priors
    6. Calculate all the Phase 1-7 features
    """
    print(f"      ⚠️  Feature engineering not yet implemented for {player_name}")

    # Return None for now - needs full implementation
    return None

def predict_props_for_player(models, features, player_name):
    """Generate predictions for all props for one player."""
    if features is None:
        return None

    predictions = {
        'player': player_name
    }

    for prop_name, model in models.items():
        try:
            # Make prediction
            pred = model.predict(features)

            # Get uncertainty if available
            if hasattr(model, 'predict') and hasattr(model, 'sigma_model'):
                try:
                    pred_val, uncertainty = model.predict(features, return_uncertainty=True)
                    predictions[prop_name] = {
                        'prediction': float(pred_val[0]),
                        'uncertainty': float(uncertainty[0]),
                        'lower_bound': float(pred_val[0] - uncertainty[0]),
                        'upper_bound': float(pred_val[0] + uncertainty[0])
                    }
                except:
                    predictions[prop_name] = {
                        'prediction': float(pred[0]),
                        'uncertainty': None
                    }
            else:
                predictions[prop_name] = {
                    'prediction': float(pred[0]),
                    'uncertainty': None
                }

        except Exception as e:
            print(f"      Error predicting {prop_name} for {player_name}: {e}")
            predictions[prop_name] = None

    return predictions

def format_predictions_table(all_predictions):
    """Format predictions as a nice table."""
    if not all_predictions:
        print("\n   No predictions generated")
        return

    print(f"\n{'='*120}")
    print(f"{'Player':<25} {'PTS':<12} {'REB':<12} {'AST':<12} {'3PM':<12} {'MIN':<12}")
    print(f"{'-'*120}")

    for pred in all_predictions:
        player = pred['player'][:24]  # Truncate long names

        pts = pred.get('points', {}).get('prediction', 0)
        reb = pred.get('rebounds', {}).get('prediction', 0)
        ast = pred.get('assists', {}).get('prediction', 0)
        threes = pred.get('threes', {}).get('prediction', 0)
        mins = pred.get('minutes', {}).get('prediction', 0)

        print(f"{player:<25} {pts:<12.1f} {reb:<12.1f} {ast:<12.1f} {threes:<12.1f} {mins:<12.1f}")

    print(f"{'-'*120}\n")

def main():
    parser = argparse.ArgumentParser(description='Generate predictions for NBA games')
    parser.add_argument('--date', type=str, default=None,
                       help='Date to predict (YYYY-MM-DD). Default: today')
    parser.add_argument('--team', type=str, default=None,
                       help='Filter by team abbreviation (e.g., LAL, BOS)')
    parser.add_argument('--models-dir', type=str, default='./models',
                       help='Directory containing trained models')
    parser.add_argument('--output', type=str, default=None,
                       help='Save predictions to CSV file')
    args = parser.parse_args()

    print("\n" + "="*120)
    print("🏀 NBA LIVE PREDICTIONS")
    print("="*120)

    # Load models
    print("\n📦 Loading models...")
    models = load_models(args.models_dir)

    if not models:
        print("\n❌ No models found. Please ensure models are in ./models/ directory")
        print("   Download nba_models_trained.zip from Colab and extract it")
        return

    print(f"\n✅ Loaded {len(models)} models: {', '.join(models.keys())}")

    # Get today's games
    games = get_todays_games(args.date)

    if games.empty:
        print("\n❌ No games found for the specified date")
        return

    # Display games
    print(f"\n📊 Games Schedule:")
    print(f"{'─'*120}")
    for idx, game in games.iterrows():
        home = game['home_team_abbr']
        away = game['away_team_abbr']
        time = game['game_time']
        print(f"   {away} @ {home} - {time}")
    print(f"{'─'*120}")

    # Filter by team if specified
    if args.team:
        team_upper = args.team.upper()
        games = games[
            (games['home_team_abbr'] == team_upper) |
            (games['away_team_abbr'] == team_upper)
        ]
        if games.empty:
            print(f"\n❌ No games found for team {team_upper}")
            return
        print(f"\n🔍 Filtered to {len(games)} game(s) for {team_upper}")

    # Generate predictions
    print(f"\n🎯 Generating Predictions...")
    print(f"{'='*120}")

    all_predictions = []

    for _, game in games.iterrows():
        home_team = game['home_team_abbr']
        away_team = game['away_team_abbr']

        print(f"\n🏟️  {away_team} @ {home_team}")
        print(f"{'─'*120}")

        # This is where you'd:
        # 1. Get rosters for both teams
        # 2. For each player, prepare features
        # 3. Generate predictions
        # 4. Format output

        print(f"\n   ⚠️  IMPLEMENTATION NEEDED:")
        print(f"   This script is a template. To make it work, you need to:")
        print(f"   1. Implement get_team_roster() to fetch current rosters")
        print(f"   2. Implement prepare_features_for_player() with full feature engineering")
        print(f"   3. Load player history, team stats, opponent matchups")
        print(f"   4. Calculate all Phase 1-7 features used in training")
        print(f"\n   The models are ready - they just need properly formatted features!")

    # Save to file if requested
    if args.output and all_predictions:
        df = pd.DataFrame(all_predictions)
        df.to_csv(args.output, index=False)
        print(f"\n💾 Predictions saved to: {args.output}")

    print(f"\n{'='*120}")
    print("📝 NEXT STEPS TO MAKE THIS WORK:")
    print("="*120)
    print("""
1. Feature Engineering Pipeline:
   - Extract the feature engineering code from train_auto.py
   - Create a reusable function that takes player data and returns features
   - Ensure it matches the 56 features your model expects

2. Data Sources:
   - Use nba_api to get player game logs
   - Fetch team stats for context features
   - Load Basketball Reference priors for historical context

3. Real-time Data:
   - Get today's probable starters
   - Calculate recent form (L3, L5, L10 games)
   - Get opponent defensive ratings

4. Validation:
   - Test on historical games first
   - Compare predictions to actual results
   - Calibrate if needed

Example feature engineering:
    from train_auto import build_players_from_playerstats
    # Use same function that created training features
""")
    print("="*120 + "\n")

if __name__ == '__main__':
    main()
