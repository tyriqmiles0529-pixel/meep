"""
Fetch Actual Results for Unsettled Bets

This script:
1. Loads unsettled predictions from bets_ledger.pkl
2. Fetches actual stats from nba_api
3. Updates the ledger with actuals and win/loss
4. Saves updated ledger

Run this daily to keep ledger up to date!
"""

import pickle
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import time

print("=" * 70)
print("FETCH BET RESULTS - Update Ledger with Actuals")
print("=" * 70)

# Load ledger
ledger_file = Path("bets_ledger.pkl")
if not ledger_file.exists():
    print("ERROR: bets_ledger.pkl not found!")
    exit(1)

with open(ledger_file, 'rb') as f:
    ledger_data = pickle.load(f)

# Handle ledger format
if isinstance(ledger_data, dict) and 'bets' in ledger_data:
    bets = ledger_data['bets']
else:
    bets = ledger_data if isinstance(ledger_data, list) else [ledger_data]

print(f"\nTotal predictions in ledger: {len(bets):,}")

# Convert to DataFrame
df = pd.DataFrame(bets)

# Find unsettled bets from past games
df['game_datetime'] = pd.to_datetime(df['game_date'], utc=True)
now = pd.Timestamp.now(tz='UTC')
df['hours_ago'] = (now - df['game_datetime']).dt.total_seconds() / 3600

# Bets that should be settled (game was >3 hours ago)
ready_to_settle = df[(df['settled'] == False) & (df['hours_ago'] > 3)].copy()

print(f"Unsettled bets: {(df['settled'] == False).sum():,}")
print(f"Ready to fetch results (game >3hrs ago): {len(ready_to_settle):,}")

if len(ready_to_settle) == 0:
    print("\nNo bets ready to settle. Check back after games finish!")
    exit(0)

# Group by player and game date for efficient API calls
games_to_fetch = ready_to_settle.groupby(['player', 'game_date']).size()
print(f"\nUnique player-games to fetch: {len(games_to_fetch)}")

print("\nFetching actual stats from NBA API...")
print("(This may take a few minutes...)")

# Import nba_api
try:
    from nba_api.stats.endpoints import playergamelog
    from nba_api.stats.static import players as nba_players
except ImportError:
    print("\nERROR: nba_api not installed!")
    print("Install with: pip install nba-api")
    exit(1)

# Get all NBA players
all_players = nba_players.get_players()
player_lookup = {p['full_name'].lower(): p['id'] for p in all_players}

# Track updates
updates_made = 0
api_calls = 0

# Process each unique player
for player_name in ready_to_settle['player'].unique():
    if api_calls >= 50:  # Rate limit
        print("\nRate limit reached (50 API calls). Run again later for more.")
        break
    
    print(f"\nFetching: {player_name}...", end=" ")
    
    # Find NBA player ID
    player_id = player_lookup.get(player_name.lower())
    if not player_id:
        print(f"NOT FOUND in NBA database")
        continue
    
    try:
        # Fetch 2025-26 season game log
        gamelog = playergamelog.PlayerGameLog(
            player_id=player_id,
            season='2025-26',
            season_type_all_star='Regular Season'
        )
        api_calls += 1
        
        games_df = gamelog.get_data_frames()[0]
        
        if games_df.empty:
            print("No games found")
            continue
        
        print(f"Found {len(games_df)} games")
        
        # Process each prediction for this player
        player_bets = ready_to_settle[ready_to_settle['player'] == player_name]
        
        for idx, bet in player_bets.iterrows():
            # Match game by date
            game_date = pd.to_datetime(bet['game_date']).date()
            
            # Find matching game (nba_api uses GAME_DATE column)
            matching_games = games_df[pd.to_datetime(games_df['GAME_DATE']).dt.date == game_date]
            
            if len(matching_games) == 0:
                continue
            
            game = matching_games.iloc[0]
            
            # Get actual stat
            stat_map = {
                'points': 'PTS',
                'assists': 'AST',
                'rebounds': 'REB',
                'threes': 'FG3M'
            }
            
            nba_col = stat_map.get(bet['prop_type'])
            if not nba_col:
                continue
            
            actual_value = float(game[nba_col])
            
            # Determine if bet won
            # Assuming 'pick' field indicates over/under
            # If pick is None, we need to infer from predicted_prob vs 50%
            if bet['predicted_prob'] > 0.5:
                # Model predicted OVER
                won = actual_value > bet['line']
            else:
                # Model predicted UNDER
                won = actual_value < bet['line']
            
            # Update the bet in the original list
            for i, original_bet in enumerate(bets):
                if original_bet['prop_id'] == bet['prop_id']:
                    bets[i]['actual'] = actual_value
                    bets[i]['won'] = won
                    bets[i]['settled'] = True
                    updates_made += 1
                    break
        
        # Rate limiting
        time.sleep(0.6)  # 100 requests per minute max
        
    except Exception as e:
        print(f"ERROR: {e}")
        continue

print(f"\n\nResults:")
print(f"  API calls made: {api_calls}")
print(f"  Predictions updated: {updates_made}")

if updates_made > 0:
    # Save updated ledger
    if isinstance(ledger_data, dict):
        ledger_data['bets'] = bets
        updated_ledger = ledger_data
    else:
        updated_ledger = bets
    
    # Backup old ledger
    backup_file = Path("bets_ledger_backup.pkl")
    if ledger_file.exists():
        import shutil
        shutil.copy(ledger_file, backup_file)
        print(f"\n  Backup saved: {backup_file}")
    
    # Save updated ledger
    with open(ledger_file, 'wb') as f:
        pickle.dump(updated_ledger, f)
    
    print(f"  Updated ledger saved: {ledger_file}")
    print(f"\n  Run analyze_ledger.py to see updated results!")
else:
    print(f"\n  No updates made - results may not be available yet")

print("\n" + "=" * 70)
print("COMPLETE")
print("=" * 70)
