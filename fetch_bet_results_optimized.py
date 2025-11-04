"""
Fetch Bet Results - OPTIMIZED (by game, not player)

This version:
1. Groups predictions by GAME (not player)
2. Fetches box score for entire game (1 API call)
3. Updates all predictions from that game
4. Much more efficient - 10-20 API calls instead of 50+

Run daily to update ledger!
"""

import pickle
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import time

print("=" * 70)
print("FETCH BET RESULTS - Optimized (by game)")
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

# Group by GAME and DATE (not player!)
unique_games = ready_to_settle.groupby(['game', 'game_date']).size()
print(f"\nUnique games to fetch: {len(unique_games)} (much better than per-player!)")

print("\nFetching box scores from NBA API...")
print("(This will be much faster...)")

# Import nba_api
try:
    from nba_api.stats.endpoints import boxscoretraditionalv2, leaguegamefinder
    from nba_api.stats.static import teams as nba_teams
except ImportError:
    print("\nERROR: nba_api not installed!")
    print("Install with: pip install nba-api")
    exit(1)

# Get all teams for lookup
all_teams = nba_teams.get_teams()
team_lookup = {t['full_name']: t['id'] for t in all_teams}

# Track updates
updates_made = 0
api_calls = 0
games_processed = 0

# Process each unique game
for (game_name, game_date), count in unique_games.items():
    if api_calls >= 100:  # Higher limit since we're doing fewer calls
        print(f"\nAPI limit reached ({api_calls} calls). Run again for more.")
        break
    
    print(f"\nGame: {game_name} ({game_date.split('T')[0]})...", end=" ")
    
    # Parse team names from game string (e.g., "Team A at Team B")
    if ' at ' in game_name:
        away_team, home_team = game_name.split(' at ')
    elif ' vs ' in game_name:
        home_team, away_team = game_name.split(' vs ')
    else:
        print("Can't parse teams")
        continue
    
    # Get team IDs
    home_team_id = team_lookup.get(home_team.strip())
    away_team_id = team_lookup.get(away_team.strip())
    
    if not home_team_id or not away_team_id:
        print(f"Team not found")
        continue
    
    try:
        # Find game ID using LeagueGameFinder
        game_date_obj = pd.to_datetime(game_date).date()
        
        # Search for games on this date for home team
        finder = leaguegamefinder.LeagueGameFinder(
            team_id_nullable=home_team_id,
            season_nullable='2025-26',
            season_type_nullable='Regular Season'
        )
        api_calls += 1
        
        games_found = finder.get_data_frames()[0]
        
        # Filter to exact date
        games_found['GAME_DATE_DT'] = pd.to_datetime(games_found['GAME_DATE']).dt.date
        game_match = games_found[games_found['GAME_DATE_DT'] == game_date_obj]
        
        if len(game_match) == 0:
            print("Game not found")
            continue
        
        game_id = game_match.iloc[0]['GAME_ID']
        
        # Fetch box score for this game
        boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
        api_calls += 1
        
        player_stats = boxscore.get_data_frames()[0]  # Player stats
        
        print(f"Found! ({len(player_stats)} players)")
        
        # Now update all predictions from this game
        game_bets = ready_to_settle[
            (ready_to_settle['game'] == game_name) & 
            (ready_to_settle['game_date'] == game_date)
        ]
        
        for idx, bet in game_bets.iterrows():
            # Find player in box score
            player_row = player_stats[player_stats['PLAYER_NAME'] == bet['player']]
            
            if len(player_row) == 0:
                continue
            
            player_row = player_row.iloc[0]
            
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
            
            actual_value = float(player_row[nba_col])
            
            # Determine if bet won
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
        
        games_processed += 1
        
        # Rate limiting (600 requests per minute = 1 per 0.1 sec)
        time.sleep(0.6)
        
    except Exception as e:
        print(f"ERROR: {e}")
        continue

print(f"\n\nResults:")
print(f"  Games processed: {games_processed}")
print(f"  API calls made: {api_calls}")
print(f"  Predictions updated: {updates_made}")
print(f"  Efficiency: {updates_made / api_calls:.1f} predictions per API call" if api_calls > 0 else "")

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
print("COMPLETE - Much more efficient!")
print("=" * 70)
