"""
Fetch Bet Results - INCREMENTAL VERSION

This version:
- Tracks which players have already been fetched
- Skips already-processed players
- Continues where the last run left off
- Saves progress to avoid re-fetching same players
"""

import pickle
import pandas as pd
from datetime import datetime
from pathlib import Path
import time
import json

print("=" * 70)
print("FETCH BET RESULTS - Incremental (Skip Already Fetched)")
print("=" * 70)

# Load ledger
ledger_file = Path("bets_ledger.pkl")
if not ledger_file.exists():
    print("ERROR: bets_ledger.pkl not found!")
    raise FileNotFoundError("bets_ledger.pkl not found")

with open(ledger_file, 'rb') as f:
    ledger_data = pickle.load(f)

if isinstance(ledger_data, dict) and 'bets' in ledger_data:
    bets = ledger_data['bets']
else:
    bets = ledger_data if isinstance(ledger_data, list) else [ledger_data]

print(f"\nTotal predictions: {len(bets):,}")

df = pd.DataFrame(bets)

# Find unsettled bets from past games
df['game_datetime'] = pd.to_datetime(df['game_date'], utc=True)
now = pd.Timestamp.now(tz='UTC')
df['hours_ago'] = (now - df['game_datetime']).dt.total_seconds() / 3600

ready_to_settle = df[(df['settled'] == False) & (df['hours_ago'] > 3)].copy()

print(f"Unsettled: {(df['settled'] == False).sum():,}")
print(f"Ready to fetch: {len(ready_to_settle):,}")

if len(ready_to_settle) == 0:
    print("\nNo bets ready to settle!")
    pass  # Success

# Load/create fetch progress tracker
progress_file = Path("fetch_progress.json")
if progress_file.exists():
    with open(progress_file, 'r') as f:
        fetch_progress = json.load(f)
    fetched_players = set(fetch_progress.get('fetched_players', []))
    print(f"\nProgress file found: {len(fetched_players)} players already fetched")
else:
    fetched_players = set()
    print(f"\nNo progress file - starting fresh")

# Get unique players that need fetching
all_players_needed = ready_to_settle['player'].unique()
remaining_players = [p for p in all_players_needed if p not in fetched_players]

print(f"\nPlayers:")
print(f"  Total unique: {len(all_players_needed)}")
print(f"  Already fetched: {len(fetched_players)}")
print(f"  Remaining: {len(remaining_players)}")

if len(remaining_players) == 0:
    print("\n✅ All players already fetched!")
    print("   If you still have unsettled bets, they may be:")
    print("   - Future games (not played yet)")
    print("   - Players not in NBA database")
    pass  # Success

# Import nba_api
try:
    from nba_api.stats.endpoints import playergamelog
    from nba_api.stats.static import players as nba_players
    from player_name_mapping import find_player_id
except ImportError as e:
    print(f"\nERROR: {e}")
    print("Install with: pip install nba-api")
    raise RuntimeError("Script terminated")

# Get all NBA players
all_players = nba_players.get_players()

# Track updates
updates_made = 0
api_calls = 0
players_processed = 0

print(f"\nFetching up to 50 players (API limit)...")

# Process remaining players (up to 50 API calls)
for player_name in remaining_players:
    if api_calls >= 50:
        print("\nRate limit reached (50 API calls).")
        break
    
    print(f"\nFetching: {player_name}...", end=" ")
    
    # Find NBA player ID
    player_id = find_player_id(player_name, all_players)
    
    if not player_id:
        print(f"NOT FOUND")
        # Mark as fetched even if not found (don't retry forever)
        fetched_players.add(player_name)
        continue
    
    try:
        # Fetch 2025-26 season game log
        gamelog = playergamelog.PlayerGameLog(
            player_id=player_id,
            season='2025-26'
        )
        api_calls += 1
        time.sleep(0.6)
        
        games = gamelog.get_data_frames()[0]
        
        if len(games) == 0:
            print(f"No games found")
            fetched_players.add(player_name)
            continue
        
        print(f"Found {len(games)} games")
        
        # Parse game dates
        games['GAME_DATE_DT'] = pd.to_datetime(games['GAME_DATE']).dt.date
        
        # Update predictions for this player
        player_bets = ready_to_settle[ready_to_settle['player'] == player_name]
        
        for idx, bet in player_bets.iterrows():
            bet_date = pd.to_datetime(bet['game_date']).date()
            
            # Find matching game with FUZZY DATE (±1 day)
            # Sportsbook scheduled time != NBA official game date
            from datetime import timedelta
            min_date = bet_date - timedelta(days=1)
            max_date = bet_date + timedelta(days=1)
            
            game_match = games[
                (games['GAME_DATE_DT'] >= min_date) &
                (games['GAME_DATE_DT'] <= max_date)
            ]
            
            if len(game_match) == 0:
                continue
            
            # If multiple matches, take closest date
            if len(game_match) > 1:
                game_match = game_match.copy()
                game_match['date_diff'] = game_match['GAME_DATE_DT'].apply(lambda x: abs((x - bet_date).days))
                game_match = game_match.sort_values('date_diff').head(1)
            
            game_row = game_match.iloc[0]
            
            # Get actual stat
            stat_map = {
                'points': 'PTS',
                'assists': 'AST',
                'rebounds': 'REB',
                'threes': 'FG3M'
            }
            
            nba_col = stat_map.get(bet['prop_type'])
            if not nba_col or nba_col not in game_row:
                continue
            
            actual_value = float(game_row[nba_col])
            
            # Determine if won
            if bet['predicted_prob'] > 0.5:
                won = actual_value > bet['line']
            else:
                won = actual_value < bet['line']
            
            # Update in original list
            for i, original_bet in enumerate(bets):
                if original_bet['prop_id'] == bet['prop_id']:
                    bets[i]['actual'] = actual_value
                    bets[i]['won'] = won
                    bets[i]['settled'] = True
                    updates_made += 1
                    break
        
        # Mark player as fetched
        fetched_players.add(player_name)
        players_processed += 1
        
    except Exception as e:
        print(f"ERROR: {str(e)[:50]}")
        # Mark as fetched anyway (don't retry on errors)
        fetched_players.add(player_name)
        continue

print(f"\n\nResults:")
print(f"  API calls made: {api_calls}")
print(f"  Players processed: {players_processed}")
print(f"  Predictions updated: {updates_made}")

# Save progress
fetch_progress = {
    'fetched_players': list(fetched_players),
    'last_updated': datetime.now().isoformat(),
    'total_players_fetched': len(fetched_players),
    'total_predictions_updated': updates_made
}

with open(progress_file, 'w') as f:
    json.dump(fetch_progress, f, indent=2)

print(f"\n  Progress saved: {progress_file}")
print(f"  Players fetched so far: {len(fetched_players)}/{len(all_players_needed)}")

if updates_made > 0:
    # Save updated ledger
    if isinstance(ledger_data, dict):
        ledger_data['bets'] = bets
        updated_ledger = ledger_data
    else:
        updated_ledger = bets
    
    # Backup
    backup_file = Path("bets_ledger_backup.pkl")
    import shutil
    if ledger_file.exists():
        shutil.copy(ledger_file, backup_file)
        print(f"  Backup saved: {backup_file}")
    
    # Save
    with open(ledger_file, 'wb') as f:
        pickle.dump(updated_ledger, f)
    
    print(f"  Updated ledger saved: {ledger_file}")

if len(remaining_players) > players_processed:
    remaining_count = len(remaining_players) - players_processed
    print(f"\n⚠️  {remaining_count} players remaining - run script again to continue!")
else:
    print(f"\n✅ All available players fetched!")

print(f"\n  Run analyze_ledger.py to see updated results!")
print("=" * 70)
