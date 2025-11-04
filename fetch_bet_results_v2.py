"""
Fetch Bet Results - SIMPLIFIED (Using ScoreboardV2 + BoxScore)

More reliable approach:
1. Get scoreboard for each date
2. Match games by teams
3. Fetch box scores
4. Update predictions

This avoids the LeagueGameFinder complexity.
"""

import pickle
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import time

print("=" * 70)
print("FETCH BET RESULTS - Simplified Approach")
print("=" * 70)

# Load ledger
ledger_file = Path("bets_ledger.pkl")
if not ledger_file.exists():
    print("ERROR: bets_ledger.pkl not found!")
    exit(1)

with open(ledger_file, 'rb') as f:
    ledger_data = pickle.load(f)

if isinstance(ledger_data, dict) and 'bets' in ledger_data:
    bets = ledger_data['bets']
else:
    bets = ledger_data if isinstance(ledger_data, list) else [ledger_data]

print(f"\nTotal predictions: {len(bets):,}")

df = pd.DataFrame(bets)

# Find unsettled bets
df['game_datetime'] = pd.to_datetime(df['game_date'], utc=True)
now = pd.Timestamp.now(tz='UTC')
df['hours_ago'] = (now - df['game_datetime']).dt.total_seconds() / 3600

ready_to_settle = df[(df['settled'] == False) & (df['hours_ago'] > 3)].copy()

print(f"Unsettled: {(df['settled'] == False).sum():,}")
print(f"Ready to fetch: {len(ready_to_settle):,}")

if len(ready_to_settle) == 0:
    print("\nNo bets to settle!")
    exit(0)

# Group by date
ready_to_settle['game_date_only'] = pd.to_datetime(ready_to_settle['game_date']).dt.date
dates_to_fetch = ready_to_settle['game_date_only'].unique()

print(f"\nUnique dates to fetch: {len(dates_to_fetch)}")

# Import nba_api
try:
    from nba_api.stats.endpoints import scoreboardv2, boxscoretraditionalv2
except ImportError:
    print("\nERROR: nba_api not installed!")
    exit(1)

updates_made = 0
api_calls = 0

# Process each date
for game_date in sorted(dates_to_fetch):
    print(f"\n{'='*70}")
    print(f"DATE: {game_date}")
    print(f"{'='*70}")
    
    # Get scoreboard for this date
    try:
        # Format: YYYY-MM-DD -> MM/DD/YYYY
        date_str = game_date.strftime('%m/%d/%Y')
        
        scoreboard = scoreboardv2.ScoreboardV2(game_date=date_str)
        api_calls += 1
        time.sleep(0.6)
        
        games = scoreboard.get_data_frames()[0]  # GameHeader
        
        if len(games) == 0:
            print(f"  No games found for {game_date}")
            continue
        
        print(f"  Found {len(games)} games on this date")
        
        # Process each game
        for _, game_row in games.iterrows():
            game_id = game_row['GAME_ID']
            home_team = game_row.get('HOME_TEAM_NAME', '')
            away_team = game_row.get('VISITOR_TEAM_NAME', '')
            
            print(f"\n  Game: {away_team} @ {home_team}")
            
            # Fetch box score
            try:
                box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
                api_calls += 1
                time.sleep(0.6)
                
                player_stats = box.get_data_frames()[0]
                
                if len(player_stats) == 0:
                    print(f"    ⚠️  Empty box score")
                    continue
                
                print(f"    ✓ Box score: {len(player_stats)} players")
                
                # Match predictions for this game
                game_bets = ready_to_settle[
                    ready_to_settle['game_date_only'] == game_date
                ]
                
                # Try to match by team names in game string
                matched_bets = []
                for idx, bet in game_bets.iterrows():
                    game_str = bet['game'].lower()
                    if (home_team.lower() in game_str) or (away_team.lower() in game_str):
                        matched_bets.append((idx, bet))
                
                if len(matched_bets) == 0:
                    print(f"    No predictions matched")
                    continue
                
                print(f"    Matched {len(matched_bets)} predictions")
                
                # Update predictions
                for idx, bet in matched_bets:
                    # Find player in box score
                    player_match = player_stats[
                        player_stats['PLAYER_NAME'].str.lower() == bet['player'].lower()
                    ]
                    
                    if len(player_match) == 0:
                        continue
                    
                    player_row = player_match.iloc[0]
                    
                    # Get actual stat
                    stat_map = {
                        'points': 'PTS',
                        'assists': 'AST',
                        'rebounds': 'REB',
                        'threes': 'FG3M'
                    }
                    
                    nba_col = stat_map.get(bet['prop_type'])
                    if not nba_col or nba_col not in player_row:
                        continue
                    
                    actual_value = float(player_row[nba_col])
                    
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
                            print(f"      ✓ {bet['player']} {bet['prop_type']}: {actual_value} (line: {bet['line']}) - {'WON' if won else 'LOST'}")
                            break
                
            except Exception as e:
                print(f"    ERROR fetching box score: {str(e)[:50]}")
                continue
        
    except Exception as e:
        print(f"  ERROR fetching scoreboard: {str(e)[:50]}")
        continue

print(f"\n{'='*70}")
print(f"RESULTS")
print(f"{'='*70}")
print(f"  Dates processed: {len(dates_to_fetch)}")
print(f"  API calls: {api_calls}")
print(f"  Predictions updated: {updates_made}")

if updates_made > 0:
    # Save
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
    
    with open(ledger_file, 'wb') as f:
        pickle.dump(updated_ledger, f)
    
    print(f"\n  ✅ Ledger updated and backed up")
    print(f"\n  Run: python analyze_ledger.py")
else:
    print(f"\n  ⚠️  No updates made")

print(f"\n{'='*70}")
