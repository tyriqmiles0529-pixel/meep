import pickle
import pandas as pd
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players as nba_players

# Load ledger
with open('bets_ledger.pkl', 'rb') as f:
    ledger = pickle.load(f)

bets = ledger['bets'] if isinstance(ledger, dict) else ledger
df = pd.DataFrame(bets)

# Get one unsettled player
unsettled = df[df['settled'] == False]
test_player = 'Jalen Brunson'
player_bets = unsettled[unsettled['player'] == test_player]

print(f'Testing player: {test_player}')
print(f'Predictions in ledger: {len(player_bets)}')
print()
print('Dates in ledger for this player:')
for idx, bet in player_bets.head(5).iterrows():
    bet_date = pd.to_datetime(bet['game_date']).date()
    print(f'  {bet_date} - {bet["prop_type"]} {bet["line"]}')

# Fetch player from API
all_players = nba_players.get_players()
player_obj = [p for p in all_players if p['full_name'] == test_player]

if player_obj:
    player_id = player_obj[0]['id']
    print(f'\nFetching from NBA API (player_id: {player_id})...')
    gamelog = playergamelog.PlayerGameLog(player_id=player_id, season='2025-26')
    games = gamelog.get_data_frames()[0]
    
    print(f'\nGames from NBA API ({len(games)} total):')
    if len(games) > 0:
        games['GAME_DATE_DT'] = pd.to_datetime(games['GAME_DATE']).dt.date
        for idx, game in games.head(10).iterrows():
            print(f'  {game["GAME_DATE_DT"]} - {game["MATCHUP"]} - {game["PTS"]} PTS, {game["AST"]} AST, {game["REB"]} REB')
        
        # Check for matches with FUZZY DATE (±1 day)
        print('\nChecking for date matches (with ±1 day fuzzy matching):')
        for idx, bet in player_bets.head(5).iterrows():
            bet_date = pd.to_datetime(bet['game_date']).date()
            
            # Exact match
            exact_match = games[games['GAME_DATE_DT'] == bet_date]
            
            # Fuzzy match (±1 day)
            fuzzy_match = games[
                (games['GAME_DATE_DT'] >= bet_date - pd.Timedelta(days=1)) &
                (games['GAME_DATE_DT'] <= bet_date + pd.Timedelta(days=1))
            ]
            
            if len(exact_match) > 0:
                print(f'  ✅ EXACT MATCH: {bet_date}')
            elif len(fuzzy_match) > 0:
                actual_date = fuzzy_match.iloc[0]['GAME_DATE_DT']
                print(f'  ⚠️  FUZZY MATCH: {bet_date} → actual game on {actual_date}')
            else:
                print(f'  ❌ NO MATCH: {bet_date}')
    else:
        print('  No games found!')
else:
    print('Player not found in NBA database')
