
import pandas as pd
from datetime import datetime
import time
import os
import re

# Use nba_api to fetch actual results
from nba_api.stats.endpoints import playergamelogs
from nba_api.stats.static import players as nba_players

LEDGER_PATH = "betting_ledger.csv"

def find_player_id(player_name, all_players):
    import difflib
    name_lower = player_name.lower().strip()
    for player in all_players:
        if player['full_name'].lower() == name_lower:
            return player['id']
    player_names = [p['full_name'] for p in all_players]
    matches = difflib.get_close_matches(player_name, player_names, n=1, cutoff=0.8)
    if matches:
        for player in all_players:
            if player['full_name'] == matches[0]:
                return player['id']
    return None

def grade_ledger():
    if not os.path.exists(LEDGER_PATH):
        print("No ledger found.")
        return

    print(f"Loading {LEDGER_PATH}...")
    df = pd.read_csv(LEDGER_PATH)
    
    # Identify pending bets
    # Outcome column might be empty string or NaN
    pending = df[df['Outcome'].isna() | (df['Outcome'] == '')]
    if pending.empty:
        print("No pending bets to grade.")
        print_summary(df)
        return

    print(f"Found {len(pending)} pending bets.")
    
    all_players_ref = nba_players.get_players()
    
    # Cache player logs to avoid spamming API
    # Key: PlayerID, Value: DataFrame of game logs
    gamelog_cache = {}
    
    updates = 0
    
    unique_players = pending['Player'].unique()
    
    print("Fetching player logs for pending bets from local espn cache...")
    
    csv_path = "data/standardized_espn_logs.csv"
    if not os.path.exists(csv_path):
        print(f"Missing {csv_path}, cannot grade.")
        return
        
    league_logs_df = pd.read_csv(csv_path)
    
    # Map ESPN CSV columns to expected format
    # player_name -> PLAYER_NAME
    # date -> GAME_DATE_DT
    # points -> PTS
    # assists -> AST
    # reboundsTotal -> REB
    # three_pointers -> FG3M
    league_logs_df = league_logs_df.rename(columns={
        'player_name': 'PLAYER_NAME',
        'points': 'PTS',
        'assists': 'AST',
        'reboundsTotal': 'REB',
        'three_pointers': 'FG3M'
    })
    league_logs_df['GAME_DATE_DT'] = pd.to_datetime(league_logs_df['date']).dt.date
    
    # Pre-fetch cache for all unique players
    for player_name in unique_players:
        if "Parlay" in player_name or "Round Robin" in player_name or "Lotto" in player_name or "MULTI" in player_name:
            continue
            
        clean_name = re.sub(r'\s*\([A-Z]{2,4}\)$', '', player_name).strip()
        p_logs = league_logs_df[league_logs_df['PLAYER_NAME'].str.lower() == clean_name.lower()]
        if not p_logs.empty:
            gamelog_cache[player_name] = p_logs
            
    def fetch_player_logs(p_name_raw):
        clean_name = re.sub(r'\s*\([A-Z]{2,4}\)$', '', p_name_raw).strip()
        p_logs = league_logs_df[league_logs_df['PLAYER_NAME'].str.lower() == clean_name.lower()]
        return p_logs if not p_logs.empty else None
            
    # Grade rows
    for idx, row in pending.iterrows():
        p_name = row['Player']
        is_parlay = "Parlay" in p_name or "RR" in p_name or "Lotto" in p_name or "MULTI" in p_name or row['Market'] == "Parlay"
        
        try:
            run_date_str = row['Run Date']
            run_date = datetime.strptime(run_date_str, "%m.%d.%y").date()
        except:
            continue
            
        def get_game_stat(player, r_date):
            if player not in gamelog_cache: return None
            games = gamelog_cache[player]
            if games.empty: return None
            from datetime import timedelta
            match = games[games['GAME_DATE_DT'] == r_date]
            if match.empty: match = games[games['GAME_DATE_DT'] == (r_date + timedelta(days=1))]
            if match.empty: return None
            return match.iloc[0]

        if not is_parlay:
            if p_name not in gamelog_cache: continue
            game_stat = get_game_stat(p_name, run_date)
            if game_stat is None: continue
            
            side_str = str(row['Side']).lower()
            market_str = str(row['Market']).lower()
            target = None
            if "points" in market_str or "points" in side_str: target = 'PTS'
            elif "rebounds" in market_str or "rebounds" in side_str: target = 'REB'
            elif "assists" in market_str or "assists" in side_str: target = 'AST'
            elif "three_pointers" in market_str or "three_pointers" in side_str or "3pt" in market_str: target = 'FG3M'
            if not target: continue
            
            leg_line = float(row['Line']) if pd.notna(row['Line']) else 0.0
            if leg_line == 0.0:
                match = re.search(r'(over|under).+?([\d.]+)', side_str)
                if match: leg_line = float(match.group(2))
                else: continue
                    
            direction = "Over" if "over" in side_str else ("Under" if "under" in side_str else "Over")
            actual = float(game_stat[target])
            
            outcome = "Push"
            if direction == "Over":
                if actual > leg_line: outcome = "Win"
                elif actual < leg_line: outcome = "Loss"
            elif direction == "Under":
                if actual < leg_line: outcome = "Win"
                elif actual > leg_line: outcome = "Loss"
                
            df.at[idx, 'Outcome'] = outcome
            updates += 1
            print(f"  Graded {p_name}: {target} {direction} {leg_line} vs Act {actual} -> {outcome}")
            
        else:
            # Grade Parlay
            side_str = str(row['Side'])
            legs = [leg.strip() for leg in side_str.split('|')]
            all_won = True
            any_loss = False
            
            for leg in legs:
                match_leg = re.search(r'^(.*?)\s*\([A-Z]{2,4}\)?\s*\((.*?)\s+(Over|Under)\b', leg, re.I)
                if not match_leg: match_leg = re.search(r'^(.*?)\s*\([A-Z]{2,4}\)?\s*-\s*(.*?)\s+(Over|Under)', leg, re.I)
                if not match_leg: 
                    all_won = False; break
                    
                leg_player_raw = match_leg.group(1).strip()
                leg_target_str = match_leg.group(2).strip().lower()
                leg_dir = match_leg.group(3).capitalize()
                
                target = None
                if "points" in leg_target_str: target = 'PTS'
                elif "rebounds" in leg_target_str: target = 'REB'
                elif "assists" in leg_target_str: target = 'AST'
                elif "three_pointers" in leg_target_str or "3pt" in leg_target_str: target = 'FG3M'
                if not target: all_won = False; break
                
                # Fetch Line from single in ledger
                same_day = df[df['Run Date'] == run_date_str]
                # Try exact player or starting with
                single_match = same_day[(same_day['Player'].str.startswith(leg_player_raw)) & (same_day['Market'].str.lower() == leg_target_str.replace(" ", "_"))]
                if single_match.empty:
                    # fallback fuzzy target
                    single_match = same_day[(same_day['Player'].str.startswith(leg_player_raw)) & (same_day['Side'].str.lower().str.contains(leg_target_str[:4]))]
                
                if single_match.empty:
                    # try to extract line from side string if available (dash format)
                    match_line = re.search(r'(Over|Under)\s+([\d.]+)', leg, re.I)
                    if match_line:
                        leg_line = float(match_line.group(2))
                    else:
                        all_won = False
                        break
                else:
                    leg_line = float(single_match.iloc[0]['Line'])

                game_stat = get_game_stat(leg_player_raw, run_date)
                if game_stat is None:
                    # Player might not be in cache (fuzzy team name issues handled below) -> fallback
                    clean_name = re.sub(r'\s*\([A-Z]{2,4}\)$', '', leg_player_raw).strip()
                    # We might have not cached this player if they were ONLY in parlays!
                    df_logs = fetch_player_logs(leg_player_raw)
                    if df_logs is not None and not df_logs.empty:
                        gamelog_cache[leg_player_raw] = df_logs
                        game_stat = get_game_stat(leg_player_raw, run_date)
                        
                if game_stat is None: all_won = False; break
                
                actual = float(game_stat[target])
                if leg_dir == "Over" and actual < leg_line: any_loss = True; break
                if leg_dir == "Under" and actual > leg_line: any_loss = True; break
                # Push effectively means not a win yet not a loss, we simplify (loss of parlay edge usually, or reduced payout)
                if (leg_dir == "Over" and actual == leg_line) or (leg_dir == "Under" and actual == leg_line):
                    pass # Push logic: often reduces odds, but we won't count as loss
                
            if any_loss:
                df.at[idx, 'Outcome'] = "Loss"
                updates += 1
                print(f"  Graded Parlay {p_name} ({len(legs)} legs) -> Loss")
            elif all_won:
                df.at[idx, 'Outcome'] = "Win"
                updates += 1
                print(f"  Graded Parlay {p_name} ({len(legs)} legs) -> Win")

    if updates > 0:
        print(f"Saving {updates} graded bets...")
        df.to_csv(LEDGER_PATH, index=False)
        print_summary(df)
    else:
        print("No matches found to grade.")

def print_summary(df):
    graded = df[df['Outcome'].isin(['Win', 'Loss'])]
    if graded.empty:
        print("No graded bets.")
        return
        
    wins = len(graded[graded['Outcome'] == 'Win'])
    losses = len(graded[graded['Outcome'] == 'Loss'])
    total = wins + losses
    rate = wins / total if total > 0 else 0
    
    print("\n=== PAPER TRADING SUMMARY ===")
    print(f"Total Graded: {total}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Win Rate: {rate:.1%}")
    
    # Calculate Unit PNL
    # Stake * (Odds/100) if +Odds...
    # Simplified PnL
    pnl = 0.0
    for _, row in graded.iterrows():
        stake = row['Stake Size']
        odds = row['Odds']
        if row['Outcome'] == 'Win':
            if odds > 0:
                profit = stake * (odds / 100)
            else:
                profit = stake * (100 / abs(odds))
            pnl += profit
        else:
            pnl -= stake
            
    print(f"Total PnL: {pnl:.2f} Units")
    print("=============================")

if __name__ == "__main__":
    grade_ledger()
