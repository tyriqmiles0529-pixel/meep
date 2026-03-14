import pandas as pd
import requests
import time
import os
import json
from datetime import datetime

# --- CONFIGURATION ---
THE_ODDS_API_KEY = "5648e037727b48a34a679ac87a1d0edb"
THE_ODDS_API_HOST = "api.the-odds-api.com"
LEDGER_PATH = "paper_ledger.csv" 

def get_live_events():
    url = f"https://{THE_ODDS_API_HOST}/v4/sports/basketball_nba/events?apiKey={THE_ODDS_API_KEY}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            events = response.json()
            return events
        else:
            print(f"[ERROR] Failed to fetch events: {response.status_code}")
            return []
    except Exception as e:
        print(f"[ERROR] Exception: {e}")
        return []

def get_player_props_for_event(event_id):
    # Fetch Standard + Alternates to maximize chance of finding Lines
    markets = "player_points,player_rebounds,player_assists,player_threes,player_points_alternate,player_rebounds_alternate,player_assists_alternate,player_threes_alternate"
    url = f"https://{THE_ODDS_API_HOST}/v4/sports/basketball_nba/events/{event_id}/odds?apiKey={THE_ODDS_API_KEY}&regions=us&markets={markets}&oddsFormat=american"
    
    props_lookup = {} # Key: Player|Market -> {Line: Odds, 'Main': {Line, Odds}}
    
    market_map_rev = {
        'player_points': 'points', 'player_points_alternate': 'points',
        'player_rebounds': 'rebounds', 'player_rebounds_alternate': 'rebounds',
        'player_assists': 'assists', 'player_assists_alternate': 'assists',
        'player_threes': 'three_pointers', 'player_threes_alternate': 'three_pointers'
    }

    try:
        response = requests.get(url)
        if response.status_code != 200:
            return {}
            
        data = response.json()
        bookmakers = data.get('bookmakers', [])
        
        # Priority: FanDuel > DraftKings > Any
        selected_book = None
        for b in bookmakers:
            if b['key'] == 'fanduel': selected_book = b; break
        if not selected_book:
            for b in bookmakers: 
                if b['key'] == 'draftkings': selected_book = b; break
        if not selected_book and bookmakers: selected_book = bookmakers[0]
        
        if not selected_book: return {}
        
        for market in selected_book.get('markets', []):
            m_key = market['key']
            target = market_map_rev.get(m_key)
            if not target: continue
            
            for outcome in market.get('outcomes', []):
                player = outcome.get('description')
                label = outcome.get('name') # Over/Under
                line = outcome.get('point')
                odds = outcome.get('price')
                
                if not player or not line: continue
                
                key = f"{player}|{target}"
                if key not in props_lookup: props_lookup[key] = []
                
                props_lookup[key].append({
                    'side': label,
                    'line': line,
                    'odds': odds
                })
                
    except Exception as e:
        print(f"[WARN] Error fetching odds for {event_id}: {e}")
        
    return props_lookup

def update_clv():
    if not os.path.exists(LEDGER_PATH):
        print("No ledger found.")
        return

    print("Loading Ledger...")
    df = pd.read_csv(LEDGER_PATH)
    
    mask = df['Closing_Line'].isna() | (df['Closing_Line'] == '')
    active_rows = df[mask]
    
    if active_rows.empty:
        print("No pending bets to update CLV for.")
        return

    print(f"Found {len(active_rows)} bets needing CLV update.")
    
    # 2. Get Live Odds Snapshot
    print("Fetching Live Odds Snapshot...")
    events = get_live_events()
    odds_cache = {} 
    
    for evt in events:
        eid = evt['id']
        print(f"  Fetching props for {evt['away_team']} @ {evt['home_team']}...")
        props = get_player_props_for_event(eid)
        for k, v in props.items():
            if k not in odds_cache: odds_cache[k] = []
            odds_cache[k].extend(v)
        time.sleep(0.2) 

    # 3. Match and Update
    updates_count = 0
    
    for idx, row in active_rows.iterrows():
        raw_player = row['Player']
        market_type = row['Market']
        
        if market_type in ['Parlay', 'Round Robin', 'Prop']:
            # Handle Single Props and Parlays alike if we can parse legs
            # For 'Prop', Side is just one leg.
            # For 'Parlay', Side is "Leg | Leg"
            
            legs_str = row['Side']
            legs = legs_str.split(' | ')
            
            current_dec_odds = 1.0
            all_legs_found = True
            current_parlay_lines = []
            
            for leg in legs:
                try:
                    # Parsing varies by Single vs Parlay.
                    # Single might be "Kevin Porter (MIL) - points Over 14.5 (-110)"
                    # Parlay legs are cleaner in ledger usually? Need to check run_phase_i output.
                    
                    target = None
                    if "points" in leg: target = "points"
                    elif "rebounds" in leg: target = "rebounds"
                    elif "assists" in leg: target = "assists"
                    elif "three_pointers" in leg: target = "three_pointers"
                    
                    side = "Over" if "Over" in leg else "Under"
                    
                    if not target: 
                        all_legs_found = False
                        break
                        
                    p_name_clean = leg.split('(')[0].strip()
                    cache_key = f"{p_name_clean}|{target}"
                    
                    if cache_key not in odds_cache:
                        for k in odds_cache.keys():
                            if p_name_clean in k and target in k:
                                cache_key = k
                                break
                    
                    if cache_key in odds_cache:
                        options = odds_cache[cache_key]
                        same_side = [o for o in options if o['side'].lower() == side.lower()]
                        if same_side:
                            same_side.sort(key=lambda x: abs(x['odds'] + 110))
                            best_opt = same_side[0]
                            
                            o_val = best_opt['odds']
                            d_val = (1 + o_val/100) if o_val > 0 else (1 + 100/abs(o_val))
                            current_dec_odds *= d_val
                            current_parlay_lines.append(f"{best_opt['line']}@{best_opt['odds']}")
                        else:
                            all_legs_found = False
                    else:
                        all_legs_found = False
                        
                except Exception as e:
                    all_legs_found = False
            
            if all_legs_found:
                final_us = int((current_dec_odds - 1) * 100) if current_dec_odds >= 2.0 else int(-100 / (current_dec_odds - 1))
                df.at[idx, 'Closing_Odds'] = final_us
                df.at[idx, 'Closing_Line'] = " | ".join(current_parlay_lines)
                
                bet_odds = row['Odds']
                bet_dec = (1 + bet_odds/100) if bet_odds > 0 else (1 + 100/abs(bet_odds))
                clv_pct = (bet_dec / current_dec_odds) - 1
                df.at[idx, 'CLV_%'] = round(clv_pct * 100, 2)
                
                updates_count += 1

    if updates_count > 0:
        print(f"Updated CLV for {updates_count} bets.")
        df.to_csv(LEDGER_PATH, index=False)
    else:
        print("No matches found to update.")

if __name__ == "__main__":
    print("--- Paper Trading: CLV Tracker ---")
    update_clv()
