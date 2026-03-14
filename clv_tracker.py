import pandas as pd
import requests
import time
import os
import json
from datetime import datetime

# --- CONFIGURATION ---
THE_ODDS_API_KEY = "5648e037727b48a34a679ac87a1d0edb"
THE_ODDS_API_HOST = "api.the-odds-api.com"
LEDGER_PATH = "betting_ledger.csv"

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
    
    # 1. Identify active bets (No Outcome, or No Closing Line)
    # We focus on bets where we haven't locked a closing line yet.
    # We only update if the 'Run Date' is today? Or if the game is live?
    # Simplified Logic: If Closing_Line is NULL, try to fetch it.
    
    mask = df['Closing_Line'].isna() | (df['Closing_Line'] == '')
    active_rows = df[mask]
    
    if active_rows.empty:
        print("No pending bets to update CLV for.")
        return

    print(f"Found {len(active_rows)} bets needing CLV update.")
    
    # 2. Get Live Odds Snapshot
    print("Fetching Live Odds Snapshot...")
    events = get_live_events()
    odds_cache = {} # Player|Market -> List of lines
    
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
        # Ledger Format: Player like "Kevin Porter Jr. (MIL)" -> extract name
        raw_player = row['Player']
        
        # Handle Parlays - We can't easily track CLV for entire parlays yet without breaking them down.
        # Phase K.1 MVP: Track CLV for Singles? 
        # But Phase J only logs Parlays.
        # To track Parlay CLV, we need to re-price the *entire parlay*.
        # The ledger stores the legs in 'Side' column for parlays. 
        # "Leg 1 | Leg 2 | Leg 3"
        
        market_type = row['Market']
        
        if market_type == 'Parlay':
            # Complex: Need to parse legs, find odds for each, re-calculate parlay odds.
            # Legs format: "Kevin Porter Jr. (MIL) - points Over 14.5 (-280.0)<br>..." (from report)
            # Ledger 'Side' format: "Isaiah Collier (UTA) (points Over @ -114.0) | ... "
            
            legs_str = row['Side']
            legs = legs_str.split(' | ')
            
            current_dec_odds = 1.0
            all_legs_found = True
            
            # Debug
            # print(f"Checking Parlay: {legs_str[:50]}...")
            
            new_prob_accum = 1.0
            
            current_parlay_lines = []
            
            for leg in legs:
                # Parse: "Player Name (TEAM) (target Side @ Odds)"
                try:
                    # Very rough parsing
                    part1, part2 = leg.split('(')
                    p_name = part1.strip() # "Isaiah Collier"
                    
                    # rest: "UTA) (points Over @ -114.0)"
                    # We need target and side.
                    # This is brittle. Robust parsing needed.
                    # Let's try to extract from inside the last parenthesis
                    
                    # Normalized name check
                    # We have odds_cache keys: "Name|points"
                    
                    # Let's iterate cache keys to find partial match for player name
                    found_leg_odds = None
                    
                    # Extract target from string: "points", "rebounds", "assists", "three_pointers"
                    target = None
                    if "points" in leg: target = "points"
                    elif "rebounds" in leg: target = "rebounds"
                    elif "assists" in leg: target = "assists"
                    elif "three_pointers" in leg: target = "three_pointers"
                    
                    side = "Over" if "Over" in leg else "Under"
                    
                    if not target: 
                        all_legs_found = False
                        break
                        
                    # Find player in cache
                    # Cache Key: "Isaiah Collier|points"
                    # We need to handle "Name (Team)" in ledger vs "Name" in Odds API
                    # The Ledger P_Name has (Team).
                    p_name_clean = leg.split('(')[0].strip()
                    
                    cache_key = f"{p_name_clean}|{target}"
                    
                    # Fallback search if exact key missing
                    if cache_key not in odds_cache:
                        # try simple name match
                        for k in odds_cache.keys():
                            if p_name_clean in k and target in k:
                                cache_key = k
                                break
                    
                    if cache_key in odds_cache:
                        # Found player props. Look for matching line or closest line.
                        # We want the *main* line odds, or the *exact* line odds?
                        # CLV usually compares to the widely available line.
                        # Let's grab the best odds for the SAME side.
                        
                        options = odds_cache[cache_key] # List of dicts {side, line, odds}
                        
                        # Just grab the first one matching 'side' as the "Current Market Price"?
                        # Ideally we find the Line closest to 50/50 (-110) as the "Main Line"
                        
                        # Filter by Side
                        same_side = [o for o in options if o['side'].lower() == side.lower()]
                        if same_side:
                            # Heuristic: Pick the one with odds closest to -110 as "Main"
                            # Or just pick the one with the *same line*?
                            
                            # 1. Try exact line match
                            # Need to extract original line from string? "14.5"
                            # Ledger doesn't store legs line explicitly in columns, simpler to just get "Closing Odds of Parlay" 
                            
                            # Let's just assume the first available "Main" line is the market.
                            # Sort by distance to -110
                            same_side.sort(key=lambda x: abs(x['odds'] + 110))
                            best_opt = same_side[0]
                            
                            o_val = best_opt['odds']
                            d_val = (1 + o_val/100) if o_val > 0 else (1 + 100/abs(o_val))
                            current_dec_odds *= d_val
                            
                            # We construct a string for Closing Lines?
                            current_parlay_lines.append(f"{best_opt['line']}@{best_opt['odds']}")
                        else:
                            all_legs_found = False
                    else:
                        all_legs_found = False
                        
                except Exception as e:
                    # print(f"Parse error: {e}")
                    all_legs_found = False
            
            if all_legs_found:
                # Calc US Odds
                final_us = int((current_dec_odds - 1) * 100) if current_dec_odds >= 2.0 else int(-100 / (current_dec_odds - 1))
                
                df.at[idx, 'Closing_Odds'] = final_us
                df.at[idx, 'Closing_Line'] = " | ".join(current_parlay_lines)
                
                # CLV % = (Current Dec / Mn - 1) ? 
                # Simplest: Closing Odds vs Bet Odds
                # If I bet +1000 and it closes +800, good.
                # If I bet +1000 and it closes +1200, bad.
                
                bet_odds = row['Odds']
                # Decimal conv
                bet_dec = (1 + bet_odds/100) if bet_odds > 0 else (1 + 100/abs(bet_odds))
                
                # CLV Ratio = (Bet Dec / Closing Dec) - 1 ?
                # If Bet 2.0, Close 1.8 -> 2.0/1.8 = 1.11 (+11% CLV)
                # If Bet 2.0, Close 2.2 -> 2.0/2.2 = 0.90 (-10% CLV)
                
                clv_pct = (bet_dec / current_dec_odds) - 1
                df.at[idx, 'CLV_%'] = round(clv_pct * 100, 2)
                
                updates_count += 1

    if updates_count > 0:
        print(f"Updated CLV for {updates_count} bets.")
        df.to_csv(LEDGER_PATH, index=False)
    else:
        print("No matches found to update.")

if __name__ == "__main__":
    print("--- Phase K.1: CLV Tracker ---")
    update_clv()
