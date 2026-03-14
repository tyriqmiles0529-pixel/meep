import pandas as pd
import os
import sys
import requests
import json
import time
from datetime import datetime
from betting_strategy import BettingStrategy

# --- CONFIGURATION ---
# Mode Toggle: 'live' or 'simulated'
ODDS_MODE = os.getenv("ODDS_MODE", "live") 
THE_ODDS_API_KEY = "feb98d2672d0505df5dbb1cfa8d06ccd"
THE_ODDS_API_HOST = "api.the-odds-api.com"

# Paths
PREDICTIONS_PATH = "predictions/live_ensemble_2025.csv"
SIM_ODDS_PATH = "historical_data/odds_2024-12-15.csv" 
LEDGER_PATH = "betting_ledger.csv"

# J.9: Global Strategy Parameters
BANKROLL = 1000.0
KELLY_FRACTION = 0.25 # User risk setting

def get_live_events():
    """Fetch today's NBA events from The Odds API."""
    print("  Fetching today's events from The Odds API...")
    url = f"https://{THE_ODDS_API_HOST}/v4/sports/basketball_nba/events?apiKey={THE_ODDS_API_KEY}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"  [ERROR] Failed to fetch events: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        print(f"  [ERROR] Exception fetching events: {e}")
        return []

def get_live_player_odds(event_id):
    """Fetch player prop odds (including alternates) for a specific event."""
    # J.9: Expanded to include alternate markets for heavy-favorite discovery
    standard_markets = "player_points,player_rebounds,player_assists,player_threes"
    alt_markets = "player_points_alternate,player_rebounds_alternate,player_assists_alternate,player_threes_alternate"
    markets = f"{standard_markets},{alt_markets}"
    
    url = f"https://{THE_ODDS_API_HOST}/v4/sports/basketball_nba/events/{event_id}/odds?apiKey={THE_ODDS_API_KEY}&regions=us&markets={markets}&oddsFormat=american"
    
    all_props = []
    market_map = {
        'player_points': 'points',
        'player_rebounds': 'rebounds',
        'player_assists': 'assists',
        'player_threes': 'three_pointers',
        'player_points_alternate': 'points',
        'player_rebounds_alternate': 'rebounds',
        'player_assists_alternate': 'assists',
        'player_threes_alternate': 'three_pointers'
    }

    try:
        time.sleep(0.2) # Light rate limiting
        response = requests.get(url)
        if response.status_code != 200:
            print(f"    [WARN] Props failed for event {event_id}: {response.status_code}")
            return []
            
        data = response.json()
        bookmakers = data.get('bookmakers', [])
        
        # J.5 DIRECTIVE: FanDuel = PRIMARY, DraftKings = SECONDARY
        selected_book = None
        for b in bookmakers:
            if b['key'] == 'fanduel':
                selected_book = b
                break
        if not selected_book:
            for b in bookmakers:
                if b['key'] == 'draftkings':
                    selected_book = b
                    break
        if not selected_book and bookmakers:
            selected_book = bookmakers[0]
            
        if not selected_book:
            return []

        for market in selected_book.get('markets', []):
            m_key = market['key']
            m_name = market_map.get(m_key)
            if not m_name:
                continue
                
            # The Odds API outcomes for props: Over/Under
            # Outcomes are paired by player
            player_outcomes = {}
            for outcome in market.get('outcomes', []):
                p_name = outcome.get('description')
                if p_name not in player_outcomes:
                    player_outcomes[p_name] = {}
                
                label = outcome.get('name') # 'Over' or 'Under'
                player_outcomes[p_name][label] = {
                    'price': outcome.get('price'),
                    'point': outcome.get('point')
                }
            
            for p_name, sides in player_outcomes.items():
                over = sides.get('Over')
                under = sides.get('Under')
                if over and over.get('point'):
                    all_props.append({
                        'player_name': p_name,
                        'event_id': event_id,
                        'market': m_name,
                        'line': over['point'],
                        'odds_over': over['price'],
                        'odds_under': under['price'] if under else None,
                        'bookmaker': selected_book['key']
                    })
    except Exception as e:
        print(f"    [WARN] Exception for event {event_id}: {e}")
        
    return all_props

def fetch_live_odds_pivoted():
    """Fetch live odds and return as a list of betting opportunities."""
    events = get_live_events()
    if not events:
        print("  [ERROR] No live events found.")
        return []

    print(f"  Found {len(events)} games. Fetching props...")
    all_props = []
    for event in events:
        eid = event['id']
        home = event['home_team']
        away = event['away_team']
        print(f"    Processing {away} @ {home}...")
        all_props.extend(get_live_player_odds(eid))

    if not all_props:
        return []
    
    # Normalize for merging
    for p in all_props:
        p['player_name_key'] = p['player_name'].lower().strip()
        
    return all_props

def run():
    print(f"--- Phase I: Production Readiness ({ODDS_MODE.upper()} MODE) ---")
    print(f"Main Provider: The Odds API")
    
    # 1. Load Predictions
    if not os.path.exists(PREDICTIONS_PATH):
        print(f"[FAIL] Predictions missing: {PREDICTIONS_PATH}")
        return
        
    print(f"Loading predictions from {PREDICTIONS_PATH}...")
    preds = pd.read_csv(PREDICTIONS_PATH)
    
    # Update col_map to match PRODUCTION V4 SCHEMA
    col_map = {
        'player': 'player_name', 
        'proj_PTS': 'pred_points', 
        'proj_AST': 'pred_assists', 
        'proj_REB': 'pred_rebounds',
        'proj_FG3M': 'pred_three_pointers'
    }
    preds = preds.rename(columns=col_map)
    preds['player_name_key'] = preds['player_name'].str.lower().str.strip()
    
    # 1.5 Eligibility Filter
    ELIGIBILITY_PATH = "data/eligibility_lookup.csv"
    if os.path.exists(ELIGIBILITY_PATH):
        print(f"Filtering by eligibility (>= 20 MPG) from {ELIGIBILITY_PATH}...")
        lookup = pd.read_csv(ELIGIBILITY_PATH)
        # Use today's date or the date from preds if available
        # Assuming predictions are for "today"
        today_str = datetime.now().strftime("%Y-%m-%d")
        
        # Check if lookup has today's date, otherwise fallback or warn
        if 'game_date_str' in lookup.columns:
            # Robust Live Logic: Get latest eligibility status for each player <= Today
            lookup['game_date'] = pd.to_datetime(lookup['game_date_str'])
            today_dt = pd.to_datetime(today_str)
            
            # Filter for dates in the past/today
            valid_lookup = lookup[lookup['game_date'] <= today_dt].copy()
            
            if not valid_lookup.empty:
                # Sort by date desc to get latest first
                valid_lookup = valid_lookup.sort_values('game_date', ascending=False)
                # Drop duplicates to keep most recent status
                latest_status = valid_lookup.drop_duplicates(subset=['PLAYER_NAME'])
                
                eligible_players = latest_status[latest_status['eligible'] == True]['PLAYER_NAME'].tolist()
                eligible_keys = [p.lower().strip() for p in eligible_players]
                
                initial_count = len(preds)
                preds = preds[preds['player_name_key'].isin(eligible_keys)]
                print(f"  Eligible players (Latest Status): {len(preds)} (Dropped {initial_count - len(preds)})")
            else:
                 print(f"[WARN] No past eligibility data found relative to {today_str}.")
    else:
        print("[WARN] Eligibility lookup missing. Proceeding with all players.")

    # 2. Get Odds
    if ODDS_MODE == "live":
        print(f"Fetching LIVE odds (Standard + Alternates)...")
        odds_list = fetch_live_odds_pivoted()
        odds_source = "The Odds API Live"
    else:
        odds_list = [] # Simulator not updated for this normalized format yet
        odds_source = "Simulated"

    if not odds_list:
        print("[FAIL] No odds available. Execution halted.")
        return

    # 3. Merge & Bet
    print("Merging predictions and odds...")
    odds_df = pd.DataFrame(odds_list)
    
    # Identify common columns to avoid suffixes or handle them
    # Predictions has 'player_name', Odds has 'player_name'. 
    # We join on 'player_name_key'.
    merged = pd.merge(preds, odds_df, on='player_name_key', how='inner', suffixes=('', '_odds'))
    
    # Ensure 'player_name' is present (it will be from 'preds' due to empty suffix)
    print(f"Generated {len(merged)} betting candidate lines.")
    
    bs = BettingStrategy(load_models=False)

    # J.9: Using global BANKROLL and KELLY_FRACTION
    bets = bs.generate_bets(merged, bankroll=BANKROLL, confidence_threshold=10, kelly_fraction=KELLY_FRACTION, min_ev=0.05)
    
    if bets.empty:
        print("[INFO] No qualifying bets found today.")
        # We'll still generate the report header though
        qualified_bets = 0
    else:
        qualified_bets = len(bets)

    # 4. Generate Outputs
    short_date = datetime.now().strftime('%m.%d.%y')
    md_filename = f"{short_date} - Riq's Picks.md"
    
    # J.9: select_top_props now handles favorites filtering
    top_props_df = bs.select_top_props(bets, n=7) if qualified_bets > 0 else pd.DataFrame()
    # J.9: Targeted parlay construction with bankroll allocation - Use Top Props ONLY per user request
    parlay_results = bs.generate_optimal_targeted_parlays(top_props_df, bankroll=BANKROLL, kelly_fraction=KELLY_FRACTION) if not top_props_df.empty else {'rr': [], 'traditional': []}

    print(f"Generating Report: {md_filename}")
    with open(md_filename, 'w', encoding='utf-8') as f:
        f.write(f"# Riq's Picks — {short_date} (Phase J.9)\n\n")
        f.write(f"**Primary Book:** FanDuel | **Run Time:** {datetime.now().strftime('%H:%M:%S ET')}\n\n")
        
        # Team List
        f.write("## Teams Playing Today\n")
        live_events = get_live_events() if ODDS_MODE == "live" else []
        if live_events:
            for ev in live_events:
                f.write(f"- {ev['away_team']} @ {ev['home_team']}\n")
        else:
            f.write("See event details below.\n")
        f.write("\n")

        f.write("## Top Favorites (Pool for Parlay Construction)\n")
        f.write("> **Constraint:** Favorites (-500 to -200) only. EV >= 5%. Parlays capped at +700.\n\n")
        
        if top_props_df.empty:
            f.write("No qualifying heavy-favorite single bets found today.\n")
        else:
            for target in bs.targets:
                df_t = top_props_df[top_props_df['target'] == target]
                if not df_t.empty:
                    f.write(f"### {target.replace('_', ' ').title()}\n")
                    f.write("| Player | Model | Side | Line | Odds | Prob | EV | Stake |\n")
                    f.write("|---|---|---|---|---|---|---|---|\n")
                    for _, row in df_t.iterrows():
                        o_str = f"{row['odds']}" # Known to be negative per J.9 filter
                        # J.9: Display the calculated stake_amt instead of hardcoded $0.00
                        f.write(f"| {row['player']} ({row.get('team', 'UNK')}) | {row['prediction']:.1f} | {row['side']} | {row['line']} | {o_str} | {row['win_prob']:.1%} | {row['ev']:.2f} | **${row['stake_amt']:.2f}** |\n")
                    f.write("\n")
            
        # J.9: Round Robin Section
        f.write(f"## Round Robin Parlays (Targets: +100 to +500)\n")
        rrs = parlay_results['rr']
        if rrs:
            rr_df = pd.DataFrame(rrs)
            for t in sorted(rr_df['type'].unique(), reverse=True): # 4-leg then 3-leg
                sub = rr_df[rr_df['type'] == t]
                if not sub.empty:
                    f.write(f"### {t}\n")
                    f.write("| Legs | Odds | Prob | EV | Stake |\n")
                    f.write("|---|---|---|---|---|\n")
                    for _, row in sub.iterrows():
                        l_str = "<br>".join(row['legs'])
                        o_str = f"+{row['combined_odds']}" if row['combined_odds'] > 0 else f"{row['combined_odds']}"
                        f.write(f"| {l_str} | {o_str} | {row['combined_prob']:.1%} | {row['ev']:.2f} | **${row['stake_amt']:.2f}** |\n")
                    f.write("\n")
        else:
            f.write("No qualifying Round Robin combinations found.\n\n")

        # J.9: Traditional Parlay Section
        f.write(f"## Traditional Parlays (Target Scale: +100 → +500)\n")
        ptrad = parlay_results['traditional']
        if ptrad:
            f.write("| Type | Legs | Odds | Prob | EV | Stake |\n")
            f.write("|---|---|---|---|---|---|\n")
            for row in ptrad:
                l_str = "<br>".join(row['legs'])
                o_str = f"+{row['combined_odds']}" if row['combined_odds'] > 0 else f"{row['combined_odds']}"
                f.write(f"| {row['type']} | {l_str} | {o_str} | {row['combined_prob']:.1%} | {row['ev']:.2f} | **${row['stake_amt']:.2f}** |\n")
            f.write("\n")
        else:
            f.write("No qualifying Traditional Parlays found.\n\n")

    # 5. Update Ledger (J.9: PARLAYS ONLY)
    if qualified_bets > 0:
        print(f"Updating Ledger: {LEDGER_PATH}")
        ledger_rows = []
        
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        # J.9: NEW - Include top singles in ledger too
        for _, row in top_props_df.iterrows():
            o = row['odds']
            imp = (100 / (abs(o) + 100)) # Simple implied for neg odds
            ledger_rows.append({
                'Run Date': short_date,
                'Timestamp_ET': timestamp,
                'Book': 'FanDuel',
                'Player': row['player'],
                'Team': row.get('team', 'UNK'),
                'Market': row['target'],
                'Side': row['side'],
                'Line': row['line'],
                'Odds': o,
                'Implied Prob': round(imp, 4),
                'Model Prob': round(row['win_prob'], 4),
                'Stake Size': round(row['stake_amt'], 2),
                'EV': round(row['ev'], 4),
                'Outcome': None
            })

        # J.9: Appending parlay stakes
        all_parlays = parlay_results['rr'] + parlay_results['traditional']
        for p in all_parlays:
            o = p['combined_odds']
            imp = (100 / (o + 100)) if o > 0 else (abs(o) / (abs(o) + 100))
            ledger_rows.append({
                'Run Date': short_date,
                'Timestamp_ET': timestamp,
                'Book': 'FanDuel',
                'Player': f"Parlay ({p['type']})", 
                'Team': 'MULTI',
                'Market': 'Parlay', 
                'Side': " | ".join(p['legs']),
                'Line': p['size'],
                'Odds': o, 
                'Implied Prob': round(imp, 4), 
                'Model Prob': round(p['combined_prob'], 4), 
                'Stake Size': round(p['stake_amt'], 2), 
                'EV': round(p['ev'], 4), 
                'Outcome': None
            })
        
        if ledger_rows:
            new_ledger_df = pd.DataFrame(ledger_rows)
            if os.path.exists(LEDGER_PATH):
                new_ledger_df.to_csv(LEDGER_PATH, mode='a', header=False, index=False)
            else:
                new_ledger_df.to_csv(LEDGER_PATH, mode='w', header=True, index=False)
        else:
            print("[INFO] No parlay stakes to log today.")
    
    print(f"[SUCCESS] Phase J.9 Complete. Capital concentrated in Parlays.")

if __name__ == "__main__":
    run()
