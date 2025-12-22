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
    """Fetch player prop odds for a specific event from The Odds API."""
    markets = "player_points,player_rebounds,player_assists,player_threes"
    url = f"https://{THE_ODDS_API_HOST}/v4/sports/basketball_nba/events/{event_id}/odds?apiKey={THE_ODDS_API_KEY}&regions=us&markets={markets}&oddsFormat=american"
    
    all_props = []
    market_map = {
        'player_points': 'points',
        'player_rebounds': 'rebounds',
        'player_assists': 'assists',
        'player_threes': 'three_pointers'
    }

    try:
        time.sleep(0.2) # Light rate limiting
        response = requests.get(url)
        if response.status_code != 200:
            print(f"    [WARN] Props failed for event {event_id}: {response.status_code}")
            return []
            
        data = response.json()
        bookmakers = data.get('bookmakers', [])
        
        # We'll take the first available bookmaker that has props (usually DraftKings or FanDuel)
        # Or we could find the 'best' odds. For simplicity, we'll take DraftKings if available, else first.
        selected_book = None
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
                if over and over.get('point'):
                    all_props.append({
                        'player_name': p_name,
                        'event_id': event_id,
                        'market': m_name,
                        'line': over['point'],
                        'odds': over['price'], # American price
                        'bookmaker': selected_book['key']
                    })
    except Exception as e:
        print(f"    [WARN] Exception for event {event_id}: {e}")
        
    return all_props

def fetch_live_odds_pivoted():
    """Fetch live odds and pivot into strategy format."""
    events = get_live_events()
    if not events:
        print("  [ERROR] No live events found.")
        return pd.DataFrame()

    print(f"  Found {len(events)} games. Fetching props...")
    raw_props = []
    for event in events:
        eid = event['id']
        home = event['home_team']
        away = event['away_team']
        print(f"    Processing {away} @ {home}...")
        raw_props.extend(get_live_player_odds(eid))

    if not raw_props:
        return pd.DataFrame()

    # Pivot logic
    pivoted_data = []
    df_raw = pd.DataFrame(raw_props)
    for player in df_raw['player_name'].unique():
        p_rows = df_raw[df_raw['player_name'] == player]
        first_row = p_rows.iloc[0]
        row_dict = {
            'player_name': player, 
            'gameId': first_row.get('event_id', '')
        }
        for _, r in p_rows.iterrows():
            m = r['market']
            row_dict[f'line_{m}'] = r['line']
            row_dict[f'odds_{m}'] = int(r['odds'])
        pivoted_data.append(row_dict)
    
    return pd.DataFrame(pivoted_data)

def run():
    print(f"--- Phase I: Production Readiness ({ODDS_MODE.upper()} MODE) ---")
    print(f"Main Provider: The Odds API")
    
    # 1. Load Predictions
    if not os.path.exists(PREDICTIONS_PATH):
        print(f"[FAIL] Predictions missing: {PREDICTIONS_PATH}")
        return
        
    print(f"Loading predictions from {PREDICTIONS_PATH}...")
    preds = pd.read_csv(PREDICTIONS_PATH)
    col_map = {
        'PLAYER_NAME': 'player_name', 
        'pred_PTS': 'pred_points', 
        'pred_AST': 'pred_assists', 
        'pred_REB': 'pred_rebounds', 
        'pred_3PM': 'pred_three_pointers'
    }
    preds = preds.rename(columns=col_map)

    # 2. Get Odds
    if ODDS_MODE == "live":
        print(f"Fetching LIVE odds...")
        odds_pivoted = fetch_live_odds_pivoted()
        odds_source = "The Odds API Live"
    else:
        print(f"Using SIMULATED odds from {SIM_ODDS_PATH}...")
        if not os.path.exists(SIM_ODDS_PATH): 
            print("[FAIL] Sim odds missing.")
            return
            
        sim_raw = pd.read_csv(SIM_ODDS_PATH)
        market_map = {
            'Points': 'points', 
            'Rebounds': 'rebounds', 
            'Assists': 'assists', 
            'Threes': 'three_pointers', 
            '3-Point Field Goals Made': 'three_pointers'
        }
        sim_raw['clean_market'] = sim_raw['market'].map(market_map)
        sim_raw = sim_raw.dropna(subset=['clean_market'])
        
        pivoted_data = []
        for player in sim_raw['player_name'].unique():
            p_rows = sim_raw[sim_raw['player_name'] == player]
            first_row = p_rows.iloc[0]
            row_dict = {
                'player_name': player, 
                'playerteamName': first_row.get('team', ''), 
                'gameId': first_row.get('event_id', '')
            }
            for _, r in p_rows.iterrows():
                m = r['clean_market']
                dec = float(r.get('over_odds', 1.91))
                if dec >= 2.0:
                    us_odds = (dec - 1) * 100
                elif dec > 1.0:
                    us_odds = -100 / (dec - 1)
                else:
                    us_odds = -110
                row_dict[f'line_{m}'] = float(r.get('line', 0.0))
                row_dict[f'odds_{m}'] = int(us_odds)
            pivoted_data.append(row_dict)
        odds_pivoted = pd.DataFrame(pivoted_data)
        odds_source = f"Simulated ({SIM_ODDS_PATH})"

    if odds_pivoted.empty:
        print("[FAIL] No odds available. Execution halted.")
        return

    # 3. Merge & Bet
    print("Merging predictions and odds...")
    merged = pd.merge(preds, odds_pivoted, on='player_name', how='inner')
    print(f"Matched {len(merged)} players.")
    
    bs = BettingStrategy(load_models=False)
    bets = bs.generate_bets(merged, bankroll=1000.0, confidence_threshold=10, kelly_fraction=0.25, min_ev=0.0)
    
    if bets.empty:
        print("[INFO] No qualifying bets found today.")
        # We'll still generate the report header though
        qualified_bets = 0
    else:
        qualified_bets = len(bets)

    # 4. Generate Outputs
    short_date = datetime.now().strftime('%m.%d.%y')
    md_filename = f"{short_date} - Riq's Picks.md"
    
    top_props_df = bs.select_top_props(bets, n=5) if qualified_bets > 0 else pd.DataFrame()
    rrs = bs.generate_calibrated_round_robins(bets, n_candidates=5) if qualified_bets > 0 else []

    print(f"Generating Report: {md_filename}")
    with open(md_filename, 'w', encoding='utf-8') as f:
        f.write(f"# Riq's Picks — {short_date}\n\n")
        f.write(f"**Odds Source:** {odds_source} | **Timestamp:** {datetime.now().strftime('%H:%M:%S')}\n\n")
        
        # Team List
        f.write("## Teams Playing Today\n")
        live_events = get_live_events() if ODDS_MODE == "live" else []
        if live_events:
            for ev in live_events:
                f.write(f"- {ev['away_team']} @ {ev['home_team']}\n")
        else:
            f.write("See event details below.\n")
        f.write("\n")

        f.write("## Top Bets Per Prop\n")
        f.write("> **Constraint:** All selected players are filtered for >20 minutes expected playing time.\n\n")
        
        if qualified_bets == 0:
            f.write("No qualifying bets met the edge and confidence thresholds today.\n")
        else:
            for target in bs.targets:
                tgt_bets = top_props_df[top_props_df['target'] == target] if not top_props_df.empty else pd.DataFrame()
                if not tgt_bets.empty:
                    f.write(f"### {target.title()}\n")
                    f.write("| Player | Side | Line | Odds | Prob | EV | Stake |\n")
                    f.write("|---|---|---|---|---|---|---|\n")
                    for _, row in tgt_bets.iterrows():
                        odds_str = f"+{row['odds']}" if row['odds'] > 0 else f"{row['odds']}"
                        f.write(f"| {row['player']} ({row.get('team', 'UNK')}) | {row['side']} | {row['line']} | {odds_str} | {row['win_prob']:.1%} | {row['ev']:.2f} | ${row['stake_amt']:.2f} |\n")
                    f.write("\n")
            
            f.write(f"## Round Robin Parlays\n")
            if rrs:
                rr_df = pd.DataFrame(rrs)
                for t in rr_df['type'].unique():
                    f.write(f"### {t}\n")
                    f.write("| Legs | Odds | Prob | EV |\n")
                    f.write("|---|---|---|---|\n")
                    for _, row in rr_df[rr_df['type'] == t].iterrows():
                        legs_str = "<br>".join(row['legs'])
                        odds_str = f"+{row['combined_odds']}" if row['combined_odds'] > 0 else f"{row['combined_odds']}"
                        f.write(f"| {legs_str} | {odds_str} | {row['combined_prob']:.1%} | {row['ev']:.2f} |\n")
                    f.write("\n")
            else:
                f.write("No valid parlay combinations found.\n")

    # 5. Update Ledger
    if qualified_bets > 0:
        print(f"Updating Ledger: {LEDGER_PATH}")
        ledger_rows = []
        for _, row in bets.iterrows():
            o = row['odds']
            if o > 0:
                imp = 100 / (o + 100)
            else:
                imp = abs(o) / (abs(o) + 100)
                
            ledger_rows.append({
                'Run Date': short_date, 
                'Player': row['player'], 
                'Market': row['target'], 
                'Side': row['side'], 
                'Odds': o, 
                'Implied Prob': round(imp, 4), 
                'Model Prob': round(row['win_prob'], 4), 
                'Stake Size': round(row['stake_amt'], 2), 
                'EV': round(row['ev'], 4), 
                'Odds Source': odds_source, 
                'Outcome': None
            })
        
        new_ledger_df = pd.DataFrame(ledger_rows)
        if os.path.exists(LEDGER_PATH):
            new_ledger_df.to_csv(LEDGER_PATH, mode='a', header=False, index=False)
        else:
            new_ledger_df.to_csv(LEDGER_PATH, mode='w', header=True, index=False)
    
    print(f"[SUCCESS] Phase I Complete. Final capital allocation saved.")

if __name__ == "__main__":
    run()
