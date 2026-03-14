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
THE_ODDS_API_KEY = "5648e037727b48a34a679ac87a1d0edb"
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
    # J.11: Re-enable Alternates (FanDuel Specific)
    # User clarification: Alts allowed, but MUST be from FanDuel.
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
        
        # J.10 DIRECTIVE: FanDuel ONLY (Strict)
        # User explicitly requested only FanDuel lines to ensure availability.
        selected_book = None
        for b in bookmakers:
            if b['key'] == 'fanduel':
                selected_book = b
                break
                
        if not selected_book:
            # If FanDuel lines aren't out, we skip.
            return []
            
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
    print(f"--- Phase J : Institution-Grade Execution ({ODDS_MODE.upper()}) ---")
    print(f"Main Provider: The Odds API")
    
    # 1. Load Predictions
    if not os.path.exists(PREDICTIONS_PATH):
        print(f"[FAIL] Predictions missing: {PREDICTIONS_PATH}")
        return
        
    print(f"Loading predictions from {PREDICTIONS_PATH}...")
    preds = pd.read_csv(PREDICTIONS_PATH)
    
    # Handle Long Format (Long -> Wide Pivot)
    if 'prop_type' in preds.columns and 'prediction' in preds.columns:
        print("    [FORMAT] Detected long-format predictions. Orienting for BettingStrategy...")
        # Pivot the dataframe
        # We include all possible base columns in index
        index_cols = ['player_name', 'team', 'opponent', 'game_date', 'minutes']
        actual_index = [c for c in index_cols if c in preds.columns]
        
        preds_wide = preds.pivot_table(
            index=actual_index, 
            columns='prop_type', 
            values='prediction'
        ).reset_index()
        
        # Add 'pred_' prefix to prop columns
        target_map = {
            'points': 'pred_points',
            'assists': 'pred_assists',
            'rebounds': 'pred_rebounds',
            'three_pointers': 'pred_three_pointers',
            'minutes': 'pred_minutes'
        }
        preds_wide = preds_wide.rename(columns=target_map)
        preds = preds_wide
    else:
        # Legacy Wide Format Support
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
        today_str = datetime.now().strftime("%Y-%m-%d")
        
        if 'game_date_str' in lookup.columns:
            # Robust Live Logic: Get latest eligibility status for each player <= Today
            lookup['game_date'] = pd.to_datetime(lookup['game_date_str'])
            today_dt = pd.to_datetime(today_str)
            
            # Filter for dates in the past/today
            valid_lookup = lookup[lookup['game_date'] <= today_dt].copy()
            
            if not valid_lookup.empty:
                valid_lookup = valid_lookup.sort_values('game_date', ascending=False)
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
        odds_list = []
        odds_source = "Simulated"

    if not odds_list:
        print("[FAIL] No odds available. Execution halted.")
        return

    # 3. Merge & Bet
    print("Merging predictions and odds...")
    odds_df = pd.DataFrame(odds_list)
    merged = pd.merge(preds, odds_df, on='player_name_key', how='inner', suffixes=('', '_odds'))
    
    print(f"Generated {len(merged)} betting candidate lines.")
    print(f"Merged columns: {merged.columns.tolist()}")
    if not merged.empty:
        print(f"Sample row keys: {merged.iloc[0].keys().tolist()}")
    
    bs = BettingStrategy(load_models=False)

    # Generate Initial Bets (Pool)
    # Note: Phase J asks for favorites (-200 to -500). Strategy handles specific filtering.
    print("Calling bs.generate_bets...")
    bets = bs.generate_bets(merged, bankroll=BANKROLL, confidence_threshold=10, kelly_fraction=KELLY_FRACTION, min_ev=0.0)
    
    if bets.empty:
        print("[INFO] No qualifying bets found today.")
        qualified_bets = 0
    else:
        qualified_bets = len(bets)

    # 4. Generate Outputs
    short_date = datetime.now().strftime('%m.%d.%y')
    md_filename = f"{short_date} - Riq's Picks.md"
    
    # Select Top Props (The "Pool")
    top_props_df = bs.select_top_props(bets, n=7) if qualified_bets > 0 else pd.DataFrame()
    
    # Phase S2: Slate Quality Filter
    daily_units = 10.0
    slate_status = "Standard Volume"
    
    if not top_props_df.empty:
        median_prob = top_props_df['win_prob'].median()
        if median_prob < 0.56:
            daily_units = 5.0
            slate_status = "**REDUCED VOLUME (Low Confidence Slate)**"
            print(f"[Phase S2] Slate Filter Triggered: Median Prob {median_prob:.1%} < 56%. Reducing Volume to 50%.")

    # Generate Core Parlays (Tiered)
    parlay_results = bs.generate_optimal_targeted_parlays(
        top_props_df, 
        bankroll=BANKROLL, 
        kelly_fraction=KELLY_FRACTION,
        target_units=daily_units
    ) if not top_props_df.empty else {'rr': [], 'traditional': []}
    
    # Generate Lotto Slips
    lotto_slips = bs.generate_lotto_parlays(top_props_df, n=3) if not top_props_df.empty else []

    print(f"Generating Report: {md_filename}")
    with open(md_filename, 'w', encoding='utf-8') as f:
        f.write(f"# Riq's Picks — {short_date} (Phase S2 - Hardened)\n\n")
        f.write(f"**Strategy:** Survival Mode (Flat Staking) | **Run Time:** {datetime.now().strftime('%H:%M:%S ET')}\n")
        f.write(f"**Allocated:** {daily_units:.2f} Units (Total) | **Status:** {slate_status}\n\n")
        
        # 1. CORE PORTFOLIO (Phase S: Singles)
        f.write("## 1. Core Portfolio (Survival Mode - Singles)\n")
        f.write("Target: 65% Allocation. Flat Sizing. High Hit Rate Focus.\n\n")
        
        singles = parlay_results.get('singles', [])
        traditional = parlay_results.get('traditional', [])
        rrs = parlay_results.get('rr', [])
        
        # Classification for Display
        core_display = singles
        growth_display = [p for p in traditional if len(p['legs']) == 2]
        moonshot_display = [p for p in traditional if len(p['legs']) > 2] + rrs
        
        if core_display:
            f.write("| Tier | Bet | Odds | Prob | Units |\n")
            f.write("|---|---|---|---|---|\n")
            for row in core_display:
                l_str = "<br>".join(row['legs'])
                # Extract odds from leg string if needed, or just show N/A since it's in the string
                f.write(f"| **Single** | {l_str} | - | {row['prob']:.1%} | **{row.get('units', 0):.2f} U** |\n")
            f.write("\n")
        else:
            f.write("No qualifying Core Singles found.\n\n")
            
        # 2. GROWTH (2-Leg Parlays)
        f.write("## 2. Growth Portfolio (2-Legs)\n")
        f.write("Target: 25% Allocation. Controlled Upside.\n\n")
        
        if growth_display:
            f.write("| Tier | Legs | Odds | Prob | Units |\n")
            f.write("|---|---|---|---|---|\n")
            for row in growth_display:
                l_str = "<br>".join(row['legs'])
                o_str = f"+{row['combined_odds']}" if row['combined_odds'] > 0 else f"{row['combined_odds']}"
                f.write(f"| **2-Leg** | {l_str} | {o_str} | {row['prob']:.1%} | **{row.get('units', 0):.2f} U** |\n")
            f.write("\n")
        else:
             f.write("No 2-Leg Parlays found.\n\n")

        # 3. MOONSHOT (3+ Legs & RR)
        f.write("## 3. Moonshot Portfolio (High Variance)\n")
        f.write("Target: 10% Allocation. 3-4 Legs & Hedged RRs.\n\n")
        
        if moonshot_display:
             f.write("| Tier | Legs | Odds | Prob | Units |\n")
             f.write("|---|---|---|---|---|\n")
             for row in moonshot_display:
                l_str = "<br>".join(row['legs'])
                o_str = f"+{row.get('combined_odds', 'RR')}" if row.get('combined_odds', 0) != 0 else "RR"
                f.write(f"| **{row['name']}** | {l_str} | {o_str} | {row['prob']:.1%} | **{row.get('units', 0):.2f} U** |\n")
             f.write("\n")
        else:
             f.write("No Moonshots found.\n\n")

        # SKIP OLD LOGIC
        """
        ptrad = parlay_results['traditional']
        
        # Filter Buckets for Display
        # Add RRs to Core Display
        rr_list = parlay_results.get('rr', [])
        all_core_candidates = ptrad + rr_list
        
        core_display = [p for p in all_core_candidates if p['name'] in ['Core 1', 'Core 2', 'Round Robin (3x2)']]
        growth_display = [p for p in ptrad if p['name'] == 'Growth']
        moonshot_display = [p for p in ptrad if p['name'] in ['Moonshot', 'Hail Mary']]
        
        if core_display:
            f.write("| Tier | Legs | Odds | Prob | Units |\n")
            f.write("|---|---|---|---|---|\n")
            for row in core_display:
                l_str = "<br>".join(row['legs'])
                o_str = f"+{row['combined_odds']}" if row['combined_odds'] > 0 else f"{row['combined_odds']}"
                f.write(f"| **{row['name']}** | {l_str} | {o_str} | {row['prob']:.1%} | **{row.get('units', 0):.2f} U** |\n")
            f.write("\n")
        else:
            f.write("No qualifying Core Parlays found.\n\n")
        """

        # 2. GROWTH (Tier 3)
        f.write("## 2. Growth Portfolio (3-Legs)\n")
        f.write("Balanced Risk/Reward (Target: 15% Allocation).\n\n")
        if growth_display:
            f.write("| Tier | Legs | Odds | Prob | Units |\n")
            f.write("|---|---|---|---|---|\n")
            for row in growth_display:
                l_str = "<br>".join(row['legs'])
                o_str = f"+{row['combined_odds']}" if row['combined_odds'] > 0 else f"{row['combined_odds']}"
                f.write(f"| **{row['name']}** | {l_str} | {o_str} | {row['prob']:.1%} | **{row.get('units', 0):.2f} U** |\n")
            f.write("\n")
        else:
            f.write("No qualifying Growth Parlays found.\n\n")

        # 3. MOONSHOTS
        f.write("## 3. Moonshot Portfolio (High Variance)\n")
        f.write("High odds 4-5 Leg Parlays (Target: 5% Allocation).\n\n")
        if moonshot_display:
            f.write("| Tier | Legs | Odds | Prob | Units |\n")
            f.write("|---|---|---|---|---|\n")
            for row in moonshot_display:
                l_str = "<br>".join(row['legs'])
                o_str = f"+{row['combined_odds']}" if row['combined_odds'] > 0 else f"{row['combined_odds']}"
                f.write(f"| **{row['name']}** | {l_str} | {o_str} | {row['prob']:.1%} | **{row.get('units', 0):.2f} U** |\n")
            f.write("\n")
        else:
            f.write("No qualifying Moonshot Parlays found.\n\n")

        # 2. LOTTO SLIPS
        f.write("## 2. Lotto Slips (Moonshots)\n")
        f.write("High odds (+1000+), small stake, disjoint legs. Fun variance.\n\n")
        if lotto_slips:
            f.write("| Legs | Odds | EV | Prob |\n")
            f.write("|---|---|---|---|\n")
            for slip in lotto_slips:
                 l_str = "<br>".join(slip['legs'])
                 o_str = f"+{slip['combined_odds']}" if slip['combined_odds'] > 0 else f"{slip['combined_odds']}"
                 f.write(f"| {l_str} | {o_str} | {slip['ev']:.2f} | {slip['combined_prob']:.1%} |\n")
            f.write("\n")
        else:
            f.write("No qualifying Lotto Slips found.\n\n")



        # APPENDIX: CANDIDATE POOL
        f.write("## Appendix: Candidate Pool\n")
        f.write("Top 7 props per category used to construct the card.\n\n")
        
        if top_props_df.empty:
            f.write("No qualifying candidates found.\n")
        else:
            for target in bs.targets:
                df_t = top_props_df[top_props_df['target'] == target]
                if not df_t.empty:
                    f.write(f"### {target.replace('_', ' ').title()}\n")
                    f.write("| Player | Target | Side | Line | Odds | Prob | Edge |\n")
                    f.write("|---|---|---|---|---|---|---|\n")
                    for _, row in df_t.iterrows():
                        o_str = f"{row['odds']}"
                        f.write(f"| {row['player']} | {row['target']} | {row['side']} | {row['line']} | {o_str} | {row['win_prob']:.1%} | {row['delta']:.2f} |\n")
                    f.write("\n")

    # 5. Update Ledger (Parlays + Lotto only)
    if qualified_bets > 0:
        print(f"Updating Ledger: {LEDGER_PATH}")
        ledger_rows = []
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        # 0. Singles (Phase S Core)
        import re
        for s in parlay_results.get('singles', []):
             leg_str = s['legs'][0]
             # Parse Odds from string "Player ... (Odds)"
             odds_match = re.search(r'\(([-+]?\d+\.?\d*)\)$', leg_str)
             odds_val = float(odds_match.group(1)) if odds_match else 0
             
             imp = (100 / (odds_val + 100)) if odds_val > 0 else (abs(odds_val) / (abs(odds_val) + 100))
             
             ledger_rows.append({
                'Run Date': short_date,
                'Timestamp_ET': timestamp,
                'Book': 'FanDuel',
                'Player': s['name'].replace('Single: ', ''), 
                'Team': 'N/A', 
                'Market': 'Prop', 
                'Side': leg_str,
                'Line': 1,
                'Odds': odds_val, 
                'Implied Prob': round(imp, 4), 
                'Model Prob': round(s['prob'], 4), 
                'Stake Size': round(s.get('stake_amt', 0), 2), 
                'EV': 0,
                'Outcome': None,
                'Closing_Line': None,
                'Closing_Odds': None,
                'CLV_%': None
             })

        # 0.5 Round Robins
        for p in parlay_results.get('rr', []):
            o = 0 # RR doesn't have a single "Odds"
            ledger_rows.append({
                'Run Date': short_date,
                'Timestamp_ET': timestamp,
                'Book': 'FanDuel',
                'Player': f"RR ({p['name']})", 
                'Team': 'MULTI',
                'Market': 'Round Robin', 
                'Side': " | ".join(p['legs']),
                'Line': len(p['legs']), 
                'Odds': 0, 
                'Implied Prob': 0, 
                'Model Prob': round(p['prob'], 4), 
                'Stake Size': round(p.get('stake_amt', 0), 2), 
                'EV': 0, 
                'Outcome': None,
                'Closing_Line': None,
                'Closing_Odds': None,
                'CLV_%': None
            })

        # 1. Traditional Parlays
        for p in parlay_results.get('traditional', []):
            o = p['combined_odds']
            imp = (100 / (o + 100)) if o > 0 else (abs(o) / (abs(o) + 100))
            ledger_rows.append({
                'Run Date': short_date,
                'Timestamp_ET': timestamp,
                'Book': 'FanDuel',
                'Player': f"Parlay ({p['name']})", 
                'Team': 'MULTI',
                'Market': 'Parlay', 
                'Side': " | ".join(p['legs']),
                'Line': len(p['legs']), # Leg count
                'Odds': o, 
                'Implied Prob': round(imp, 4), 
                'Model Prob': round(p['prob'], 4), 
                'Stake Size': round(p.get('stake_amt', 0), 2), 
                'EV': 0, # Complex to calc for parlay here
                'Outcome': None,
                'Closing_Line': None,
                'Closing_Odds': None,
                'CLV_%': None
            })

        # 2. Lotto Slips 
        for ls in lotto_slips:
             o = ls['combined_odds']
             ledger_rows.append({
                'Run Date': short_date,
                'Timestamp_ET': timestamp,
                'Book': 'FanDuel',
                'Player': f"Lotto Slip", 
                'Team': 'MULTI',
                'Market': 'Parlay', 
                'Side': " | ".join(ls['legs']),
                'Line': len(ls['legs']),
                'Odds': o, 
                'Implied Prob': 0, 
                'Model Prob': round(ls['combined_prob'], 4), 
                'Stake Size': 0.00, # Tracking only
                'EV': round(ls['ev'], 4), 
                'Outcome': None,
                'Closing_Line': None,
                'Closing_Odds': None,
                'CLV_%': None
             })
        
        if ledger_rows:
            new_ledger_df = pd.DataFrame(ledger_rows)
            if os.path.exists(LEDGER_PATH):
                new_ledger_df.to_csv(LEDGER_PATH, mode='a', header=False, index=False)
            else:
                new_ledger_df.to_csv(LEDGER_PATH, mode='w', header=True, index=False)
        else:
            print("[INFO] No parlay stakes to log today.")
    
    print(f"[SUCCESS] Phase J Complete. Report Generated.")

if __name__ == "__main__":
    run()
