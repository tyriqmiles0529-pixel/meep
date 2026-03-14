"""
Paper Trading with Live Odds - FINAL VERSION
Uses correct RapidAPI endpoints.
"""
import http.client
import json
import os
from datetime import datetime
from team_stats_integration import load_team_stats, enrich_predictions
from paper_ledger import PaperLedger
import pandas as pd

API_KEY = "9ef7289093msh76adf5ee5bedb5fp15e0d6jsnc2a0d0ed9abe"
HOST = "nba-player-props-odds.p.rapidapi.com"

def get_events(date="2025-12-06"):
    """Fetch events for a specific date."""
    conn = http.client.HTTPSConnection(HOST)
    headers = {'x-rapidapi-key': API_KEY, 'x-rapidapi-host': HOST}
    conn.request("GET", f"/get-events-for-date?date={date}", headers=headers)
    res = conn.getresponse()
    data = json.loads(res.read().decode("utf-8"))
    return data

def get_player_odds(event_id, home_team, away_team):
    """Fetch player odds for a specific event looping valid markets."""
    conn = http.client.HTTPSConnection("nba-player-props-odds.p.rapidapi.com")
    
    headers = {
        'x-rapidapi-key': API_KEY,
        'x-rapidapi-host': "nba-player-props-odds.p.rapidapi.com"
    }

    # 1: Assists, 3: Points, 4: Rebounds, 6: Threes
    market_ids = [1, 3, 4, 6]
    all_props = []

    for mk_id in market_ids:
        try:
            # Request specific market
            conn.request("GET", f"/get-player-odds-for-event?eventId={event_id}&marketId={mk_id}&decimal=true&best=true", headers=headers)
            res = conn.getresponse()
            data = res.read().decode("utf-8")
            
            # Parse response
            props = json.loads(data)
            
            if props and isinstance(props, list):
                for prop in props:
                    player_info = prop.get('player', {})
                    player_name = player_info.get('name', 'Unknown')
                    team = player_info.get('team', '')
                    market = prop.get('market_label', '')
                    
                    # Get over/under odds from selections
                    over_odds = 1.91
                    line = 0.0
                    
                    for selection in prop.get('selections', []):
                        if selection.get('label') == 'Over':
                            books = selection.get('books', [])
                            if books:
                                try:
                                    over_odds = float(books[0].get('line', {}).get('cost', 1.91))
                                    line = float(books[0].get('line', {}).get('line', 0.0))
                                except (ValueError, TypeError):
                                    continue
                    
                    # Determine opponent
                    if team in home_team:
                        opponent = away_team
                    else:
                        opponent = home_team
                        
                    # Add to list
                    all_props.append({
                        'event_id': event_id,
                        'player_name': player_name,
                        'team': team,
                        'opponent': opponent,
                        'market': market,
                        'line': line,
                        'over_odds': over_odds
                    })
        except Exception as e:
            print(f"    Error fetching market {mk_id}: {e}")
            continue

    return all_props

import argparse
import sys

# ... imports ...

# ... API functions ...

def run_paper_trading(target_date):
    print("=" * 50)
    print(f"PAPER TRADING - {target_date}")
    print("=" * 50)
    
    # 1. Load team stats
    print("\n[1] Loading team stats...")
    team_stats = load_team_stats(2025)
    print(f"Loaded stats for {len(team_stats)} teams")
    
    # 2. Get events for target date
    print(f"\n[2] Fetching {target_date} events...")
    events = get_events(target_date) # Uses the dynamic date
    
    if not events:
        print("No events found for this date.")
        return

    print(f"Found {len(events)} games")
    
    for e in events:
        home = e['teams']['home']['city'] + ' ' + e['teams']['home']['name']
        away = e['teams']['away']['city'] + ' ' + e['teams']['away']['name']
        print(f"  ID {e['id']}: {away} @ {home}")
    
    # 3. Fetch player odds for each event
    print("\n[3] Fetching player props...")
    all_props = []
    
    for event in events:
        event_id = event['id']
        home_team = event['teams']['home']['city'] + ' ' + event['teams']['home']['name']
        away_team = event['teams']['away']['city'] + ' ' + event['teams']['away']['name']
        
        try:
            # Call the updated get_player_odds function
            odds_data = get_player_odds(event_id, home_team, away_team)
            
            if odds_data:
                all_props.extend(odds_data)
                print(f"  Event {event_id}: {len(odds_data)} props")
            else:
                print(f"  Event {event_id}: 0 props (no data)")
                
        except Exception as e:
            print(f"  Event {event_id}: Error - {str(e)[:50]}")
    
    print(f"\nTotal props: {len(all_props)}")
    
    if not all_props:
        print("No props available yet.")
        return
    
    # 4. Convert to DataFrame and enrich
    print("\n[4] Enriching and Archiving Data...")
    df = pd.DataFrame(all_props)
    
    # --- ARCHIVE RAW ODDS ---
    archive_path = f"historical_data/odds_{target_date}.csv"
    
    os.makedirs("historical_data", exist_ok=True)
    df.to_csv(archive_path, index=False)
    print(f"[OK] Archived {len(df)} props to {archive_path}")
    
    df = enrich_predictions(df, team_stats)
    
    # 5. Select best bets
    print("\n[5] Selecting bets...")
    league_avg_pts = 113.8
    
    all_bets = []
    for idx, row in df.iterrows():
        opp_pts = row.get('opp_pts_allowed', league_avg_pts)
        matchup_factor = opp_pts / league_avg_pts
        line = row.get('line', 0)
        
        # FILTER: Broaden to include Rebounds/Assists if needed, but sticking to Points for safety first
        # Added Rebounds check: (Market contains 'Rebound'?)
        is_points = 'point' in str(row.get('market', '')).lower()
        is_rebs = 'rebound' in str(row.get('market', '')).lower()
        
        if line > 0.5 and (is_points or is_rebs) and matchup_factor > 1.0:
            # 1. Estimate Win Probability (Heuristic)
            # Base 50% + bonus for bad defense
            est_win_prob = 0.50 + (matchup_factor - 1.0) * 0.8  # 10% diff -> 58% win prob
            est_win_prob = min(0.75, est_win_prob) # Cap at 75%
            
            # 2. Kelly Criterion
            # Odds are decimal (e.g. 1.91) -> b = 0.91
            decimal_odds = row.get('over_odds', 1.91)
            b = decimal_odds - 1
            q = 1 - est_win_prob
            if b > 0:
                kelly_f = (b * est_win_prob - q) / b
            else:
                kelly_f = 0
            
            # Fractional Kelly (safer) - 1/4 Kelly
            stake_pct = max(0, kelly_f * 0.25)
            
            # 3. Alt Line Alert
            is_blowout = matchup_factor > 1.10
            
            all_bets.append({
                'player': row.get('player_name'),
                'market': row.get('market'),
                'line': line,
                'over_odds': decimal_odds,
                'opponent': row.get('opponent'),
                'event_id': row.get('event_id'),
                'matchup_factor': matchup_factor,
                'opp_pts_allowed': opp_pts,
                'est_win_prob': est_win_prob,
                'stake_pct': stake_pct,
                'is_blowout': is_blowout
            })
    
    # Sort by Kelly Stake (highest value first)
    all_bets = sorted(all_bets, key=lambda x: x['stake_pct'], reverse=True)
    
    # 6. Display all picks
    print("\n" + "=" * 50)
    print("ALL QUALIFYING PICKS (Sorted by Value)")
    print("=" * 50)
    
    if not all_bets:
        print("No qualifying bets found.")
    else:
        for i, bet in enumerate(all_bets[:15], 1):
            blowout_tag = " [ALT LINE PREDICTED]" if bet['is_blowout'] else ""
            print(f"{i}. {bet['player']} {bet['market']} {bet['line']} OVER @ {bet['over_odds']:.2f}{blowout_tag}")
            print(f"   vs {bet['opponent']} (Matchup: {bet['matchup_factor']:.2f}x)")
            print(f"   Est. Win: {bet['est_win_prob']:.0%} | Rec. Stake: {bet['stake_pct']:.1%} of Bankroll")
    
    # 7. Build NON-CORRELATED parlays
    print("\n" + "=" * 50)
    print("SUGGESTED PARLAYS (no same-game)")
    print("=" * 50)
    
    # Build 2-leg parlays from different games
    from itertools import combinations
    
    parlay_legs = []
    used_events = set()
    for bet in all_bets:
        if bet['event_id'] not in used_events and len(parlay_legs) < 6:
            parlay_legs.append(bet)
            used_events.add(bet['event_id'])
    
    if len(parlay_legs) >= 2:
        print("\n2-LEG PARLAYS:")
        for i, combo in enumerate(combinations(parlay_legs[:4], 2), 1):
            combined_odds = combo[0]['over_odds'] * combo[1]['over_odds']
            print(f"  Parlay {i}: {combo[0]['player']} + {combo[1]['player']} @ {combined_odds:.2f}")
    
    if len(parlay_legs) >= 3:
        print("\n3-LEG PARLAY:")
        combo = parlay_legs[:3]
        combined_odds = combo[0]['over_odds'] * combo[1]['over_odds'] * combo[2]['over_odds']
        players = " + ".join([b['player'] for b in combo])
        print(f"  {players} @ {combined_odds:.2f}")
    
    # 8. Show bankroll
    print("\n" + "=" * 50)
    ledger = PaperLedger(initial_bankroll=25.0)
    bankroll = ledger.state['bankroll']
    daily_budget = bankroll / 5
    print(f"Bankroll: ${bankroll:.2f} | Daily Budget: ${daily_budget:.2f}")
    print("=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NBA Paper Trading")
    parser.add_argument("--date", type=str, default="2025-12-06", help="Date in YYYY-MM-DD format")
    args = parser.parse_args()
    
    run_paper_trading(args.date)
