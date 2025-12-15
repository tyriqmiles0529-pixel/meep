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
from player_status_filter import filter_active_players_local
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
    
    # Filter out injured/inactive players
    print("\n[4.5] Filtering inactive players...")
    all_players = df['player_name'].unique().tolist()
    player_status = filter_active_players_local(all_players, lookback_days=7)
    
    inactive_players = [p for p, status in player_status.items() if not status['active']]
    if inactive_players:
        print(f"Removed {len(inactive_players)} inactive players:")
        for player in inactive_players[:5]:  # Show first 5
            print(f"  ✗ {player}: {player_status[player]['reason']}")
        if len(inactive_players) > 5:
            print(f"  ... and {len(inactive_players) - 5} more")
    
    # Filter dataframe
    df = df[df['player_name'].isin([p for p, s in player_status.items() if s['active']])]
    print(f"Remaining props: {len(df)}")
    
    # 5. Select best bets (EVALUATE BOTH OVER AND UNDER)
    print("\n[5] Selecting bets...")
    league_avg_pts = 113.8
    
    all_bets = []
    for idx, row in df.iterrows():
        opp_pts = row.get('opp_pts_allowed', league_avg_pts)
        matchup_factor = opp_pts / league_avg_pts
        line = row.get('line', 0)
        
        # Market type check
        is_points = 'point' in str(row.get('market', '')).lower()
        is_rebs = 'rebound' in str(row.get('market', '')).lower()
        is_3pm = '3' in str(row.get('market', '')).lower() and 'point' in str(row.get('market', '')).lower()
        
        if line > 0.5 and (is_points or is_rebs or is_3pm):
            over_odds = row.get('over_odds', 1.91)
            under_odds = row.get('under_odds', 1.91)
            
            # EVALUATE OVER
            if matchup_factor > 1.0:  # Favorable for OVER
                est_win_prob_over = 0.50 + (matchup_factor - 1.0) * 0.8
                est_win_prob_over = min(0.75, est_win_prob_over)
                
                b_over = over_odds - 1
                if b_over > 0:
                    kelly_over = ((b_over * est_win_prob_over - (1 - est_win_prob_over)) / b_over) * 0.25
                    kelly_over = max(0, kelly_over)
                else:
                    kelly_over = 0
                
                if kelly_over > 0.01:  # Minimum 1% stake threshold
                    all_bets.append({
                        'player': row.get('player_name'),
                        'market': row.get('market'),
                        'line': line,
                        'side': 'OVER',
                        'odds': over_odds,
                        'opponent': row.get('opponent'),
                        'event_id': row.get('event_id'),
                        'matchup_factor': matchup_factor,
                        'est_win_prob': est_win_prob_over,
                        'stake_pct': kelly_over,
                        'is_blowout': matchup_factor > 1.10
                    })
            
            # EVALUATE UNDER
            if matchup_factor < 1.0:  # Favorable for UNDER (tough defense)
                est_win_prob_under = 0.50 + (1.0 - matchup_factor) * 0.8
                est_win_prob_under = min(0.75, est_win_prob_under)
                
                b_under = under_odds - 1
                if b_under > 0:
                    kelly_under = ((b_under * est_win_prob_under - (1 - est_win_prob_under)) / b_under) * 0.25
                    kelly_under = max(0, kelly_under)
                else:
                    kelly_under = 0
                
                if kelly_under > 0.01:
                    all_bets.append({
                        'player': row.get('player_name'),
                        'market': row.get('market'),
                        'line': line,
                        'side': 'UNDER',
                        'odds': under_odds,
                        'opponent': row.get('opponent'),
                        'event_id': row.get('event_id'),
                        'matchup_factor': matchup_factor,
                        'est_win_prob': est_win_prob_under,
                        'stake_pct': kelly_under,
                        'is_blowout': matchup_factor < 0.90
                    })
    
    # Sort by Kelly Stake (highest value first)
    all_bets = sorted(all_bets, key=lambda x: x['stake_pct'], reverse=True)
    
    # 6. Display picks with GAME DIVERSITY
    print("\n" + "=" * 50)
    print("ALL QUALIFYING PICKS (Sorted by Value)")
    print("=" * 50)
    
    if not all_bets:
        print("No qualifying bets found.")
    else:
        # Apply game diversity: Max 2 picks per game in top 15
        displayed_picks = []
        game_count = {}
        
        for bet in all_bets:
            game_id = bet['event_id']
            if game_count.get(game_id, 0) < 2:  # Max 2 per game
                displayed_picks.append(bet)
                game_count[game_id] = game_count.get(game_id, 0) + 1
            
            if len(displayed_picks) >= 15:
                break
        
        for i, bet in enumerate(displayed_picks, 1):
            blowout_tag = " [ALT LINE ALERT]" if bet['is_blowout'] else ""
            print(f"{i}. {bet['player']} {bet['market']} {bet['line']} {bet['side']} @ {bet['odds']:.2f}{blowout_tag}")
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
            combined_odds = combo[0]['odds'] * combo[1]['odds']
            print(f"  Parlay {i}: {combo[0]['player']} {combo[0]['side']} + {combo[1]['player']} {combo[1]['side']} @ {combined_odds:.2f}")
    
    if len(parlay_legs) >= 3:
        print("\n3-LEG PARLAY:")
        combo = parlay_legs[:3]
        combined_odds = combo[0]['odds'] * combo[1]['odds'] * combo[2]['odds']
        players = " + ".join([f"{b['player']} {b['side']}" for b in combo])
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
