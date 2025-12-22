import requests
import pandas as pd
import os
import time
from datetime import datetime, timedelta

# API Config
API_KEY = "feb98d2672d0505df5dbb1cfa8d06ccd"
BASE_URL = "https://api.the-odds-api.com/v4/historical/sports/basketball_nba"
REGIONS = "us"
MARKETS = "player_points,player_rebounds,player_assists,player_threes"
ODDS_FORMAT = "american"
OUTPUT_FILE = "historical_data/the_odds_api_historical.csv"

def fetch_events(date_str):
    """
    Fetch NBA events for a specific snapshot timestamp.
    date_str: ISO8601 format (e.g., 2024-12-10T12:00:00Z)
    """
    url = f"{BASE_URL}/events?apiKey={API_KEY}&date={date_str}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"  [ERROR] Events API failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"  [ERROR] Exception fetching events: {e}")
        return None

def fetch_event_props(event_id, date_str):
    """
    Fetch all prop markets for a single event at a timestamp.
    """
    url = f"{BASE_URL}/events/{event_id}/odds?apiKey={API_KEY}&regions={REGIONS}&markets={MARKETS}&date={date_str}&oddsFormat={ODDS_FORMAT}"
    try:
        # Respect rate limits/quota - basic sleep
        time.sleep(0.5) 
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 422:
            # Often means props aren't available for this event yet
            return None
        else:
            print(f"    [ERROR] Props API failed for {event_id}: {response.status_code}")
            return None
    except Exception as e:
        print(f"    [ERROR] Exception fetching props for {event_id}: {e}")
        return None

def parse_props(resp_json):
    """
    Parse the complex Odds API JSON into a list of simplified records.
    """
    records = []
    if not resp_json or 'data' not in resp_json:
        return records
        
    data = resp_json['data']
    event_id = data.get('id')
    home_team = data.get('home_team')
    away_team = data.get('away_team')
    commence_time = data.get('commence_time')
    snapshot_time = resp_json.get('timestamp')

    # Bookmakers list
    for book in data.get('bookmakers', []):
        book_key = book.get('key')
        # Markets list
        for market in book.get('markets', []):
            m_key = market.get('key')
            # Outcomes list (Over/Under for players)
            for outcome in market.get('outcomes', []):
                player = outcome.get('description')
                side = outcome.get('name')
                price = outcome.get('price')
                line = outcome.get('point')
                
                records.append({
                    'event_id': event_id,
                    'commence_time': commence_time,
                    'snapshot_time': snapshot_time,
                    'home_team': home_team,
                    'away_team': away_team,
                    'bookmaker': book_key,
                    'market': m_key,
                    'player_name': player,
                    'side': side,
                    'line': line,
                    'odds': price
                })
    return records

def run_backfill(start_date_str, days=7):
    """
    Loop through dates and collect data.
    """
    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
    all_records = []
    
    os.makedirs("historical_data", exist_ok=True)

    for i in range(days):
        current_date = start_dt + timedelta(days=i)
        # We target ~12:00 PM UTC, which is morning in the US, when lines are usually sharp/available.
        ts = current_date.strftime("%Y-%m-%dT12:00:00Z")
        print(f"--- Fetching Data for {ts} ---")
        
        events_resp = fetch_events(ts)
        if not events_resp or 'data' not in events_resp:
            print(f"  No events for {ts}")
            continue
            
        events = events_resp['data']
        print(f"  Found {len(events)} events.")
        
        for event in events:
            eid = event['id']
            print(f"    Processing {event['away_team']} @ {event['home_team']} ({eid})...")
            props_resp = fetch_event_props(eid, ts)
            if props_resp:
                recs = parse_props(props_resp)
                print(f"      Extracted {len(recs)} prop outcomes.")
                all_records.extend(recs)
        
        # Save progress every day
        if all_records:
            df = pd.DataFrame(all_records)
            # Append if file exists
            if os.path.exists(OUTPUT_FILE):
                df.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
            else:
                df.to_csv(OUTPUT_FILE, mode='w', header=True, index=False)
            all_records = [] # Reset to avoid duplicate saves
            print(f"  [SUCCESS] Saved data for {ts} to {OUTPUT_FILE}")

if __name__ == "__main__":
    # Example: Fetch first week of Dec 2024
    run_backfill("2024-12-01", days=3)
