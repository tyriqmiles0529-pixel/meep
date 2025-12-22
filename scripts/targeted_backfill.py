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
BOOKS = ["draftkings", "fanduel", "betrivers", "williamhill_us"] # williamhill_us is Caesars
ODDS_FORMAT = "american"

OUTPUT_FILE = "historical_data/the_odds_api_historical.csv"
LOOKUP_PATH = "data/eligibility_lookup.csv"

def load_lookup():
    print(f"Loading eligibility lookup from {LOOKUP_PATH}...")
    df = pd.read_csv(LOOKUP_PATH)
    # Create key: (player_name, game_date_str) -> (avg_min, eligible, player_id)
    # Using a dict for O(1) lookups
    lookup = {}
    for _, row in df.iterrows():
        key = (row['PLAYER_NAME'], row['game_date_str'])
        lookup[key] = {
            'avg_min': row['avg_min'],
            'eligible': bool(row['eligible']),
            'player_id': row['PLAYER_ID']
        }
    return lookup

def load_existing_keys():
    if not os.path.exists(OUTPUT_FILE):
        return set()
    print(f"Loading existing keys from {OUTPUT_FILE} for idempotency...")
    try:
        df = pd.read_csv(OUTPUT_FILE, usecols=['player_id', 'game_date', 'market', 'book', 'line'])
        # Create a set of tuples for fast containment checking
        keys = set()
        for _, row in df.iterrows():
            keys.add((str(row['player_id']), str(row['game_date']), str(row['market']), str(row['book']), float(row['line'])))
        return keys
    except Exception as e:
        print(f"  Warning: Could not load existing keys: {e}")
        return set()

REQUEST_COUNT = 0

def fetch_events(date_str):
    global REQUEST_COUNT
    url = f"{BASE_URL}/events?apiKey={API_KEY}&date={date_str}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            REQUEST_COUNT += 1
            remaining = response.headers.get('x-requests-remaining', 'unknown')
            print(f"  [Quota] Remaining: {remaining}")
            return response.json()
        return None
    except:
        return None

def fetch_event_odds(event_id, date_str):
    global REQUEST_COUNT
    url = f"{BASE_URL}/events/{event_id}/odds?apiKey={API_KEY}&regions={REGIONS}&markets={MARKETS}&date={date_str}&oddsFormat={ODDS_FORMAT}"
    try:
        time.sleep(0.5) # Rate limiting
        response = requests.get(url)
        if response.status_code == 200:
            REQUEST_COUNT += 1
            return response.json()
        elif response.status_code == 429:
            print("  [429] Rate limit hit. Cooling down...")
            time.sleep(10)
            return None
        return None
    except:
        return None

def run_backfill(start_date_str, end_date_str):
    lookup = load_lookup()
    existing_keys = load_existing_keys()
    
    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    total_processed = 0
    total_skipped_min = 0
    total_new_records = 0
    
    current_dt = start_dt
    while current_dt <= end_dt:
        date_str = current_dt.strftime("%Y-%m-%d")
        ts = date_str + "T12:00:00Z"
        print(f"--- Processing {date_str} ---")
        
        events_resp = fetch_events(ts)
        if not events_resp or 'data' not in events_resp:
            current_dt += timedelta(days=1)
            continue
            
        events = events_resp['data']
        batch_records = []
        
        for event in events:
            eid = event['id']
            # Before calling props, check if we have ANY potentially eligible players in this game
            # This is hard without a roster list, so we proceed to call props once per game.
            
            odds_resp = fetch_event_odds(eid, ts)
            if not odds_resp or 'data' not in odds_resp:
                continue
                
            data = odds_resp['data']
            snapshot_ts = odds_resp.get('timestamp')
            
            for bookmaker in data.get('bookmakers', []):
                b_key = bookmaker['key']
                if b_key not in BOOKS:
                    continue
                    
                for market in bookmaker.get('markets', []):
                    m_key = market['key']
                    for outcome in market.get('outcomes', []):
                        p_name = outcome.get('description')
                        line = outcome.get('point')
                        price = outcome.get('price')
                        
                        total_processed += 1
                        
                        # 1. Eligibility Check
                        l_data = lookup.get((p_name, date_str))
                        if not l_data or not l_data['eligible']:
                            total_skipped_min += 1
                            continue
                        
                        pid = str(l_data['player_id'])
                        
                        # 2. Idempotency Check
                        # (player_id, game_date, market, book, line)
                        key = (pid, date_str, m_key, b_key, float(line))
                        if key in existing_keys:
                            continue
                            
                        # 3. Build Record
                        record = {
                            'game_date': date_str,
                            'player_id': pid,
                            'player_name': p_name,
                            'market': m_key,
                            'line': line,
                            'odds': price,
                            'book': b_key,
                            'market_id': m_key,
                            'timestamp': snapshot_ts,
                            'source': 'historical_backfill',
                            'minutes_avg_at_time': round(l_data['avg_min'], 2)
                        }
                        batch_records.append(record)
                        existing_keys.add(key)
                        total_new_records += 1
        
        if batch_records:
            df = pd.DataFrame(batch_records)
            df.to_csv(OUTPUT_FILE, mode='a', header=not os.path.exists(OUTPUT_FILE), index=False)
            print(f"  Added {len(batch_records)} new records.")
            
        current_dt += timedelta(days=1)

    print("\n--- Backfill Complete ---")
    print(f"Total API Requests used: {REQUEST_COUNT}")
    print(f"Total prop outcomes processed: {total_processed}")
    print(f"Skipped due to minutes filter: {total_skipped_min}")
    print(f"New records saved: {total_new_records}")

if __name__ == "__main__":
    # Nov 2024: Approx 300-400 requests
    run_backfill("2025-12-15", "2025-12-15")
