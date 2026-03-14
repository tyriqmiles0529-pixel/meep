import requests
import json
import re
import pandas as pd
from datetime import datetime, timedelta
import os

def get_game_urls(date_str):
    """Get all game URLs for a specific date from Yahoo Scoreboard."""
    url = f"https://sports.yahoo.com/nba/scoreboard/?date={date_str}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    
    try:
        r = requests.get(url, headers=headers, timeout=20)
        if r.status_code != 200:
            print(f"Failed to fetch scoreboard for {date_str}: {r.status_code}")
            return []
            
        # Look for game URLs in the HTML
        # Links look like: /nba/cleveland-cavaliers-charlotte-hornets-2026022001/
        matches = re.findall(r'/nba/[a-z0-9\-]+-[0-9]{10}/', r.text)
        urls = [f"https://sports.yahoo.com{m}" for m in set(matches)]
        print(f"Found {len(urls)} games for {date_str}.")
        return urls
    except Exception as e:
        print(f"Error fetching scoreboard: {e}")
        return []

def scrape_game_json(game_url):
    """Extract the player statistics JSON from a Yahoo game page."""
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        r = requests.get(game_url, headers=headers, timeout=20)
        if r.status_code != 200:
            return None
            
        # The data is in a massive JSON blob inside a script tag
        # We look for the "boxscore" key
        text = r.text
        # Find the start of the JSON-like structure
        start_idx = text.find('root.App.main = ')
        if start_idx == -1:
            return None
        
        start_idx += len('root.App.main = ')
        # We need to find the end of this object - usually ends with ;
        end_idx = text.find(';\n', start_idx)
        if end_idx == -1:
            end_idx = text.find('</script>', start_idx)
            
        json_str = text[start_idx:end_idx].strip()
        data = json.loads(json_str)
        return data
    except Exception as e:
        print(f"Error scraping {game_url}: {e}")
        return None

def parse_yahoo_data(data, date_str):
    """Parse the extracted JSON into the format expected by the engine."""
    # Path: context.dispatcher.stores.BoxscoreStore.boxscore
    try:
        store = data.get('context', {}).get('dispatcher', {}).get('stores', {}).get('BoxscoreStore', {})
        boxscore = store.get('boxscore', {})
        if not boxscore:
            return []
            
        game_id = boxscore.get('id')
        teams = boxscore.get('teams', [])
        
        all_logs = []
        for team in teams:
            team_abbr = team.get('abbr')
            opp_abbr = next((t.get('abbr') for t in teams if t.get('abbr') != team_abbr), 'UNK')
            is_home = team.get('isHome', False)
            matchup = f"{team_abbr} {'vs.' if is_home else '@'} {opp_abbr}"
            
            # Players
            players = team.get('players', [])
            for p in players:
                stats = p.get('statistics', {})
                if not stats: continue
                
                # Check for DNP
                if stats.get('dnp') or stats.get('minutes') == '0':
                    continue
                    
                log = {
                    'player_id': p.get('id'), # Yahoo ID - we will remap this
                    'player_name': p.get('displayName'),
                    'team': team_abbr,
                    'gameId': game_id,
                    'date': date_str,
                    'matchup': matchup,
                    'WL': 'W' if team.get('isWinner') else 'L',
                    'minutes': stats.get('minutes'),
                    'points': stats.get('points'),
                    'assists': stats.get('assists'),
                    'reboundsTotal': stats.get('reboundsTotal'),
                    'three_pointers': stats.get('threePointersMade'),
                    'FGA': stats.get('fieldGoalsAttempted'),
                    'FG3A': stats.get('threePointersAttempted'),
                    'TOV': stats.get('turnovers'),
                    'FTA': stats.get('freeThrowsAttempted'),
                    'SEASON_ID': '22025' # Hardcoded for now
                }
                all_logs.append(log)
        return all_logs
    except Exception as e:
        print(f"Error parsing data: {e}")
        return []

def run_fallback_sync(start_date, end_date):
    """Sync data between two dates using web scraping."""
    current_date = start_date
    all_new_logs = []
    
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        print(f"\n>>> Processing {date_str}...")
        
        game_urls = get_game_urls(date_str)
        for url in game_urls:
            print(f"   Scraping {url}...")
            data = scrape_game_json(url)
            if data:
                logs = parse_yahoo_data(data, date_str)
                all_new_logs.extend(logs)
            
        current_date += timedelta(days=1)
        
    if not all_new_logs:
        print("No logs successfully scraped.")
        return
        
    df = pd.DataFrame(all_new_logs)
    
    # REMAP IDs to stay consistent with Master
    # We need the maps we created earlier
    if os.path.exists('player_id_map.json'):
        with open('player_id_map.json', 'r') as f:
            p_map = json.load(f)
            df['player_id'] = df['player_name'].map(p_map).fillna(df['player_id'])
            
    if os.path.exists('team_id_map.json'):
        with open('team_id_map.json', 'r') as f:
            t_map = json.load(f)
            df['TEAM_ID'] = df['team'].map(t_map)

    output_path = f"data/fallback_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    os.makedirs('data', exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n>>> SYNC COMPLETE: Saved {len(df)} logs to {output_path}")
    return output_path

if __name__ == "__main__":
    # Sync from Feb 14 to today
    start = datetime(2026, 2, 14)
    end = datetime.now()
    run_fallback_sync(start, end)
