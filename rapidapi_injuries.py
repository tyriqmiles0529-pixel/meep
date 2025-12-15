"""
RapidAPI NBA Injuries Integration
Uses the same API key as the odds endpoint
"""
import http.client
import json
from datetime import datetime, timedelta

API_KEY = "9ef7289093msh76adf5ee5bedb5fp15e0d6jsnc2a0d0ed9abe"
HOST = "nba-injuries-reports.p.rapidapi.com"

def get_injuries_for_date(date_str=None):
    """
    Fetches injury report for a specific date.
    
    Args:
        date_str: Date in YYYY-MM-DD format (default: today)
    
    Returns:
        dict: {player_name: {'status': str, 'injury': str, 'team': str}}
    """
    if date_str is None:
        date_str = datetime.now().strftime('%Y-%m-%d')
    
    conn = http.client.HTTPSConnection(HOST)
    headers = {
        'x-rapidapi-key': API_KEY,
        'x-rapidapi-host': HOST
    }
    
    injuries = {}
    
    try:
        conn.request("GET", f"/injuries/nba/{date_str}", headers=headers)
        res = conn.getresponse()
        data = res.read()
        
        injury_data = json.loads(data.decode("utf-8"))
        
        # Parse the response
        # Structure varies, but typically: [{player_name, status, injury_description, team}]
        if isinstance(injury_data, list):
            for entry in injury_data:
                player_name = entry.get('player_name') or entry.get('playerName') or entry.get('name')
                status = entry.get('status') or entry.get('injury_status') or 'OUT'
                injury_desc = entry.get('injury') or entry.get('injury_description') or entry.get('description') or 'Unknown'
                team = entry.get('team') or entry.get('team_name') or 'Unknown'
                
                if player_name:
                    injuries[player_name] = {
                        'status': status.upper(),
                        'injury': injury_desc,
                        'team': team
                    }
        elif isinstance(injury_data, dict):
            # Handle dict response format
            for key, value in injury_data.items():
                if isinstance(value, list):
                    for entry in value:
                        player_name = entry.get('player_name') or entry.get('playerName') or entry.get('name')
                        status = entry.get('status') or entry.get('injury_status') or 'OUT'
                        injury_desc = entry.get('injury') or entry.get('injury_description') or 'Unknown'
                        team = entry.get('team') or entry.get('team_name') or key
                        
                        if player_name:
                            injuries[player_name] = {
                                'status': status.upper(),
                                'injury': injury_desc,
                                'team': team
                            }
                            
    except Exception as e:
        print(f"Error fetching injuries: {e}")
        return {}
    finally:
        conn.close()
    
    return injuries


def filter_active_players_rapidapi(player_list, date_str=None):
    """
    Filters players based on RapidAPI injury report.
    
    Args:
        player_list: List of player names to check
        date_str: Date to check (default: today)
    
    Returns:
        dict: {player_name: {'active': True/False, 'reason': str}}
    """
    injuries = get_injuries_for_date(date_str)
    
    results = {}
    
    for player in player_list:
        if player in injuries:
            injury_info = injuries[player]
            status = injury_info['status']
            
            # OUT, DOUBTFUL = inactive
            if status in ['OUT', 'DOUBTFUL', 'QUESTIONABLE']:
                results[player] = {
                    'active': False,
                    'reason': f"{status}: {injury_info['injury']}"
                }
            else:
                results[player] = {
                    'active': True,
                    'reason': f"Active ({status})"
                }
        else:
            # Not in injury report = active
            results[player] = {
                'active': True,
                'reason': 'Active (not on injury report)'
            }
    
    return results


if __name__ == "__main__":
    # Test
    print("Testing RapidAPI Injury Report...")
    
    # Test with today's date
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"\nFetching injuries for {today}...")
    
    injuries = get_injuries_for_date(today)
    print(f"Found {len(injuries)} injured players")
    
    if injuries:
        print("\nSample injuries:")
        for player, info in list(injuries.items())[:5]:
            print(f"  {player} ({info['team']}): {info['status']} - {info['injury']}")
    
    # Test filter
    print("\n\nTesting filter...")
    test_players = ["LeBron James", "Tyrese Haliburton", "Stephen Curry", "Kristaps Porzingis"]
    
    results = filter_active_players_rapidapi(test_players, today)
    
    print("\nResults:")
    for player, status in results.items():
        icon = '✓' if status['active'] else '✗'
        print(f"{icon} {player}: {status['reason']}")
