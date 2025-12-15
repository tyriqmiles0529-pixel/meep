"""
BalldontLie API - Free NBA Stats API
No API key required, reliable injury/status data
"""
import requests
from datetime import datetime, timedelta

def get_active_players_balldontlie():
    """
    Fetches list of active NBA players from BalldontLie API.
    Returns set of active player names.
    """
    url = "https://www.balldontlie.io/api/v1/players?per_page=100"
    
    active_players = set()
    page = 1
    
    try:
        while page <= 5:  # Max 5 pages (500 players total, covers entire NBA)
            response = requests.get(f"{url}&page={page}", timeout=10)
            response.raise_for_status()
            data = response.json()
            
            players = data.get('data', [])
            if not players:
                break
            
            for player in players:
                # Combine first and last name
                first_name = player.get('first_name', '')
                last_name = player.get('last_name', '')
                full_name = f"{first_name} {last_name}".strip()
                
                if full_name:
                    active_players.add(full_name)
            
            # Check if there are more pages
            meta = data.get('meta', {})
            if page >= meta.get('total_pages', 1):
                break
                
            page += 1
            
    except Exception as e:
        print(f"Error fetching from BalldontLie: {e}")
        return set()
    
    return active_players


def get_recent_game_logs(days=3):
    """
    Fetches recent game logs to see who actually played.
    Returns dict: {player_name: games_played_count}
    """
    url = "https://www.balldontlie.io/api/v1/stats"
    
    # Get date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Format dates for API
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    player_games = {}
    page = 1
    
    try:
        while page <= 10:  # Limit to 10 pages
            params = {
                'start_date': start_str,
                'end_date': end_str,
                'per_page': 100,
                'page': page
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            stats = data.get('data', [])
            if not stats:
                break
            
            for stat in stats:
                player_info = stat.get('player', {})
                first_name = player_info.get('first_name', '')
                last_name = player_info.get('last_name', '')
                full_name = f"{first_name} {last_name}".strip()
                
                if full_name:
                    player_games[full_name] = player_games.get(full_name, 0) + 1
            
            # Check pagination
            meta = data.get('meta', {})
            if page >= meta.get('total_pages', 1):
                break
                
            page += 1
            
    except Exception as e:
        print(f"Error fetching game logs: {e}")
        return {}
    
    return player_games


def filter_active_players_balldontlie(player_list, min_games=1):
    """
    Filters players based on recent game activity from BalldontLie API.
    
    Args:
        player_list: List of player names to check
        min_games: Minimum games in last 3 days to be considered active
    
    Returns:
        dict: {player_name: {'active': True/False, 'reason': str, 'games': int}}
    """
    print("Fetching recent NBA game logs from BalldontLie API...")
    recent_games = get_recent_game_logs(days=3)
    
    results = {}
    
    for player in player_list:
        games_played = recent_games.get(player, 0)
        
        if games_played >= min_games:
            results[player] = {
                'active': True,
                'reason': f'Played {games_played} game(s) in last 3 days',
                'games': games_played
            }
        else:
            results[player] = {
                'active': False,
                'reason': f'No games in last 3 days (likely injured/inactive)',
                'games': 0
            }
    
    return results


if __name__ == "__main__":
    # Test
    print("Testing BalldontLie API...")
    
    test_players = ["LeBron James", "Tyrese Haliburton", "Stephen Curry", "Giannis Antetokounmpo"]
    
    results = filter_active_players_balldontlie(test_players)
    
    print("\nResults:")
    for player, status in results.items():
        icon = '✓' if status['active'] else '✗'
        print(f"{icon} {player}: {status['reason']}")
