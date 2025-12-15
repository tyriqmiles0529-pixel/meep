"""
NBA.com Stats API Integration for Player Status and Recent Activity
Free API - No key required
"""
import requests
import json
from datetime import datetime, timedelta

def get_recent_player_stats(days=3):
    """
    Fetches recent game logs for all players to identify who's active.
    Returns dict: {player_name: {'games_played': int, 'avg_minutes': float, 'last_game': date}}
    """
    # NBA Stats API endpoint for recent games
    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Referer': 'https://www.nba.com/',
        'Origin': 'https://www.nba.com'
    }
    
    # Get date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    date_from = start_date.strftime('%m/%d/%Y')
    date_to = end_date.strftime('%m/%d/%Y')
    
    url = f"https://stats.nba.com/stats/leaguegamelog?Counter=1000&DateFrom={date_from}&DateTo={date_to}&Direction=DESC&LeagueID=00&PlayerOrTeam=P&Season=2024-25&SeasonType=Regular+Season&Sorter=DATE"
    
    player_stats = {}
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        headers_list = data['resultSets'][0]['headers']
        rows = data['resultSets'][0]['rowSet']
        
        # Find column indices
        player_name_idx = headers_list.index('PLAYER_NAME')
        min_idx = headers_list.index('MIN')
        game_date_idx = headers_list.index('GAME_DATE')
        
        # Aggregate stats per player
        for row in rows:
            player_name = row[player_name_idx]
            minutes_str = row[min_idx]
            game_date = row[game_date_idx]
            
            # Parse minutes (format: "35:24" -> 35.4)
            if minutes_str and ':' in str(minutes_str):
                mins, secs = str(minutes_str).split(':')
                minutes = float(mins) + float(secs) / 60
            else:
                minutes = 0.0
            
            if player_name not in player_stats:
                player_stats[player_name] = {
                    'games_played': 0,
                    'total_minutes': 0,
                    'last_game': game_date
                }
            
            player_stats[player_name]['games_played'] += 1
            player_stats[player_name]['total_minutes'] += minutes
        
        # Calculate averages
        for player in player_stats:
            games = player_stats[player]['games_played']
            if games > 0:
                player_stats[player]['avg_minutes'] = player_stats[player]['total_minutes'] / games
            else:
                player_stats[player]['avg_minutes'] = 0
                
    except Exception as e:
        print(f"Error fetching player stats: {e}")
        return {}
    
    return player_stats


def filter_active_players(player_list, min_games=1, min_avg_minutes=15.0):
    """
    Filters players based on recent activity.
    
    Args:
        player_list: List of player names
        min_games: Minimum games played in last 3 days (default 1)
        min_avg_minutes: Minimum average minutes (default 15)
    
    Returns:
        dict: {player_name: {'active': True/False, 'reason': str, 'avg_min': float}}
    """
    recent_stats = get_recent_player_stats(days=3)
    
    results = {}
    
    for player in player_list:
        if player not in recent_stats:
            # Player hasn't played in last 3 days
            results[player] = {
                'active': False,
                'reason': 'No games in last 3 days (likely injured/inactive)',
                'avg_min': 0.0
            }
        else:
            stats = recent_stats[player]
            avg_min = stats['avg_minutes']
            games = stats['games_played']
            
            if games < min_games:
                results[player] = {
                    'active': False,
                    'reason': f'Only {games} game(s) in last 3 days',
                    'avg_min': avg_min
                }
            elif avg_min < min_avg_minutes:
                results[player] = {
                    'active': False,
                    'reason': f'Low minutes ({avg_min:.1f} avg)',
                    'avg_min': avg_min
                }
            else:
                results[player] = {
                    'active': True,
                    'reason': f'Active ({avg_min:.1f} min/game)',
                    'avg_min': avg_min
                }
    
    return results


if __name__ == "__main__":
    # Test
    print("Fetching recent NBA player activity...")
    
    test_players = ["LeBron James", "Tyrese Haliburton", "Stephen Curry", "Giannis Antetokounmpo"]
    
    print(f"\nTesting filter on {len(test_players)} players...")
    results = filter_active_players(test_players)
    
    print("\nResults:")
    for player, status in results.items():
        icon = '✓' if status['active'] else '✗'
        print(f"{icon} {player}: {status['reason']}")

