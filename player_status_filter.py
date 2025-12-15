"""
Player Activity Filter - Uses Local Historical Data
No external API required - 100% reliable
"""
import pandas as pd
import os
from datetime import datetime, timedelta

def filter_active_players_local(player_list, historical_odds_dir='historical_data', min_avg_minutes=20.0, lookback_days=5):
    """
    Filters players based on recent appearance in our own historical odds data.
    
    Logic: If a player has props offered by bookmakers recently, they're likely active.
    If they haven't appeared in 5+ days, they're likely injured/inactive.
    
    Args:
        player_list: List of player names to check
        historical_odds_dir: Directory with odds_YYYY-MM-DD.csv files
        min_avg_minutes: Minimum average minutes threshold (not used currently, placeholder)
        lookback_days: How many days back to check (default 5)
    
    Returns:
        dict: {player_name: {'active': True/False, 'reason': str, 'last_seen': date}}
    """
    results = {}
    
    # Get list of recent CSV files
    if not os.path.exists(historical_odds_dir):
        print(f"Warning: {historical_odds_dir} not found. Assuming all players active.")
        return {player: {'active': True, 'reason': 'No historical data', 'last_seen': None} for player in player_list}
    
    # Get recent files
    cutoff_date = datetime.now() - timedelta(days=lookback_days)
    recent_files = []
    
    for filename in os.listdir(historical_odds_dir):
        if filename.startswith('odds_') and filename.endswith('.csv'):
            try:
                date_str = filename.replace('odds_', '').replace('.csv', '')
                file_date = datetime.strptime(date_str, '%Y-%m-%d')
                
                if file_date >= cutoff_date:
                    recent_files.append(os.path.join(historical_odds_dir, filename))
            except:
                continue
    
    if not recent_files:
        print(f"Warning: No recent odds files found in {historical_odds_dir}. Assuming all players active.")
        return {player: {'active': True, 'reason': 'No recent data', 'last_seen': None} for player in player_list}
    
    # Load and combine recent data
    player_appearances = {}
    
    for filepath in recent_files:
        try:
            df = pd.read_csv(filepath)
            
            if 'player_name' not in df.columns:
                continue
            
            # Extract date from filename
            filename = os.path.basename(filepath)
            date_str = filename.replace('odds_', '').replace('.csv', '')
            
            for player_name in df['player_name'].unique():
                if player_name not in player_appearances:
                    player_appearances[player_name] = []
                player_appearances[player_name].append(date_str)
                
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue
    
    # Evaluate each player
    for player in player_list:
        if player in player_appearances:
            appearances = player_appearances[player]
            last_seen = max(appearances)
            num_days = len(appearances)
            
            results[player] = {
                'active': True,
                'reason': f'Active (seen {num_days} days in last {lookback_days})',
                'last_seen': last_seen
            }
        else:
            results[player] = {
                'active': False,
                'reason': f'Not seen in last {lookback_days} days (likely injured/inactive)',
                'last_seen': None
            }
    
    return results


if __name__ == "__main__":
    # Test
    print("Testing local player activity filter...")
    
    test_players = ["LeBron James", "Tyrese Haliburton", "Stephen Curry", "Kristaps Porzingis"]
    
    results = filter_active_players_local(test_players)
    
    print("\nResults:")
    for player, status in results.items():
        icon = '✓' if status['active'] else '✗'
        last_seen = status['last_seen'] if status['last_seen'] else 'Never'
        print(f"{icon} {player}: {status['reason']} (Last: {last_seen})")
